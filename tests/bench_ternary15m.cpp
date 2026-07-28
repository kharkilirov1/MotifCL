// Throughput benchmark: one full training step of a stories15M-shaped
// transformer (dim 288, 6 layers, 6 heads MHA, SwiGLU hidden 768, vocab 32k,
// seq 256) with EVERY attention/FFN linear as CounterStateLinear (ternary
// counter synapses) on Vulkan. Purpose: measure tokens/sec on the local GPU to
// size the ternary15M-from-scratch pretrain run (reference: 655M tokens,
// hard-ternary val loss 1.6074 on 1xL40S).
//
// Notes: RoPE is omitted (negligible FLOPs, not part of the timed question);
// the LM head runs untied as a plain FP32 matmul 288x32000 — same compute as
// the tied head. Everything runs immediate (no command batching) because the
// counter fused update path is exercised through Tensor::backward like the
// memory-native witnesses. env: BENCH_B (batch, default 8), BENCH_STEPS
// (timed steps, default 20).
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>

namespace {

void fill_deterministic(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t s = seed | 1u;
    for (auto& x : v) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        x = scale * (static_cast<float>(static_cast<std::int32_t>(s % 2001) - 1000) / 1000.0f);
    }
}

motifcl::Tensor make_param(motifcl::Backend& backend, std::int64_t rows, std::int64_t cols,
                           std::uint32_t seed, float scale) {
    std::vector<float> host(static_cast<std::size_t>(rows * cols));
    fill_deterministic(host, seed, scale);
    auto t = motifcl::Tensor::from_cpu(backend, {rows, cols}, motifcl::DType::F32, host.data());
    t.set_requires_grad(true);
    return t;
}

int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::atoi(v) : fallback;
}

} // namespace

int main() {
    using namespace motifcl;
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cerr << "no Vulkan device: " << probe.error << "\n";
        return 77;
    }
    try {
        Backend backend = Backend::create_vulkan();
        auto& runtime = backend.vulkan_runtime();
        const bool use_batched = env_int("BENCH_BATCHED", 0) != 0;

        const std::int64_t B = env_int("BENCH_B", 8);
        const int steps = env_int("BENCH_STEPS", 20);
        const int warmup = 5;
        const std::int64_t T = env_int("BENCH_T", 256);
        const std::int64_t dim = 288;
        const int n_head = 6;
        const int n_kv_head = 6;
        const std::int64_t hidden = 768;
        const std::int64_t vocab = env_int("BENCH_VOCAB", 32000);
        const int layer_count = env_int("BENCH_LAYERS", 6);
        const std::int64_t N = B * T;
        const float eps = 1e-5f;
        const float counter_lr = 0.005f;
        const float head_lr = 0.01f;
        const int C = 11;

        std::cout << "bench_ternary15m: B=" << B << " T=" << T
                  << " layers=" << layer_count << " steps=" << steps
                  << " (+" << warmup << " warmup)\n";

        const bool use_fp = env_int("BENCH_FP", 0) != 0;
        struct Block {
            std::unique_ptr<nn::CounterStateLinear> wq, wk, wv, wo, wup, wdown;
            Tensor norm1, norm2;
            std::vector<Tensor> fpw;  // BENCH_FP=1: plain FP32 weights instead
        };
        auto lin = [&](Block& blk, int idx, const Tensor& in) {
            if (idx < (int)blk.fpw.size()) return matmul(in, blk.fpw[(std::size_t)idx]);
            nn::CounterStateLinear* ls[6] = {blk.wq.get(), blk.wk.get(), blk.wv.get(),
                                             blk.wo.get(), blk.wup.get(), blk.wdown.get()};
            return ls[idx]->forward(in);
        };
        std::vector<float> ones(static_cast<std::size_t>(dim), 1.0f);
        std::vector<Block> blocks;
        for (int l = 0; l < layer_count; ++l) {
            Block blk{
                std::make_unique<nn::CounterStateLinear>(backend, dim, dim, C, counter_lr),
                std::make_unique<nn::CounterStateLinear>(backend, dim, dim, C, counter_lr),
                std::make_unique<nn::CounterStateLinear>(backend, dim, dim, C, counter_lr),
                std::make_unique<nn::CounterStateLinear>(backend, dim, dim, C, counter_lr),
                std::make_unique<nn::CounterStateLinear>(backend, dim, hidden * 2, C, counter_lr),
                std::make_unique<nn::CounterStateLinear>(backend, hidden, dim, C, counter_lr),
                Tensor::from_cpu(backend, {dim}, DType::F32, ones.data()),
                Tensor::from_cpu(backend, {dim}, DType::F32, ones.data()),
                {},
            };
            blk.norm1.set_requires_grad(true);
            blk.norm2.set_requires_grad(true);
            if (use_fp) {
                const std::int64_t shapes[6][2] = {{dim, dim}, {dim, dim}, {dim, dim},
                                                   {dim, dim}, {dim, hidden * 2}, {hidden, dim}};
                for (int i = 0; i < 6; ++i)
                    blk.fpw.push_back(make_param(backend, shapes[i][0], shapes[i][1],
                                                 0x100u * (unsigned)l + (unsigned)i, 0.05f));
            }
            blocks.push_back(std::move(blk));
        }
        auto norm_f = Tensor::from_cpu(backend, {dim}, DType::F32, ones.data());
        norm_f.set_requires_grad(true);
        auto Whead = make_param(backend, dim, vocab, 0x77u, 0.02f);

        std::vector<float> x_host(static_cast<std::size_t>(N * dim));
        fill_deterministic(x_host, 0x88u, 0.5f);
        auto X = Tensor::from_cpu(backend, {N, dim}, DType::F32, x_host.data());
        X.set_requires_grad(true);
        std::vector<std::int32_t> target_host(static_cast<std::size_t>(N));
        for (std::int64_t t = 0; t < N; ++t)
            target_host[static_cast<std::size_t>(t)] = static_cast<std::int32_t>((t * 7 + 3) % vocab);
        auto Targets = Tensor::from_cpu(backend, {N}, DType::I32, target_host.data());

        std::vector<Tensor*> fp_params = {&Whead, &norm_f, &X};
        for (auto& blk : blocks) {
            fp_params.push_back(&blk.norm1);
            fp_params.push_back(&blk.norm2);
            for (auto& w : blk.fpw) fp_params.push_back(&w);
        }

        double timed_ms = 0.0;
        float last_loss = 0.0f;
        for (int step = 0; step < warmup + steps; ++step) {
            const auto t0 = std::chrono::steady_clock::now();
            for (auto* p : fp_params) p->zero_grad();
            if (use_batched && !runtime.batch_begin()) {
                std::cerr << "batch_begin failed\n";
                return 1;
            }

            Tensor cur = X;
            for (auto& blk : blocks) {
                auto a = rmsnorm(cur, blk.norm1, eps);
                auto q = lin(blk, 0, a);
                auto k = lin(blk, 1, a);
                auto v = lin(blk, 2, a);
                // Vulkan dispatch selects the general batched forward and its
                // matching causal/batched backward automatically.
                auto attn = grouped_query_attention(q, k, v, n_head, n_kv_head, false,
                                                    static_cast<int>(B), T, T, 0, 0.0f);
                auto o = lin(blk, 3, attn);
                auto h = add(cur, o);
                auto b = rmsnorm(h, blk.norm2, eps);
                auto packed = lin(blk, 4, b);
                auto m = swiglu(packed);
                auto mo = lin(blk, 5, m);
                cur = add(h, mo);
            }
            auto xf = rmsnorm(cur, norm_f, eps);
            auto logits = matmul(xf, Whead);
            auto loss = softmax_cross_entropy(logits, Targets);
            loss.backward();
            for (auto* p : fp_params) {
                if (p->grad()) sgd_update(*p, *p->grad(), head_lr);
            }
            if (use_batched) {
                auto submit = runtime.batch_end();
                if (!submit.success) {
                    std::cerr << "batch submit failed: " << submit.error << "\n";
                    return 1;
                }
            }
            last_loss = loss.item();
            if (!std::isfinite(last_loss)) {
                std::cerr << "non-finite loss at step " << step << "\n";
                return 1;
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (step >= warmup) timed_ms += ms;
            if (step == 0 || step == warmup + steps - 1)
                std::cout << "  step " << step << ": " << ms << " ms, loss " << last_loss << "\n";
        }

        const double ms_per_step = timed_ms / steps;
        const double tok_per_s = double(N) * 1000.0 / ms_per_step;
        const double target_tokens = 655e6;
        std::cout << "ms/step=" << ms_per_step << "  tokens/step=" << N
                  << "  tok/s=" << tok_per_s << "\n";
        std::cout << "ETA for 655M tokens: " << target_tokens / tok_per_s / 3600.0 << " h\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "bench failed: " << e.what() << "\n";
        return 1;
    }
}
