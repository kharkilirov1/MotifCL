#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>

namespace {

template <typename Fn>
void run_case(motifcl::Backend& backend, const std::string& name, int iters, Fn&& fn) {
    for (int i = 0; i < 2; ++i) {
        auto out = fn();
        (void)out;
    }
    backend.finish();

    double wall_total = 0.0;
    double kernel_total = 0.0;
    backend.profiler.set_enabled(true);
    for (int i = 0; i < iters; ++i) {
        backend.profiler.clear();
        const auto t0 = std::chrono::steady_clock::now();
        auto out = fn();
        (void)out;
        backend.finish();
        const auto t1 = std::chrono::steady_clock::now();
        wall_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        for (const auto& item : backend.profiler.summary()) kernel_total += item.total_ms;
    }
    backend.profiler.set_enabled(false);

    std::cout << std::left << std::setw(34) << name
              << std::right << " wall_ms=" << std::setw(10) << (wall_total / static_cast<double>(iters))
              << " kernel_ms=" << std::setw(10) << (kernel_total / static_cast<double>(iters))
              << "\n";
}

std::size_t count_params(const std::vector<motifcl::nn::Parameter*>& params) {
    std::size_t total = 0;
    for (auto* p : params) {
        if (p) total += static_cast<std::size_t>(p->data.numel());
    }
    return total;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int iters = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5;
        auto backend = motifcl::Backend::create_opencl();
        const auto info = backend.device_info();
        std::cout << "device=" << info.device_name << " driver=" << info.driver_version
                  << " fp16=" << (motifcl::backend_supports_fp16(backend) ? "yes" : "no")
                  << " iters=" << iters << "\n";

        constexpr int batch = 1;
        constexpr int seq = 128;
        constexpr int vocab = 256;
        constexpr int n_embd = 512;
        constexpr int n_head = 8;
        constexpr int n_kv_head = 2;
        constexpr int mlp_hidden = 2304;
        constexpr int n_layer = 3;
        constexpr int head_dim = n_embd / n_head;
        constexpr int kv_dim = n_kv_head * head_dim;

        auto q = motifcl::Tensor::randn(backend, {batch * seq, n_embd}, 0.02f);
        auto k = motifcl::Tensor::randn(backend, {batch * seq, kv_dim}, 0.02f);
        auto v = motifcl::Tensor::randn(backend, {batch * seq, kv_dim}, 0.02f);
        std::vector<std::int32_t> mask_host(seq * seq, 0);
        for (int r = 0; r < seq; ++r) {
            for (int c = r + 1; c < seq; ++c) mask_host[static_cast<std::size_t>(r * seq + c)] = 1;
        }
        auto mask = motifcl::Tensor::from_cpu(backend, {seq, seq}, motifcl::DType::I32, mask_host.data());

        run_case(backend, "gqa_forward", iters, [&]() {
            return motifcl::grouped_query_attention(q, k, v, n_head, n_kv_head, true, batch, seq, seq, 0);
        });
        run_case(backend, "gqa_masked_forward", iters, [&]() {
            return motifcl::grouped_query_attention_masked(q, k, v, mask, n_head, n_kv_head, false, batch, seq, seq, 0, false);
        });

        motifcl::nn::TransformerConfig cfg;
        cfg.vocab_size = vocab;
        cfg.block_size = seq;
        cfg.n_embd = n_embd;
        cfg.n_head = n_head;
        cfg.n_kv_head = n_kv_head;
        cfg.n_layer = n_layer;
        cfg.mlp_hidden = mlp_hidden;
        cfg.dropout = 0.0f;
        cfg.use_rope = true;
        cfg.use_swiglu = true;
        cfg.use_qkv_bias = true;
        cfg.learned_position_embeddings = false;

        motifcl::nn::ModernGPTModel model(backend, cfg);
        std::vector<std::int32_t> token_host(batch * seq);
        for (int i = 0; i < batch * seq; ++i) token_host[static_cast<std::size_t>(i)] = i % vocab;
        auto tokens = motifcl::Tensor::from_cpu(backend, {batch, seq}, motifcl::DType::I32, token_host.data());
        const auto params = model.parameters();
        std::cout << "modern_gpt params=" << count_params(params)
                  << " seq=" << seq
                  << " n_embd=" << n_embd
                  << " n_layer=" << n_layer
                  << " n_head=" << n_head
                  << " n_kv_head=" << n_kv_head << "\n";

        run_case(backend, "modern_gpt_forward", iters, [&]() {
            return model.forward(tokens);
        });
        run_case(backend, "modern_gpt_masked_forward", iters, [&]() {
            return model.forward_masked(tokens, mask);
        });

        motifcl::nn::ModernGPTModel train_model(backend, cfg);
        motifcl::optim::Adam opt(train_model.parameters(), 3e-4f);
        std::vector<std::int32_t> target_host(batch * seq);
        for (int i = 0; i < batch * seq; ++i) target_host[static_cast<std::size_t>(i)] = (i + 1) % vocab;
        auto targets = motifcl::Tensor::from_cpu(backend, {batch * seq}, motifcl::DType::I32, target_host.data());
        const int train_iters = std::min(iters, 3);
        run_case(backend, "modern_gpt_train_step", train_iters, [&]() {
            auto logits = train_model.forward(tokens).view({batch * seq, vocab});
            auto loss = motifcl::softmax_cross_entropy(logits, targets);
            loss.backward();
            opt.step();
            opt.zero_grad();
            return loss;
        });
        motifcl::nn::ModernGPTModel masked_train_model(backend, cfg);
        motifcl::optim::Adam masked_opt(masked_train_model.parameters(), 3e-4f);
        run_case(backend, "modern_gpt_masked_train_step", train_iters, [&]() {
            auto logits = masked_train_model.forward_masked(tokens, mask).view({batch * seq, vocab});
            auto loss = motifcl::softmax_cross_entropy(logits, targets);
            loss.backward();
            masked_opt.step();
            masked_opt.zero_grad();
            return loss;
        });

        std::vector<std::int32_t> one_host = {42};
        auto one_token = motifcl::Tensor::from_cpu(backend, {batch, 1}, motifcl::DType::I32, one_host.data());
        double decode_wall = 0.0;
        for (int i = 0; i < iters; ++i) {
            auto caches = model.create_kv_cache(backend, batch);
            const auto t0 = std::chrono::steady_clock::now();
            for (int t = 0; t < 16; ++t) {
                auto out = model.forward_with_cache(one_token, caches);
                (void)out;
            }
            backend.finish();
            const auto t1 = std::chrono::steady_clock::now();
            decode_wall += std::chrono::duration<double, std::milli>(t1 - t0).count() / 16.0;
        }
        std::cout << std::left << std::setw(34) << "modern_gpt_decode_1tok"
                  << std::right << " wall_ms=" << std::setw(10) << (decode_wall / static_cast<double>(iters))
                  << " token/s=" << (1000.0 / (decode_wall / static_cast<double>(iters)))
                  << "\n";
    } catch (const std::exception& e) {
        std::cerr << "OpenCL unavailable or benchmark failed: " << e.what() << "\n";
        return 1;
    }
}
