// Slice 4 end-to-end witness: SGD training steps of a small transformer
// block with EVERY tensor allocated via Backend::create_vulkan() — no OpenCL
// context is created. Forward is recorded into one command buffer and
// submitted once per step (batch_begin/batch_end), the backward+optimizer
// pass into a second one (invariant 9). The witness asserts the loss
// decreases on a fixed synthetic batch. Skips cleanly (exit 77) when no
// Vulkan device is present.
//
// Model (bias-free, batch=1, T=16 tokens, n_embd=64):
//   h1 = h0 + Wo * GQA(rmsnorm(h0) @ {Wq,Wk,Wv})   (4 heads / 2 KV heads)
//   h2 = h1 + Wdown * swiglu(rmsnorm(h1) @ Wup)
//   loss = softmax_cross_entropy(h2 @ lm_head, targets)
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>

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

} // namespace

int main() {
    using namespace motifcl;
    // Robust no-device skip: gate on the probe rather than exception-message
    // matching, so "loader present but zero devices" runners also skip.
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cerr << "Skipping Vulkan train step test: "
                  << (probe.error.empty() ? "no Vulkan device" : probe.error) << "\n";
        return 77;
    }
    try {
        Backend backend = Backend::create_vulkan();
        auto& runtime = backend.vulkan_runtime();

        const std::int64_t T = 16;
        const std::int64_t n_embd = 64;
        const int n_head = 4;
        const int n_kv_head = 2;
        const std::int64_t kv_dim = n_embd / n_head * n_kv_head;  // 32
        const std::int64_t hidden = 128;
        const std::int64_t vocab = 32;
        const float eps = 1e-5f;
        const float lr = 0.05f;
        const int steps = 30;

        auto Wq = make_param(backend, n_embd, n_embd, 0x11u, 0.15f);
        auto Wk = make_param(backend, n_embd, kv_dim, 0x22u, 0.15f);
        auto Wv = make_param(backend, n_embd, kv_dim, 0x33u, 0.15f);
        auto Wo = make_param(backend, n_embd, n_embd, 0x44u, 0.15f);
        auto Wup = make_param(backend, n_embd, hidden * 2, 0x55u, 0.15f);
        auto Wdown = make_param(backend, hidden, n_embd, 0x66u, 0.15f);
        auto Whead = make_param(backend, n_embd, vocab, 0x77u, 0.15f);
        std::vector<float> norm_host(static_cast<std::size_t>(n_embd), 1.0f);
        auto Norm1 = Tensor::from_cpu(backend, {n_embd}, DType::F32, norm_host.data());
        auto Norm2 = Tensor::from_cpu(backend, {n_embd}, DType::F32, norm_host.data());
        Norm1.set_requires_grad(true);
        Norm2.set_requires_grad(true);

        std::vector<Tensor*> params = {&Wq, &Wk, &Wv, &Wo, &Wup, &Wdown, &Whead, &Norm1, &Norm2};

        std::vector<float> x_host(static_cast<std::size_t>(T * n_embd));
        fill_deterministic(x_host, 0x88u, 0.5f);
        auto X = Tensor::from_cpu(backend, {T, n_embd}, DType::F32, x_host.data());
        std::vector<std::int32_t> target_host(static_cast<std::size_t>(T));
        for (std::int64_t t = 0; t < T; ++t) target_host[static_cast<std::size_t>(t)] = static_cast<std::int32_t>((t * 7 + 3) % vocab);
        auto Targets = Tensor::from_cpu(backend, {T}, DType::I32, target_host.data());

        // MOTIFCL_TRAIN_STEP_NO_BATCH=1 runs each dispatch immediate
        // (diagnostic); the default batched mode is the invariant-9 witness.
        const char* no_batch_env = std::getenv("MOTIFCL_TRAIN_STEP_NO_BATCH");
        const bool use_batches = !(no_batch_env && *no_batch_env && *no_batch_env != '0');

        std::vector<float> losses;
        losses.reserve(steps);
        for (int step = 0; step < steps; ++step) {
            for (auto* p : params) p->zero_grad();

            // Forward + backward + optimizer: one command buffer, one submit.
            // (Slice J #4 optimization: previously this was two batches with a
            // synchronous loss.item() host stall between them. Folding forward
            // and backward into a single batch removes one full submit+fence
            // round-trip and the device->host staging copy of the scalar loss
            // that previously serialized the GPU between the two passes. The
            // scalar loss is downloaded once after the whole step completes.)
            if (use_batches && !runtime.batch_begin()) {
                std::cerr << "batch_begin failed\n";
                return 1;
            }
            auto a = rmsnorm(X, Norm1, eps);
            auto q = matmul(a, Wq);
            auto k = matmul(a, Wk);
            auto v = matmul(a, Wv);
            auto attn = grouped_query_attention(q, k, v, n_head, n_kv_head, false, 1, T, T, 0, 0.0f);
            auto o = matmul(attn, Wo);
            auto h1 = add(X, o);
            auto b = rmsnorm(h1, Norm2, eps);
            auto packed = matmul(b, Wup);
            auto m = swiglu(packed);
            auto mo = matmul(m, Wdown);
            auto h2 = add(h1, mo);
            auto logits = matmul(h2, Whead);
            auto loss = softmax_cross_entropy(logits, Targets);
            // Backward + SGD inside the same batch — no host stall between
            // forward and backward.
            loss.backward();
            for (auto* p : params) {
                if (!p->grad()) {
                    std::cerr << "missing gradient for a parameter at step " << step << "\n";
                    return 1;
                }
                sgd_update(*p, *p->grad(), lr);
            }
            if (use_batches) {
                auto step_submit = runtime.batch_end();
                if (!step_submit.success) {
                    std::cerr << "train step batch submit failed: " << step_submit.error << "\n";
                    return 1;
                }
            }

            if (step == 0 && std::getenv("MOTIFCL_TRAIN_STEP_DEBUG")) {
                auto dump = [](const char* name, const Tensor& t) {
                    const auto host = t.to_vector<float>();
                    double sum = 0.0;
                    for (float x : host) sum += std::fabs(x);
                    std::cout << name << " |sum|=" << sum << "\n";
                };
                dump("a", a);
                dump("q", q);
                dump("attn", attn);
                dump("h1", h1);
                dump("m", m);
                dump("h2", h2);
                dump("logits", logits);
            }

            // Scalar loss is downloaded once per step, after the entire
            // forward+backward+optimizer batch already submitted. This is the
            // only host stall in the steady-state step, and it is unavoidable
            // (we need the scalar for logging / non-finite guard).
            const float loss_value = loss.item();
            if (!std::isfinite(loss_value)) {
                std::cerr << "non-finite loss at step " << step << "\n";
                return 1;
            }
            losses.push_back(loss_value);
        }

        const float first = losses.front();
        const float last = losses.back();
        float first3 = 0.0f;
        float last3 = 0.0f;
        for (int i = 0; i < 3; ++i) {
            first3 += losses[static_cast<std::size_t>(i)];
            last3 += losses[losses.size() - 1 - static_cast<std::size_t>(i)];
        }
        std::cout << "vulkan train step: loss " << first << " -> " << last << " over " << steps
                  << " steps\n";
        if (!(last3 < first3)) {
            std::cerr << "loss did not decrease: first3 " << first3 / 3.0f << ", last3 " << last3 / 3.0f
                      << "\n";
            return 1;
        }
        if (!(last < first * 0.9f)) {
            std::cerr << "loss decreased less than 10%: " << first << " -> " << last << "\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        const std::string what = e.what();
        if (what.find("Vulkan") != std::string::npos &&
            (what.find("not available") != std::string::npos || what.find("loader") != std::string::npos ||
             what.find("No Vulkan") != std::string::npos || what.find("vkCreate") != std::string::npos ||
             what.find("vkEnumerate") != std::string::npos)) {
            std::cerr << "Skipping Vulkan train step test: " << what << "\n";
            return 77;
        }
        std::cerr << "Vulkan train step test failed: " << what << "\n";
        return 1;
    }
}
