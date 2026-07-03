// Regression reproducer for the fog-qkv-split HEAD (29f2702) backward
// training segfault: multi-step training with split qk_head_dim=32 /
// v_head_dim=64 and use_qk_norm=true crashed (SIGSEGV) in
// grouped_query_attention_backward between step ~6 and ~500 on the OpenCL
// path, taking the OpenCL ICD down with it.  Single-step smokes passed, so
// this test runs a bounded multi-step loop.  Passing means the cumulative
// crash is gone in the current tree; crashing here is the regression signal.
//
// Step count defaults to a bounded value and can be raised via
// MOTIFCL_SPLIT_REPRO_STEPS for longer soak runs.
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

namespace {

int env_steps(int fallback) {
    const char* raw = std::getenv("MOTIFCL_SPLIT_REPRO_STEPS");
    if (!raw || !*raw) return fallback;
    const int parsed = std::atoi(raw);
    return parsed > 0 ? parsed : fallback;
}

} // namespace

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        motifcl::manual_seed(20260703);

        motifcl::nn::TransformerConfig cfg;
        cfg.vocab_size = 64;
        cfg.block_size = 16;
        cfg.n_embd = 128;
        cfg.n_head = 4;
        cfg.n_kv_head = 2;
        cfg.n_layer = 2;
        cfg.mlp_hidden = 256;
        cfg.dropout = 0.0f;
        cfg.use_rope = true;
        cfg.use_swiglu = true;
        cfg.use_qkv_bias = false;
        cfg.learned_position_embeddings = false;
        // The crash signature: narrow Q/K, wide V, qk-norm enabled.
        cfg.head_dim = 32;
        cfg.v_head_dim = 64;
        cfg.use_qk_norm = true;

        motifcl::nn::ModernGPTModel model(backend, cfg);
        motifcl::optim::Adam opt(model.parameters(), 3e-4f);

        const int steps = env_steps(120);
        const int tokens = cfg.block_size;
        std::uint64_t rng = 0x9e3779b97f4a7c15ull;
        auto next_token = [&rng](int mod) {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            return static_cast<std::int32_t>(rng % static_cast<std::uint64_t>(mod));
        };

        float last_loss = 0.0f;
        for (int step = 0; step < steps; ++step) {
            std::vector<std::int32_t> ids(tokens);
            std::vector<std::int32_t> targets(tokens);
            for (int t = 0; t < tokens; ++t) {
                ids[t] = next_token(cfg.vocab_size);
                targets[t] = next_token(cfg.vocab_size);
            }
            auto X = motifcl::Tensor::from_cpu(backend, {1, tokens}, motifcl::DType::I32, ids.data());
            auto T = motifcl::Tensor::from_cpu(backend, {tokens}, motifcl::DType::I32, targets.data());

            model.zero_grad();
            auto logits = model.forward(X);
            auto loss = motifcl::softmax_cross_entropy(logits.view({tokens, cfg.vocab_size}), T);
            loss.backward();
            opt.step();

            last_loss = loss.item();
            if (!std::isfinite(last_loss)) {
                std::cerr << "split-qkv train produced non-finite loss at step " << step << "\n";
                return 1;
            }
            if (step % 20 == 0) {
                std::cout << "step " << step << " loss " << last_loss << std::endl;
            }
        }
        std::cout << "split qk=32/v=64 training survived " << steps
                  << " steps, final loss " << last_loss << std::endl;
        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
