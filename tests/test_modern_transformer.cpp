#include <cmath>
#include <cstdint>
#include <vector>

#include <motifcl/motifcl.hpp>

#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();

        {
            std::vector<float> packed_host = {
                1, 2, 3, 4, 10, 11, 20, 21,
                5, 6, 7, 8, 12, 13, 22, 23,
            };
            auto packed = motifcl::Tensor::from_cpu(backend, {2, 8}, motifcl::DType::F32, packed_host.data());
            auto split = motifcl::qkv_split(packed, 4, 2);
            auto q = split.q.to_vector<float>();
            auto k = split.k.to_vector<float>();
            auto v = split.v.to_vector<float>();
            if (q != std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8})) return 1;
            if (k != std::vector<float>({10, 11, 12, 13})) return 1;
            if (v != std::vector<float>({20, 21, 22, 23})) return 1;
        }

        {
            std::vector<float> x_host = {
                1, 2, 3, 4,
                5, 6, 7, 8,
            };
            auto x = motifcl::Tensor::from_cpu(backend, {2, 4}, motifcl::DType::F32, x_host.data());
            auto y = motifcl::rope(x, 2, 1, 2, 10000.0f, 2, 0).to_vector<float>();
            if (std::fabs(y[0] - 1.0f) > 1e-5f || std::fabs(y[1] - 2.0f) > 1e-5f) return 1;
        }

        {
            auto q = motifcl::Tensor::randn(backend, {3, 16}, 0.02f);
            auto k = motifcl::Tensor::randn(backend, {3, 8}, 0.02f);
            auto v = motifcl::Tensor::randn(backend, {3, 8}, 0.02f);
            q.set_requires_grad(true);
            k.set_requires_grad(true);
            v.set_requires_grad(true);
            auto y = motifcl::grouped_query_attention(q, k, v, 4, 2, true, 1, 3, 3, 0);
            if (y.shape() != motifcl::Shape({3, 16})) return 1;
            y.backward(motifcl::Tensor::ones(backend, y.shape()));
            if (!q.grad() || !k.grad() || !v.grad()) return 1;
        }

        {
            std::vector<float> qkv = {
                1, 0,
                0, 1,
            };
            std::vector<float> v_host = {
                10, 0,
                0, 20,
            };
            auto q = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, qkv.data());
            auto k = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, qkv.data());
            auto v = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, v_host.data());
            q.set_requires_grad(true);
            k.set_requires_grad(true);
            v.set_requires_grad(true);
            std::vector<std::int32_t> mask_host = {
                0, 1,
                0, 0,
            };
            auto mask = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::I32, mask_host.data());
            auto y = motifcl::grouped_query_attention_masked(q, k, v, mask, 1, 1, false, 1, 2, 2, 0, false);
            auto y_host = y.to_vector<float>();
            if (std::fabs(y_host[0] - 10.0f) > 1e-5f || std::fabs(y_host[1]) > 1e-5f) return 1;
            y.backward(motifcl::Tensor::ones(backend, y.shape()));
            if (!q.grad() || !k.grad() || !v.grad()) return 1;

            std::vector<float> bias_host = {
                0.0f, -1.0e30f,
                0.0f, 0.0f,
            };
            auto bias = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, bias_host.data());
            auto y_bias = motifcl::grouped_query_attention_masked(q, k, v, bias, 1, 1, false, 1, 2, 2, 0, true).to_vector<float>();
            if (std::fabs(y_bias[0] - 10.0f) > 1e-5f || std::fabs(y_bias[1]) > 1e-5f) return 1;
        }

        {
            std::vector<float> x_host = {1, 2, 3, 4};
            std::vector<std::int32_t> mask_host = {0, 1, 1, 0};
            auto x = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, x_host.data());
            x.set_requires_grad(true);
            auto mask = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::I32, mask_host.data());
            auto y = motifcl::masked_fill(x, mask, -99.0f);
            auto y_host = y.to_vector<float>();
            if (y_host != std::vector<float>({1.0f, -99.0f, -99.0f, 4.0f})) return 1;
            y.backward(motifcl::Tensor::ones(backend, y.shape()));
            auto grad = x.grad()->to_vector<float>();
            if (grad != std::vector<float>({1.0f, 0.0f, 0.0f, 1.0f})) return 1;
        }

        {
            auto x = motifcl::Tensor::ones(backend, {64}, motifcl::DType::F32);
            x.set_requires_grad(true);
            auto y = motifcl::dropout(x, 0.5f, true);
            y.backward(motifcl::Tensor::ones(backend, y.shape()));
            auto y_host = y.to_vector<float>();
            auto grad = x.grad()->to_vector<float>();
            for (std::size_t i = 0; i < y_host.size(); ++i) {
                if (std::fabs(y_host[i] - grad[i]) > 1e-6f) return 1;
                if (!(std::fabs(y_host[i]) < 1e-6f || std::fabs(y_host[i] - 2.0f) < 1e-6f)) return 1;
            }
        }

        motifcl::nn::TransformerConfig cfg;
        cfg.vocab_size = 32;
        cfg.block_size = 8;
        cfg.n_embd = 32;
        cfg.n_head = 4;
        cfg.n_kv_head = 2;
        cfg.n_layer = 1;
        cfg.mlp_hidden = 64;
        cfg.dropout = 0.0f;
        cfg.use_rope = true;
        cfg.use_swiglu = true;
        cfg.use_qkv_bias = true;
        cfg.learned_position_embeddings = false;

        {
            motifcl::nn::ModernSelfAttention attn(backend, cfg);
            motifcl::nn::KVCache cache(backend, 1, cfg.block_size, cfg.n_kv_head, cfg.n_embd / cfg.n_head);
            auto x1 = motifcl::Tensor::randn(backend, {1, cfg.n_embd}, 0.02f);
            auto y1 = attn.forward_with_cache(x1, cache, 1, 1);
            if (cache.length != 1 || y1.shape() != motifcl::Shape({1, cfg.n_embd})) return 1;
            auto x2 = motifcl::Tensor::randn(backend, {1, cfg.n_embd}, 0.02f);
            auto y2 = attn.forward_with_cache(x2, cache, 1, 1);
            if (cache.length != 2 || y2.shape() != motifcl::Shape({1, cfg.n_embd})) return 1;
        }

        {
            motifcl::nn::ModernGPTModel model(backend, cfg);
            std::vector<std::int32_t> ids = {1, 2, 3, 4};
            std::vector<std::int32_t> targets = {2, 3, 4, 5};
            auto tokens = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::I32, ids.data());
            auto target_tensor = motifcl::Tensor::from_cpu(backend, {4}, motifcl::DType::I32, targets.data());
            auto logits = model.forward(tokens);
            if (logits.shape() != motifcl::Shape({1, 4, cfg.vocab_size})) return 1;
            auto loss = motifcl::softmax_cross_entropy(logits.view({4, cfg.vocab_size}), target_tensor);
            loss.backward();
            if (!model.lm_head.grad()) return 1;

            std::vector<std::int32_t> full_mask_host = {
                0, 1, 1, 1,
                0, 0, 1, 1,
                0, 0, 0, 1,
                0, 0, 0, 0,
            };
            auto mask = motifcl::Tensor::from_cpu(backend, {4, 4}, motifcl::DType::I32, full_mask_host.data());
            auto masked_logits = model.forward_masked(tokens, mask);
            if (masked_logits.shape() != motifcl::Shape({1, 4, cfg.vocab_size})) return 1;

            auto caches = model.create_kv_cache(backend, 1);
            auto step1 = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, ids.data());
            auto step_logits1 = model.forward_with_cache(step1, caches);
            if (step_logits1.shape() != motifcl::Shape({1, 1, cfg.vocab_size}) || caches[0].length != 1) return 1;
            auto step2 = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, ids.data() + 1);
            auto step_logits2 = model.forward_with_cache(step2, caches);
            if (step_logits2.shape() != motifcl::Shape({1, 1, cfg.vocab_size}) || caches[0].length != 2) return 1;

            auto masked_caches = model.create_kv_cache(backend, 1);
            std::vector<std::int32_t> cache_mask_host(static_cast<std::size_t>(cfg.block_size), 0);
            auto cache_mask = motifcl::Tensor::from_cpu(backend, {1, 1, cfg.block_size}, motifcl::DType::I32, cache_mask_host.data());
            auto masked_step_logits = model.forward_with_cache_masked(step1, cache_mask, masked_caches);
            if (masked_step_logits.shape() != motifcl::Shape({1, 1, cfg.vocab_size}) || masked_caches[0].length != 1) return 1;
        }
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
