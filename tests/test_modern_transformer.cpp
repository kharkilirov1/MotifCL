#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <motifcl/motifcl.hpp>

#include "test_utils.hpp"

namespace {

void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if (value && *value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

void require_close_vec(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) std::exit(1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) std::exit(1);
    }
}

} // namespace

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
            constexpr int NB = 1;
            constexpr int QT = 3;
            constexpr int KT = 3;
            constexpr int NH = 4;
            constexpr int NKH = 2;
            constexpr int HD = 64;
            constexpr int QC = NH * HD;
            constexpr int KVC = NKH * HD;
            std::vector<float> qh(NB * QT * QC);
            std::vector<float> kh(NB * KT * KVC);
            std::vector<float> vh(kh.size());
            std::vector<float> goh(qh.size());
            for (std::size_t i = 0; i < qh.size(); ++i) {
                qh[i] = 0.003f * static_cast<float>(static_cast<int>(i % 17) - 8);
                goh[i] = 0.002f * static_cast<float>(static_cast<int>(i % 19) - 9);
            }
            for (std::size_t i = 0; i < kh.size(); ++i) {
                kh[i] = 0.004f * static_cast<float>(static_cast<int>(i % 13) - 6);
                vh[i] = 0.005f * static_cast<float>(static_cast<int>(i % 11) - 5);
            }
            set_test_env("MOTIFCL_ENABLE_GQA_NO_TMP_BWD", "");
            set_test_env("MOTIFCL_DISABLE_GQA_HD64_DQDS_FUSION", "1");
            auto q0 = motifcl::Tensor::from_cpu(backend, {NB * QT, QC}, motifcl::DType::F32, qh.data());
            auto k0 = motifcl::Tensor::from_cpu(backend, {NB * KT, KVC}, motifcl::DType::F32, kh.data());
            auto v0 = motifcl::Tensor::from_cpu(backend, {NB * KT, KVC}, motifcl::DType::F32, vh.data());
            auto go0 = motifcl::Tensor::from_cpu(backend, {NB * QT, QC}, motifcl::DType::F32, goh.data());
            q0.set_requires_grad(true);
            k0.set_requires_grad(true);
            v0.set_requires_grad(true);
            auto y0 = motifcl::grouped_query_attention(q0, k0, v0, NH, NKH, true, NB, QT, KT, 0);
            y0.backward(go0);
            auto ref_q = q0.grad()->to_vector<float>();
            auto ref_k = k0.grad()->to_vector<float>();
            auto ref_v = v0.grad()->to_vector<float>();

            set_test_env("MOTIFCL_DISABLE_GQA_HD64_DQDS_FUSION", "");
            auto qf = motifcl::Tensor::from_cpu(backend, {NB * QT, QC}, motifcl::DType::F32, qh.data());
            auto kf = motifcl::Tensor::from_cpu(backend, {NB * KT, KVC}, motifcl::DType::F32, kh.data());
            auto vf = motifcl::Tensor::from_cpu(backend, {NB * KT, KVC}, motifcl::DType::F32, vh.data());
            auto gof = motifcl::Tensor::from_cpu(backend, {NB * QT, QC}, motifcl::DType::F32, goh.data());
            qf.set_requires_grad(true);
            kf.set_requires_grad(true);
            vf.set_requires_grad(true);
            auto yf = motifcl::grouped_query_attention(qf, kf, vf, NH, NKH, true, NB, QT, KT, 0);
            yf.backward(gof);
            require_close_vec(qf.grad()->to_vector<float>(), ref_q, 2e-5f);
            require_close_vec(kf.grad()->to_vector<float>(), ref_k, 2e-5f);
            require_close_vec(vf.grad()->to_vector<float>(), ref_v, 2e-5f);

            set_test_env("MOTIFCL_ENABLE_GQA_PARTIAL_KV_BWD", "1");
            auto qp = motifcl::Tensor::from_cpu(backend, {NB * QT, QC}, motifcl::DType::F32, qh.data());
            auto kp = motifcl::Tensor::from_cpu(backend, {NB * KT, KVC}, motifcl::DType::F32, kh.data());
            auto vp = motifcl::Tensor::from_cpu(backend, {NB * KT, KVC}, motifcl::DType::F32, vh.data());
            auto gop = motifcl::Tensor::from_cpu(backend, {NB * QT, QC}, motifcl::DType::F32, goh.data());
            qp.set_requires_grad(true);
            kp.set_requires_grad(true);
            vp.set_requires_grad(true);
            auto yp = motifcl::grouped_query_attention(qp, kp, vp, NH, NKH, true, NB, QT, KT, 0);
            yp.backward(gop);
            require_close_vec(qp.grad()->to_vector<float>(), ref_q, 2e-5f);
            require_close_vec(kp.grad()->to_vector<float>(), ref_k, 2e-5f);
            require_close_vec(vp.grad()->to_vector<float>(), ref_v, 2e-5f);
            set_test_env("MOTIFCL_ENABLE_GQA_PARTIAL_KV_BWD", "");

            set_test_env("MOTIFCL_ENABLE_GQA_NO_TMP_BWD", "1");
            auto q1 = motifcl::Tensor::from_cpu(backend, {NB * QT, QC}, motifcl::DType::F32, qh.data());
            auto k1 = motifcl::Tensor::from_cpu(backend, {NB * KT, KVC}, motifcl::DType::F32, kh.data());
            auto v1 = motifcl::Tensor::from_cpu(backend, {NB * KT, KVC}, motifcl::DType::F32, vh.data());
            auto go1 = motifcl::Tensor::from_cpu(backend, {NB * QT, QC}, motifcl::DType::F32, goh.data());
            q1.set_requires_grad(true);
            k1.set_requires_grad(true);
            v1.set_requires_grad(true);
            auto y1 = motifcl::grouped_query_attention(q1, k1, v1, NH, NKH, true, NB, QT, KT, 0);
            y1.backward(go1);
            require_close_vec(q1.grad()->to_vector<float>(), ref_q, 2e-5f);
            require_close_vec(k1.grad()->to_vector<float>(), ref_k, 2e-5f);
            require_close_vec(v1.grad()->to_vector<float>(), ref_v, 2e-5f);
            set_test_env("MOTIFCL_ENABLE_GQA_NO_TMP_BWD", "");
            set_test_env("MOTIFCL_ENABLE_GQA_PARTIAL_KV_BWD", "");
            set_test_env("MOTIFCL_DISABLE_GQA_HD64_DQDS_FUSION", "");
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
