#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/autograd/node.hpp>

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

motifcl::Tensor make_q4_0_tile8_weight(motifcl::Backend& backend,
                                       int rows,
                                       int cols,
                                       const std::vector<float>& values) {
    constexpr int block = 32;
    constexpr int tile_cols = 8;
    if (rows % block != 0 || cols % tile_cols != 0 ||
        static_cast<int>(values.size()) != rows * cols) {
        std::exit(1);
    }
    const int blocks_per_col = rows / block;
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(rows * cols + 1) / 2, 0);
    std::vector<float> scales(static_cast<std::size_t>(cols * blocks_per_col), 1.0f);
    for (int c0 = 0; c0 < cols; c0 += tile_cols) {
        const int tile = c0 / tile_cols;
        for (int kb = 0; kb < blocks_per_col; ++kb) {
            const int begin = kb * block;
            for (int tc = 0; tc < tile_cols; ++tc) {
                const int c = c0 + tc;
                float max_abs = 0.0f;
                for (int kk = 0; kk < block; ++kk) {
                    max_abs = std::max(max_abs, std::fabs(values[(begin + kk) * cols + c]));
                }
                scales[static_cast<std::size_t>(tile * blocks_per_col * tile_cols + kb * tile_cols + tc)] =
                    max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
            }
            for (int kk = 0; kk < block; ++kk) {
                const int r = begin + kk;
                for (int tc = 0; tc < tile_cols; ++tc) {
                    const int c = c0 + tc;
                    const float scale =
                        scales[static_cast<std::size_t>(tile * blocks_per_col * tile_cols + kb * tile_cols + tc)];
                    int q = static_cast<int>(std::lrint(values[r * cols + c] / scale));
                    q = std::max(-7, std::min(7, q));
                    const auto code = static_cast<std::uint8_t>((q + 8) & 0x0f);
                    const int pos = tile * blocks_per_col * block * tile_cols + (kb * block + kk) * tile_cols + tc;
                    if ((pos & 1) == 0) {
                        packed[static_cast<std::size_t>(pos >> 1)] =
                            static_cast<std::uint8_t>((packed[static_cast<std::size_t>(pos >> 1)] & 0xf0u) | code);
                    } else {
                        packed[static_cast<std::size_t>(pos >> 1)] =
                            static_cast<std::uint8_t>((packed[static_cast<std::size_t>(pos >> 1)] & 0x0fu) | (code << 4));
                    }
                }
            }
        }
    }
    auto q = motifcl::Tensor::from_cpu(backend, {rows, cols}, motifcl::DType::Q4_0_COL, packed.data());
    auto s = motifcl::Tensor::from_cpu(backend, {static_cast<int64_t>(scales.size())}, motifcl::DType::F32, scales.data());
    q._set_quant_scales(s, 4, block);
    return q;
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
            constexpr int in = 128;
            constexpr int hidden = 64;
            motifcl::nn::ModernMLP mlp(backend, in, hidden, true, false, 0.0f, true);
            mlp.enable_split_projections(true);
            std::vector<float> x_host(in);
            std::vector<float> gate_w(in * hidden);
            std::vector<float> up_w(in * hidden);
            std::vector<float> down_w(hidden * in);
            for (int i = 0; i < in; ++i) x_host[i] = static_cast<float>((i % 13) - 6) * 0.01f;
            for (int i = 0; i < in * hidden; ++i) {
                gate_w[i] = static_cast<float>((i % 17) - 8) * 0.007f;
                up_w[i] = static_cast<float>((i % 19) - 9) * 0.006f;
            }
            for (int i = 0; i < hidden * in; ++i) down_w[i] = static_cast<float>((i % 23) - 11) * 0.005f;
            mlp.gate_proj().set_quantized_weight(make_q4_0_tile8_weight(backend, in, hidden, gate_w));
            mlp.up_proj().set_quantized_weight(make_q4_0_tile8_weight(backend, in, hidden, up_w));
            mlp.down_proj.weight.data =
                motifcl::Tensor::from_cpu(backend, {hidden, in}, motifcl::DType::F32, down_w.data());
            auto x = motifcl::Tensor::from_cpu(backend, {1, in}, motifcl::DType::F32, x_host.data());
            motifcl::autograd::NoGradGuard no_grad;
            set_test_env("MOTIFCL_DISABLE_FUSED_SWIGLU_PRODUCT_PAIR", "1");
            auto ref = mlp.forward(x).to_vector<float>();
            set_test_env("MOTIFCL_DISABLE_FUSED_SWIGLU_PRODUCT_PAIR", "");
            auto fused = mlp.forward(x).to_vector<float>();
            require_close_vec(fused, ref, 5e-4f);
        }

        {
            motifcl::nn::TransformerConfig ple_cfg = cfg;
            ple_cfg.vocab_size = 24;
            ple_cfg.block_size = 4;
            ple_cfg.n_embd = 16;
            ple_cfg.n_head = 4;
            ple_cfg.n_kv_head = 2;
            ple_cfg.n_layer = 1;
            ple_cfg.mlp_hidden = 32;
            ple_cfg.use_per_layer_inputs = true;
            ple_cfg.per_layer_input_dim = 8;
            ple_cfg.per_layer_input_vocab_size = 24;
            motifcl::nn::ModernGPTModel ple_model(backend, ple_cfg);
            std::int32_t token_value = 3;
            auto token = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::I32, &token_value);
            set_test_env("MOTIFCL_DISABLE_FUSED_PLE_M1", "1");
            set_test_env("MOTIFCL_DISABLE_FUSED_PLE_GATE_M1", "1");
            auto ref_cache = ple_model.create_kv_cache(backend, 1);
            auto ref = ple_model.forward_with_cache(token, ref_cache).to_vector<float>();
            set_test_env("MOTIFCL_DISABLE_FUSED_PLE_M1", "");
            set_test_env("MOTIFCL_DISABLE_FUSED_PLE_GATE_M1", "");
            auto fused_cache = ple_model.create_kv_cache(backend, 1);
            auto fused = ple_model.forward_with_cache(token, fused_cache).to_vector<float>();
            require_close_vec(fused, ref, 3e-4f);

            {
                motifcl::autograd::NoGradGuard no_grad;
                std::int32_t token_values[2] = {3, 4};
                auto token_a = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, token_values);
                auto token_b = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, token_values + 1);

                set_test_env("MOTIFCL_ENABLE_DECODE_BLOCK_GRAPH_REPLAY", "1");
                set_test_env("MOTIFCL_DISABLE_DECODE_BLOCK_GRAPH_REPLAY", "1");
                auto eager_cache = ple_model.create_kv_cache(backend, 1);
                auto eager_a = ple_model.forward_with_cache(token_a, eager_cache).to_vector<float>();
                auto eager_b = ple_model.forward_with_cache(token_b, eager_cache).to_vector<float>();

                set_test_env("MOTIFCL_DISABLE_DECODE_BLOCK_GRAPH_REPLAY", "");
                auto replay_cache = ple_model.create_kv_cache(backend, 1);
                auto replay_a = ple_model.forward_with_cache(token_a, replay_cache).to_vector<float>();
                auto replay_b = ple_model.forward_with_cache(token_b, replay_cache).to_vector<float>();
                require_close_vec(replay_a, eager_a, 3e-4f);
                require_close_vec(replay_b, eager_b, 3e-4f);
                set_test_env("MOTIFCL_ENABLE_DECODE_BLOCK_GRAPH_REPLAY", "");
                set_test_env("MOTIFCL_DISABLE_DECODE_BLOCK_GRAPH_REPLAY", "");
            }
        }

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
            motifcl::nn::ModernSelfAttention attn(backend, cfg);
            auto x1 = motifcl::Tensor::randn(backend, {1, cfg.n_embd}, 0.02f);
            auto x2 = motifcl::Tensor::randn(backend, {1, cfg.n_embd}, 0.02f);
            motifcl::autograd::NoGradGuard no_grad;

            set_test_env("MOTIFCL_DISABLE_FUSED_ROPE_CACHE_APPEND_DECODE", "1");
            motifcl::nn::KVCache ref_cache(backend, 1, cfg.block_size, cfg.n_kv_head, cfg.n_embd / cfg.n_head);
            auto ref1 = attn.forward_with_cache(x1, ref_cache, 1, 1).to_vector<float>();
            auto ref2 = attn.forward_with_cache(x2, ref_cache, 1, 1).to_vector<float>();

            set_test_env("MOTIFCL_DISABLE_FUSED_ROPE_CACHE_APPEND_DECODE", "");
            motifcl::nn::KVCache fused_cache(backend, 1, cfg.block_size, cfg.n_kv_head, cfg.n_embd / cfg.n_head);
            auto fused1 = attn.forward_with_cache(x1, fused_cache, 1, 1).to_vector<float>();
            auto fused2 = attn.forward_with_cache(x2, fused_cache, 1, 1).to_vector<float>();

            require_close_vec(fused1, ref1, 3e-4f);
            require_close_vec(fused2, ref2, 3e-4f);
        }

        {
            motifcl::nn::TransformerConfig qk_cfg = cfg;
            qk_cfg.use_qk_norm = true;
            qk_cfg.use_rope = true;
            motifcl::nn::ModernSelfAttention attn(backend, qk_cfg);
            auto x1 = motifcl::Tensor::randn(backend, {1, qk_cfg.n_embd}, 0.02f);
            auto x2 = motifcl::Tensor::randn(backend, {1, qk_cfg.n_embd}, 0.02f);
            motifcl::autograd::NoGradGuard no_grad;

            set_test_env("MOTIFCL_DISABLE_FUSED_QK_NORM_ROPE_DECODE", "1");
            motifcl::nn::KVCache ref_cache(backend, 1, qk_cfg.block_size, qk_cfg.n_kv_head, qk_cfg.n_embd / qk_cfg.n_head);
            auto ref1 = attn.forward_with_cache(x1, ref_cache, 1, 1).to_vector<float>();
            auto ref2 = attn.forward_with_cache(x2, ref_cache, 1, 1).to_vector<float>();

            set_test_env("MOTIFCL_DISABLE_FUSED_QK_NORM_ROPE_DECODE", "");
            motifcl::nn::KVCache fused_cache(backend, 1, qk_cfg.block_size, qk_cfg.n_kv_head, qk_cfg.n_embd / qk_cfg.n_head);
            auto fused1 = attn.forward_with_cache(x1, fused_cache, 1, 1).to_vector<float>();
            auto fused2 = attn.forward_with_cache(x2, fused_cache, 1, 1).to_vector<float>();

            require_close_vec(fused1, ref1, 3e-4f);
            require_close_vec(fused2, ref2, 3e-4f);
        }

        {
            motifcl::nn::TransformerConfig post_cfg = cfg;
            post_cfg.use_post_attention_norm = true;
            post_cfg.use_post_ffw_norm = true;
            motifcl::nn::ModernTransformerBlock block(backend, post_cfg);
            auto x = motifcl::Tensor::randn(backend, {1, post_cfg.n_embd}, 0.02f);
            motifcl::autograd::NoGradGuard no_grad;

            set_test_env("MOTIFCL_DISABLE_FUSED_RMSNORM_RESIDUAL_ADD", "1");
            motifcl::nn::KVCache ref_cache(backend, 1, post_cfg.block_size, post_cfg.n_kv_head, post_cfg.n_embd / post_cfg.n_head);
            auto ref = block.forward_with_cache(x, ref_cache, 1, 1).to_vector<float>();

            set_test_env("MOTIFCL_DISABLE_FUSED_RMSNORM_RESIDUAL_ADD", "");
            motifcl::nn::KVCache fused_cache(backend, 1, post_cfg.block_size, post_cfg.n_kv_head, post_cfg.n_embd / post_cfg.n_head);
            auto fused = block.forward_with_cache(x, fused_cache, 1, 1).to_vector<float>();
            require_close_vec(fused, ref, 3e-4f);
        }

        {
            motifcl::nn::ModernGPTModel model(backend, cfg);
            auto paged = model.create_paged_kv_cache(backend, 2, 3);
            if (paged.size() != 1 || paged[0].page_size != 3 || paged[0].page_count != 3) return 1;
            if (paged[0].k_pages.shape() != motifcl::Shape({18, 16})) return 1;
            if (paged[0].v_pages.shape() != motifcl::Shape({18, 16})) return 1;
            auto table = paged[0].page_table.to_vector<std::int32_t>();
            if (table != std::vector<std::int32_t>({0, 1, 2, 3, 4, 5})) return 1;
            paged[0].length = 5;
            paged[0].reset();
            if (paged[0].length != 0 || paged[0].capacity() != 9) return 1;

            auto delta = model.create_delta_state_cache(backend, 2, 5);
            if (delta.size() != 1 || delta[0].state.shape() != motifcl::Shape({2, 4, 8, 5})) return 1;
            auto delta_values = delta[0].state.to_vector<float>();
            for (float v : delta_values) {
                if (std::fabs(v) > 1e-8f) return 1;
            }
            delta[0].zero();
            if (delta[0].state.shape() != motifcl::Shape({2, 4, 8, 5})) return 1;
        }

        {
            motifcl::nn::ModernGPTModel model(backend, cfg);
            auto paged = model.create_paged_kv_cache(backend, 1, 2);
            std::vector<std::int32_t> ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
            for (std::size_t i = 0; i < ids.size(); ++i) {
                auto token = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, ids.data() + i);
                auto logits = model.forward_with_cache(token, paged);
                if (logits.shape() != motifcl::Shape({1, 1, cfg.vocab_size})) return 1;
            }
            if (paged[0].tokens_seen != 10 || paged[0].length != cfg.block_size) return 1;
        }

        {
            motifcl::nn::MoEFFN moe(backend, cfg.n_embd, cfg.mlp_hidden, 4, 2);
            auto x = motifcl::Tensor::randn(backend, {3, cfg.n_embd}, 0.02f);
            auto y = moe.forward(x);
            if (y.shape() != motifcl::Shape({3, cfg.n_embd})) return 1;
            for (float v : y.to_vector<float>()) {
                if (!std::isfinite(v)) return 1;
            }

            motifcl::motif::Router router(backend, cfg.n_embd, 4, 2);
            auto probs = router.forward(x);
            if (probs.shape() != motifcl::Shape({3, 4})) return 1;
            auto p = probs.to_vector<float>();
            for (int r = 0; r < 3; ++r) {
                float sum = 0.0f;
                int non_zero = 0;
                for (int c = 0; c < 4; ++c) {
                    const float value = p[static_cast<std::size_t>(r * 4 + c)];
                    sum += value;
                    if (value > 0.0f) ++non_zero;
                }
                if (std::fabs(sum - 1.0f) > 1e-5f || non_zero != 2) return 1;
            }
        }

        {
            motifcl::nn::GatedDeltaNetLayer delta_layer(backend, cfg);
            motifcl::nn::DeltaStateCache state(backend, 1, cfg.n_head, cfg.n_embd / cfg.n_head, cfg.n_embd / cfg.n_head);
            auto x = motifcl::Tensor::randn(backend, {2, cfg.n_embd}, 0.02f);
            auto y = delta_layer.forward_with_state(x, state, 1, 2);
            if (y.shape() != motifcl::Shape({2, cfg.n_embd})) return 1;
            bool any_state = false;
            for (float v : state.state.to_vector<float>()) {
                if (std::fabs(v) > 1e-9f) any_state = true;
            }
            if (!any_state) return 1;

            motifcl::nn::GatedAttentionLayer gated_attn(backend, cfg);
            auto gy = gated_attn.forward(x, 1, 2, true);
            if (gy.shape() != motifcl::Shape({2, cfg.n_embd})) return 1;
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

            {
                motifcl::autograd::NoGradGuard no_grad;
                auto full_caches = model.create_kv_cache(backend, 1);
                auto last_caches = model.create_kv_cache(backend, 1);
                auto full_cached = model.forward_with_cache(tokens, full_caches).view({4, cfg.vocab_size});
                auto last_logits = model.forward_with_cache_last_logits(tokens, last_caches);
                if (last_logits.shape() != motifcl::Shape({1, cfg.vocab_size}) ||
                    full_caches[0].length != 4 || last_caches[0].length != 4) return 1;
                auto full_last = motifcl::slice_rows(full_cached, 3, 4);
                require_close_vec(last_logits.to_vector<float>(), full_last.to_vector<float>(), 1e-4f);
            }

            {
                motifcl::autograd::NoGradGuard no_grad;
                auto paged_full = model.create_paged_kv_cache(backend, 1, 2);
                auto paged_last = model.create_paged_kv_cache(backend, 1, 2);
                auto full_cached = model.forward_with_cache(tokens, paged_full).view({4, cfg.vocab_size});
                auto last_logits = model.forward_with_cache_last_logits(tokens, paged_last);
                if (last_logits.shape() != motifcl::Shape({1, cfg.vocab_size}) ||
                    paged_full[0].tokens_seen != 4 || paged_last[0].tokens_seen != 4) return 1;
                auto full_last = motifcl::slice_rows(full_cached, 3, 4);
                require_close_vec(last_logits.to_vector<float>(), full_last.to_vector<float>(), 1e-4f);
            }

            auto caches = model.create_kv_cache(backend, 1);
            auto step1 = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, ids.data());
            auto step_logits1 = model.forward_with_cache(step1, caches);
            if (step_logits1.shape() != motifcl::Shape({1, 1, cfg.vocab_size}) || caches[0].length != 1) return 1;
            auto step2 = motifcl::Tensor::from_cpu(backend, {1, 1}, motifcl::DType::I32, ids.data() + 1);
            auto step_logits2 = model.decode_step(step2, caches);
            if (step_logits2.shape() != motifcl::Shape({1, cfg.vocab_size}) || caches[0].length != 2) return 1;

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
