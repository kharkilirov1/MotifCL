#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

namespace {

std::vector<float> reference_attention(const std::vector<float>& q,
                                       const std::vector<float>& k,
                                       const std::vector<float>& v,
                                       int batch,
                                       int tokens,
                                       int channels,
                                       int n_head,
                                       bool causal) {
    std::vector<float> out(static_cast<std::size_t>(batch * tokens * channels), 0.0f);
    const int head_dim = channels / n_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < channels; ++c) {
                const int head = c / head_dim;
                float max_score = -1.0e30f;
                for (int kt = 0; kt < tokens; ++kt) {
                    if (causal && kt > t) continue;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        int qc = head * head_dim + d;
                        score += q[(b * tokens + t) * channels + qc] * k[(b * tokens + kt) * channels + qc];
                    }
                    max_score = std::max(max_score, score * scale);
                }
                float denom = 0.0f;
                float acc = 0.0f;
                for (int kt = 0; kt < tokens; ++kt) {
                    if (causal && kt > t) continue;
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        int qc = head * head_dim + d;
                        score += q[(b * tokens + t) * channels + qc] * k[(b * tokens + kt) * channels + qc];
                    }
                    float prob = std::exp(score * scale - max_score);
                    denom += prob;
                    acc += prob * v[(b * tokens + kt) * channels + c];
                }
                out[(b * tokens + t) * channels + c] = acc / denom;
            }
        }
    }
    return out;
}

float attention_loss(const std::vector<float>& q,
                     const std::vector<float>& k,
                     const std::vector<float>& v,
                     const std::vector<float>& grad_out,
                     int batch,
                     int tokens,
                     int channels,
                     int n_head,
                     bool causal) {
    auto out = reference_attention(q, k, v, batch, tokens, channels, n_head, causal);
    float loss = 0.0f;
    for (std::size_t i = 0; i < out.size(); ++i) loss += out[i] * grad_out[i];
    return loss;
}

std::vector<float> finite_difference_grad(std::vector<float> q,
                                          std::vector<float> k,
                                          std::vector<float> v,
                                          const std::vector<float>& grad_out,
                                          int batch,
                                          int tokens,
                                          int channels,
                                          int n_head,
                                          bool causal,
                                          char target) {
    std::vector<float>* values = target == 'q' ? &q : (target == 'k' ? &k : &v);
    std::vector<float> grad(values->size(), 0.0f);
    constexpr float eps = 1e-3f;
    for (std::size_t i = 0; i < values->size(); ++i) {
        float old = (*values)[i];
        (*values)[i] = old + eps;
        float plus = attention_loss(q, k, v, grad_out, batch, tokens, channels, n_head, causal);
        (*values)[i] = old - eps;
        float minus = attention_loss(q, k, v, grad_out, batch, tokens, channels, n_head, causal);
        (*values)[i] = old;
        grad[i] = (plus - minus) / (2.0f * eps);
    }
    return grad;
}

std::vector<float> reference_grouped_decode(const std::vector<float>& q,
                                            const std::vector<float>& k,
                                            const std::vector<float>& v,
                                            int batch,
                                            int key_tokens,
                                            int key_stride,
                                            int n_head,
                                            int n_kv_head,
                                            int head_dim,
                                            int query_offset,
                                            int sliding_window) {
    const int q_channels = n_head * head_dim;
    const int kv_channels = n_kv_head * head_dim;
    const int group = n_head / n_kv_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> out(static_cast<std::size_t>(batch * q_channels), 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < n_head; ++h) {
            const int kv_head = h / group;
            const int min_key = std::max(0, sliding_window > 0 ? query_offset - sliding_window + 1 : 0);
            const int max_key = std::min(query_offset, key_tokens - 1);
            float max_score = -1.0e30f;
            for (int kt = min_key; kt <= max_key; ++kt) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q[b * q_channels + h * head_dim + d] *
                             k[(b * key_stride + kt) * kv_channels + kv_head * head_dim + d];
                }
                max_score = std::max(max_score, score * scale);
            }
            float denom = 0.0f;
            std::vector<float> probs(static_cast<std::size_t>(std::max(0, max_key - min_key + 1)), 0.0f);
            for (int kt = min_key; kt <= max_key; ++kt) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q[b * q_channels + h * head_dim + d] *
                             k[(b * key_stride + kt) * kv_channels + kv_head * head_dim + d];
                }
                const float p = std::exp(score * scale - max_score);
                probs[static_cast<std::size_t>(kt - min_key)] = p;
                denom += p;
            }
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int kt = min_key; kt <= max_key; ++kt) {
                    acc += probs[static_cast<std::size_t>(kt - min_key)] *
                           v[(b * key_stride + kt) * kv_channels + kv_head * head_dim + d];
                }
                out[b * q_channels + h * head_dim + d] = acc / denom;
            }
        }
    }
    return out;
}

} // namespace

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        constexpr int B = 2;
        constexpr int T = 3;
        constexpr int C = 4;
        constexpr int H = 2;
        std::vector<float> q = {
            1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 0, 0,
        };
        std::vector<float> k = q;
        std::vector<float> v = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12,
        };
        auto Q = motifcl::Tensor::from_cpu(backend, {B * T, C}, motifcl::DType::F32, q.data());
        auto K = motifcl::Tensor::from_cpu(backend, {B * T, C}, motifcl::DType::F32, k.data());
        auto V = motifcl::Tensor::from_cpu(backend, {B * T, C}, motifcl::DType::F32, v.data());
        auto Y = motifcl::multihead_attention(Q, K, V, H, true, B, T).to_vector<float>();
        motifcl::autograd::begin_graph_capture();
        auto captured_attn = motifcl::multihead_attention(Q, K, V, H, true, B, T);
        (void)captured_attn;
        auto graph = motifcl::autograd::end_graph_capture();
        if (backend.device_info().max_work_group_size >= 128 &&
            backend.device_info().local_mem_size >= 24 * 1024 &&
            (graph.empty() || graph.nodes()[0].op != "multihead_attention_flash_f32")) return 1;
        auto ref = reference_attention(q, k, v, B, T, C, H, true);
        for (std::size_t i = 0; i < Y.size(); ++i) {
            if (std::fabs(Y[i] - ref[i]) > 1e-4f) return 1;
        }

        {
            constexpr int GB = 1;
            constexpr int KT = 5;
            constexpr int GH = 4;
            constexpr int GKVH = 2;
            constexpr int HD = 4;
            constexpr int QC = GH * HD;
            constexpr int KVC = GKVH * HD;
            std::vector<float> gq(GB * QC);
            std::vector<float> gk(GB * KT * KVC);
            std::vector<float> gv(GB * KT * KVC);
            for (std::size_t i = 0; i < gq.size(); ++i) {
                gq[i] = 0.031f * static_cast<float>(static_cast<int>(i % 9) - 4);
            }
            for (std::size_t i = 0; i < gk.size(); ++i) {
                gk[i] = 0.017f * static_cast<float>(static_cast<int>(i % 11) - 5);
                gv[i] = 0.023f * static_cast<float>(static_cast<int>(i % 13) - 6);
            }
            auto GQ = motifcl::Tensor::from_cpu(backend, {GB, QC}, motifcl::DType::F32, gq.data());
            auto GK = motifcl::Tensor::from_cpu(backend, {GB * KT, KVC}, motifcl::DType::F32, gk.data());
            auto GV = motifcl::Tensor::from_cpu(backend, {GB * KT, KVC}, motifcl::DType::F32, gv.data());
            auto full = motifcl::grouped_query_attention(GQ, GK, GV, GH, GKVH, true, GB, 1, KT, KT - 1).to_vector<float>();
            auto full_ref = reference_grouped_decode(gq, gk, gv, GB, KT, KT, GH, GKVH, HD, KT - 1, 0);
            auto windowed = motifcl::grouped_query_attention_windowed(GQ, GK, GV, GH, GKVH, 3, true, GB, 1, KT, KT - 1)
                                .to_vector<float>();
            auto windowed_ref = reference_grouped_decode(gq, gk, gv, GB, KT, KT, GH, GKVH, HD, KT - 1, 3);
            for (std::size_t i = 0; i < full.size(); ++i) {
                if (std::fabs(full[i] - full_ref[i]) > 1e-4f) return 1;
                if (std::fabs(windowed[i] - windowed_ref[i]) > 1e-4f) return 1;
            }
        }

        {
            constexpr int GB = 2;
            constexpr int KT = 3;
            constexpr int KSTRIDE = 5;
            constexpr int GH = 4;
            constexpr int GKVH = 2;
            constexpr int HD = 4;
            constexpr int QC = GH * HD;
            constexpr int KVC = GKVH * HD;
            std::vector<float> gq(GB * QC);
            std::vector<float> gk(GB * KSTRIDE * KVC, 99.0f);
            std::vector<float> gv(GB * KSTRIDE * KVC, -99.0f);
            for (std::size_t i = 0; i < gq.size(); ++i) {
                gq[i] = 0.019f * static_cast<float>(static_cast<int>(i % 7) - 3);
            }
            for (int b = 0; b < GB; ++b) {
                for (int kt = 0; kt < KT; ++kt) {
                    for (int c = 0; c < KVC; ++c) {
                        const int idx = (b * KSTRIDE + kt) * KVC + c;
                        gk[static_cast<std::size_t>(idx)] =
                            0.011f * static_cast<float>(((b + 3 * kt + c) % 13) - 6);
                        gv[static_cast<std::size_t>(idx)] =
                            0.013f * static_cast<float>(((2 * b + kt + 5 * c) % 17) - 8);
                    }
                }
            }
            auto GQ = motifcl::Tensor::from_cpu(backend, {GB, QC}, motifcl::DType::F32, gq.data());
            auto GK = motifcl::Tensor::from_cpu(backend, {GB * KSTRIDE, KVC}, motifcl::DType::F32, gk.data());
            auto GV = motifcl::Tensor::from_cpu(backend, {GB * KSTRIDE, KVC}, motifcl::DType::F32, gv.data());
            auto full = motifcl::grouped_query_attention(GQ, GK, GV, GH, GKVH, true, GB, 1, KT, KT - 1).to_vector<float>();
            auto full_ref = reference_grouped_decode(gq, gk, gv, GB, KT, KSTRIDE, GH, GKVH, HD, KT - 1, 0);
            auto windowed = motifcl::grouped_query_attention_windowed(GQ, GK, GV, GH, GKVH, 2, true, GB, 1, KT, KT - 1)
                                .to_vector<float>();
            auto windowed_ref = reference_grouped_decode(gq, gk, gv, GB, KT, KSTRIDE, GH, GKVH, HD, KT - 1, 2);
            for (std::size_t i = 0; i < full.size(); ++i) {
                if (std::fabs(full[i] - full_ref[i]) > 1e-4f) return 1;
                if (std::fabs(windowed[i] - windowed_ref[i]) > 1e-4f) return 1;
            }
        }

        std::vector<float> grad_out = {
            0.1f, -0.2f, 0.3f, -0.4f, 0.2f, 0.1f, -0.1f, 0.5f, -0.3f, 0.4f, 0.2f, -0.2f,
            -0.2f, 0.3f, -0.4f, 0.1f, 0.5f, -0.1f, 0.2f, -0.3f, 0.4f, 0.2f, -0.5f, 0.1f,
        };
        auto GO = motifcl::Tensor::from_cpu(backend, {B * T, C}, motifcl::DType::F32, grad_out.data());
        auto GQ = motifcl::multihead_attention_backward_q(Q, K, V, GO, H, true, B, T).to_vector<float>();
        auto GK = motifcl::multihead_attention_backward_k(Q, K, V, GO, H, true, B, T).to_vector<float>();
        auto GV = motifcl::multihead_attention_backward_v(Q, K, V, GO, H, true, B, T).to_vector<float>();
        auto ref_gq = finite_difference_grad(q, k, v, grad_out, B, T, C, H, true, 'q');
        auto ref_gk = finite_difference_grad(q, k, v, grad_out, B, T, C, H, true, 'k');
        auto ref_gv = finite_difference_grad(q, k, v, grad_out, B, T, C, H, true, 'v');
        for (std::size_t i = 0; i < GQ.size(); ++i) {
            if (std::fabs(GQ[i] - ref_gq[i]) > 3e-2f) return 1;
            if (std::fabs(GK[i] - ref_gk[i]) > 3e-2f) return 1;
            if (std::fabs(GV[i] - ref_gv[i]) > 3e-2f) return 1;
        }

        {
            constexpr int NB = 1;
            constexpr int NT = 4;
            constexpr int NC = 4;
            constexpr int NH = 2;
            std::vector<float> q2(NB * NT * NC);
            std::vector<float> k2(q2.size());
            std::vector<float> v2(q2.size());
            std::vector<float> go2(q2.size());
            for (std::size_t i = 0; i < q2.size(); ++i) {
                q2[i] = 0.03f * static_cast<float>(static_cast<int>(i % 7) - 3);
                k2[i] = 0.02f * static_cast<float>(static_cast<int>(i % 5) - 2);
                v2[i] = 0.04f * static_cast<float>(static_cast<int>(i % 11) - 5);
                go2[i] = 0.01f * static_cast<float>(static_cast<int>(i % 13) - 6);
            }
            auto Q2 = motifcl::Tensor::from_cpu(backend, {NB * NT, NC}, motifcl::DType::F32, q2.data());
            auto K2 = motifcl::Tensor::from_cpu(backend, {NB * NT, NC}, motifcl::DType::F32, k2.data());
            auto V2 = motifcl::Tensor::from_cpu(backend, {NB * NT, NC}, motifcl::DType::F32, v2.data());
            auto GO2 = motifcl::Tensor::from_cpu(backend, {NB * NT, NC}, motifcl::DType::F32, go2.data());
            auto GQ2 = motifcl::multihead_attention_backward_q(Q2, K2, V2, GO2, NH, false, NB, NT).to_vector<float>();
            auto GK2 = motifcl::multihead_attention_backward_k(Q2, K2, V2, GO2, NH, false, NB, NT).to_vector<float>();
            auto GV2 = motifcl::multihead_attention_backward_v(Q2, K2, V2, GO2, NH, false, NB, NT).to_vector<float>();
            auto ref_gq2 = finite_difference_grad(q2, k2, v2, go2, NB, NT, NC, NH, false, 'q');
            auto ref_gk2 = finite_difference_grad(q2, k2, v2, go2, NB, NT, NC, NH, false, 'k');
            auto ref_gv2 = finite_difference_grad(q2, k2, v2, go2, NB, NT, NC, NH, false, 'v');
            for (std::size_t i = 0; i < GQ2.size(); ++i) {
                if (std::fabs(GQ2[i] - ref_gq2[i]) > 3e-2f) return 1;
                if (std::fabs(GK2[i] - ref_gk2[i]) > 3e-2f) return 1;
                if (std::fabs(GV2[i] - ref_gv2[i]) > 3e-2f) return 1;
            }
        }

        Q.set_requires_grad(true);
        K.set_requires_grad(true);
        V.set_requires_grad(true);
        auto autograd_out = motifcl::multihead_attention(Q, K, V, H, true, B, T);
        autograd_out.backward(GO);
        if (!Q.grad() || !K.grad() || !V.grad()) return 1;
        auto AGQ = Q.grad()->to_vector<float>();
        auto AGK = K.grad()->to_vector<float>();
        auto AGV = V.grad()->to_vector<float>();
        for (std::size_t i = 0; i < GQ.size(); ++i) {
            if (std::fabs(AGQ[i] - GQ[i]) > 1e-5f) return 1;
            if (std::fabs(AGK[i] - GK[i]) > 1e-5f) return 1;
            if (std::fabs(AGV[i] - GV[i]) > 1e-5f) return 1;
        }
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
