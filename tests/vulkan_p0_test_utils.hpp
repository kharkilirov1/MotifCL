#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/autograd/node.hpp>

namespace motifcl_vulkan_p0_test {

inline bool graph_has_op(const motifcl::autograd::CapturedGraph& graph,
                         const std::string& op) {
    for (const auto& node : graph.nodes()) {
        if (node.op == op) return true;
    }
    return false;
}

inline void require_close(const std::vector<float>& actual,
                          const std::vector<float>& expected,
                          float tolerance,
                          const std::string& label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    float max_error = 0.0f;
    std::size_t max_index = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        if (error > max_error) {
            max_error = error;
            max_index = i;
        }
    }
    if (!std::isfinite(max_error) || max_error > tolerance) {
        throw std::runtime_error(label + " max error " + std::to_string(max_error) +
                                 " at index " + std::to_string(max_index) +
                                 " exceeds " + std::to_string(tolerance));
    }
}

inline void fill_deterministic(std::vector<float>& values,
                               std::uint32_t seed,
                               float scale = 0.25f) {
    std::uint32_t state = seed;
    for (float& value : values) {
        state = state * 1664525u + 1013904223u;
        const float unit = static_cast<float>((state >> 8) & 0xffffu) / 32767.5f - 1.0f;
        value = unit * scale;
    }
}

inline std::vector<float> reference_gqa(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int batch,
    int query_tokens,
    int key_tokens,
    int key_stride,
    int n_head,
    int n_kv_head,
    int head_dim,
    int v_head_dim,
    bool causal,
    int query_offset,
    int sliding_window,
    const std::function<bool(int, int, int)>& keep = {},
    const std::function<float(int, int, int)>& additive = {},
    float scale_override = 0.0f) {
    const int q_channels = n_head * head_dim;
    const int kv_channels = n_kv_head * head_dim;
    const int v_channels = n_kv_head * v_head_dim;
    const int out_channels = n_head * v_head_dim;
    const int heads_per_kv = n_head / n_kv_head;
    const float scale = scale_override > 0.0f
        ? scale_override
        : 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> out(static_cast<std::size_t>(batch * query_tokens * out_channels), 0.0f);
    std::vector<float> scores(static_cast<std::size_t>(key_tokens));

    for (int b = 0; b < batch; ++b) {
        for (int tq = 0; tq < query_tokens; ++tq) {
            const int absolute_query = query_offset + tq;
            for (int h = 0; h < n_head; ++h) {
                const int kv_head = h / heads_per_kv;
                float max_score = -std::numeric_limits<float>::infinity();
                for (int tk = 0; tk < key_tokens; ++tk) {
                    bool visible = !causal || tk <= absolute_query;
                    if (visible && sliding_window > 0) {
                        visible = tk >= absolute_query - sliding_window + 1;
                    }
                    if (visible && keep) visible = keep(b, tq, tk);
                    if (!visible) {
                        scores[static_cast<std::size_t>(tk)] =
                            -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    float score = 0.0f;
                    const std::size_t q_base =
                        static_cast<std::size_t>((b * query_tokens + tq) * q_channels +
                                                 h * head_dim);
                    const std::size_t k_base =
                        static_cast<std::size_t>((b * key_stride + tk) * kv_channels +
                                                 kv_head * head_dim);
                    for (int d = 0; d < head_dim; ++d) {
                        score += q[q_base + static_cast<std::size_t>(d)] *
                                 k[k_base + static_cast<std::size_t>(d)];
                    }
                    score *= scale;
                    if (additive) score += additive(b, tq, tk);
                    scores[static_cast<std::size_t>(tk)] = score;
                    max_score = std::max(max_score, score);
                }

                float denom = 0.0f;
                for (int tk = 0; tk < key_tokens; ++tk) {
                    float& score = scores[static_cast<std::size_t>(tk)];
                    if (std::isfinite(score)) {
                        score = std::exp(score - max_score);
                        denom += score;
                    } else {
                        score = 0.0f;
                    }
                }
                const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
                for (int d = 0; d < v_head_dim; ++d) {
                    float value = 0.0f;
                    for (int tk = 0; tk < key_tokens; ++tk) {
                        const std::size_t v_index =
                            static_cast<std::size_t>((b * key_stride + tk) * v_channels +
                                                     kv_head * v_head_dim + d);
                        value += scores[static_cast<std::size_t>(tk)] * inv_denom * v[v_index];
                    }
                    const std::size_t out_index =
                        static_cast<std::size_t>((b * query_tokens + tq) * out_channels +
                                                 h * v_head_dim + d);
                    out[out_index] = value;
                }
            }
        }
    }
    return out;
}

inline std::vector<float> q4_dequantize_cols(const std::vector<float>& dense,
                                             int rows,
                                             int cols) {
    std::vector<float> scales(static_cast<std::size_t>(cols), 1.0f);
    for (int c = 0; c < cols; ++c) {
        float max_abs = 0.0f;
        for (int r = 0; r < rows; ++r) {
            max_abs = std::max(max_abs,
                               std::fabs(dense[static_cast<std::size_t>(r * cols + c)]));
        }
        scales[static_cast<std::size_t>(c)] = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    }
    std::vector<float> dequant(dense.size());
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const std::size_t index = static_cast<std::size_t>(r * cols + c);
            const float scale = scales[static_cast<std::size_t>(c)];
            int q = static_cast<int>(std::nearbyint(dense[index] / scale));
            q = std::max(-7, std::min(7, q));
            dequant[index] = static_cast<float>(q) * scale;
        }
    }
    return dequant;
}

inline std::vector<float> q4_dequantize_row(const std::vector<float>& dense) {
    float max_abs = 0.0f;
    for (float value : dense) max_abs = std::max(max_abs, std::fabs(value));
    const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
    std::vector<float> dequant(dense.size());
    for (std::size_t i = 0; i < dense.size(); ++i) {
        int q = static_cast<int>(std::nearbyint(dense[i] / scale));
        q = std::max(-7, std::min(7, q));
        dequant[i] = static_cast<float>(q) * scale;
    }
    return dequant;
}

inline void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} // namespace motifcl_vulkan_p0_test
