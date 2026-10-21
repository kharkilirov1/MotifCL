// Independent CPU reference for the sparse-MoE semantics used by the Vulkan
// sparse path (mission Phase B). Two references live here:
//
//   moe7_reference_forward  — nn::MoE7FFN semantics: hard top-1 over seven
//       experts with the balancing bias shifting selection only, winner output
//       from decoded counter weights (w = scale[o] * t[o,i]), SwiGLU
//       gate*sigmoid(gate)*up, plus the always-on shared expert.
//   moe_topk_reference_forward — general MoEFFN semantics (the OpenCL kernel
//       kernels/motif.cl:moe_swiglu_forward_f32): top-k by logit, softmax over
//       the SELECTED logits only, weighted sum of expert outputs.
//
// Tie-breaking matches fog_hard_route7_f32.comp (`if (v > bv)` — strict), i.e.
// the lowest expert index wins equal logits. Nothing here calls into the
// library under test.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace motifcl_test {

struct MoE7ReferenceResult {
    std::vector<float> y;                 // [rows, dim]
    std::vector<int> picks;               // winning expert per row
    std::array<std::int64_t, 7> counts{}; // tokens per expert
};

// weights: w[o * in + i] (row-major [out, in]); x: [rows, in].
inline std::vector<float> ref_linear(const std::vector<float>& x, int rows, int in,
                                     const std::vector<float>& w, int out) {
    std::vector<float> y(static_cast<std::size_t>(rows) * out, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int o = 0; o < out; ++o) {
            float s = 0.0f;
            for (int i = 0; i < in; ++i)
                s += w[static_cast<std::size_t>(o) * in + i] * x[static_cast<std::size_t>(r) * in + i];
            y[static_cast<std::size_t>(r) * out + o] = s;
        }
    }
    return y;
}

// packed: [rows, 2*hidden] (gate first, up second) -> [rows, hidden].
inline std::vector<float> ref_swiglu(const std::vector<float>& packed, int rows, int hidden) {
    std::vector<float> act(static_cast<std::size_t>(rows) * hidden, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < hidden; ++c) {
            const float gate = packed[static_cast<std::size_t>(r) * 2 * hidden + c];
            const float up = packed[static_cast<std::size_t>(r) * 2 * hidden + hidden + c];
            act[static_cast<std::size_t>(r) * hidden + c] = gate * (1.0f / (1.0f + std::exp(-gate))) * up;
        }
    }
    return act;
}

// Decoded counter weights per expert pair, plus the shared expert.
// up_w/down_w: decoded [out, in] weights of each CounterStateLinear.
inline MoE7ReferenceResult moe7_reference_forward(const std::vector<float>& x, int rows, int dim,
                                                  const std::vector<float>& router_w,  // [dim, 7]
                                                  const std::vector<float>& router_b,  // [7]
                                                  const std::vector<std::vector<float>>& up_w,   // [7]
                                                  const std::vector<std::vector<float>>& down_w, // [7]
                                                  const std::vector<float>& shared_up_w,
                                                  const std::vector<float>& shared_down_w,
                                                  int hidden_expert, int hidden_shared) {
    MoE7ReferenceResult result;
    result.y.assign(static_cast<std::size_t>(rows) * dim, 0.0f);
    result.picks.assign(rows, 0);

    std::array<std::vector<float>, 7> expert_out{};
    for (int e = 0; e < 7; ++e) {
        const auto packed = ref_linear(x, rows, dim, up_w[e], 2 * hidden_expert);
        const auto act = ref_swiglu(packed, rows, hidden_expert);
        expert_out[e] = ref_linear(act, rows, hidden_expert, down_w[e], dim);
    }
    const auto shared_out =
        ref_linear(ref_swiglu(ref_linear(x, rows, dim, shared_up_w, 2 * hidden_shared), rows, hidden_shared),
                   rows, hidden_shared, shared_down_w, dim);

    for (int r = 0; r < rows; ++r) {
        int best = 0;
        float best_value = 0.0f;
        for (int e = 0; e < 7; ++e) {
            float s = router_b[e];
            for (int i = 0; i < dim; ++i)
                s += router_w[static_cast<std::size_t>(i) * 7 + e] * x[static_cast<std::size_t>(r) * dim + i];
            if (e == 0 || s > best_value) {  // strict '>' => lowest index wins ties
                best_value = s;
                best = e;
            }
        }
        result.picks[r] = best;
        result.counts[best] += 1;
        for (int i = 0; i < dim; ++i)
            result.y[static_cast<std::size_t>(r) * dim + i] =
                expert_out[best][static_cast<std::size_t>(r) * dim + i] + shared_out[static_cast<std::size_t>(r) * dim + i];
    }
    return result;
}

struct TopKReferenceResult {
    std::vector<float> y;                  // [rows, in]
    std::vector<int> indices;              // [rows, top_k]
    std::vector<float> weights;            // [rows, top_k]
};

// General top-k MoE (MoEFFN / kernels/motif.cl semantics).
// router_w: [in, experts]; gate/up/down_w: [experts, in, hidden] / [experts, hidden, in].
inline TopKReferenceResult moe_topk_reference_forward(const std::vector<float>& x, int rows, int in,
                                                      const std::vector<float>& router_w,
                                                      const std::vector<float>& gate_w,
                                                      const std::vector<float>& up_w,
                                                      const std::vector<float>& down_w,
                                                      int experts, int hidden, int top_k) {
    TopKReferenceResult result;
    result.y.assign(static_cast<std::size_t>(rows) * in, 0.0f);
    result.indices.assign(static_cast<std::size_t>(rows) * top_k, 0);
    result.weights.assign(static_cast<std::size_t>(rows) * top_k, 0.0f);

    std::vector<float> logits(static_cast<std::size_t>(rows) * experts, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int e = 0; e < experts; ++e) {
            float s = 0.0f;
            for (int i = 0; i < in; ++i)
                s += x[static_cast<std::size_t>(r) * in + i] * router_w[static_cast<std::size_t>(i) * experts + e];
            logits[static_cast<std::size_t>(r) * experts + e] = s;
        }
    }

    for (int r = 0; r < rows; ++r) {
        const auto base = static_cast<std::size_t>(r) * experts;
        // Selection: descending logit, strict '>' so ties keep the lower index.
        std::array<int, 8> selected{};
        std::array<float, 8> selected_logits{};
        for (int k = 0; k < 8; ++k) { selected[k] = -1; selected_logits[k] = -3.402823466e+38F; }
        for (int pick = 0; pick < top_k; ++pick) {
            int best_e = -1;
            float best = -3.402823466e+38F;
            for (int e = 0; e < experts; ++e) {
                bool already = false;
                for (int j = 0; j < pick; ++j)
                    if (selected[j] == e) already = true;
                if (already) continue;
                const float logit = logits[base + e];
                if (best_e < 0 || logit > best) { best = logit; best_e = e; }
            }
            selected[pick] = best_e;
            selected_logits[pick] = best;
        }
        // Softmax over the SELECTED logits only (kernel lines 95-99).
        float max_logit = selected_logits[0];
        for (int k = 1; k < top_k; ++k) max_logit = std::fmax(max_logit, selected_logits[k]);
        float denom = 0.0f;
        for (int k = 0; k < top_k; ++k) denom += std::exp(selected_logits[k] - max_logit);
        denom = std::fmax(denom, 1e-20f);

        std::array<std::vector<float>, 8> expert_out{};
        for (int pick = 0; pick < top_k; ++pick) {
            const int e = selected[pick];
            const float route = std::exp(selected_logits[pick] - max_logit) / denom;
            result.indices[static_cast<std::size_t>(r) * top_k + pick] = e;
            result.weights[static_cast<std::size_t>(r) * top_k + pick] = route;
            if (expert_out[pick].empty()) {
                std::vector<float> packed(static_cast<std::size_t>(in) * 2 * hidden, 0.0f);
                for (int h = 0; h < 2 * hidden; ++h) {
                    float g = 0.0f;
                    for (int i = 0; i < in; ++i)
                        g += x[static_cast<std::size_t>(r) * in + i] *
                             gate_w[static_cast<std::size_t>(e) * in * hidden + static_cast<std::size_t>(i) * hidden + h];
                    packed[static_cast<std::size_t>(h)] = g;
                }
                std::vector<float> up(static_cast<std::size_t>(in) * hidden, 0.0f);
                for (int h = 0; h < hidden; ++h) {
                    float u = 0.0f;
                    for (int i = 0; i < in; ++i)
                        u += x[static_cast<std::size_t>(r) * in + i] *
                             up_w[static_cast<std::size_t>(e) * in * hidden + static_cast<std::size_t>(i) * hidden + h];
                    up[static_cast<std::size_t>(h)] = u;
                }
                std::vector<float> act(hidden, 0.0f);
                for (int h = 0; h < hidden; ++h) {
                    const float gate = packed[static_cast<std::size_t>(h)];
                    act[h] = gate * (1.0f / (1.0f + std::exp(-gate))) * up[h];
                }
                std::vector<float> y(in, 0.0f);
                for (int o = 0; o < in; ++o) {
                    float s = 0.0f;
                    for (int h = 0; h < hidden; ++h)
                        s += act[h] * down_w[static_cast<std::size_t>(e) * hidden * in +
                                            static_cast<std::size_t>(h) * in + o];
                    y[o] = s;
                }
                expert_out[pick] = std::move(y);
            }
            for (int o = 0; o < in; ++o)
                result.y[static_cast<std::size_t>(r) * in + o] += route * expert_out[pick][o];
        }
    }
    return result;
}

}  // namespace motifcl_test
