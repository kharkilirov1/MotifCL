#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

#include "vulkan_p0_test_utils.hpp"

namespace {

struct GqaResult {
    std::vector<float> out;
    std::vector<float> dq;
    std::vector<float> dk;
    std::vector<float> dv;
    bool general_forward = false;
    bool fast_forward = false;
    bool vulkan_backward = false;
};

std::vector<float> slice(const std::vector<float>& values,
                         std::size_t offset,
                         std::size_t count) {
    return {values.begin() + static_cast<std::ptrdiff_t>(offset),
            values.begin() + static_cast<std::ptrdiff_t>(offset + count)};
}

void append(std::vector<float>& dst, const std::vector<float>& src) {
    dst.insert(dst.end(), src.begin(), src.end());
}

GqaResult reference_gqa_backward(const std::vector<float>& q,
                                 const std::vector<float>& k,
                                 const std::vector<float>& v,
                                 const std::vector<float>& grad_out,
                                 int batch,
                                 int query_tokens,
                                 int key_tokens,
                                 int n_head,
                                 int n_kv_head,
                                 int head_dim,
                                 bool causal,
                                 int query_offset) {
    const int q_channels = n_head * head_dim;
    const int kv_channels = n_kv_head * head_dim;
    const int group_size = n_head / n_kv_head;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    GqaResult result;
    result.out.assign(q.size(), 0.0f);
    result.dq.assign(q.size(), 0.0f);
    result.dk.assign(k.size(), 0.0f);
    result.dv.assign(v.size(), 0.0f);
    std::vector<float> probs(static_cast<std::size_t>(key_tokens));
    std::vector<float> dp(static_cast<std::size_t>(key_tokens));
    std::vector<float> ds(static_cast<std::size_t>(key_tokens));

    for (int b = 0; b < batch; ++b) {
        for (int tq = 0; tq < query_tokens; ++tq) {
            for (int h = 0; h < n_head; ++h) {
                const int kv_head = h / group_size;
                const std::size_t q_base = static_cast<std::size_t>(
                    (b * query_tokens + tq) * q_channels + h * head_dim);
                float max_score = -std::numeric_limits<float>::infinity();
                for (int tk = 0; tk < key_tokens; ++tk) {
                    if (causal && tk > query_offset + tq) {
                        probs[static_cast<std::size_t>(tk)] = 0.0f;
                        continue;
                    }
                    const std::size_t kv_base = static_cast<std::size_t>(
                        (b * key_tokens + tk) * kv_channels +
                        kv_head * head_dim);
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        score += q[q_base + static_cast<std::size_t>(d)] *
                                 k[kv_base + static_cast<std::size_t>(d)];
                    }
                    score *= scale;
                    probs[static_cast<std::size_t>(tk)] = score;
                    max_score = std::max(max_score, score);
                }
                float sum = 0.0f;
                for (int tk = 0; tk < key_tokens; ++tk) {
                    if (causal && tk > query_offset + tq) continue;
                    float& prob = probs[static_cast<std::size_t>(tk)];
                    prob = std::exp(prob - max_score);
                    sum += prob;
                }
                const float inv_sum = 1.0f / sum;
                float dot_dp_p = 0.0f;
                for (int tk = 0; tk < key_tokens; ++tk) {
                    float& prob = probs[static_cast<std::size_t>(tk)];
                    prob *= inv_sum;
                    float dp_value = 0.0f;
                    if (prob != 0.0f) {
                        const std::size_t kv_base = static_cast<std::size_t>(
                            (b * key_tokens + tk) * kv_channels +
                            kv_head * head_dim);
                        for (int d = 0; d < head_dim; ++d) {
                            const std::size_t di = static_cast<std::size_t>(d);
                            result.out[q_base + di] += prob * v[kv_base + di];
                            dp_value += grad_out[q_base + di] * v[kv_base + di];
                            result.dv[kv_base + di] +=
                                prob * grad_out[q_base + di];
                        }
                    }
                    dp[static_cast<std::size_t>(tk)] = dp_value;
                    dot_dp_p += dp_value * prob;
                }
                for (int tk = 0; tk < key_tokens; ++tk) {
                    const float ds_value =
                        probs[static_cast<std::size_t>(tk)] *
                        (dp[static_cast<std::size_t>(tk)] - dot_dp_p);
                    ds[static_cast<std::size_t>(tk)] = ds_value;
                    const std::size_t kv_base = static_cast<std::size_t>(
                        (b * key_tokens + tk) * kv_channels +
                        kv_head * head_dim);
                    for (int d = 0; d < head_dim; ++d) {
                        const std::size_t di = static_cast<std::size_t>(d);
                        result.dq[q_base + di] +=
                            ds_value * k[kv_base + di] * scale;
                        result.dk[kv_base + di] +=
                            ds_value * q[q_base + di] * scale;
                    }
                }
            }
        }
    }
    return result;
}

GqaResult run_autograd_case(motifcl::Backend& backend,
                            const std::vector<float>& q,
                            const std::vector<float>& k,
                            const std::vector<float>& v,
                            const std::vector<float>& grad_out,
                            int batch,
                            int query_tokens,
                            int key_tokens,
                            int n_head,
                            int n_kv_head,
                            int head_dim,
                            bool causal,
                            int query_offset) {
    using namespace motifcl;
    using motifcl_vulkan_p0_test::graph_has_op;
    const int q_channels = n_head * head_dim;
    const int kv_channels = n_kv_head * head_dim;
    auto qt = Tensor::from_cpu(
        backend, {batch * query_tokens, q_channels}, DType::F32, q.data());
    auto kt = Tensor::from_cpu(
        backend, {batch * key_tokens, kv_channels}, DType::F32, k.data());
    auto vt = Tensor::from_cpu(
        backend, {batch * key_tokens, kv_channels}, DType::F32, v.data());
    auto got = Tensor::from_cpu(
        backend, {batch * query_tokens, q_channels}, DType::F32,
        grad_out.data());
    qt.set_requires_grad(true);
    kt.set_requires_grad(true);
    vt.set_requires_grad(true);

    autograd::begin_graph_capture();
    auto out = grouped_query_attention(
        qt, kt, vt, n_head, n_kv_head, causal, batch, query_tokens,
        key_tokens, query_offset);
    const auto forward_graph = autograd::end_graph_capture();

    autograd::begin_graph_capture();
    out.backward(got);
    const auto backward_graph = autograd::end_graph_capture();
    backend.finish();
    if (!qt.grad() || !kt.grad() || !vt.grad()) {
        throw std::runtime_error("Vulkan GQA autograd did not produce all gradients");
    }

    GqaResult result;
    result.out = out.to_vector<float>();
    result.dq = qt.grad()->to_vector<float>();
    result.dk = kt.grad()->to_vector<float>();
    result.dv = vt.grad()->to_vector<float>();
    result.general_forward = graph_has_op(
        forward_graph, "grouped_query_attention_general_vulkan_f32");
    result.fast_forward = graph_has_op(
        forward_graph, "grouped_query_attention_vulkan_f32");
    result.vulkan_backward = graph_has_op(
        backward_graph, "grouped_query_attention_backward_vulkan_f32");
    return result;
}

} // namespace

int main() {
    using namespace motifcl;
    using namespace motifcl_vulkan_p0_test;

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cout << "SKIP test_vulkan_gqa_training: " << probe.error << '\n';
        return 77;
    }

    try {
        Backend backend = Backend::create_vulkan();

        // (a) Causal gradients against an independent CPU microkernel.
        constexpr int causal_batch = 2;
        constexpr int causal_qt = 4;
        constexpr int causal_kt = 5;
        constexpr int heads = 4;
        constexpr int kv_heads = 2;
        constexpr int dim = 8;
        constexpr int causal_offset = 1;
        constexpr int q_channels = heads * dim;
        constexpr int kv_channels = kv_heads * dim;
        std::vector<float> causal_q(
            static_cast<std::size_t>(causal_batch * causal_qt * q_channels));
        std::vector<float> causal_k(
            static_cast<std::size_t>(causal_batch * causal_kt * kv_channels));
        std::vector<float> causal_v(causal_k.size());
        std::vector<float> causal_go(causal_q.size());
        fill_deterministic(causal_q, 0x101u, 0.35f);
        fill_deterministic(causal_k, 0x202u, 0.30f);
        fill_deterministic(causal_v, 0x303u, 0.40f);
        fill_deterministic(causal_go, 0x404u, 0.45f);
        const auto causal_ref = reference_gqa_backward(
            causal_q, causal_k, causal_v, causal_go, causal_batch, causal_qt,
            causal_kt, heads, kv_heads, dim, true, causal_offset);
        const auto causal_gpu = run_autograd_case(
            backend, causal_q, causal_k, causal_v, causal_go, causal_batch,
            causal_qt, causal_kt, heads, kv_heads, dim, true, causal_offset);
        if (!causal_gpu.general_forward || !causal_gpu.vulkan_backward) {
            throw std::runtime_error(
                "causal/batched GQA did not engage automatic Vulkan forward/backward");
        }
        require_close(causal_gpu.out, causal_ref.out, 4.0e-4f,
                      "causal GQA forward");
        require_close(causal_gpu.dq, causal_ref.dq, 4.0e-4f,
                      "causal GQA dQ");
        require_close(causal_gpu.dk, causal_ref.dk, 4.0e-4f,
                      "causal GQA dK");
        require_close(causal_gpu.dv, causal_ref.dv, 4.0e-4f,
                      "causal GQA dV");

        // (b) Batch=N gradients against concatenated independent batch=1
        // runs. Batch=3 selects the general path; each batch=1 run selects
        // the established fast noncausal path.
        constexpr int batch = 3;
        constexpr int qt_count = 4;
        constexpr int kt_count = 4;
        const std::size_t q_per_batch =
            static_cast<std::size_t>(qt_count * q_channels);
        const std::size_t kv_per_batch =
            static_cast<std::size_t>(kt_count * kv_channels);
        std::vector<float> batch_q(static_cast<std::size_t>(batch) * q_per_batch);
        std::vector<float> batch_k(static_cast<std::size_t>(batch) * kv_per_batch);
        std::vector<float> batch_v(batch_k.size());
        std::vector<float> batch_go(batch_q.size());
        fill_deterministic(batch_q, 0x505u, 0.25f);
        fill_deterministic(batch_k, 0x606u, 0.30f);
        fill_deterministic(batch_v, 0x707u, 0.35f);
        fill_deterministic(batch_go, 0x808u, 0.40f);
        const auto batched = run_autograd_case(
            backend, batch_q, batch_k, batch_v, batch_go, batch, qt_count,
            kt_count, heads, kv_heads, dim, false, 0);
        if (!batched.general_forward || !batched.vulkan_backward) {
            throw std::runtime_error(
                "batch=N GQA did not engage general Vulkan autograd");
        }
        GqaResult concatenated;
        for (int b = 0; b < batch; ++b) {
            const auto single = run_autograd_case(
                backend,
                slice(batch_q, static_cast<std::size_t>(b) * q_per_batch,
                      q_per_batch),
                slice(batch_k, static_cast<std::size_t>(b) * kv_per_batch,
                      kv_per_batch),
                slice(batch_v, static_cast<std::size_t>(b) * kv_per_batch,
                      kv_per_batch),
                slice(batch_go, static_cast<std::size_t>(b) * q_per_batch,
                      q_per_batch),
                1, qt_count, kt_count, heads, kv_heads, dim, false, 0);
            if (!single.fast_forward || !single.vulkan_backward) {
                throw std::runtime_error(
                    "batch=1 comparison did not engage existing fast Vulkan path");
            }
            append(concatenated.out, single.out);
            append(concatenated.dq, single.dq);
            append(concatenated.dk, single.dk);
            append(concatenated.dv, single.dv);
        }
        require_close(batched.out, concatenated.out, 4.0e-4f,
                      "batch=N forward vs concatenated batch=1");
        require_close(batched.dq, concatenated.dq, 4.0e-4f,
                      "batch=N dQ vs concatenated batch=1");
        require_close(batched.dk, concatenated.dk, 4.0e-4f,
                      "batch=N dK vs concatenated batch=1");
        require_close(batched.dv, concatenated.dv, 4.0e-4f,
                      "batch=N dV vs concatenated batch=1");

        // (c) Explicit general-forward + new backward at B=1 must match the
        // established fast noncausal forward + autograd combination.
        const auto q1 = slice(batch_q, 0, q_per_batch);
        const auto k1 = slice(batch_k, 0, kv_per_batch);
        const auto v1 = slice(batch_v, 0, kv_per_batch);
        const auto go1 = slice(batch_go, 0, q_per_batch);
        const auto fast = run_autograd_case(
            backend, q1, k1, v1, go1, 1, qt_count, kt_count, heads,
            kv_heads, dim, false, 0);

        auto q_tensor = Tensor::from_cpu(
            backend, {qt_count, q_channels}, DType::F32, q1.data());
        auto k_tensor = Tensor::from_cpu(
            backend, {kt_count, kv_channels}, DType::F32, k1.data());
        auto v_tensor = Tensor::from_cpu(
            backend, {kt_count, kv_channels}, DType::F32, v1.data());
        auto go_tensor = Tensor::from_cpu(
            backend, {qt_count, q_channels}, DType::F32, go1.data());
        auto general_out = Tensor::empty(
            backend, {qt_count, q_channels}, DType::F32);
        const float attention_scale = 1.0f / std::sqrt(static_cast<float>(dim));
        const auto general_forward = run_vulkan_grouped_query_attention_general(
            backend.vulkan_runtime(), q_tensor.storage().vulkan_buffer,
            k_tensor.storage().vulkan_buffer, v_tensor.storage().vulkan_buffer,
            nullptr, nullptr, nullptr, general_out.storage().vulkan_buffer,
            1, qt_count, kt_count, kt_count, heads, kv_heads, dim, dim, false,
            0, 0, 0, 0, 0, 0, attention_scale);
        if (!general_forward.success) {
            throw std::runtime_error(
                "explicit general GQA forward failed: " + general_forward.error);
        }
        auto probs = Tensor::empty(
            backend, {heads * qt_count, kt_count}, DType::F32);
        auto ds = Tensor::empty(
            backend, {heads * qt_count, kt_count}, DType::F32);
        auto dq = Tensor::empty(backend, q_tensor.shape(), DType::F32);
        auto dk = Tensor::empty(backend, k_tensor.shape(), DType::F32);
        auto dv = Tensor::empty(backend, v_tensor.shape(), DType::F32);
        const auto general_backward = run_vulkan_grouped_query_attention_backward(
            backend.vulkan_runtime(), q_tensor.storage().vulkan_buffer,
            k_tensor.storage().vulkan_buffer, v_tensor.storage().vulkan_buffer,
            go_tensor.storage().vulkan_buffer, probs.storage().vulkan_buffer,
            ds.storage().vulkan_buffer, dq.storage().vulkan_buffer,
            dk.storage().vulkan_buffer, dv.storage().vulkan_buffer, 1,
            qt_count, kt_count, heads, kv_heads, dim, false, 0,
            attention_scale);
        if (!general_backward.success) {
            throw std::runtime_error(
                "explicit general GQA backward failed: " +
                general_backward.error);
        }
        backend.finish();
        require_close(general_out.to_vector<float>(), fast.out, 4.0e-4f,
                      "general vs fast B=1 forward");
        require_close(dq.to_vector<float>(), fast.dq, 4.0e-4f,
                      "general vs fast B=1 dQ");
        require_close(dk.to_vector<float>(), fast.dk, 4.0e-4f,
                      "general vs fast B=1 dK");
        require_close(dv.to_vector<float>(), fast.dv, 4.0e-4f,
                      "general vs fast B=1 dV");

        std::cout
            << "PASS test_vulkan_gqa_training: causal_cpu, batch_concat, general_fast 3/3\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_gqa_training: " << e.what() << '\n';
        return 1;
    }
}
