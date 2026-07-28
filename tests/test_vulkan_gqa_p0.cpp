#include <cstdint>
#include <iostream>
#include <vector>

#include "vulkan_p0_test_utils.hpp"

int main() {
    using namespace motifcl;
    using namespace motifcl_vulkan_p0_test;

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cout << "SKIP test_vulkan_gqa_p0: " << probe.error << '\n';
        return 77;
    }

    try {
        Backend backend = Backend::create_vulkan();
        constexpr int batch = 2;
        constexpr int query_tokens = 3;
        constexpr int key_tokens = 5;
        constexpr int n_head = 4;
        constexpr int n_kv_head = 2;
        constexpr int head_dim = 16;
        constexpr int v_head_dim = 12;
        constexpr int query_offset = 1;
        constexpr int q_channels = n_head * head_dim;
        constexpr int kv_channels = n_kv_head * head_dim;
        constexpr int v_channels = n_kv_head * v_head_dim;

        std::vector<float> q(static_cast<std::size_t>(batch * query_tokens * q_channels));
        std::vector<float> k(static_cast<std::size_t>(batch * key_tokens * kv_channels));
        std::vector<float> v(static_cast<std::size_t>(batch * key_tokens * v_channels));
        fill_deterministic(q, 0x101u, 0.35f);
        fill_deterministic(k, 0x202u, 0.30f);
        fill_deterministic(v, 0x303u, 0.45f);
        auto q_tensor = Tensor::from_cpu(backend, {batch * query_tokens, q_channels},
                                         DType::F32, q.data());
        auto k_tensor = Tensor::from_cpu(backend, {batch * key_tokens, kv_channels},
                                         DType::F32, k.data());
        auto v_tensor = Tensor::from_cpu(backend, {batch * key_tokens, v_channels},
                                         DType::F32, v.data());

        autograd::begin_graph_capture();
        auto causal = grouped_query_attention(q_tensor, k_tensor, v_tensor,
                                              n_head, n_kv_head, true, batch,
                                              query_tokens, key_tokens, query_offset);
        auto causal_graph = autograd::end_graph_capture();
        require_close(causal.to_vector<float>(),
                      reference_gqa(q, k, v, batch, query_tokens, key_tokens, key_tokens,
                                    n_head, n_kv_head, head_dim, v_head_dim, true,
                                    query_offset, 0),
                      4e-4f, "causal GQA");
        if (!graph_has_op(causal_graph, "grouped_query_attention_general_vulkan_f32")) {
            throw std::runtime_error("causal GQA did not engage Vulkan general kernel");
        }

        autograd::begin_graph_capture();
        auto windowed = grouped_query_attention_windowed(
            q_tensor, k_tensor, v_tensor, n_head, n_kv_head, 2, true, batch,
            query_tokens, key_tokens, query_offset);
        auto windowed_graph = autograd::end_graph_capture();
        require_close(windowed.to_vector<float>(),
                      reference_gqa(q, k, v, batch, query_tokens, key_tokens, key_tokens,
                                    n_head, n_kv_head, head_dim, v_head_dim, true,
                                    query_offset, 2),
                      4e-4f, "windowed GQA");
        if (!graph_has_op(windowed_graph, "grouped_query_attention_windowed_vulkan_f32")) {
            throw std::runtime_error("windowed GQA did not engage Vulkan general kernel");
        }

        std::vector<float> additive_mask(
            static_cast<std::size_t>(batch * query_tokens * key_tokens));
        fill_deterministic(additive_mask, 0x404u, 0.2f);
        auto additive_tensor = Tensor::from_cpu(
            backend, {batch, query_tokens, key_tokens}, DType::F32, additive_mask.data());
        autograd::begin_graph_capture();
        auto additive = grouped_query_attention_masked(
            q_tensor, k_tensor, v_tensor, additive_tensor, n_head, n_kv_head, false,
            batch, query_tokens, key_tokens, query_offset, true);
        auto additive_graph = autograd::end_graph_capture();
        require_close(
            additive.to_vector<float>(),
            reference_gqa(
                q, k, v, batch, query_tokens, key_tokens, key_tokens, n_head,
                n_kv_head, head_dim, v_head_dim, false, query_offset, 0, {},
                [&](int b, int tq, int tk) {
                    return additive_mask[static_cast<std::size_t>(
                        (b * query_tokens + tq) * key_tokens + tk)];
                }),
            4e-4f, "additive masked GQA");
        if (!graph_has_op(additive_graph, "grouped_query_attention_mask_vulkan_f32")) {
            throw std::runtime_error("additive mask did not engage Vulkan general kernel");
        }

        std::vector<std::uint8_t> keep_mask(
            static_cast<std::size_t>(query_tokens * key_tokens), 1u);
        keep_mask[1] = 0u;
        keep_mask[6] = 0u;
        keep_mask[8] = 0u;
        keep_mask[12] = 0u;
        auto keep_tensor = Tensor::from_cpu(
            backend, {query_tokens, key_tokens}, DType::U8, keep_mask.data());
        autograd::begin_graph_capture();
        auto masked = grouped_query_attention_masked(
            q_tensor, k_tensor, v_tensor, keep_tensor, n_head, n_kv_head, true,
            batch, query_tokens, key_tokens, query_offset, false);
        auto mask_graph = autograd::end_graph_capture();
        require_close(
            masked.to_vector<float>(),
            reference_gqa(
                q, k, v, batch, query_tokens, key_tokens, key_tokens, n_head,
                n_kv_head, head_dim, v_head_dim, true, query_offset, 0,
                [&](int, int tq, int tk) {
                    return keep_mask[static_cast<std::size_t>(tq * key_tokens + tk)] == 0u;
                }),
            4e-4f, "boolean masked GQA");
        if (!graph_has_op(mask_graph, "grouped_query_attention_mask_vulkan_f32")) {
            throw std::runtime_error("boolean mask did not engage Vulkan general kernel");
        }

        constexpr int long_keys = 1025;
        constexpr int long_heads = 2;
        constexpr int long_kv_heads = 1;
        constexpr int long_head_dim = 8;
        std::vector<float> long_q(static_cast<std::size_t>(long_heads * long_head_dim));
        std::vector<float> long_k(static_cast<std::size_t>(long_keys * long_head_dim));
        std::vector<float> long_v(static_cast<std::size_t>(long_keys * long_head_dim));
        fill_deterministic(long_q, 0x505u, 0.2f);
        fill_deterministic(long_k, 0x606u, 0.2f);
        fill_deterministic(long_v, 0x707u, 0.25f);
        auto long_q_tensor = Tensor::from_cpu(
            backend, {1, long_heads * long_head_dim}, DType::F32, long_q.data());
        auto long_k_tensor = Tensor::from_cpu(
            backend, {long_keys, long_head_dim}, DType::F32, long_k.data());
        auto long_v_tensor = Tensor::from_cpu(
            backend, {long_keys, long_head_dim}, DType::F32, long_v.data());
        autograd::begin_graph_capture();
        auto long_attention = grouped_query_attention(
            long_q_tensor, long_k_tensor, long_v_tensor, long_heads, long_kv_heads,
            true, 1, 1, long_keys, long_keys - 1);
        auto long_graph = autograd::end_graph_capture();
        require_close(
            long_attention.to_vector<float>(),
            reference_gqa(long_q, long_k, long_v, 1, 1, long_keys, long_keys,
                          long_heads, long_kv_heads, long_head_dim, long_head_dim,
                          true, long_keys - 1, 0),
            6e-4f, "long-context causal GQA");
        if (!graph_has_op(long_graph, "grouped_query_attention_general_vulkan_f32")) {
            throw std::runtime_error("long-context GQA did not engage Vulkan streaming kernel");
        }

        std::cout << "PASS test_vulkan_gqa_p0: 5/5 causal/windowed/masked/long GQA parity cases\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_gqa_p0: " << e.what() << '\n';
        return 1;
    }
}
