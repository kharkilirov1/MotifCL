#include <iostream>
#include <vector>

#include "vulkan_p0_test_utils.hpp"

namespace {

void assign_attention_weights(motifcl::nn::ModernSelfAttention& attention,
                              motifcl::Backend& backend,
                              const motifcl::nn::TransformerConfig& config) {
    const int head_dim = config.n_embd / config.n_head;
    const int kv_dim = config.n_kv_head * head_dim;
    std::vector<float> q(static_cast<std::size_t>(config.n_embd * config.n_embd));
    std::vector<float> k(static_cast<std::size_t>(config.n_embd * kv_dim));
    std::vector<float> v(static_cast<std::size_t>(config.n_embd * kv_dim));
    std::vector<float> identity(static_cast<std::size_t>(config.n_embd * config.n_embd), 0.0f);
    motifcl_vulkan_p0_test::fill_deterministic(q, 0x515u, 0.08f);
    motifcl_vulkan_p0_test::fill_deterministic(k, 0x525u, 0.08f);
    motifcl_vulkan_p0_test::fill_deterministic(v, 0x535u, 0.08f);
    for (int i = 0; i < config.n_embd; ++i) {
        identity[static_cast<std::size_t>(i * config.n_embd + i)] = 1.0f;
    }
    attention.q_proj().weight.data = motifcl::Tensor::from_cpu(
        backend, {config.n_embd, config.n_embd}, motifcl::DType::F32, q.data());
    attention.k_proj().weight.data = motifcl::Tensor::from_cpu(
        backend, {config.n_embd, kv_dim}, motifcl::DType::F32, k.data());
    attention.v_proj().weight.data = motifcl::Tensor::from_cpu(
        backend, {config.n_embd, kv_dim}, motifcl::DType::F32, v.data());
    attention.o_proj().weight.data = motifcl::Tensor::from_cpu(
        backend, {config.n_embd, config.n_embd}, motifcl::DType::F32, identity.data());
    if (config.use_qk_norm) {
        std::vector<float> norm(static_cast<std::size_t>(head_dim), 1.0f);
        attention.q_norm().weight.data = motifcl::Tensor::from_cpu(
            backend, {head_dim}, motifcl::DType::F32, norm.data());
        attention.k_norm().weight.data = motifcl::Tensor::from_cpu(
            backend, {head_dim}, motifcl::DType::F32, norm.data());
    }
}

void set_all_fused_decode_disabled(bool disabled) {
    using motifcl_vulkan_p0_test::set_env;
    const char* value = disabled ? "1" : "";
    set_env("MOTIFCL_DISABLE_FUSED_QK_NORM_ROPE_DECODE", value);
    set_env("MOTIFCL_DISABLE_FUSED_QK_NORM_ROPE_CACHE_APPEND_DECODE", value);
    set_env("MOTIFCL_DISABLE_FUSED_ROPE_CACHE_APPEND_DECODE", value);
}

void verify_transformer_decode_path(motifcl::Backend& backend,
                                    bool qk_norm,
                                    const std::string& expected_op,
                                    bool standalone_qk) {
    using namespace motifcl;
    using namespace motifcl_vulkan_p0_test;
    nn::TransformerConfig config;
    config.vocab_size = 8;
    config.block_size = 4;
    config.n_embd = 64;
    config.n_head = 4;
    config.n_kv_head = 2;
    config.n_layer = 1;
    config.mlp_hidden = 128;
    config.use_rope = true;
    config.use_qk_norm = qk_norm;
    config.split_qkv_projections = true;
    config.skip_weight_init = true;
    nn::ModernSelfAttention attention(backend, config);
    assign_attention_weights(attention, backend, config);

    std::vector<float> x0(static_cast<std::size_t>(config.n_embd));
    std::vector<float> x1(static_cast<std::size_t>(config.n_embd));
    fill_deterministic(x0, 0x616u, 0.2f);
    fill_deterministic(x1, 0x717u, 0.2f);
    auto t0 = Tensor::from_cpu(backend, {1, config.n_embd}, DType::F32, x0.data());
    auto t1 = Tensor::from_cpu(backend, {1, config.n_embd}, DType::F32, x1.data());
    autograd::NoGradGuard no_grad;

    set_all_fused_decode_disabled(true);
    nn::KVCache reference_cache(backend, 1, config.block_size, config.n_kv_head,
                                config.n_embd / config.n_head);
    (void)attention.forward_with_cache(t0, reference_cache, 1, 1);
    const auto reference = attention.forward_with_cache(t1, reference_cache, 1, 1)
                               .to_vector<float>();

    nn::KVCache fused_cache(backend, 1, config.block_size, config.n_kv_head,
                            config.n_embd / config.n_head);
    (void)attention.forward_with_cache(t0, fused_cache, 1, 1);
    if (qk_norm) {
        set_env("MOTIFCL_DISABLE_FUSED_QK_NORM_ROPE_DECODE", "");
        if (!standalone_qk) {
            set_env("MOTIFCL_DISABLE_FUSED_QK_NORM_ROPE_CACHE_APPEND_DECODE", "");
        }
    } else {
        set_env("MOTIFCL_DISABLE_FUSED_ROPE_CACHE_APPEND_DECODE", "");
    }
    autograd::begin_graph_capture();
    auto fused = attention.forward_with_cache(t1, fused_cache, 1, 1);
    auto graph = autograd::end_graph_capture();
    require_close(fused.to_vector<float>(), reference, 7e-4f, expected_op);
    if (!graph_has_op(graph, expected_op)) {
        throw std::runtime_error(expected_op + " did not engage");
    }
    set_all_fused_decode_disabled(false);
}

} // namespace

int main() {
    using namespace motifcl;
    using namespace motifcl_vulkan_p0_test;

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cout << "SKIP test_vulkan_kv_decode_p0: " << probe.error << '\n';
        return 77;
    }

    try {
        Backend backend = Backend::create_vulkan();
        constexpr int max_tokens = 6;
        constexpr int key_tokens = 4;
        constexpr int n_head = 4;
        constexpr int n_kv_head = 2;
        constexpr int head_dim = 16;
        constexpr int channels = n_kv_head * head_dim;
        std::vector<float> dense_k(static_cast<std::size_t>(key_tokens * channels));
        std::vector<float> dense_v(static_cast<std::size_t>(key_tokens * channels));
        std::vector<float> query(static_cast<std::size_t>(n_head * head_dim));
        fill_deterministic(dense_k, 0x818u, 0.35f);
        fill_deterministic(dense_v, 0x919u, 0.4f);
        fill_deterministic(query, 0xa1au, 0.3f);
        auto query_tensor = Tensor::from_cpu(
            backend, {1, n_head * head_dim}, DType::F32, query.data());

        nn::KVCache f32_cache(backend, 1, max_tokens, n_kv_head, head_dim, DType::F32);
        autograd::begin_graph_capture();
        for (int token = 0; token < key_tokens; ++token) {
            auto new_k = Tensor::from_cpu(backend, {1, channels}, DType::F32,
                                          dense_k.data() + token * channels);
            auto new_v = Tensor::from_cpu(backend, {1, channels}, DType::F32,
                                          dense_v.data() + token * channels);
            kv_cache_append(new_k, new_v, f32_cache.k, f32_cache.v,
                            1, 1, max_tokens, token);
        }
        auto f32_decode = grouped_query_attention(
            query_tensor, f32_cache.k, f32_cache.v, n_head, n_kv_head, true,
            1, 1, key_tokens, key_tokens - 1);
        auto f32_graph = autograd::end_graph_capture();
        std::vector<float> padded_k(static_cast<std::size_t>(max_tokens * channels), 0.0f);
        std::vector<float> padded_v(static_cast<std::size_t>(max_tokens * channels), 0.0f);
        std::copy(dense_k.begin(), dense_k.end(), padded_k.begin());
        std::copy(dense_v.begin(), dense_v.end(), padded_v.begin());
        require_close(
            f32_decode.to_vector<float>(),
            reference_gqa(query, padded_k, padded_v, 1, 1, key_tokens, max_tokens,
                          n_head, n_kv_head, head_dim, head_dim, true, key_tokens - 1, 0),
            4e-4f, "f32 KV decode");
        if (!graph_has_op(f32_graph, "kv_cache_append_vulkan_f32") ||
            !graph_has_op(f32_graph, "grouped_query_attention_general_vulkan_f32")) {
            throw std::runtime_error("f32 KV decode did not engage both Vulkan paths");
        }

        nn::KVCache q4_cache(backend, 1, max_tokens, n_kv_head, head_dim, DType::Q4_0);
        std::vector<float> dequant_k(static_cast<std::size_t>(max_tokens * channels), 0.0f);
        std::vector<float> dequant_v(static_cast<std::size_t>(max_tokens * channels), 0.0f);
        autograd::begin_graph_capture();
        for (int token = 0; token < key_tokens; ++token) {
            std::vector<float> k_row(
                dense_k.begin() + token * channels,
                dense_k.begin() + (token + 1) * channels);
            std::vector<float> v_row(
                dense_v.begin() + token * channels,
                dense_v.begin() + (token + 1) * channels);
            auto new_k = Tensor::from_cpu(backend, {1, channels}, DType::F32, k_row.data());
            auto new_v = Tensor::from_cpu(backend, {1, channels}, DType::F32, v_row.data());
            kv_cache_append(new_k, new_v, q4_cache.k, q4_cache.v,
                            1, 1, max_tokens, token);
            const auto dk = q4_dequantize_row(k_row);
            const auto dv = q4_dequantize_row(v_row);
            std::copy(dk.begin(), dk.end(), dequant_k.begin() + token * channels);
            std::copy(dv.begin(), dv.end(), dequant_v.begin() + token * channels);
        }
        auto q4_decode = grouped_query_attention(
            query_tensor, q4_cache.k, q4_cache.v, n_head, n_kv_head, true,
            1, 1, key_tokens, key_tokens - 1);
        auto q4_graph = autograd::end_graph_capture();
        require_close(
            q4_decode.to_vector<float>(),
            reference_gqa(query, dequant_k, dequant_v, 1, 1, key_tokens, max_tokens,
                          n_head, n_kv_head, head_dim, head_dim, true, key_tokens - 1, 0),
            8e-4f, "Q4_0 KV decode");
        if (!graph_has_op(q4_graph, "kv_cache_append_vulkan_q4_0") ||
            !graph_has_op(q4_graph, "grouped_query_attention_general_vulkan_f32")) {
            throw std::runtime_error("Q4_0 KV decode did not engage both Vulkan paths");
        }

        constexpr int long_tokens = 1025;
        std::vector<float> long_k(static_cast<std::size_t>(long_tokens * channels));
        std::vector<float> long_v(static_cast<std::size_t>(long_tokens * channels));
        fill_deterministic(long_k, 0xa2au, 0.25f);
        fill_deterministic(long_v, 0xb2bu, 0.3f);
        auto long_new_k = Tensor::from_cpu(
            backend, {long_tokens, channels}, DType::F32, long_k.data());
        auto long_new_v = Tensor::from_cpu(
            backend, {long_tokens, channels}, DType::F32, long_v.data());
        nn::KVCache long_q4_cache(
            backend, 1, long_tokens, n_kv_head, head_dim, DType::Q4_0);
        std::vector<float> long_dequant_k(long_k.size());
        std::vector<float> long_dequant_v(long_v.size());
        for (int token = 0; token < long_tokens; ++token) {
            std::vector<float> k_row(
                long_k.begin() + token * channels,
                long_k.begin() + (token + 1) * channels);
            std::vector<float> v_row(
                long_v.begin() + token * channels,
                long_v.begin() + (token + 1) * channels);
            const auto dk = q4_dequantize_row(k_row);
            const auto dv = q4_dequantize_row(v_row);
            std::copy(dk.begin(), dk.end(), long_dequant_k.begin() + token * channels);
            std::copy(dv.begin(), dv.end(), long_dequant_v.begin() + token * channels);
        }
        autograd::begin_graph_capture();
        kv_cache_append(long_new_k, long_new_v, long_q4_cache.k, long_q4_cache.v,
                        1, long_tokens, long_tokens, 0);
        auto long_q4_decode = grouped_query_attention(
            query_tensor, long_q4_cache.k, long_q4_cache.v, n_head, n_kv_head, true,
            1, 1, long_tokens, long_tokens - 1);
        auto long_q4_graph = autograd::end_graph_capture();
        require_close(
            long_q4_decode.to_vector<float>(),
            reference_gqa(query, long_dequant_k, long_dequant_v, 1, 1, long_tokens,
                          long_tokens, n_head, n_kv_head, head_dim, head_dim, true,
                          long_tokens - 1, 0),
            1e-3f, "long-context Q4_0 KV decode");
        if (!graph_has_op(long_q4_graph, "kv_cache_append_vulkan_q4_0") ||
            !graph_has_op(long_q4_graph, "grouped_query_attention_general_vulkan_f32")) {
            throw std::runtime_error("long Q4_0 KV decode did not engage Vulkan streaming path");
        }

        verify_transformer_decode_path(
            backend, true, "qk_norm_rope_decode_vulkan_f32", true);
        verify_transformer_decode_path(
            backend, true, "qk_norm_rope_cache_append_decode_vulkan_f32", false);
        verify_transformer_decode_path(
            backend, false, "rope_cache_append_decode_vulkan_f32", false);

        std::cout << "PASS test_vulkan_kv_decode_p0: 6/6 f32/Q4 KV and fused decode parity cases\n";
        return 0;
    } catch (const std::exception& e) {
        set_all_fused_decode_disabled(false);
        std::cerr << "FAIL test_vulkan_kv_decode_p0: " << e.what() << '\n';
        return 1;
    }
}
