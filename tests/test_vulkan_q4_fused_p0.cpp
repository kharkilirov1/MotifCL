#include <cmath>
#include <iostream>
#include <vector>

#include "vulkan_p0_test_utils.hpp"

int main() {
    using namespace motifcl;
    using namespace motifcl_vulkan_p0_test;

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cout << "SKIP test_vulkan_q4_fused_p0: " << probe.error << '\n';
        return 77;
    }

    try {
        Backend backend = Backend::create_vulkan();
        autograd::NoGradGuard no_grad;

        constexpr int in = 128;
        constexpr int n_head = 4;
        constexpr int n_kv_head = 2;
        constexpr int head_dim = in / n_head;
        constexpr int kv_dim = n_kv_head * head_dim;
        constexpr int total = in + 2 * kv_dim;
        nn::TransformerConfig config;
        config.vocab_size = 8;
        config.block_size = 4;
        config.n_embd = in;
        config.n_head = n_head;
        config.n_kv_head = n_kv_head;
        config.n_layer = 1;
        config.mlp_hidden = 256;
        config.use_rope = false;
        config.use_qkv_bias = false;
        config.split_qkv_projections = false;
        config.skip_weight_init = true;
        nn::ModernSelfAttention attention(backend, config);

        std::vector<float> input(static_cast<std::size_t>(in));
        std::vector<float> qkv_weight(static_cast<std::size_t>(in * total));
        std::vector<float> identity(static_cast<std::size_t>(in * in), 0.0f);
        fill_deterministic(input, 0xb1bu, 0.25f);
        fill_deterministic(qkv_weight, 0xc1cu, 0.07f);
        for (int i = 0; i < in; ++i) identity[static_cast<std::size_t>(i * in + i)] = 1.0f;
        auto dense_qkv = Tensor::from_cpu(
            backend, {in, total}, DType::F32, qkv_weight.data());
        attention.qkv_proj().set_quantized_weight(quantize_q4_symmetric_cols(dense_qkv));
        if (autograd::is_enabled()) {
            throw std::runtime_error("NoGradGuard did not disable autograd");
        }
        if (!attention.qkv_proj().quantized_inference_enabled() ||
            attention.qkv_proj().quantized_weight_dtype() != DType::Q4_0 ||
            attention.qkv_proj().has_bias() ||
            !attention.qkv_proj().quantized_weight().has_quant_scales() ||
            attention.qkv_proj().quantized_weight().quant_scale_axis() != 1 ||
            backend.device_info().max_work_group_size < 64) {
            throw std::runtime_error("packed Q4_0 QKV public eligibility preconditions failed");
        }
        attention.o_proj().weight.data = Tensor::from_cpu(
            backend, {in, in}, DType::F32, identity.data());
        auto x = Tensor::from_cpu(backend, {1, in}, DType::F32, input.data());
        nn::KVCache cache(backend, 1, config.block_size, n_kv_head, head_dim);

        autograd::begin_graph_capture();
        auto attention_out = attention.forward_with_cache(x, cache, 1, 1);
        auto attention_graph = autograd::end_graph_capture();
        const auto dequant_qkv = q4_dequantize_cols(qkv_weight, in, total);
        std::vector<float> expected_attention(static_cast<std::size_t>(in), 0.0f);
        const int v_start = in + kv_dim;
        for (int c = 0; c < kv_dim; ++c) {
            float sum = 0.0f;
            for (int r = 0; r < in; ++r) {
                sum += input[static_cast<std::size_t>(r)] *
                       dequant_qkv[static_cast<std::size_t>(r * total + v_start + c)];
            }
            for (int h = 0; h < n_head; ++h) {
                if (h / (n_head / n_kv_head) == c / head_dim) {
                    const int d = c % head_dim;
                    expected_attention[static_cast<std::size_t>(h * head_dim + d)] = sum;
                }
            }
        }
        require_close(attention_out.to_vector<float>(), expected_attention, 8e-4f,
                      "packed Q4_0 QKV");
        if (!graph_has_op(attention_graph, "matmul_f32_q4_0_packed_qkv_vulkan_f32")) {
            throw std::runtime_error("packed Q4_0 QKV Vulkan kernel did not engage");
        }

        constexpr int hidden = 64;
        nn::ModernMLP mlp(backend, hidden, hidden, true, false, 0.0f, true);
        std::vector<float> mlp_input(static_cast<std::size_t>(hidden));
        std::vector<float> gate_up(static_cast<std::size_t>(hidden * hidden * 2));
        std::vector<float> down_identity(static_cast<std::size_t>(hidden * hidden), 0.0f);
        fill_deterministic(mlp_input, 0xd1du, 0.3f);
        fill_deterministic(gate_up, 0xe1eu, 0.08f);
        for (int i = 0; i < hidden; ++i) {
            down_identity[static_cast<std::size_t>(i * hidden + i)] = 1.0f;
        }
        auto dense_gate_up = Tensor::from_cpu(
            backend, {hidden, hidden * 2}, DType::F32, gate_up.data());
        mlp.gate_up_proj.set_quantized_weight(quantize_q4_symmetric_cols(dense_gate_up));
        mlp.down_proj.weight.data = Tensor::from_cpu(
            backend, {hidden, hidden}, DType::F32, down_identity.data());
        auto mlp_x = Tensor::from_cpu(
            backend, {1, hidden}, DType::F32, mlp_input.data());

        autograd::begin_graph_capture();
        auto mlp_out = mlp.forward(mlp_x);
        auto mlp_graph = autograd::end_graph_capture();
        const auto dequant_gate_up = q4_dequantize_cols(gate_up, hidden, hidden * 2);
        std::vector<float> expected_mlp(static_cast<std::size_t>(hidden));
        for (int c = 0; c < hidden; ++c) {
            float gate = 0.0f;
            float up = 0.0f;
            for (int r = 0; r < hidden; ++r) {
                const float xv = mlp_input[static_cast<std::size_t>(r)];
                gate += xv * dequant_gate_up[static_cast<std::size_t>(r * hidden * 2 + c)];
                up += xv * dequant_gate_up[
                    static_cast<std::size_t>(r * hidden * 2 + hidden + c)];
            }
            const float silu = gate / (1.0f + std::exp(-gate));
            expected_mlp[static_cast<std::size_t>(c)] = silu * up;
        }
        require_close(mlp_out.to_vector<float>(), expected_mlp, 8e-4f,
                      "packed Q4_0 SwiGLU");
        if (!graph_has_op(mlp_graph, "matmul_swiglu_f32_q4_0_packed_vulkan_f32")) {
            throw std::runtime_error("packed Q4_0 SwiGLU Vulkan kernel did not engage");
        }

        std::cout << "PASS test_vulkan_q4_fused_p0: 2/2 packed QKV/SwiGLU CPU parity cases\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_q4_fused_p0: " << e.what() << '\n';
        return 1;
    }
}
