#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/autograd/node.hpp>
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>

#include "vulkan_p0_test_utils.hpp"

namespace {

template <typename T>
void require_exact(const std::vector<T>& actual,
                   const std::vector<T>& expected,
                   const std::string& label) {
    if (actual != expected) throw std::runtime_error(label + " exact mismatch");
}

std::vector<float> reference_grad_x(const std::vector<float>& grad_out,
                                    const std::vector<float>& weight,
                                    int rows,
                                    int in_features,
                                    int out_features) {
    std::vector<float> grad_x(static_cast<std::size_t>(rows * in_features), 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < in_features; ++i) {
            float acc = 0.0f;
            for (int o = 0; o < out_features; ++o) {
                acc += grad_out[static_cast<std::size_t>(r * out_features + o)] *
                       weight[static_cast<std::size_t>(o * in_features + i)];
            }
            grad_x[static_cast<std::size_t>(r * in_features + i)] = acc;
        }
    }
    return grad_x;
}

} // namespace

int main() {
    using namespace motifcl;
    using namespace motifcl_vulkan_p0_test;

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cout << "SKIP test_vulkan_counter_fused: " << probe.error << '\n';
        return 77;
    }

    try {
        Backend backend = Backend::create_vulkan();
        constexpr int rows = 37;
        constexpr int in_features = 68;
        constexpr int out_features = 35;
        constexpr int C = 3;
        constexpr std::uint32_t seed = 777u;

        std::vector<float> x_host(static_cast<std::size_t>(rows * in_features));
        std::vector<float> grad_out_host(static_cast<std::size_t>(rows * out_features));
        fill_deterministic(x_host, 0x1234u, 0.8f);
        fill_deterministic(grad_out_host, 0x5678u, 0.9f);

        nn::CounterStateLinear fast(
            backend, in_features, out_features, C, 1.0f, 0.05f,
            1.0f, 0.9f, 1.0e-6f, seed);
        nn::CounterStateLinear reference(
            backend, in_features, out_features, C, 1.0f, 0.05f,
            1.0f, 0.9f, 1.0e-6f, seed);

        const auto state_before = reference.state.to_vector<std::uint8_t>();
        const auto scale_before = reference.scale.to_vector<float>();
        require_exact(fast.state.to_vector<std::uint8_t>(), state_before,
                      "initial packed state");
        require_exact(fast.scale.to_vector<float>(), scale_before,
                      "initial scale");

        auto x_fast = Tensor::from_cpu(
            backend, {rows, in_features}, DType::F32, x_host.data());
        auto x_reference = Tensor::from_cpu(
            backend, {rows, in_features}, DType::F32, x_host.data());
        auto grad_out = Tensor::from_cpu(
            backend, {rows, out_features}, DType::F32, grad_out_host.data());
        x_fast.set_requires_grad(true);

        Tensor dense_weight;
        Tensor y_reference;
        {
            autograd::NoGradGuard no_grad;
            dense_weight = reference.decode_weight();
            y_reference = matmul_transpose_b(x_reference, dense_weight);
        }
        const auto weight_host = dense_weight.to_vector<float>();

        autograd::begin_graph_capture();
        auto y_fast = fast.forward(x_fast);
        const auto forward_graph = autograd::end_graph_capture();
        if (!graph_has_op(forward_graph,
                          "compact_counter_forward_vulkan_f32_rb4")) {
            throw std::runtime_error(
                "fused counter forward did not engage Vulkan rb4 kernel");
        }
        require_close(y_fast.to_vector<float>(), y_reference.to_vector<float>(),
                      2.0e-5f, "counter fused forward");

        y_fast.backward(grad_out);
        if (!x_fast.grad()) {
            throw std::runtime_error("counter fused backward did not produce grad_x");
        }
        require_close(
            x_fast.grad()->to_vector<float>(),
            reference_grad_x(grad_out_host, weight_host, rows, in_features,
                             out_features),
            2.0e-5f, "counter fused grad_x");

        // The reference executes the established fused counter update directly
        // with the same pre-update state, activations, gradient, and seed.
        reference.apply_update_backward(grad_out, x_reference, seed);
        backend.finish();

        const auto expected_state = reference.state.to_vector<std::uint8_t>();
        const auto expected_scale = reference.scale.to_vector<float>();
        const auto expected_v = reference.v.to_vector<float>();
        require_exact(fast.state.to_vector<std::uint8_t>(), expected_state,
                      "counter state transition");
        require_exact(fast.scale.to_vector<float>(), expected_scale,
                      "counter scale transition");
        require_exact(fast.v.to_vector<float>(), expected_v,
                      "counter RMS transition");

        const bool state_changed = expected_state != state_before;
        const bool scale_changed = expected_scale != scale_before;
        if (!state_changed || !scale_changed) {
            throw std::runtime_error(
                "counter update witness produced no packed-state/scale delta");
        }

        std::cout
            << "PASS test_vulkan_counter_fused: forward, grad_x, exact update 3/3\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_counter_fused: " << e.what() << '\n';
        return 1;
    }
}
