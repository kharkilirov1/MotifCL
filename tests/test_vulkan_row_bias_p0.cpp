#include <iostream>
#include <vector>

#include "vulkan_p0_test_utils.hpp"

int main() {
    using namespace motifcl;
    using namespace motifcl_vulkan_p0_test;

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cout << "SKIP test_vulkan_row_bias_p0: " << probe.error << '\n';
        return 77;
    }

    try {
        Backend backend = Backend::create_vulkan();
        constexpr int rows = 3;
        constexpr int cols = 70;
        std::vector<float> x(static_cast<std::size_t>(rows * cols));
        std::vector<float> bias(static_cast<std::size_t>(cols));
        fill_deterministic(x, 0xf1fu, 0.5f);
        fill_deterministic(bias, 0x123u, 0.2f);
        auto x_tensor = Tensor::from_cpu(backend, {rows, cols}, DType::F32, x.data());
        auto bias_tensor = Tensor::from_cpu(backend, {cols}, DType::F32, bias.data());
        autograd::begin_graph_capture();
        auto y = add_bias_rows(x_tensor, bias_tensor);
        auto graph = autograd::end_graph_capture();
        std::vector<float> expected(x.size());
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                expected[static_cast<std::size_t>(r * cols + c)] =
                    x[static_cast<std::size_t>(r * cols + c)] +
                    bias[static_cast<std::size_t>(c)];
            }
        }
        require_close(y.to_vector<float>(), expected, 1e-6f, "row bias rb4 tail");
        if (!graph_has_op(graph, "add_bias_rows_vulkan_f32")) {
            throw std::runtime_error("row bias Vulkan rb4 kernel did not engage");
        }

        constexpr int in_features = 65;
        constexpr int out_features = 70;
        nn::Linear linear(backend, in_features, out_features, true, true);
        std::vector<float> linear_x(static_cast<std::size_t>(rows * in_features));
        std::vector<float> weight(static_cast<std::size_t>(in_features * out_features));
        std::vector<float> linear_bias(static_cast<std::size_t>(out_features));
        fill_deterministic(linear_x, 0x234u, 0.25f);
        fill_deterministic(weight, 0x345u, 0.08f);
        fill_deterministic(linear_bias, 0x456u, 0.15f);
        linear.weight.data = Tensor::from_cpu(
            backend, {in_features, out_features}, DType::F32, weight.data());
        linear.bias.data = Tensor::from_cpu(
            backend, {out_features}, DType::F32, linear_bias.data());
        auto linear_input = Tensor::from_cpu(
            backend, {rows, in_features}, DType::F32, linear_x.data());
        autograd::begin_graph_capture();
        auto linear_y = linear.forward(linear_input);
        auto linear_graph = autograd::end_graph_capture();
        std::vector<float> linear_expected(
            static_cast<std::size_t>(rows * out_features), 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < out_features; ++c) {
                float value = linear_bias[static_cast<std::size_t>(c)];
                for (int d = 0; d < in_features; ++d) {
                    value += linear_x[static_cast<std::size_t>(r * in_features + d)] *
                             weight[static_cast<std::size_t>(d * out_features + c)];
                }
                linear_expected[static_cast<std::size_t>(r * out_features + c)] = value;
            }
        }
        require_close(linear_y.to_vector<float>(), linear_expected, 7e-4f,
                      "Linear row bias");
        if (!graph_has_op(linear_graph, "add_bias_rows_vulkan_f32")) {
            throw std::runtime_error("Linear did not route bias through Vulkan rb4 kernel");
        }

        std::cout << "PASS test_vulkan_row_bias_p0: 2/2 direct/Linear row-bias parity cases\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_row_bias_p0: " << e.what() << '\n';
        return 1;
    }
}
