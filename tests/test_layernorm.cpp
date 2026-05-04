#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> x = {1, 2, 3, 4, 2, 4, 6, 8};
        std::vector<float> weight = {1, 1, 1, 1};
        std::vector<float> bias = {0.5f, -0.5f, 1.0f, -1.0f};
        auto X = motifcl::Tensor::from_cpu(backend, {2, 4}, motifcl::DType::F32, x.data());
        auto W = motifcl::Tensor::from_cpu(backend, {4}, motifcl::DType::F32, weight.data());
        auto B = motifcl::Tensor::from_cpu(backend, {4}, motifcl::DType::F32, bias.data());
        auto Y = motifcl::layernorm(X, W, B, 1e-6f).to_vector<float>();
        motifcl::autograd::begin_graph_capture();
        auto captured = motifcl::layernorm(X, W, B, 1e-6f);
        (void)captured;
        auto graph = motifcl::autograd::end_graph_capture();
        if (backend.device_info().max_work_group_size >= 256 &&
            (graph.empty() || graph.nodes()[0].op != "layernorm_rowwise_wg_f32")) return 1;

        for (int row = 0; row < 2; ++row) {
            float mean = 0.0f;
            for (int c = 0; c < 4; ++c) mean += x[row * 4 + c];
            mean /= 4.0f;
            float var = 0.0f;
            for (int c = 0; c < 4; ++c) {
                float centered = x[row * 4 + c] - mean;
                var += centered * centered;
            }
            var /= 4.0f;
            float inv = 1.0f / std::sqrt(var + 1e-6f);
            for (int c = 0; c < 4; ++c) {
                float expected = (x[row * 4 + c] - mean) * inv + bias[c];
                if (std::fabs(Y[row * 4 + c] - expected) > 1e-4f) return 1;
            }
        }

        constexpr int Rows = 2;
        constexpr int Cols = 513;
        std::vector<float> big_x(Rows * Cols);
        std::vector<float> big_w(Cols);
        std::vector<float> big_b(Cols);
        for (int i = 0; i < Rows * Cols; ++i) big_x[i] = static_cast<float>((i % 17) - 8) * 0.03f;
        for (int i = 0; i < Cols; ++i) {
            big_w[i] = 0.5f + static_cast<float>(i % 7) * 0.05f;
            big_b[i] = static_cast<float>((i % 5) - 2) * 0.02f;
        }
        auto BigX = motifcl::Tensor::from_cpu(backend, {Rows, Cols}, motifcl::DType::F32, big_x.data());
        auto BigW = motifcl::Tensor::from_cpu(backend, {Cols}, motifcl::DType::F32, big_w.data());
        auto BigB = motifcl::Tensor::from_cpu(backend, {Cols}, motifcl::DType::F32, big_b.data());
        auto BigY = motifcl::layernorm(BigX, BigW, BigB, 1e-6f).to_vector<float>();
        for (int row = 0; row < Rows; ++row) {
            float mean = 0.0f;
            for (int c = 0; c < Cols; ++c) mean += big_x[row * Cols + c];
            mean /= static_cast<float>(Cols);
            float var = 0.0f;
            for (int c = 0; c < Cols; ++c) {
                float centered = big_x[row * Cols + c] - mean;
                var += centered * centered;
            }
            float inv = 1.0f / std::sqrt(var / static_cast<float>(Cols) + 1e-6f);
            for (int c = 0; c < Cols; ++c) {
                float expected = (big_x[row * Cols + c] - mean) * inv * big_w[c] + big_b[c];
                if (std::fabs(BigY[row * Cols + c] - expected) > 1e-4f) return 1;
            }
        }
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
