#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> x = {1, 2, 3, 4};
        auto X = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::F32, x.data());
        auto W = motifcl::Tensor::ones(backend, {4});
        auto Y = motifcl::rmsnorm(X, W).to_vector<float>();
        float rms = std::sqrt((1 + 4 + 9 + 16) / 4.0f + 1e-6f);
        for (int i = 0; i < 4; ++i) if (std::fabs(Y[i] - x[i] / rms) > 1e-4f) return 1;
        motifcl::autograd::begin_graph_capture();
        auto captured_norm = motifcl::rmsnorm(X, W);
        (void)captured_norm;
        auto graph = motifcl::autograd::end_graph_capture();
        if (backend.device_info().max_work_group_size >= 256 &&
            (graph.empty() || graph.nodes()[0].op != "rmsnorm_rowwise_wg_f32")) return 1;

        std::vector<float> grad = {0.5f, -0.25f, 0.75f, -1.0f};
        auto G = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::F32, grad.data());
        auto GX = motifcl::rmsnorm_backward_x(X, W, G).to_vector<float>();
        auto GW = motifcl::rmsnorm_backward_weight(X, W, G).to_vector<float>();
        std::vector<float> residual_grad = {1.0f, -2.0f, 0.5f, 3.0f};
        auto RG = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::F32, residual_grad.data());
        auto GXFused = motifcl::rmsnorm_backward_x_residual(X, W, G, RG).to_vector<float>();
        float dot = 0.0f;
        for (int i = 0; i < 4; ++i) dot += grad[i] * x[i];
        float inv = 1.0f / rms;
        for (int i = 0; i < 4; ++i) {
            float expected_x = grad[i] * inv - x[i] * inv * inv * inv * dot / 4.0f;
            float expected_w = grad[i] * x[i] * inv;
            if (std::fabs(GX[i] - expected_x) > 1e-4f) return 1;
            if (std::fabs(GXFused[i] - (expected_x + residual_grad[i])) > 1e-4f) return 1;
            if (std::fabs(GW[i] - expected_w) > 1e-4f) return 1;
        }

        X.set_requires_grad(true);
        W.set_requires_grad(true);
        auto loss = motifcl::mse_loss(motifcl::rmsnorm(X, W), motifcl::Tensor::zeros(backend, {1, 4}));
        loss.backward();
        if (!X.grad() || !W.grad()) return 1;

        constexpr int Rows = 3;
        constexpr int Cols = 513;
        std::vector<float> big_x(Rows * Cols);
        std::vector<float> big_w(Cols);
        for (int i = 0; i < Rows * Cols; ++i) big_x[i] = static_cast<float>((i % 13) - 6) * 0.05f;
        for (int i = 0; i < Cols; ++i) big_w[i] = 0.75f + static_cast<float>(i % 5) * 0.1f;
        auto BigX = motifcl::Tensor::from_cpu(backend, {Rows, Cols}, motifcl::DType::F32, big_x.data());
        auto BigW = motifcl::Tensor::from_cpu(backend, {Cols}, motifcl::DType::F32, big_w.data());
        auto BigY = motifcl::rmsnorm(BigX, BigW).to_vector<float>();
        auto BigRms = motifcl::rms_per_row(BigX).to_vector<float>();
        for (int r = 0; r < Rows; ++r) {
            float ss = 0.0f;
            for (int c = 0; c < Cols; ++c) ss += big_x[r * Cols + c] * big_x[r * Cols + c];
            float row_rms = std::sqrt(ss / static_cast<float>(Cols) + 1e-6f);
            if (std::fabs(BigRms[r] - row_rms) > 1e-5f) return 1;
            for (int c = 0; c < Cols; ++c) {
                float expected = big_x[r * Cols + c] / row_rms * big_w[c];
                if (std::fabs(BigY[r * Cols + c] - expected) > 1e-4f) return 1;
            }
        }
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
