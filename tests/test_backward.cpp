#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> xh = {1, 2, 3, 4};
        std::vector<float> wh = {1, 0, 0, 1};
        std::vector<float> yh = {0, 0, 0, 0};
        auto X = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, xh.data());
        auto W = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, wh.data());
        auto Y = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, yh.data());
        X.set_requires_grad(true);
        W.set_requires_grad(true);
        auto loss = motifcl::mse_loss(motifcl::matmul(X, W), Y);
        loss.backward();
        if (!X.grad() || !W.grad()) return 1;
        auto wg = W.grad()->to_vector<float>();
        for (float v : wg) if (!std::isfinite(v)) return 1;

        auto shared_x = motifcl::Tensor::from_cpu(backend, {2, 2}, motifcl::DType::F32, xh.data());
        shared_x.set_requires_grad(true);
        auto shared_y = motifcl::gelu(shared_x);
        auto shared_z = motifcl::add(shared_y, shared_y);
        auto shared_loss = motifcl::sum_all(shared_z);
        backend.profiler.clear();
        backend.profiler.set_enabled(true);
        shared_loss.backward();
        backend.finish();
        backend.profiler.set_enabled(false);
        std::size_t gelu_backward_count = 0;
        for (const auto& row : backend.profiler.summary()) {
            if (row.name == "gelu_backward_f32") gelu_backward_count = row.count;
        }
        if (gelu_backward_count != 1) {
            std::cerr << "expected topological backward to execute gelu backward once, got "
                      << gelu_backward_count << "\n";
            return 2;
        }
        if (!shared_x.grad()) return 3;
        for (float v : shared_x.grad()->to_vector<float>()) if (!std::isfinite(v)) return 4;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
