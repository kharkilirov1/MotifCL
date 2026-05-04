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
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
