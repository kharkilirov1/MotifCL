#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        motifcl::nn::GPTModel model(backend, 16, 8, 16, 4, 1, 32);
        std::vector<std::int32_t> ids = {1, 2, 3, 4, 4, 3, 2, 1};
        auto X = motifcl::Tensor::from_cpu(backend, {2, 4}, motifcl::DType::I32, ids.data());
        auto Y = model.forward(X);
        if (Y.shape() != motifcl::Shape({2, 4, 16})) return 1;
        std::vector<std::int32_t> targets = {2, 3, 4, 5, 5, 4, 3, 2};
        auto T = motifcl::Tensor::from_cpu(backend, {8}, motifcl::DType::I32, targets.data());
        auto loss = motifcl::softmax_cross_entropy(Y.view({8, 16}), T);
        loss.backward();
        int grad_count = 0;
        for (auto* p : model.parameters()) {
            if (p && p->grad()) {
                ++grad_count;
                auto grad = p->grad()->to_vector<float>();
                for (float v : grad) if (!std::isfinite(v)) return 1;
            }
        }
        if (grad_count < 6) return 1;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
