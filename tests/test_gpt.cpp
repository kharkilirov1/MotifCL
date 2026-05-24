#include <cstdint>
#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        motifcl::manual_seed(123);
        motifcl::nn::GPTModel model(backend, 16, 8, 16, 4, 1, 32);
        for (auto* p : model.parameters()) {
            if (!p) continue;
            p->data = motifcl::Tensor::randn(backend, p->data.shape(), 0.005f);
            p->data.set_requires_grad(p->trainable);
        }
        std::vector<std::int32_t> ids = {1, 2, 3, 4, 4, 3, 2, 1};
        auto X = motifcl::Tensor::from_cpu(backend, {2, 4}, motifcl::DType::I32, ids.data());
        auto Y = model.forward(X);
        if (Y.shape() != motifcl::Shape({2, 4, 16})) {
            std::cerr << "GPT forward shape mismatch\n";
            return 1;
        }
        std::vector<std::int32_t> targets = {2, 3, 4, 5, 5, 4, 3, 2};
        auto T = motifcl::Tensor::from_cpu(backend, {8}, motifcl::DType::I32, targets.data());
        auto loss = motifcl::softmax_cross_entropy(Y.view({8, 16}), T);
        loss.backward();
        int grad_count = 0;
        for (auto* p : model.parameters()) {
            if (p && p->grad()) {
                ++grad_count;
                auto grad = p->grad()->to_vector<float>();
                for (float v : grad) {
                    if (!std::isfinite(v)) {
                        std::cerr << "GPT backward produced non-finite gradient\n";
                        return 1;
                    }
                }
            }
        }
        if (grad_count < 6) {
            std::cerr << "GPT backward populated too few gradients: " << grad_count << "\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
