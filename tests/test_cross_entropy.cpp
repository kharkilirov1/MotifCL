#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> logits = {2.0f, 0.0f, 0.0f, 0.0f, 2.0f, 0.0f};
        std::vector<std::int32_t> targets = {0, 1};
        auto L = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, logits.data());
        auto T = motifcl::Tensor::from_cpu(backend, {2}, motifcl::DType::I32, targets.data());
        L.set_requires_grad(true);
        auto loss = motifcl::softmax_cross_entropy(L, T);
        float got = loss.item();
        float row_loss = std::log(std::exp(2.0f) + 2.0f) - 2.0f;
        if (std::fabs(got - row_loss) > 1e-4f) return 1;
        loss.backward();
        if (!L.grad()) return 1;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
