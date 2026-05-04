#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> ph = {1, 2, 3, 4};
        std::vector<float> gh = {0.1f, -0.2f, 0.3f, -0.4f};
        auto P = motifcl::Tensor::from_cpu(backend, {4}, motifcl::DType::F32, ph.data());
        auto G = motifcl::Tensor::from_cpu(backend, {4}, motifcl::DType::F32, gh.data());
        auto M = motifcl::Tensor::zeros(backend, {4});
        auto V = motifcl::Tensor::zeros(backend, {4});
        motifcl::adam_update(P, G, M, V, 1e-3f, 0.9f, 0.999f, 1e-8f, 1);
        auto out = P.to_vector<float>();
        for (float v : out) if (!std::isfinite(v)) return 1;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
