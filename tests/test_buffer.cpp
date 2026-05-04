#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> in = {1, 2, 3, 4};
        auto t = motifcl::Tensor::from_cpu(backend, {4}, motifcl::DType::F32, in.data());
        auto out = t.to_vector<float>();
        for (int i = 0; i < 4; ++i) if (std::fabs(out[i] - in[i]) > 1e-6f) return 1;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
