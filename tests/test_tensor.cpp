#include <cmath>
#include <iostream>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        auto t = motifcl::Tensor::ones(backend, {8});
        auto v = t.to_vector<float>();
        for (auto x : v) if (std::fabs(x - 1.0f) > 1e-6f) return 1;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
