#include <iostream>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        motifcl::nn::TransformerBlock block(backend, 16, 4, 32);
        auto x = motifcl::Tensor::randn(backend, {4, 16});
        auto y = block.forward(x);
        if (y.shape() != motifcl::Shape({4, 16})) return 1;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
