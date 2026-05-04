#include <iostream>
#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        motifcl::motif::MotifLoRA layer(backend, 16, 8, 4, 3);
        auto x = motifcl::Tensor::randn(backend, {5, 16});
        auto y = layer.forward(x);
        std::cout << "MotifLoRA forward shape = " << y.shape().str() << "\n";
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "06_motif_lora");
    }
}
