#include <cstdint>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        motifcl::nn::TransformerBlock block(backend, 32, 4, 64);
        auto x = motifcl::Tensor::randn(backend, {8, 32});
        auto y = block.forward(x);
        std::cout << "TransformerBlock forward shape = " << y.shape().str() << "\n";

        motifcl::nn::GPTModel gpt(backend, 32, 16, 32, 4, 2, 64);
        std::vector<std::int32_t> ids = {1, 2, 3, 4, 5, 6};
        auto tokens = motifcl::Tensor::from_cpu(backend, {1, 6}, motifcl::DType::I32, ids.data());
        auto logits = gpt.forward(tokens);
        std::cout << "GPTModel logits shape = " << logits.shape().str() << "\n";
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "05_tiny_transformer");
    }
}
