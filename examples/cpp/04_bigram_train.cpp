#include <cstdint>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();

        constexpr int vocab = 8;
        constexpr int T = 8;
        std::vector<std::int32_t> x_host = {0, 1, 2, 3, 4, 5, 6, 7};
        std::vector<std::int32_t> y_host = {1, 2, 3, 4, 5, 6, 7, 0};
        auto x = motifcl::Tensor::from_cpu(backend, {1, T}, motifcl::DType::I32, x_host.data());
        auto y = motifcl::Tensor::from_cpu(backend, {T}, motifcl::DType::I32, y_host.data());

        motifcl::nn::GPTModel model(backend, vocab, T, 32, 4, 1, 64);
        motifcl::optim::Adam opt(model.parameters(), 5e-3f);

        for (int step = 0; step < 10; ++step) {
            auto logits3 = model.forward(x);
            auto logits = logits3.view({T, vocab});
            auto loss = motifcl::softmax_cross_entropy(logits, y);
            loss.backward();
            opt.step();
            opt.zero_grad();
            std::cout << "step " << step << " loss=" << loss.item() << "\n";
        }
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "04_bigram_train");
    }
}
