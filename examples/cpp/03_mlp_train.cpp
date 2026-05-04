#include <iostream>
#include <memory>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();

        // Simple regression toy: learn y = linear-ish target from fixed random batch.
        auto x = motifcl::Tensor::randn(backend, {16, 4}, 1.0f);
        std::vector<float> target_host(16 * 2, 0.0f);
        auto x_host = x.to_vector<float>();
        for (int r = 0; r < 16; ++r) {
            target_host[r * 2 + 0] = 0.5f * x_host[r * 4 + 0] - 0.25f * x_host[r * 4 + 1];
            target_host[r * 2 + 1] = -0.7f * x_host[r * 4 + 2] + 0.1f * x_host[r * 4 + 3];
        }
        auto y_true = motifcl::Tensor::from_cpu(backend, {16, 2}, motifcl::DType::F32, target_host.data());

        motifcl::nn::Sequential model({
            std::make_shared<motifcl::nn::Linear>(backend, 4, 16),
            std::make_shared<motifcl::nn::GELU>(),
            std::make_shared<motifcl::nn::Linear>(backend, 16, 2)
        });
        motifcl::optim::Adam opt(model.parameters(), 1e-2f);

        for (int step = 0; step < 40; ++step) {
            auto pred = model.forward(x);
            auto loss = motifcl::mse_loss(pred, y_true);
            loss.backward();
            opt.step();
            opt.zero_grad();
            if (step % 10 == 0) std::cout << "step " << step << " loss=" << loss.item() << "\n";
        }
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "03_mlp_train");
    }
}
