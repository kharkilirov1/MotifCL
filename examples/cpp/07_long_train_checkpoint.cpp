#include <iostream>
#include <memory>
#include <vector>

#include <motifcl/motifcl.hpp>
#include "example_utils.hpp"

int main() {
    try {
        motifcl::manual_seed(2026);
        auto backend = motifcl::Backend::create_opencl();

        constexpr int rows = 128;
        auto x = motifcl::Tensor::randn(backend, {rows, 4}, 1.0f);
        auto x_host = x.to_vector<float>();
        std::vector<float> y_host(rows * 2, 0.0f);
        for (int r = 0; r < rows; ++r) {
            y_host[r * 2 + 0] = 0.8f * x_host[r * 4 + 0] - 0.3f * x_host[r * 4 + 1] + 0.1f;
            y_host[r * 2 + 1] = -0.4f * x_host[r * 4 + 2] + 0.6f * x_host[r * 4 + 3] - 0.2f;
        }
        auto y = motifcl::Tensor::from_cpu(backend, {rows, 2}, motifcl::DType::F32, y_host.data());

        motifcl::nn::Sequential model({
            std::make_shared<motifcl::nn::Linear>(backend, 4, 32),
            std::make_shared<motifcl::nn::GELU>(),
            std::make_shared<motifcl::nn::Linear>(backend, 32, 2)
        });
        motifcl::optim::Adam opt(model.parameters(), 2e-2f);
        motifcl::train::TensorDataLoader loader(x, y, 32);
        motifcl::train::CosineLR schedule(2e-2f, 80, 2e-3f);
        motifcl::train::Trainer trainer(
            model,
            opt,
            [&loader] { return loader.next(); },
            [](const motifcl::Tensor& pred, const motifcl::Tensor& target) {
                return motifcl::mse_loss(pred, target);
            });

        auto history = trainer.fit_with_history(80, 20, 1.0f, &schedule);
        motifcl::save_parameters(model.parameters(), "motifcl_long_train_checkpoint.mclp");
        motifcl::train::write_history_csv(history, "motifcl_long_train_history.csv");
        std::cout << "initial_loss=" << history.losses.front()
                  << " final_loss=" << history.losses.back()
                  << " checkpoint=motifcl_long_train_checkpoint.mclp\n";
    } catch (const std::exception& e) {
        return motifcl_example::handle_exception(e, "07_long_train_checkpoint");
    }
}
