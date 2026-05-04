#include <motifcl/train/trainer.hpp>

#include <iostream>

#include <motifcl/core/error.hpp>

namespace motifcl::train {

Trainer::Trainer(nn::Module& model, optim::Optimizer& optimizer, DataLoader loader, LossFn loss_fn)
    : model_(&model), optimizer_(&optimizer), loader_(std::move(loader)), loss_fn_(std::move(loss_fn)) {}

void Trainer::fit(int steps, int log_every, LogFn log_fn) {
    MCL_CHECK(model_ != nullptr && optimizer_ != nullptr, "Trainer has null model or optimizer");
    MCL_CHECK(loader_ != nullptr, "Trainer needs a DataLoader");
    MCL_CHECK(loss_fn_ != nullptr, "Trainer needs a LossFn");

    for (int step = 1; step <= steps; ++step) {
        auto batch = loader_();
        auto pred = model_->forward(batch.x);
        auto loss = loss_fn_(pred, batch.y);
        loss.backward();
        optimizer_->step();
        optimizer_->zero_grad();
        if (log_every > 0 && (step == 1 || step % log_every == 0)) {
            float value = loss.item();
            if (log_fn) log_fn(step, value);
            else std::cout << "step " << step << " loss=" << value << '\n';
        }
    }
}

} // namespace motifcl::train
