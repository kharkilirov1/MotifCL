#include <motifcl/train/trainer.hpp>

#include <iostream>
#include <utility>

#include <motifcl/core/error.hpp>

namespace motifcl::train {

Trainer::Trainer(nn::Module& model, optim::Optimizer& optimizer, DataLoader loader, LossFn loss_fn)
    : model_(&model), optimizer_(&optimizer), loader_(std::move(loader)), loss_fn_(std::move(loss_fn)) {}

void Trainer::fit(int steps, int log_every, LogFn log_fn) {
    (void)fit_with_history(steps, log_every, 0.0f, nullptr, std::move(log_fn));
}

TrainingHistory Trainer::fit_with_history(int steps, int log_every, float max_grad_norm, LRScheduler* scheduler, LogFn log_fn) {
    MCL_CHECK(model_ != nullptr && optimizer_ != nullptr, "Trainer has null model or optimizer");
    MCL_CHECK(loader_ != nullptr, "Trainer needs a DataLoader");
    MCL_CHECK(loss_fn_ != nullptr, "Trainer needs a LossFn");
    TrainingHistory history;
    history.losses.reserve(static_cast<std::size_t>(steps));
    history.learning_rates.reserve(static_cast<std::size_t>(steps));

    for (int step = 1; step <= steps; ++step) {
        if (scheduler) optimizer_->set_lr(scheduler->lr_at(step));
        auto batch = loader_();
        auto pred = model_->forward(batch.x);
        auto loss = loss_fn_(pred, batch.y);
        loss.backward();
        if (max_grad_norm > 0.0f) clip_grad_norm(optimizer_->parameters(), max_grad_norm);
        optimizer_->step();
        optimizer_->zero_grad();
        const float value = loss.item();
        history.losses.push_back(value);
        history.learning_rates.push_back(optimizer_->lr());
        if (log_every > 0 && (step == 1 || step % log_every == 0)) {
            if (log_fn) log_fn(step, value);
            else std::cout << "step " << step << " loss=" << value << '\n';
        }
    }
    return history;
}

} // namespace motifcl::train
