#pragma once

#include <functional>

#include <motifcl/train/dataloader.hpp>
#include <motifcl/nn/module.hpp>
#include <motifcl/train/optimizer.hpp>
#include <motifcl/train/scheduler.hpp>
#include <motifcl/train/training_utils.hpp>

namespace motifcl::train {

using DataLoader = std::function<Batch()>;
using LossFn = std::function<Tensor(const Tensor&, const Tensor&)>;
using LogFn = std::function<void(int, float)>;

class Trainer {
public:
    Trainer(nn::Module& model, optim::Optimizer& optimizer, DataLoader loader, LossFn loss_fn);

    void fit(int steps, int log_every = 10, LogFn log_fn = {});
    TrainingHistory fit_with_history(int steps,
                                     int log_every = 10,
                                     float max_grad_norm = 0.0f,
                                     LRScheduler* scheduler = nullptr,
                                     LogFn log_fn = {});

private:
    nn::Module* model_ = nullptr;
    optim::Optimizer* optimizer_ = nullptr;
    DataLoader loader_;
    LossFn loss_fn_;
};

} // namespace motifcl::train
