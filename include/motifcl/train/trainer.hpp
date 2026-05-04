#pragma once

#include <functional>

#include <motifcl/nn/module.hpp>
#include <motifcl/train/optimizer.hpp>

namespace motifcl::train {

struct Batch {
    Tensor x;
    Tensor y;
};

using DataLoader = std::function<Batch()>;
using LossFn = std::function<Tensor(const Tensor&, const Tensor&)>;
using LogFn = std::function<void(int, float)>;

class Trainer {
public:
    Trainer(nn::Module& model, optim::Optimizer& optimizer, DataLoader loader, LossFn loss_fn);

    void fit(int steps, int log_every = 10, LogFn log_fn = {});

private:
    nn::Module* model_ = nullptr;
    optim::Optimizer* optimizer_ = nullptr;
    DataLoader loader_;
    LossFn loss_fn_;
};

} // namespace motifcl::train
