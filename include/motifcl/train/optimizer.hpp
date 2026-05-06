#pragma once

#include <vector>

#include <motifcl/nn/parameter.hpp>

namespace motifcl::optim {

class Optimizer {
public:
    explicit Optimizer(std::vector<nn::Parameter*> params) : params_(std::move(params)) {}
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    virtual void zero_grad();
    virtual void set_lr(float lr) = 0;
    virtual float lr() const = 0;
    const std::vector<nn::Parameter*>& parameters() const { return params_; }

protected:
    std::vector<nn::Parameter*> params_;
};

} // namespace motifcl::optim
