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

protected:
    std::vector<nn::Parameter*> params_;
};

} // namespace motifcl::optim
