#pragma once

#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

class RMSNorm : public Module {
public:
    Parameter weight;
    float eps = 1e-6f;

    RMSNorm(Backend& backend, int features, float eps = 1e-6f);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override { return {&weight}; }
};

} // namespace motifcl::nn
