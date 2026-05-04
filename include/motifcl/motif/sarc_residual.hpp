#pragma once

#include <memory>

#include <motifcl/nn/module.hpp>

namespace motifcl::motif {

Tensor sarc_residual(const Tensor& x, const Tensor& fx, const Tensor& gamma, float eps = 1e-6f);

class SARCResidual : public nn::Module {
public:
    std::shared_ptr<nn::Module> branch;
    nn::Parameter gamma;
    float eps = 1e-6f;

    SARCResidual(Backend& backend, std::shared_ptr<nn::Module> branch, int features, float eps = 1e-6f);
    Tensor forward(const Tensor& x) override;
    std::vector<nn::Parameter*> parameters() override;
};

} // namespace motifcl::motif
