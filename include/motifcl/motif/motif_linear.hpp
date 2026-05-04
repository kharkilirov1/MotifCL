#pragma once

#include <vector>

#include <motifcl/nn/module.hpp>

namespace motifcl::motif {

class MotifLinear : public nn::Module {
public:
    std::vector<nn::Parameter> motifs;
    nn::Parameter router;

    MotifLinear(Backend& backend, int in_features, int out_features, int motif_count);
    Tensor forward(const Tensor& x) override;
    std::vector<nn::Parameter*> parameters() override;

private:
    int in_features_ = 0;
    int out_features_ = 0;
    int motif_count_ = 0;
};

} // namespace motifcl::motif
