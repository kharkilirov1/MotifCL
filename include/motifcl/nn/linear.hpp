#pragma once

#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

class Linear : public Module {
public:
    Parameter weight;
    Parameter bias;

    Linear(Backend& backend, int in_features, int out_features, bool use_bias = true);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }
    bool has_bias() const { return use_bias_; }

private:
    int in_features_ = 0;
    int out_features_ = 0;
    bool use_bias_ = true;
};

} // namespace motifcl::nn
