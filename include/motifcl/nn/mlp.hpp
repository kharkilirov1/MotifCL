#pragma once

#include <motifcl/nn/linear.hpp>

namespace motifcl::nn {

class MLP : public Module {
public:
    Linear fc1;
    Linear fc2;

    MLP(Backend& backend, int in_features, int hidden_features, int out_features)
        : fc1(backend, in_features, hidden_features), fc2(backend, hidden_features, out_features) {}

    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;
};

} // namespace motifcl::nn
