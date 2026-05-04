#pragma once

#include <motifcl/nn/module.hpp>

namespace motifcl::motif {

Tensor router_soft(const Tensor& x, const Tensor& router_weight);
Tensor router_topk(const Tensor& x, const Tensor& router_weight, int k);

class Router : public nn::Module {
public:
    nn::Parameter weight;
    int motif_count = 0;
    int top_k = 0;

    Router(Backend& backend, int in_features, int motif_count, int top_k = 0);
    Tensor forward(const Tensor& x) override;
    std::vector<nn::Parameter*> parameters() override { return {&weight}; }
};

} // namespace motifcl::motif
