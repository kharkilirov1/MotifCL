#pragma once

#include <vector>

#include <motifcl/nn/module.hpp>

namespace motifcl::motif {

class MotifLoRA : public nn::Module {
public:
    Tensor frozen_weight;
    std::vector<nn::Parameter> lora_A;
    std::vector<nn::Parameter> lora_B;
    nn::Parameter router;
    int rank = 0;
    int motif_count = 0;
    float alpha = 1.0f;

    MotifLoRA(Backend& backend, int in_features, int out_features, int rank, int motif_count, float alpha = 1.0f);
    Tensor forward(const Tensor& x) override;
    std::vector<nn::Parameter*> parameters() override;
};

} // namespace motifcl::motif
