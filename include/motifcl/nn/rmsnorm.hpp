#pragma once

#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

class RMSNorm : public Module {
public:
    Parameter weight;
    Parameter bias;
    float eps = 1e-6f;

    RMSNorm(Backend& backend, int features, float eps = 1e-6f, bool layer_norm = false);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;
    void set_layer_norm(bool enabled = true) { layer_norm_ = enabled; }
    bool is_layer_norm() const { return layer_norm_; }

private:
    bool layer_norm_ = false;
};

} // namespace motifcl::nn
