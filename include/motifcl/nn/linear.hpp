#pragma once

#include <motifcl/core/dtype.hpp>
#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

class Linear : public Module {
public:
    Parameter weight;
    Parameter bias;

    Linear(Backend& backend, int in_features, int out_features, bool use_bias = true);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

    void enable_quantized_inference(DType qdtype = DType::Q4_0);
    void set_quantized_weight(const Tensor& weight);
    void disable_quantized_inference();
    bool quantized_inference_enabled() const { return quantized_weight_.valid(); }
    DType quantized_weight_dtype() const { return quantized_weight_dtype_; }
    const Tensor& quantized_weight() const { return quantized_weight_; }

    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }
    bool has_bias() const { return use_bias_; }

private:
    int in_features_ = 0;
    int out_features_ = 0;
    bool use_bias_ = true;
    Tensor quantized_weight_;
    DType quantized_weight_dtype_ = DType::F32;
};

} // namespace motifcl::nn
