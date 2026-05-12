#pragma once

#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

class Embedding : public Module {
public:
    Parameter weight;

    Embedding(Backend& backend, int vocab_size, int embed_dim, bool skip_weight_init = false);
    Tensor forward(const Tensor& indices) override;
    std::vector<Parameter*> parameters() override { return {&weight}; }

    void set_quantized_weight_transposed(const Tensor& weight);
    void clear_quantized_weight();
    bool quantized_inference_enabled() const { return quantized_weight_transposed_.valid(); }
    DType quantized_weight_dtype() const { return quantized_weight_dtype_; }
    const Tensor& quantized_weight_transposed() const { return quantized_weight_transposed_; }

    int vocab_size() const { return vocab_size_; }
    int embed_dim() const { return embed_dim_; }

private:
    int vocab_size_ = 0;
    int embed_dim_ = 0;
    Tensor quantized_weight_transposed_;
    DType quantized_weight_dtype_ = DType::F32;
};

Tensor token_position_embedding(const Tensor& token_ids, const Tensor& token_weight, const Tensor& position_weight);

} // namespace motifcl::nn
