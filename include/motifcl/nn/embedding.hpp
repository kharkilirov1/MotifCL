#pragma once

#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

class Embedding : public Module {
public:
    Parameter weight;

    Embedding(Backend& backend, int vocab_size, int embed_dim);
    Tensor forward(const Tensor& indices) override;
    std::vector<Parameter*> parameters() override { return {&weight}; }

    int vocab_size() const { return vocab_size_; }
    int embed_dim() const { return embed_dim_; }

private:
    int vocab_size_ = 0;
    int embed_dim_ = 0;
};

Tensor token_position_embedding(const Tensor& token_ids, const Tensor& token_weight, const Tensor& position_weight);

} // namespace motifcl::nn
