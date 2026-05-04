#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <motifcl/nn/attention.hpp>
#include <motifcl/nn/embedding.hpp>
#include <motifcl/nn/mlp.hpp>
#include <motifcl/nn/rmsnorm.hpp>

namespace motifcl::nn {

class TransformerBlock : public Module {
public:
    TransformerBlock(Backend& backend, int n_embd, int n_head, int mlp_hidden);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, int64_t batch_size, int64_t seq_len);
    std::vector<Parameter*> parameters() override;

private:
    RMSNorm norm1_;
    SelfAttention attn_;
    RMSNorm norm2_;
    MLP mlp_;
};

class GPTModel : public Module {
public:
    Parameter token_embedding;
    Parameter position_embedding;
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    RMSNorm final_norm;
    Parameter lm_head;

    GPTModel(Backend& backend, int vocab_size, int block_size, int n_embd, int n_head, int n_layer, int mlp_hidden);
    Tensor forward(const Tensor& token_ids) override;
    std::vector<Parameter*> parameters() override;

    int vocab_size() const { return vocab_size_; }
    int block_size() const { return block_size_; }
    int n_embd() const { return n_embd_; }

private:
    int vocab_size_ = 0;
    int block_size_ = 0;
    int n_embd_ = 0;
};

} // namespace motifcl::nn
