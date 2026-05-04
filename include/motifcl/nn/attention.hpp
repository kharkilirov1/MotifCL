#pragma once

#include <cstdint>
#include <memory>

#include <motifcl/nn/linear.hpp>

namespace motifcl::nn {

class SelfAttention : public Module {
public:
    SelfAttention(Backend& backend, int n_embd, int n_head);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, bool causal);
    Tensor forward(const Tensor& x, int64_t batch_size, int64_t seq_len, bool causal);
    std::vector<Parameter*> parameters() override;

private:
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear o_proj_;
    int n_embd_ = 0;
    int n_head_ = 1;
    int head_dim_ = 0;
};

} // namespace motifcl::nn
