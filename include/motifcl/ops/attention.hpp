#pragma once

#include <cstdint>

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor softmax_rows(const Tensor& x);
Tensor causal_mask(const Tensor& scores, float mask_value = -1.0e30f);
Tensor attention_scores(const Tensor& q, const Tensor& k, float scale);
Tensor attention_apply(const Tensor& probs, const Tensor& v);
Tensor multihead_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                           int n_head, bool causal = true,
                           int64_t batch_size = 1, int64_t seq_len = 0);
Tensor multihead_attention_backward_q(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                      int n_head, bool causal = true,
                                      int64_t batch_size = 1, int64_t seq_len = 0);
Tensor multihead_attention_backward_k(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                      int n_head, bool causal = true,
                                      int64_t batch_size = 1, int64_t seq_len = 0);
Tensor multihead_attention_backward_v(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                      int n_head, bool causal = true,
                                      int64_t batch_size = 1, int64_t seq_len = 0);

} // namespace motifcl
