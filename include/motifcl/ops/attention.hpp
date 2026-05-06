#pragma once

#include <cstdint>

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

struct QKV {
    Tensor q;
    Tensor k;
    Tensor v;
};

Tensor softmax_rows(const Tensor& x);
Tensor causal_mask(const Tensor& scores, float mask_value = -1.0e30f);
Tensor attention_scores(const Tensor& q, const Tensor& k, float scale);
Tensor attention_apply(const Tensor& probs, const Tensor& v);
QKV qkv_split(const Tensor& packed, int64_t q_dim, int64_t kv_dim);
Tensor rope(const Tensor& x, int n_head, int64_t batch_size, int64_t seq_len,
            float theta = 10000.0f, int64_t rotary_dim = 0, int64_t token_offset = 0);
Tensor multihead_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                           int n_head, bool causal = true,
                           int64_t batch_size = 1, int64_t seq_len = 0);
Tensor grouped_query_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               int n_head, int n_kv_head, bool causal = true,
                               int64_t batch_size = 1, int64_t query_len = 0,
                               int64_t key_len = 0, int64_t query_offset = 0);
Tensor grouped_query_attention_masked(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& mask, int n_head, int n_kv_head,
                                      bool causal = true, int64_t batch_size = 1,
                                      int64_t query_len = 0, int64_t key_len = 0,
                                      int64_t query_offset = 0, bool additive_mask = false);
void kv_cache_append(const Tensor& new_k, const Tensor& new_v, Tensor& cache_k, Tensor& cache_v,
                     int64_t batch_size, int64_t new_tokens, int64_t max_tokens, int64_t start_pos);
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
