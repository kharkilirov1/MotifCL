#include <motifcl/nn/attention.hpp>

#include <cmath>

#include <motifcl/ops/attention.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/core/error.hpp>

namespace motifcl::nn {

SelfAttention::SelfAttention(Backend& backend, int n_embd, int n_head)
    : q_proj_(backend, n_embd, n_embd),
      k_proj_(backend, n_embd, n_embd),
      v_proj_(backend, n_embd, n_embd),
      o_proj_(backend, n_embd, n_embd),
      n_embd_(n_embd),
      n_head_(n_head),
      head_dim_(n_head > 0 ? n_embd / n_head : 0) {
    MCL_CHECK(n_head > 0, "SelfAttention n_head must be positive");
    MCL_CHECK(n_embd > 0, "SelfAttention n_embd must be positive");
    MCL_CHECK(n_embd % n_head == 0, "SelfAttention n_embd must be divisible by n_head");
}

Tensor SelfAttention::forward(const Tensor& x) {
    return forward(x, true);
}

Tensor SelfAttention::forward(const Tensor& x, bool causal) {
    return forward(x, 1, x.shape()[0], causal);
}

Tensor SelfAttention::forward(const Tensor& x, int64_t batch_size, int64_t seq_len, bool causal) {
    MCL_CHECK(x.ndim() == 2, "SelfAttention currently expects flattened [tokens, channels] input");
    MCL_CHECK(x.shape()[1] == n_embd_, "SelfAttention channel dimension mismatch");
    auto q = q_proj_.forward(x);
    auto k = k_proj_.forward(x);
    auto v = v_proj_.forward(x);
    auto context = multihead_attention(q, k, v, n_head_, causal, batch_size, seq_len);
    return o_proj_.forward(context);
}

std::vector<Parameter*> SelfAttention::parameters() {
    std::vector<Parameter*> result;
    for (auto* layer : {&q_proj_, &k_proj_, &v_proj_, &o_proj_}) {
        auto p = layer->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    return result;
}

} // namespace motifcl::nn
