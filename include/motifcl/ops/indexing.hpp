#pragma once

#include <cstdint>

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor sum_all(const Tensor& x);
Tensor mean_all(const Tensor& x);
Tensor slice_rows(const Tensor& x, int64_t start, int64_t end);
Tensor gather_last_token_logits(const Tensor& logits,
                                const Tensor& positions,
                                int64_t batch_size,
                                int64_t seq_len,
                                int64_t vocab_size);
Tensor dropout(const Tensor& x, float p = 0.5f, bool training = true);
Tensor masked_fill(const Tensor& x, const Tensor& mask, float value);

} // namespace motifcl
