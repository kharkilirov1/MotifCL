#pragma once

#include <cstdint>

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor sum_rows(const Tensor& x);
Tensor rowwise_sum(const Tensor& x);
Tensor rowwise_max(const Tensor& x);
Tensor rowwise_argmax(const Tensor& x);
Tensor rowwise_sample(const Tensor& x, float temperature = 0.0f, int top_k = 0, std::uint32_t seed = 1234);
Tensor rowwise_sample_top_p(const Tensor& x, float temperature = 0.0f, int top_k = 0,
                            float top_p = 1.0f, std::uint32_t seed = 1234);
Tensor rms_per_row(const Tensor& x, float eps = 1e-6f);

} // namespace motifcl
