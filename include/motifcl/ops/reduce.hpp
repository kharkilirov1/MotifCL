#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor sum_rows(const Tensor& x);
Tensor rowwise_sum(const Tensor& x);
Tensor rowwise_max(const Tensor& x);
Tensor rms_per_row(const Tensor& x, float eps = 1e-6f);

} // namespace motifcl
