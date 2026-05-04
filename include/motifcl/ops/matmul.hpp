#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matmul_transpose_a(const Tensor& a, const Tensor& b);
Tensor matmul_transpose_b(const Tensor& a, const Tensor& b);
void matmul_out(const Tensor& a, const Tensor& b, Tensor& out);
Tensor matmul_tiled_variant(const Tensor& a, const Tensor& b, int tile);

} // namespace motifcl
