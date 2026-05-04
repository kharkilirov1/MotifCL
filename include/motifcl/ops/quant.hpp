#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor quantize_q8_symmetric(const Tensor& x, float scale = 0.0f);
Tensor dequantize_q8(const Tensor& x);
Tensor quantize_q4_symmetric(const Tensor& x, float scale = 0.0f);
Tensor dequantize_q4(const Tensor& x);

Tensor quantize_q8_symmetric_axis(const Tensor& x, int axis);
Tensor quantize_q8_symmetric_rows(const Tensor& x);
Tensor quantize_q8_symmetric_cols(const Tensor& x);
Tensor quantize_q8_symmetric_blocks(const Tensor& x, int64_t block_size);

Tensor quantize_q4_symmetric_axis(const Tensor& x, int axis);
Tensor quantize_q4_symmetric_rows(const Tensor& x);
Tensor quantize_q4_symmetric_cols(const Tensor& x);
Tensor quantize_q4_symmetric_blocks(const Tensor& x, int64_t block_size);

} // namespace motifcl
