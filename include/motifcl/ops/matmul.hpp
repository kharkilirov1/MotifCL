#pragma once

#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

Tensor matmul(const Tensor& a, const Tensor& b);
Tensor matmul_transpose_a(const Tensor& a, const Tensor& b);
Tensor matmul_transpose_b(const Tensor& a, const Tensor& b);
void matmul_out(const Tensor& a, const Tensor& b, Tensor& out);
Tensor matmul_tiled_variant(const Tensor& a, const Tensor& b, int tile);
Tensor matmul_f16_accum_f32(const Tensor& a, const Tensor& b);

// Override Q4_0 M1 wg64x8 kernel selection without touching the process environment.
// state: -1 follows MOTIFCL_DISABLE_Q4_0_M1_WG64X8, 0 forces the kernel off, 1 forces it on.
// Prefer this over setenv() at runtime: mutating environ races the OpenCL runtime's worker
// threads (POCL calls getenv during kernel build) and can crash.
void set_q4_0_m1_wg64x8_override(int state);

} // namespace motifcl
