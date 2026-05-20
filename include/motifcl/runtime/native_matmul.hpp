#pragma once

#include <cstdint>

namespace motifcl {
namespace native {

// Host-native F32 decode matmul core for a single-row lhs:
//   out[0, n] = a[0, k] x b[k, n]
//
// This is intentionally a small C++ ABI boundary. It gives the runtime a stable
// replacement point for future SIMD/ASM implementations without changing the
// Tensor/OpenCL-facing op dispatch.
const char* matmul_f32_m1_kernel_name();
void matmul_f32_m1_scalar(const float* a, const float* b, float* out, std::int64_t k, std::int64_t n);
void matmul_f32_m1(const float* a, const float* b, float* out, std::int64_t k, std::int64_t n);

// Host-native F32 x packed-Q4_0 decode matmul core for a single-row lhs:
//   out[0, n] = sum_k a[0, k] * (q4_0_load(b[k, n]) * scale_b)
//
// Q4_0 uses the same logical row-major packing as the OpenCL q4_0 kernels:
// two signed 4-bit values per byte, with code 0..15 mapped to -8..7.
const char* matmul_f32_q4_0_m1_kernel_name();
void matmul_f32_q4_0_m1_scalar(const float* a,
                               const std::uint8_t* b,
                               float* out,
                               std::int64_t k,
                               std::int64_t n,
                               float scale_b);
void matmul_f32_q4_0_m1(const float* a,
                        const std::uint8_t* b,
                        float* out,
                        std::int64_t k,
                        std::int64_t n,
                        float scale_b);

} // namespace native
} // namespace motifcl
