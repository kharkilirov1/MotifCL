#pragma once

#include <cstdint>

namespace motifcl {
namespace native {

// Host-native scalar F32 decode matmul core for a single-row lhs:
//   out[0, n] = a[0, k] x b[k, n]
//
// This is intentionally a small C++ ABI boundary. It gives the runtime a stable
// replacement point for future SIMD/ASM implementations without changing the
// Tensor/OpenCL-facing op dispatch.
void matmul_f32_m1(const float* a, const float* b, float* out, std::int64_t k, std::int64_t n);

} // namespace native
} // namespace motifcl
