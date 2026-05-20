#include <motifcl/runtime/native_matmul.hpp>

#include <motifcl/core/error.hpp>

#include <cstdint>

namespace motifcl {
namespace native {

void matmul_f32_m1(const float* a, const float* b, float* out, std::int64_t k, std::int64_t n) {
    MCL_CHECK(k >= 0, "native matmul f32 m1 requires non-negative K");
    MCL_CHECK(n >= 0, "native matmul f32 m1 requires non-negative N");
    MCL_CHECK(k == 0 || a != nullptr, "native matmul f32 m1 requires non-null lhs for K > 0");
    MCL_CHECK(k == 0 || n == 0 || b != nullptr, "native matmul f32 m1 requires non-null rhs for K*N > 0");
    MCL_CHECK(n == 0 || out != nullptr, "native matmul f32 m1 requires non-null output for N > 0");

    for (std::int64_t col = 0; col < n; ++col) {
        float acc = 0.0f;
        for (std::int64_t kk = 0; kk < k; ++kk) {
            acc += a[kk] * b[kk * n + col];
        }
        out[col] = acc;
    }
}

} // namespace native
} // namespace motifcl
