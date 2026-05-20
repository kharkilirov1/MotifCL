#include <motifcl/runtime/native_matmul.hpp>

#include <motifcl/core/error.hpp>

#include <cstdint>

#if defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#define MOTIFCL_NATIVE_MATMUL_HAS_SSE 1
#include <xmmintrin.h>
#endif

namespace motifcl {
namespace native {
namespace {

void validate_matmul_f32_m1_args(const float* a, const float* b, const float* out, std::int64_t k, std::int64_t n) {
    MCL_CHECK(k >= 0, "native matmul f32 m1 requires non-negative K");
    MCL_CHECK(n >= 0, "native matmul f32 m1 requires non-negative N");
    MCL_CHECK(k == 0 || a != nullptr, "native matmul f32 m1 requires non-null lhs for K > 0");
    MCL_CHECK(k == 0 || n == 0 || b != nullptr, "native matmul f32 m1 requires non-null rhs for K*N > 0");
    MCL_CHECK(n == 0 || out != nullptr, "native matmul f32 m1 requires non-null output for N > 0");
}

#if defined(MOTIFCL_NATIVE_MATMUL_HAS_SSE)
void matmul_f32_m1_sse(const float* a, const float* b, float* out, std::int64_t k, std::int64_t n) {
    std::int64_t col = 0;
    for (; col + 4 <= n; col += 4) {
        __m128 acc = _mm_setzero_ps();
        for (std::int64_t kk = 0; kk < k; ++kk) {
            const __m128 av = _mm_set1_ps(a[kk]);
            const __m128 bv = _mm_loadu_ps(b + kk * n + col);
            acc = _mm_add_ps(acc, _mm_mul_ps(av, bv));
        }
        _mm_storeu_ps(out + col, acc);
    }
    for (; col < n; ++col) {
        float acc = 0.0f;
        for (std::int64_t kk = 0; kk < k; ++kk) {
            acc += a[kk] * b[kk * n + col];
        }
        out[col] = acc;
    }
}
#endif

} // namespace

const char* matmul_f32_m1_kernel_name() {
#if defined(MOTIFCL_NATIVE_MATMUL_HAS_SSE)
    return "sse";
#else
    return "scalar";
#endif
}

void matmul_f32_m1_scalar(const float* a, const float* b, float* out, std::int64_t k, std::int64_t n) {
    validate_matmul_f32_m1_args(a, b, out, k, n);

    for (std::int64_t col = 0; col < n; ++col) {
        float acc = 0.0f;
        for (std::int64_t kk = 0; kk < k; ++kk) {
            acc += a[kk] * b[kk * n + col];
        }
        out[col] = acc;
    }
}

void matmul_f32_m1(const float* a, const float* b, float* out, std::int64_t k, std::int64_t n) {
    validate_matmul_f32_m1_args(a, b, out, k, n);
#if defined(MOTIFCL_NATIVE_MATMUL_HAS_SSE)
    matmul_f32_m1_sse(a, b, out, k, n);
#else
    matmul_f32_m1_scalar(a, b, out, k, n);
#endif
}

} // namespace native
} // namespace motifcl
