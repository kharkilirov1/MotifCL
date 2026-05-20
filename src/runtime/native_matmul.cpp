#include <motifcl/runtime/native_matmul.hpp>

#include <motifcl/core/error.hpp>

#include <cmath>
#include <cstdint>
#include <limits>

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

void validate_matmul_f32_q4_0_m1_args(const float* a,
                                      const std::uint8_t* b,
                                      const float* out,
                                      std::int64_t k,
                                      std::int64_t n,
                                      float scale_b) {
    MCL_CHECK(k >= 0, "native matmul f32/q4_0 m1 requires non-negative K");
    MCL_CHECK(n >= 0, "native matmul f32/q4_0 m1 requires non-negative N");
    MCL_CHECK(k == 0 || a != nullptr, "native matmul f32/q4_0 m1 requires non-null lhs for K > 0");
    MCL_CHECK(k == 0 || n == 0 || b != nullptr, "native matmul f32/q4_0 m1 requires non-null packed rhs for K*N > 0");
    MCL_CHECK(n == 0 || out != nullptr, "native matmul f32/q4_0 m1 requires non-null output for N > 0");
    MCL_CHECK(std::isfinite(scale_b), "native matmul f32/q4_0 m1 requires finite quant scale");
    MCL_CHECK(k == 0 || n <= std::numeric_limits<std::int64_t>::max() / k,
              "native matmul f32/q4_0 m1 logical element count overflow");
}

float q4_0_load(const std::uint8_t* x, std::int64_t idx) {
    const std::uint8_t packed = x[static_cast<std::size_t>(idx >> 1)];
    const std::uint8_t code = (idx & 1) != 0 ? static_cast<std::uint8_t>((packed >> 4) & 15u)
                                             : static_cast<std::uint8_t>(packed & 15u);
    return static_cast<float>(static_cast<int>(code) - 8);
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

void matmul_f32_q4_0_m1_sse(const float* a,
                            const std::uint8_t* b,
                            float* out,
                            std::int64_t k,
                            std::int64_t n,
                            float scale_b) {
    std::int64_t col = 0;
    for (; col + 4 <= n; col += 4) {
        __m128 acc = _mm_setzero_ps();
        for (std::int64_t kk = 0; kk < k; ++kk) {
            const std::int64_t base = kk * n + col;
            const __m128 av = _mm_set1_ps(a[kk] * scale_b);
            const __m128 bv = _mm_set_ps(q4_0_load(b, base + 3),
                                         q4_0_load(b, base + 2),
                                         q4_0_load(b, base + 1),
                                         q4_0_load(b, base));
            acc = _mm_add_ps(acc, _mm_mul_ps(av, bv));
        }
        _mm_storeu_ps(out + col, acc);
    }
    for (; col < n; ++col) {
        float acc = 0.0f;
        for (std::int64_t kk = 0; kk < k; ++kk) {
            acc += a[kk] * q4_0_load(b, kk * n + col) * scale_b;
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

const char* matmul_f32_q4_0_m1_kernel_name() {
#if defined(MOTIFCL_NATIVE_MATMUL_HAS_SSE)
    return "sse_q4_0";
#else
    return "scalar_q4_0";
#endif
}

void matmul_f32_q4_0_m1_scalar(const float* a,
                               const std::uint8_t* b,
                               float* out,
                               std::int64_t k,
                               std::int64_t n,
                               float scale_b) {
    validate_matmul_f32_q4_0_m1_args(a, b, out, k, n, scale_b);

    for (std::int64_t col = 0; col < n; ++col) {
        float acc = 0.0f;
        for (std::int64_t kk = 0; kk < k; ++kk) {
            acc += a[kk] * q4_0_load(b, kk * n + col) * scale_b;
        }
        out[col] = acc;
    }
}

void matmul_f32_q4_0_m1(const float* a,
                        const std::uint8_t* b,
                        float* out,
                        std::int64_t k,
                        std::int64_t n,
                        float scale_b) {
    validate_matmul_f32_q4_0_m1_args(a, b, out, k, n, scale_b);
#if defined(MOTIFCL_NATIVE_MATMUL_HAS_SSE)
    matmul_f32_q4_0_m1_sse(a, b, out, k, n, scale_b);
#else
    matmul_f32_q4_0_m1_scalar(a, b, out, k, n, scale_b);
#endif
}

} // namespace native
} // namespace motifcl
