// CPU-reference parity for the Vulkan F32 GEMM family.
//
// The other Vulkan matmul parity check (test_vulkan_backend) builds its
// reference through OpenCL tensors, so it skips on hosts without an OpenCL ICD.
// This test compares every layout (A*B, A*B^T, A^T*B) against a host reference
// instead, so the double-buffered vec4 kernels that the trainers now use by
// default stay covered on Vulkan-only machines.
//
// Shapes cover: exact tile multiples (64x64, 32x32), partial output tiles,
// partial K tiles, and dimensions that are multiples of four but not of 32/64
// (the vec4 body requires multiples of four). Shapes with dimensions that are
// not multiples of four exercise the scalar fallback path.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/ops/matmul.hpp>

#include "vulkan_p0_test_utils.hpp"

int main(int argc, char** argv) {
    using namespace motifcl;
    using motifcl_vulkan_p0_test::require_close;
    try {
        motifcl_vulkan_p0_test::verify_require_close_controls();
    } catch (const std::exception& e) {
        std::cerr << "FAIL GEMM comparator controls: " << e.what() << '\n';
        return 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--comparators-only") {
        std::cout << "PASS GEMM comparator controls: finite, NaN, Inf, size, tolerance\n";
        return 0;
    }

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::printf("Skipping Vulkan GEMM parity: %s\n", probe.error.c_str());
        return 77;
    }

    try {
        Backend backend = Backend::create_vulkan();
        std::mt19937 rng(20261018u);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        struct Case {
            const char* op;
            std::int64_t m;
            std::int64_t k;
            std::int64_t n;
        };
        const std::vector<Case> cases = {
            {"nn", 64, 64, 64},    // exact 64 tile
            {"nn", 32, 32, 32},    // exact 32 tile
            {"nn", 100, 44, 68},   // partial tiles + partial K tile (all %4 == 0)
            {"nn", 2048, 288, 384},// trainer expert-up shape
            {"nn", 37, 19, 23},    // scalar fallback path (odd dimensions)
            {"nt", 64, 64, 64},
            {"nt", 100, 44, 68},
            {"nt", 2048, 32000, 288}, // trainer LM-head dA shape (large K)
            {"nt", 37, 19, 23},
            {"tn", 64, 64, 64},
            {"tn", 100, 44, 68},
            {"tn", 384, 2048, 288}, // trainer counter grad_w shape
            {"tn", 37, 19, 23},
        };

        for (const auto& c : cases) {
            const auto m = static_cast<std::size_t>(c.m);
            const auto k = static_cast<std::size_t>(c.k);
            const auto n = static_cast<std::size_t>(c.n);
            const bool transpose_a = std::string(c.op) == "tn";
            const bool transpose_b = std::string(c.op) == "nt";
            // Logical A is [m,k] and B is [k,n]; transposed layouts store the
            // operand with its leading dimension swapped, matching the ops API.
            const std::size_t a_elems = transpose_a ? k * m : m * k;
            const std::size_t b_elems = transpose_b ? n * k : k * n;

            std::vector<float> ah(a_elems);
            std::vector<float> bh(b_elems);
            for (auto& value : ah) value = dist(rng);
            for (auto& value : bh) value = dist(rng);

            std::vector<float> ref(m * n, 0.0f);
            for (std::size_t row = 0; row < m; ++row) {
                for (std::size_t col = 0; col < n; ++col) {
                    double acc = 0.0;
                    for (std::size_t kk = 0; kk < k; ++kk) {
                        const float av = transpose_a ? ah[kk * m + row] : ah[row * k + kk];
                        const float bv = transpose_b ? bh[col * k + kk] : bh[kk * n + col];
                        acc += static_cast<double>(av) * static_cast<double>(bv);
                    }
                    ref[row * n + col] = static_cast<float>(acc);
                }
            }

            const Shape a_shape = transpose_a ? Shape{c.k, c.m} : Shape{c.m, c.k};
            const Shape b_shape = transpose_b ? Shape{c.n, c.k} : Shape{c.k, c.n};
            auto va = Tensor::from_cpu(backend, a_shape, DType::F32, ah.data());
            auto vb = Tensor::from_cpu(backend, b_shape, DType::F32, bh.data());

            Tensor vc;
            if (transpose_a) {
                vc = matmul_transpose_a(va, vb);
            } else if (transpose_b) {
                vc = matmul_transpose_b(va, vb);
            } else {
                vc = matmul(va, vb);
            }
            const auto got = vc.to_vector<float>();
            const std::string label = std::string("Vulkan GEMM ") + c.op + " " +
                std::to_string(c.m) + "x" + std::to_string(c.k) + "x" +
                std::to_string(c.n);
            // Tolerances are dominated by the reference being f64: the kernel
            // accumulates in f32, so large-K shapes need a scaled bound.
            const float tol_abs = 1e-4f + 1e-6f * static_cast<float>(k);
            // The same comparator checked by the CPU-only negative controls
            // validates every element before diagnostic maxima are accumulated.
            require_close(got, ref, tol_abs, label);
            double max_abs = 0.0;
            for (std::size_t i = 0; i < ref.size(); ++i) {
                const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
                max_abs = std::max(max_abs, diff);
            }
            std::printf("  %-3s %5lld x %5lld x %5lld  max_abs=%.3e\n", c.op,
                        static_cast<long long>(c.m), static_cast<long long>(c.k),
                        static_cast<long long>(c.n), max_abs);
        }
    } catch (const std::exception& e) {
        std::cerr << "Vulkan GEMM parity threw: " << e.what() << '\n';
        return 1;
    }

    std::printf("PASS test_vulkan_gemm_parity: nn/nt/tn layouts match a host reference\n");
    return 0;
}
