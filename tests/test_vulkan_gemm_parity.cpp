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

int main() {
    using namespace motifcl;
    bool ok = true;
    auto expect = [&](bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << message << '\n';
            ok = false;
        }
    };

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
            double max_abs = 0.0;
            double max_rel = 0.0;
            for (std::size_t i = 0; i < ref.size() && i < got.size(); ++i) {
                const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
                max_abs = std::max(max_abs, diff);
                const double scale = std::max(1e-6, std::fabs(static_cast<double>(ref[i])));
                max_rel = std::max(max_rel, diff / scale);
            }
            const bool size_ok = got.size() == ref.size();
            // Tolerances are dominated by the reference being f64: the kernel
            // accumulates in f32, so large-K shapes need a scaled bound.
            const double tol_abs = 1e-4 + 1e-6 * static_cast<double>(k);
            const bool values_ok = max_abs <= tol_abs;
            expect(size_ok, std::string("Vulkan GEMM ") + c.op + " output size mismatch for " + std::to_string(c.m) +
                                "x" + std::to_string(c.k) + "x" + std::to_string(c.n));
            expect(values_ok, std::string("Vulkan GEMM ") + c.op + " parity mismatch for " + std::to_string(c.m) + "x" +
                                  std::to_string(c.k) + "x" + std::to_string(c.n) + ": max_abs=" +
                                  std::to_string(max_abs) + " max_rel=" + std::to_string(max_rel) +
                                  " tol=" + std::to_string(tol_abs));
            if (size_ok && values_ok) {
                std::printf("  %-3s %5lld x %5lld x %5lld  max_abs=%.3e\n", c.op,
                            static_cast<long long>(c.m), static_cast<long long>(c.k),
                            static_cast<long long>(c.n), max_abs);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Vulkan GEMM parity threw: " << e.what() << '\n';
        return 1;
    }

    if (!ok) {
        std::cerr << "FAIL test_vulkan_gemm_parity\n";
        return 1;
    }
    std::printf("PASS test_vulkan_gemm_parity: nn/nt/tn layouts match a host reference\n");
    return 0;
}
