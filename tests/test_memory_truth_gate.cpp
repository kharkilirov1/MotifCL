// Memory truth gate for the finite-state counter layer.
//
// The whole premise of the method is that a training STEP holds no dense weight-sized
// buffers: persistent state is packed 6-bit, and the backward update recomputes the
// weight gradient on the fly instead of materialising a dense [out,in] grad_w. This test
// measures the actual high-water mark of live device memory (every clCreateBuffer is
// counted in Buffer) across one update and asserts:
//
//   1. the fused memory-native update (apply_update_backward) allocates NO weight-sized
//      buffer  -> peak extra < 1/4 of a dense FP32 weight;
//   2. the reference path (dense matmul_transpose_a grad_w + apply_update_seed) DOES
//      allocate a weight-sized buffer -> peak extra >= 1/2 of a dense FP32 weight.
//
// (2) is the "teeth" check: it proves the gate actually detects a dense-memory regression,
// so a future change that reintroduces a dense grad_w (or a saved decoded weight) trips (1).
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/runtime/buffer.hpp>

#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();

        const int D = 256, N = 64, C = 11;

        std::mt19937 rng(0);
        std::normal_distribution<float> g(0.0f, 1.0f);
        std::vector<float> xh(static_cast<std::size_t>(N) * D), gh(static_cast<std::size_t>(N) * D);
        for (float& v : xh) v = g(rng);
        for (float& v : gh) v = 0.01f * g(rng);  // small synthetic upstream gradient

        auto x = motifcl::Tensor::from_cpu(backend, {N, D}, motifcl::DType::F32, xh.data());
        auto grad_out = motifcl::Tensor::from_cpu(backend, {N, D}, motifcl::DType::F32, gh.data());

        motifcl::nn::CounterStateLinear layer(backend, D, D, C, 0.003f);

        // Warm up both paths so one-time kernel/program setup is not charged as "extra".
        layer.apply_update_backward(grad_out, x, 1u);
        { auto gw = motifcl::matmul_transpose_a(grad_out, x); layer.apply_update_seed(gw, 1u); }
        backend.finish();

        const std::size_t weight_bytes = static_cast<std::size_t>(D) * D * sizeof(float);  // dense FP32 [D,D]
        const std::size_t gate = weight_bytes / 4;

        // (1) fused memory-native update: must NOT allocate a weight-sized buffer.
        motifcl::clear_memory_pool();
        motifcl::device_bytes_reset_peak();
        const std::size_t base_f = motifcl::device_bytes_current();
        layer.apply_update_backward(grad_out, x, 7u);
        backend.finish();
        const std::size_t peak_fused = motifcl::device_bytes_peak() - base_f;

        // (2) reference path with a dense grad_w: must blow past the gate (teeth check).
        motifcl::clear_memory_pool();
        motifcl::device_bytes_reset_peak();
        const std::size_t base_r = motifcl::device_bytes_current();
        {
            auto gw = motifcl::matmul_transpose_a(grad_out, x);  // dense [D,D] weight gradient
            layer.apply_update_seed(gw, 7u);
        }
        backend.finish();
        const std::size_t peak_ref = motifcl::device_bytes_peak() - base_r;

        // (3) in-kernel grad_x: numeric parity vs decode+matmul, and no dense [out,in] weight.
        motifcl::device_bytes_reset_peak();
        const std::size_t base_gx = motifcl::device_bytes_current();
        auto gx_kernel = layer.backward_input_from_state(grad_out);  // grad_x decoded from state
        backend.finish();
        const std::size_t peak_gx = motifcl::device_bytes_peak() - base_gx;  // ~N*in output, no [out,in]

        auto w_ref = layer.decode_weight();
        auto gx_ref = motifcl::matmul(grad_out, w_ref);             // reference: dense weight + matmul
        backend.finish();
        auto a = gx_kernel.to_vector<float>();
        auto b = gx_ref.to_vector<float>();
        double num = 0.0, den = 0.0;
        for (std::size_t t = 0; t < a.size(); ++t) {
            double d = static_cast<double>(a[t]) - static_cast<double>(b[t]);
            num += d * d; den += static_cast<double>(b[t]) * static_cast<double>(b[t]);
        }
        const double gx_relerr = den > 0.0 ? std::sqrt(num / den) : 0.0;
        std::cout << "grad_x: in-kernel vs decode+matmul rel-err=" << gx_relerr
                  << " peak_gx=" << peak_gx << "B (weight=" << weight_bytes << "B)\n";
        if (gx_relerr > 1e-4) {
            std::cerr << "GRAD_X PARITY FAILED: in-kernel grad_x rel-err " << gx_relerr << " > 1e-4\n";
            return 1;
        }
        if (peak_gx >= weight_bytes / 2) {
            std::cerr << "GRAD_X NOT MEMORY-NATIVE: path allocated a weight-sized buffer ("
                      << peak_gx << " >= " << weight_bytes / 2 << ")\n";
            return 1;
        }

        std::cout << "memory truth gate: weight_bytes=" << weight_bytes
                  << " gate(1/4)=" << gate
                  << " peak_fused=" << peak_fused
                  << " peak_ref=" << peak_ref << "\n";

        if (peak_fused >= gate) {
            std::cerr << "TRUTH GATE FAILED: fused backward materialised a weight-sized buffer ("
                      << peak_fused << " >= " << gate << " bytes)\n";
            return 1;
        }
        if (peak_ref < weight_bytes / 2) {
            std::cerr << "TRUTH GATE TOOTHLESS: reference dense-grad_w path did not allocate a "
                         "weight-sized buffer (" << peak_ref << " < " << weight_bytes / 2 << " bytes)\n";
            return 1;
        }
        std::cout << "OK: fused update peak " << peak_fused << "B < " << gate
                  << "B; reference path " << peak_ref << "B >= " << weight_bytes / 2 << "B "
                  << "(gate detects dense-memory regressions)\n";
        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
