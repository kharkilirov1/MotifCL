// Regression for the native finite-state counter linear layer (CounterStateLinear).
// Teacher-recovery: a ternary linear target must be recovered by the fused in-backward
// counter update (row-RMS + stochastic rounding) running on the OpenCL device.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/ops/loss.hpp>
#include <motifcl/autograd/node.hpp>

#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();

        const int n = 16, N = 256, C = 11, steps = 800;
        const float teacher_scale = 0.25f, lr = 0.005f;

        std::mt19937 rng(0);
        std::uniform_int_distribution<int> tern(-1, 1);
        std::normal_distribution<float> gauss(0.0f, 1.0f);

        std::vector<int> tw(n * n);
        for (int& w : tw) w = tern(rng);
        std::vector<float> xh(N * n);
        for (float& xi : xh) xi = gauss(rng);
        std::vector<float> yh(N * n, 0.0f);
        for (int r = 0; r < N; ++r)
            for (int o = 0; o < n; ++o) {
                float a = 0.0f;
                for (int i = 0; i < n; ++i) a += xh[r * n + i] * (teacher_scale * (float)tw[o * n + i]);
                yh[r * n + o] = a;
            }

        auto x = motifcl::Tensor::from_cpu(backend, {N, n}, motifcl::DType::F32, xh.data());
        x.set_requires_grad(true);
        auto y = motifcl::Tensor::from_cpu(backend, {N, n}, motifcl::DType::F32, yh.data());

        const float base = std::sqrt(3.0f / (2.0f * (float)n));
        motifcl::nn::CounterStateLinear layer(backend, n, n, C, lr, 0.0f, teacher_scale / base);

        for (int s = 0; s < steps; ++s) {
            auto pred = layer.forward(x);
            auto loss = motifcl::mse_loss(pred, y);
            loss.backward();
        }

        motifcl::autograd::set_enabled(false);
        auto pred = layer.forward(x);
        auto ph = pred.to_vector<float>();
        motifcl::autograd::set_enabled(true);

        double mse = 0.0;
        for (int k = 0; k < N * n; ++k) { double d = (double)ph[k] - (double)yh[k]; mse += d * d; }
        mse /= (double)(N * n);

        auto sh = layer.state.to_vector<std::uint8_t>();  // packed [out, in/4, 3]
        const int lv = 2 * C - 1, gpr = n / 4;
        int hit = 0;
        for (int row = 0; row < n; ++row)
            for (int g = 0; g < gpr; ++g) {
                const std::uint8_t* p = sh.data() + (static_cast<std::size_t>(row) * gpr + g) * 3;
                int code[4];
                code[0] = p[0] & 0x3F;
                code[1] = ((p[0] >> 6) | (p[1] << 2)) & 0x3F;
                code[2] = ((p[1] >> 4) | (p[2] << 4)) & 0x3F;
                code[3] = (p[2] >> 2) & 0x3F;
                for (int kk = 0; kk < 4; ++kk) {
                    int t = code[kk] / lv - 1;
                    if (t == tw[row * n + g * 4 + kk]) ++hit;
                }
            }
        const float acc = (float)hit / (float)(n * n);

        if (mse > 5e-3 || acc < 0.9f) {
            std::cerr << "counter teacher-recovery failed: mse=" << mse << " acc=" << acc << "\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
