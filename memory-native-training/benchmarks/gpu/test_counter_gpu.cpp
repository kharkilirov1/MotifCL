// Slice 4 GPU verification: train the native CounterStateLinear on the
// teacher-recovery task (same as counter_state_C_ablation.py) on the real
// OpenCL device. Pass = MSE collapses and the ternary pattern is recovered.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/ops/loss.hpp>
#include <motifcl/autograd/node.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace motifcl;

int main() {
    auto backend = Backend::create_opencl();

    const int n = 16, N = 256, C = 11, steps = 1200;
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

    auto x = Tensor::from_cpu(backend, {N, n}, DType::F32, xh.data());
    x.set_requires_grad(true);
    auto y = Tensor::from_cpu(backend, {N, n}, DType::F32, yh.data());

    // start scale near the teacher scale; keep it fixed (lr_scale=0) for a clean recovery
    const float base_scale = std::sqrt(3.0f / (2.0f * (float)n));
    const float init_gain = teacher_scale / base_scale;
    nn::CounterStateLinear layer(backend, n, n, C, lr, 0.0f, init_gain);

    for (int s = 0; s < steps; ++s) {
        auto pred = layer.forward(x);
        auto loss = mse_loss(pred, y);
        loss.backward();
        if (s % 200 == 0 || s == steps - 1) printf("step=%4d mse=%.6f\n", s, loss.item());
    }

    autograd::set_enabled(false);
    auto pred = layer.forward(x);
    auto ph = pred.to_vector<float>();
    autograd::set_enabled(true);
    double mse = 0.0;
    for (int k = 0; k < N * n; ++k) { double d = (double)ph[k] - (double)yh[k]; mse += d * d; }
    mse /= (double)(N * n);

    auto sh = layer.state.to_vector<std::uint8_t>();
    const int lv = 2 * C - 1;
    int hit = 0;
    for (int k = 0; k < n * n; ++k) { int t = (int)sh[k] / lv - 1; if (t == tw[k]) ++hit; }
    const float acc = (float)hit / (float)(n * n);

    printf("FINAL mse=%.6f ternary-acc=%.4f\n", mse, acc);
    const bool ok = (mse < 5e-3) && (acc > 0.9f);
    printf(ok ? "PASS: native GPU counter layer recovers teacher\n" : "CHECK: outside band\n");
    return ok ? 0 : 1;
}
