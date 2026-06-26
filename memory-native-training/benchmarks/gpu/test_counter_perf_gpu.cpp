// Slice 5 perf-gate: forward+backward throughput of the native counter layer vs a
// dense FP32 Linear+Adam at a realistic layer size on the GPU.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/nn/linear.hpp>
#include <motifcl/ops/loss.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

using namespace motifcl;
using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

int main() {
    auto backend = Backend::create_opencl();
    const int in = 2048, out = 2048, N = 256, iters = 20, warmup = 5;

    std::vector<float> xh(static_cast<std::size_t>(N) * in, 0.01f);
    auto x = Tensor::from_cpu(backend, {N, in}, DType::F32, xh.data());
    x.set_requires_grad(true);
    std::vector<float> yh(static_cast<std::size_t>(N) * out, 0.0f);
    auto y = Tensor::from_cpu(backend, {N, out}, DType::F32, yh.data());

    // counter layer (self-updating)
    nn::CounterStateLinear cl(backend, in, out, 11, 0.005f, 0.0f);
    for (int i = 0; i < warmup; ++i) { auto p = cl.forward(x); auto l = mse_loss(p, y); l.backward(); }
    backend.finish();
    auto t0 = clk::now();
    for (int i = 0; i < iters; ++i) { auto p = cl.forward(x); auto l = mse_loss(p, y); l.backward(); }
    backend.finish();
    double counter_ms = ms_since(t0) / iters;

    // dense FP32 Linear + Adam
    nn::Linear dl(backend, in, out, false);
    optim::Adam opt(dl.parameters(), 1e-3f);
    for (int i = 0; i < warmup; ++i) { auto p = dl.forward(x); auto l = mse_loss(p, y); l.backward(); opt.step(); opt.zero_grad(); }
    backend.finish();
    auto t1 = clk::now();
    for (int i = 0; i < iters; ++i) { auto p = dl.forward(x); auto l = mse_loss(p, y); l.backward(); opt.step(); opt.zero_grad(); }
    backend.finish();
    double dense_ms = ms_since(t1) / iters;

    printf("layer %dx%d, batch %d, fwd+bwd:\n", in, out, N);
    printf("  counter (decode+matmul+2-pass update): %.2f ms/iter\n", counter_ms);
    printf("  dense   (matmul+backward+Adam):         %.2f ms/iter\n", dense_ms);
    printf("  counter/dense = %.2fx\n", counter_ms / dense_ms);
    return 0;
}
