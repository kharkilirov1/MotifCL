// Profiler breakdown of the native counter layer: where does the 1.07x vs dense go?
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/ops/loss.hpp>

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace motifcl;

int main() {
    auto backend = Backend::create_opencl();
    const int in = 2048, out = 2048, N = 256, iters = 30, warmup = 5;

    std::vector<float> xh(static_cast<std::size_t>(N) * in, 0.01f);
    auto x = Tensor::from_cpu(backend, {N, in}, DType::F32, xh.data());
    x.set_requires_grad(true);
    std::vector<float> yh(static_cast<std::size_t>(N) * out, 0.0f);
    auto y = Tensor::from_cpu(backend, {N, out}, DType::F32, yh.data());

    nn::CounterStateLinear cl(backend, in, out, 11, 0.005f, 0.0f);
    for (int i = 0; i < warmup; ++i) { auto p = cl.forward(x); auto l = mse_loss(p, y); l.backward(); }
    backend.finish();

    backend.profiler.set_enabled(true);
    for (int i = 0; i < iters; ++i) { auto p = cl.forward(x); auto l = mse_loss(p, y); l.backward(); }
    backend.finish();

    auto summary = backend.profiler.summary();
    std::sort(summary.begin(), summary.end(),
              [](const auto& a, const auto& b) { return a.total_ms > b.total_ms; });

    double tot = 0.0;
    for (const auto& it : summary) tot += it.total_ms;
    printf("counter layer %dx%d, batch %d, %d iters fwd+bwd:\n", in, out, N, iters);
    printf("%-42s %8s %10s %10s\n", "kernel/op", "count", "total_ms", "avg_ms");
    for (const auto& it : summary) {
        if (it.total_ms < 0.05) continue;
        printf("%-42s %8d %10.2f %10.4f\n", it.name.c_str(),
               (int)it.count, it.total_ms, it.avg_ms);
    }
    printf("TOTAL profiled %.2f ms = %.3f ms/iter\n", tot, tot / iters);
    return 0;
}
