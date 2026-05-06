#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>

namespace {

struct ShapeCase {
    int m;
    int k;
    int n;
};

struct Result {
    std::string name;
    double wall_ms = 0.0;
    double kernel_ms = 0.0;
    double gflops = 0.0;
};

template <typename Fn>
Result run_case(motifcl::Backend& backend,
                const std::string& name,
                int m,
                int k,
                int n,
                int iters,
                Fn&& fn) {
    for (int i = 0; i < 2; ++i) {
        auto out = fn();
        (void)out;
    }
    backend.finish();

    double wall_total = 0.0;
    double kernel_total = 0.0;
    backend.profiler.set_enabled(true);
    for (int i = 0; i < iters; ++i) {
        backend.profiler.clear();
        const auto t0 = std::chrono::steady_clock::now();
        auto out = fn();
        (void)out;
        backend.finish();
        const auto t1 = std::chrono::steady_clock::now();
        wall_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        for (const auto& item : backend.profiler.summary()) kernel_total += item.total_ms;
    }
    backend.profiler.set_enabled(false);

    Result result;
    result.name = name;
    result.wall_ms = wall_total / static_cast<double>(iters);
    result.kernel_ms = kernel_total / static_cast<double>(iters);
    result.gflops = (2.0 * static_cast<double>(m) * static_cast<double>(k) * static_cast<double>(n)) /
                    (result.kernel_ms * 1.0e6);
    return result;
}

void print_result(const ShapeCase& shape, const Result& result) {
    std::cout << std::left << std::setw(18) << result.name
              << " M=" << std::setw(4) << shape.m
              << " K=" << std::setw(4) << shape.k
              << " N=" << std::setw(5) << shape.n
              << std::right << " wall_ms=" << std::setw(10) << result.wall_ms
              << " kernel_ms=" << std::setw(10) << result.kernel_ms
              << " GFLOP/s=" << std::setw(10) << result.gflops
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int iters = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5;
        auto backend = motifcl::Backend::create_opencl();
        const auto info = backend.device_info();
        std::cout << "device=" << info.device_name << " driver=" << info.driver_version
                  << " fp16=" << (motifcl::backend_supports_fp16(backend) ? "yes" : "no")
                  << " iters=" << iters << "\n";

        const std::vector<ShapeCase> shapes = {
            {256, 512, 2304},
            {256, 512, 512},
            {256, 512, 256},
        };

        for (const auto& shape : shapes) {
            auto a = motifcl::Tensor::randn(backend, {shape.m, shape.k}, 0.02f);
            auto b = motifcl::Tensor::randn(backend, {shape.k, shape.n}, 0.02f);

            print_result(shape, run_case(backend, "f32_default", shape.m, shape.k, shape.n, iters, [&]() {
                return motifcl::matmul(a, b);
            }));
            for (int tile : {4, 8, 16}) {
                print_result(shape, run_case(backend, "f32_tile" + std::to_string(tile), shape.m, shape.k, shape.n, iters, [&]() {
                    return motifcl::matmul_tiled_variant(a, b, tile);
                }));
            }

            if (motifcl::backend_supports_fp16(backend)) {
                auto a16 = motifcl::cast_f32_to_f16(a);
                auto b16 = motifcl::cast_f32_to_f16(b);
                backend.finish();
                print_result(shape, run_case(backend, "f16_accum_f32", shape.m, shape.k, shape.n, iters, [&]() {
                    return motifcl::matmul_f16_accum_f32(a16, b16);
                }));
            }
            std::cout << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "OpenCL unavailable or benchmark failed: " << e.what() << "\n";
        return 1;
    }
}
