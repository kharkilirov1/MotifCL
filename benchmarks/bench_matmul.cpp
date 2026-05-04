#include <chrono>
#include <iostream>
#include <motifcl/motifcl.hpp>

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        for (int n : {128, 256, 512}) {
            auto A = motifcl::Tensor::randn(backend, {n, n}, 0.02f);
            auto B = motifcl::Tensor::randn(backend, {n, n}, 0.02f);
            motifcl::matmul(A, B);
            backend.finish();
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 5; ++i) motifcl::matmul(A, B);
            backend.finish();
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
            double gflops = (2.0 * n * n * n) / (ms * 1.0e6);
            std::cout << "matmul " << n << "x" << n << ": " << ms << " ms, " << gflops << " GFLOP/s\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "OpenCL unavailable or benchmark failed: " << e.what() << "\n";
    }
}
