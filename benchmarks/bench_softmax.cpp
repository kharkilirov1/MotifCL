#include <chrono>
#include <iostream>
#include <motifcl/motifcl.hpp>

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        auto X = motifcl::Tensor::randn(backend, {1024, 1024}, 1.0f);
        motifcl::softmax_rows(X);
        backend.finish();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; ++i) motifcl::softmax_rows(X);
        backend.finish();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "softmax_rows 1024x1024: " << std::chrono::duration<double, std::milli>(t1 - t0).count() / 20.0 << " ms\n";
    } catch (const std::exception& e) {
        std::cerr << "OpenCL unavailable or benchmark failed: " << e.what() << "\n";
    }
}
