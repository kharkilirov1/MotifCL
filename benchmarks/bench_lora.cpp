#include <chrono>
#include <iostream>
#include <motifcl/motifcl.hpp>

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        motifcl::motif::MotifLoRA layer(backend, 256, 256, 8, 4);
        auto X = motifcl::Tensor::randn(backend, {128, 256}, 0.02f);
        layer.forward(X);
        backend.finish();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) layer.forward(X);
        backend.finish();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "MotifLoRA B=128 in=256 out=256 rank=8 motifs=4: " << std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0 << " ms\n";
    } catch (const std::exception& e) {
        std::cerr << "OpenCL unavailable or benchmark failed: " << e.what() << "\n";
    }
}
