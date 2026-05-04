#include <chrono>
#include <iostream>
#include <motifcl/motifcl.hpp>

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        motifcl::nn::SelfAttention attn(backend, 128, 4);
        auto X = motifcl::Tensor::randn(backend, {128, 128}, 0.02f);
        attn.forward(X);
        backend.finish();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) attn.forward(X);
        backend.finish();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "attention T=128 C=128: " << std::chrono::duration<double, std::milli>(t1 - t0).count() / 10.0 << " ms\n";
    } catch (const std::exception& e) {
        std::cerr << "OpenCL unavailable or benchmark failed: " << e.what() << "\n";
    }
}
