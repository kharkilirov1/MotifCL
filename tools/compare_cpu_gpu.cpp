#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>

namespace {
float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
}

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::cout << "Device: " << backend.device_info().device_name << "\n";

        std::vector<float> ah = {1, 2, 3, 4, 5, 6};
        std::vector<float> bh = {7, 8, 9, 10, 11, 12};
        auto A = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, ah.data());
        auto B = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32, bh.data());
        auto C = motifcl::matmul(A, B).to_vector<float>();
        std::vector<float> Cref = {58, 64, 139, 154};
        std::cout << "matmul max_abs_error=" << max_abs_diff(C, Cref) << "\n";

        std::vector<float> sx = {1, 2, 3, 1, 1, 1};
        auto S = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, sx.data());
        auto P = motifcl::softmax_rows(S).to_vector<float>();
        std::cout << "softmax row sums=" << (P[0] + P[1] + P[2]) << ", " << (P[3] + P[4] + P[5]) << "\n";

        auto X = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::F32, sx.data());
        auto W = motifcl::Tensor::ones(backend, {4});
        auto R = motifcl::rmsnorm(X, W).to_vector<float>();
        float rms = std::sqrt((1 + 4 + 9 + 1) / 4.0f + 1e-6f);
        std::vector<float> Rref = {1 / rms, 2 / rms, 3 / rms, 1 / rms};
        std::cout << "rmsnorm max_abs_error=" << max_abs_diff(R, Rref) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "OpenCL unavailable or compare failed: " << e.what() << "\n";
        return 0;
    }
}
