// Regression for f16 matmul autograd: gradients from the f16 path match a fully-f32
// reference within f16 precision. Skips (77) when the device has no cl_khr_fp16.
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/ops/fp16.hpp>
#include <motifcl/ops/loss.hpp>

#include "test_utils.hpp"

namespace {
double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, std::abs((double)a[i] - (double)b[i]));
        den = std::max(den, std::abs((double)b[i]));
    }
    return den > 0 ? num / den : num;
}
} // namespace

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        if (!motifcl::backend_supports_fp16(backend)) return 77;  // skip: no fp16

        const int M = 32, K = 64, N = 48;
        std::mt19937 rng(0);
        std::normal_distribution<float> g(0.0f, 0.5f);
        std::vector<float> ah(M * K), bh(K * N), th(M * N);
        for (auto& v : ah) v = g(rng);
        for (auto& v : bh) v = g(rng);
        for (auto& v : th) v = g(rng);

        auto a32 = motifcl::Tensor::from_cpu(backend, {M, K}, motifcl::DType::F32, ah.data());
        auto b32 = motifcl::Tensor::from_cpu(backend, {K, N}, motifcl::DType::F32, bh.data());
        auto target = motifcl::Tensor::from_cpu(backend, {M, N}, motifcl::DType::F32, th.data());

        a32.set_requires_grad(true);
        b32.set_requires_grad(true);
        { auto out = motifcl::matmul(a32, b32); auto loss = motifcl::mse_loss(out, target); loss.backward(); }
        auto ga32 = a32.grad()->to_vector<float>();
        auto gb32 = b32.grad()->to_vector<float>();

        auto a16 = motifcl::cast_f32_to_f16(a32); a16.set_requires_grad(true);
        auto b16 = motifcl::cast_f32_to_f16(b32); b16.set_requires_grad(true);
        { auto out = motifcl::matmul(a16, b16); auto loss = motifcl::mse_loss(out, target); loss.backward(); }
        auto ga16 = motifcl::cast_f16_to_f32(*a16.grad()).to_vector<float>();
        auto gb16 = motifcl::cast_f16_to_f32(*b16.grad()).to_vector<float>();

        double ea = rel_err(ga16, ga32), eb = rel_err(gb16, gb32);
        if (ea > 5e-2 || eb > 5e-2) {
            std::cerr << "f16 matmul autograd diverged: grad_a=" << ea << " grad_b=" << eb << "\n";
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
