// Witness for the new f16 matmul autograd: gradients from the f16 path must match the
// fully-f32 reference within f16 precision. Closes ROADMAP item "FP16 backward kernels".
#include <motifcl/motifcl.hpp>
#include <motifcl/ops/fp16.hpp>
#include <motifcl/ops/loss.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace motifcl;

static double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, std::abs((double)a[i] - (double)b[i]));
        den = std::max(den, std::abs((double)b[i]));
    }
    return den > 0 ? num / den : num;
}

int main() {
    auto backend = Backend::create_opencl();
    if (!backend_supports_fp16(backend)) { printf("SKIP: backend has no cl_khr_fp16\n"); return 0; }

    const int M = 32, K = 64, N = 48;
    std::mt19937 rng(0);
    std::normal_distribution<float> g(0.0f, 0.5f);
    std::vector<float> ah(M * K), bh(K * N), th(M * N);
    for (auto& v : ah) v = g(rng);
    for (auto& v : bh) v = g(rng);
    for (auto& v : th) v = g(rng);

    auto a32 = Tensor::from_cpu(backend, {M, K}, DType::F32, ah.data());
    auto b32 = Tensor::from_cpu(backend, {K, N}, DType::F32, bh.data());
    auto target = Tensor::from_cpu(backend, {M, N}, DType::F32, th.data());

    // f32 reference
    a32.set_requires_grad(true);
    b32.set_requires_grad(true);
    { auto out = matmul(a32, b32); auto loss = mse_loss(out, target); loss.backward(); }
    auto ga32 = a32.grad()->to_vector<float>();
    auto gb32 = b32.grad()->to_vector<float>();

    // f16 path (newly enabled autograd)
    auto a16 = cast_f32_to_f16(a32); a16.set_requires_grad(true);
    auto b16 = cast_f32_to_f16(b32); b16.set_requires_grad(true);
    { auto out = matmul(a16, b16); auto loss = mse_loss(out, target); loss.backward(); }
    auto ga16 = cast_f16_to_f32(*a16.grad()).to_vector<float>();
    auto gb16 = cast_f16_to_f32(*b16.grad()).to_vector<float>();

    double ea = rel_err(ga16, ga32);
    double eb = rel_err(gb16, gb32);
    printf("f16 matmul autograd vs f32 reference:\n");
    printf("  grad_a rel-err = %.3e\n", ea);
    printf("  grad_b rel-err = %.3e\n", eb);
    const bool ok = ea < 5e-2 && eb < 5e-2;
    printf(ok ? "PASS: f16 matmul autograd matches f32 within f16 precision\n"
              : "FAIL: f16 gradients diverge from f32 reference\n");
    return ok ? 0 : 1;
}
