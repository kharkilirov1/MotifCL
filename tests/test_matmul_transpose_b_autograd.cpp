// matmul_transpose_b (y = a * b^T) must carry gradients on Vulkan. Without the
// backward node a TIED LM head — logits = hidden @ embedding^T, the standard
// weight-sharing trick — silently severs the chain: the forward runs, the loss
// is finite, and nothing upstream ever gets a gradient. This witness checks the
// gradients against the same quantity computed through the plain matmul path.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <motifcl/motifcl.hpp>

namespace {

void fill_deterministic(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t s = seed | 1u;
    for (auto& x : v) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        x = scale * (static_cast<float>(static_cast<std::int32_t>(s % 2001) - 1000) / 1000.0f);
    }
}

motifcl::Tensor make(motifcl::Backend& backend, std::int64_t rows, std::int64_t cols,
                     std::uint32_t seed, bool grad) {
    std::vector<float> host(static_cast<std::size_t>(rows * cols));
    fill_deterministic(host, seed, 0.3f);
    auto t = motifcl::Tensor::from_cpu(backend, {rows, cols}, motifcl::DType::F32, host.data());
    t.set_requires_grad(grad);
    return t;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, static_cast<double>(std::fabs(a[i] - b[i])));
    return worst;
}

} // namespace

int main() {
    using namespace motifcl;
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cerr << "no Vulkan device: " << probe.error << "\n";
        return 77;
    }
    try {
        Backend backend = Backend::create_vulkan();
        const std::int64_t M = 8, K = 16, N = 12;

        // --- through matmul_transpose_b: y = a * b^T ---
        auto a1 = make(backend, M, K, 0x11u, true);
        auto b1 = make(backend, N, K, 0x22u, true);
        auto y1 = matmul_transpose_b(a1, b1);
        if (!y1.requires_grad()) {
            std::cerr << "FAIL: matmul_transpose_b produced a tensor with no grad\n";
            return 1;
        }
        std::vector<float> seed_host(static_cast<std::size_t>(M * N));
        fill_deterministic(seed_host, 0x33u, 1.0f);
        auto seed1 = Tensor::from_cpu(backend, {M, N}, DType::F32, seed_host.data());
        y1.backward(seed1);
        if (!a1.grad() || !b1.grad()) {
            std::cerr << "FAIL: missing gradient (a=" << bool(a1.grad())
                      << " b=" << bool(b1.grad()) << ")\n";
            return 1;
        }

        // --- reference: build b^T explicitly and use the plain matmul path ---
        std::vector<float> b_host(static_cast<std::size_t>(N * K));
        fill_deterministic(b_host, 0x22u, 0.3f);
        std::vector<float> bt_host(static_cast<std::size_t>(K * N));
        for (std::int64_t n = 0; n < N; ++n)
            for (std::int64_t k = 0; k < K; ++k)
                bt_host[static_cast<std::size_t>(k * N + n)] =
                    b_host[static_cast<std::size_t>(n * K + k)];
        auto a2 = make(backend, M, K, 0x11u, true);
        auto bt = Tensor::from_cpu(backend, {K, N}, DType::F32, bt_host.data());
        bt.set_requires_grad(true);
        auto y2 = matmul(a2, bt);
        auto seed2 = Tensor::from_cpu(backend, {M, N}, DType::F32, seed_host.data());
        y2.backward(seed2);

        const auto ga1 = a1.grad()->to_vector<float>();
        const auto ga2 = a2.grad()->to_vector<float>();
        const double da = max_abs_diff(ga1, ga2);

        // grad_b from the transpose path is [N,K]; the reference grad is [K,N].
        const auto gb1 = b1.grad()->to_vector<float>();
        const auto gbt = bt.grad()->to_vector<float>();
        double db = 0.0;
        for (std::int64_t n = 0; n < N; ++n)
            for (std::int64_t k = 0; k < K; ++k)
                db = std::max(db, static_cast<double>(std::fabs(
                    gb1[static_cast<std::size_t>(n * K + k)] -
                    gbt[static_cast<std::size_t>(k * N + n)])));

        std::cout << "grad_a max|diff| = " << da << "\n";
        std::cout << "grad_b max|diff| = " << db << " (transposed comparison)\n";
        if (da > 1e-4 || db > 1e-4) {
            std::cerr << "FAIL: gradients disagree with the plain-matmul reference\n";
            return 1;
        }
        std::puts("PASS test_matmul_transpose_b_autograd");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "failed: " << e.what() << "\n";
        return 1;
    }
}
