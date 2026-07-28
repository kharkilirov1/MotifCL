// Slice R2 witness: nn::ReversibleBlock on Vulkan tensors (no OpenCL context).
//
// Asserts the two PORT_PROMPT invariants specific to this slice:
//   (5) Parity: gradients produced by the NoGrad-forward + recompute-backward
//       path match a reference grad-enabled forward+backward to within 1e-4
//       relative error, on a 2-block stack with Linear+GELU coupling.
//   (2) OpenCL-free: the test does NOT call Backend::create_opencl() or any
//       OpenCL kernel directly. All ops route through Vulkan device paths.
//
// Skips cleanly (exit 77) when no Vulkan device is present.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/reversible.hpp>

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

} // namespace

int main() {
    using namespace motifcl;

    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cerr << "Skipping Vulkan reversible test: "
                  << (probe.error.empty() ? "no Vulkan device" : probe.error) << "\n";
        return 77;
    }

    bool ok = true;
    auto expect = [&](bool cond, const char* msg) {
        if (!cond) {
            std::cerr << "FAIL: " << msg << "\n";
            ok = false;
        }
    };

    try {
        Backend backend = Backend::create_vulkan();
        const std::int64_t d = 64;       // feature dim
        const std::int64_t N = 4;        // batch
        const int n_blocks = 2;

        // Coupling module: Linear -> GELU. Both Vulkan-native (Linear via
        // matmul, GELU via run_vulkan_gelu). Built once and shared between
        // the reversible and reference paths so weights are identical.
        // use_bias=false because Linear's add_bias_rows is OpenCL-only today
        // (not in scope for the memory-native Vulkan port).
        auto build_coupling = [&]() {
            auto lin = std::make_shared<nn::Linear>(backend, /*in=*/d, /*out=*/d, /*use_bias=*/false);
            auto seq = std::make_shared<nn::Sequential>();
            seq->add(lin);
            seq->add(std::make_shared<nn::GELU>());
            return seq;
        };

        std::vector<std::shared_ptr<nn::ReversibleBlock>> blocks;
        for (int i = 0; i < n_blocks; ++i) {
            auto f = build_coupling();
            auto g = build_coupling();
            blocks.push_back(std::make_shared<nn::ReversibleBlock>(f, g));
        }

        // Inputs.
        std::vector<float> x1_host(static_cast<std::size_t>(N * d));
        std::vector<float> x2_host(static_cast<std::size_t>(N * d));
        fill_deterministic(x1_host, 0xA1u, 0.5f);
        fill_deterministic(x2_host, 0xA2u, 0.5f);

        // grad-of-one loss: dL/dy1 = 1, dL/dy2 = 1 (numerically exact, exercises
        // the full recompute + inverse-recovery path; no reduction needed).
        auto run_stack_grad = [&](const Tensor& in1, const Tensor& in2,
                                  bool use_reversible) {
            Tensor cur1 = in1, cur2 = in2;
            cur1.set_requires_grad(true);
            cur2.set_requires_grad(true);
            for (auto& blk : blocks) {
                if (use_reversible) {
                    auto [y1, y2] = blk->forward(cur1, cur2);
                    cur1 = std::move(y1);
                    cur2 = std::move(y2);
                } else {
                    // Reference: same coupling, grad-enabled forward (no
                    // NoGrad, no inverse-recovery). Replicates the coupling
                    // arithmetic inline so the reference is independent of the
                    // ReversibleBlock backward implementation.
                    auto fx2 = blk->f().forward(cur2);
                    auto y1 = add(cur1, fx2);
                    auto gy1 = blk->g().forward(y1);
                    auto y2 = add(cur2, gy1);
                    cur1 = std::move(y1);
                    cur2 = std::move(y2);
                }
            }
            std::vector<float> one_host(static_cast<std::size_t>(N * d), 1.0f);
            auto grad_one = Tensor::from_cpu(backend, {N, d}, DType::F32, one_host.data());
            cur2.backward(grad_one);
            cur1.backward(grad_one);
            std::vector<float> g1, g2;
            if (auto g = in1.grad()) g1 = g->to_vector<float>();
            if (auto g = in2.grad()) g2 = g->to_vector<float>();
            return std::make_pair(std::move(g1), std::move(g2));
        };

        auto x1v = Tensor::from_cpu(backend, {N, d}, DType::F32, x1_host.data());
        auto x2v = Tensor::from_cpu(backend, {N, d}, DType::F32, x2_host.data());
        auto [g1_rev, g2_rev] = run_stack_grad(x1v, x2v, /*use_reversible=*/true);

        // Reset grads on shared Parameters so the reference pass starts clean.
        for (auto& blk : blocks) {
            for (auto* p : blk->parameters()) p->zero_grad();
        }

        auto x1r = Tensor::from_cpu(backend, {N, d}, DType::F32, x1_host.data());
        auto x2r = Tensor::from_cpu(backend, {N, d}, DType::F32, x2_host.data());
        auto [g1_ref, g2_ref] = run_stack_grad(x1r, x2r, /*use_reversible=*/false);

        // Parity check (invariant 5): grads must match within rel-err 1e-4.
        expect(g1_rev.size() == g1_ref.size(), "grad-x1 size match");
        expect(g2_rev.size() == g2_ref.size(), "grad-x2 size match");
        auto relerr = [](float a, float b) {
            const float denom = std::max(1.0e-6f, std::fabs(b));
            return std::fabs(a - b) / denom;
        };
        float max_e1 = 0.0f, max_e2 = 0.0f;
        const float tol = 1.0e-4f;
        for (std::size_t i = 0; i < g1_rev.size() && i < g1_ref.size(); ++i) {
            max_e1 = std::max(max_e1, relerr(g1_rev[i], g1_ref[i]));
        }
        for (std::size_t i = 0; i < g2_rev.size() && i < g2_ref.size(); ++i) {
            max_e2 = std::max(max_e2, relerr(g2_rev[i], g2_ref[i]));
        }
        std::cout << "reversible grad-x1 max rel-err vs reference: " << max_e1 << "\n";
        std::cout << "reversible grad-x2 max rel-err vs reference: " << max_e2 << "\n";
        expect(max_e1 <= tol, "reversible grad-x1 parity within 1e-4");
        expect(max_e2 <= tol, "reversible grad-x2 parity within 1e-4");
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << "\n";
        ok = false;
    }

    std::cout << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
