// Witness: reversible recompute-backward in the engine. A 4-block coupling chain.
// reference  : grad-enabled forward from the ORIGINAL input (stores all activations).
// reversible : forward in NoGrad (stores nothing), then recover the input via the inverse
//              chain and recompute a grad-enabled forward from the recovered input.
// Pass = reversible grads match the reference within the recovery error (~few %).
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/linear.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/loss.hpp>
#include <motifcl/autograd/node.hpp>

#include <cmath>
#include <cstdio>
#include <memory>
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
    const int N = 32, half = 64, depth = 4;

    std::vector<std::shared_ptr<nn::Linear>> F, G;
    for (int i = 0; i < depth; ++i) {
        F.push_back(std::make_shared<nn::Linear>(backend, half, half, false));
        G.push_back(std::make_shared<nn::Linear>(backend, half, half, false));
    }

    std::vector<float> x1h(N * half), x2h(N * half), t1h(N * half), t2h(N * half);
    for (int i = 0; i < N * half; ++i) {
        x1h[i] = 0.1f * std::sin(0.01f * i); x2h[i] = 0.1f * std::cos(0.017f * i);
        t1h[i] = 0.05f * std::sin(0.03f * i); t2h[i] = 0.05f * std::cos(0.02f * i);
    }
    auto t1 = Tensor::from_cpu(backend, {N, half}, DType::F32, t1h.data());
    auto t2 = Tensor::from_cpu(backend, {N, half}, DType::F32, t2h.data());

    auto fwd = [&](Tensor a, Tensor b) {
        for (int i = 0; i < depth; ++i) { auto y1 = add(a, F[i]->forward(b)); auto y2 = add(b, G[i]->forward(y1)); a = y1; b = y2; }
        return std::make_pair(a, b);
    };

    // ---- reference: grad-enabled from original input (stores all activations) ----
    auto x1 = Tensor::from_cpu(backend, {N, half}, DType::F32, x1h.data()); x1.set_requires_grad(true);
    auto x2 = Tensor::from_cpu(backend, {N, half}, DType::F32, x2h.data()); x2.set_requires_grad(true);
    { auto yz = fwd(x1, x2); auto loss = add(mse_loss(yz.first, t1), mse_loss(yz.second, t2)); loss.backward(); }
    auto ref_gx1 = x1.grad()->to_vector<float>();
    auto ref_gWF = F[0]->weight.data.grad()->to_vector<float>();
    for (int i = 0; i < depth; ++i) { F[i]->weight.data.zero_grad(); G[i]->weight.data.zero_grad(); }

    // ---- reversible: NoGrad forward (no storage) -> recover input -> recompute grads ----
    Tensor xr1, xr2;
    {
        autograd::NoGradGuard guard;
        auto x1d = Tensor::from_cpu(backend, {N, half}, DType::F32, x1h.data());
        auto x2d = Tensor::from_cpu(backend, {N, half}, DType::F32, x2h.data());
        auto yz = fwd(x1d, x2d);                       // forward stores nothing
        Tensor a = yz.first, b = yz.second;
        for (int i = depth - 1; i >= 0; --i) {         // recover input via inverse
            auto x2i = sub(b, G[i]->forward(a));
            auto x1i = sub(a, F[i]->forward(x2i));
            a = x1i; b = x2i;
        }
        xr1 = a; xr2 = b;
    }
    xr1.set_requires_grad(true); xr2.set_requires_grad(true);
    { auto yz = fwd(xr1, xr2); auto loss = add(mse_loss(yz.first, t1), mse_loss(yz.second, t2)); loss.backward(); }
    auto rev_gx1 = xr1.grad()->to_vector<float>();
    auto rev_gWF = F[0]->weight.data.grad()->to_vector<float>();

    double ex = rel_err(rev_gx1, ref_gx1);
    double ew = rel_err(rev_gWF, ref_gWF);
    printf("reversible recompute-backward (%d blocks):\n", depth);
    printf("  grad_x rel-err   = %.3e\n", ex);
    printf("  grad_WF rel-err  = %.3e\n", ew);
    const bool ok = ex < 5e-2 && ew < 5e-2;
    printf(ok ? "PASS: reversible recompute grads match stored-autograd; forward stored no activations\n"
              : "FAIL: recompute grads diverge\n");
    return ok ? 0 : 1;
}
