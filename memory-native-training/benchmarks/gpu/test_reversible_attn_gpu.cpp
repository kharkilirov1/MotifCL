// Witness: reversible recompute-backward on a TRANSFORMER block (the activation-memory
// lever applied to attention). Coupling with F = self-attention, G = MLP:
//   y1 = x1 + F(x2);  y2 = x2 + G(y1)
// reference  = grad-enabled forward from original input (stores all activations).
// reversible = NoGrad forward (no storage) -> recover input via inverse -> recompute grads.
// Pass = grads match within recovery error -> reversible works through attention.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/linear.hpp>
#include <motifcl/ops/attention.hpp>
#include <motifcl/ops/activation.hpp>
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
    const int batch = 8, seq = 16, d = 64, n_head = 4, depth = 2;
    const int N = batch * seq;

    auto mk = [&](int in, int out) { return std::make_shared<nn::Linear>(backend, in, out, false); };
    std::vector<std::shared_ptr<nn::Linear>> Q, K, V, O, FC, FC2;
    for (int i = 0; i < depth; ++i) {
        Q.push_back(mk(d, d)); K.push_back(mk(d, d)); V.push_back(mk(d, d)); O.push_back(mk(d, d));
        FC.push_back(mk(d, 4 * d)); FC2.push_back(mk(4 * d, d));
    }

    std::vector<float> x1h(N * d), x2h(N * d), t1h(N * d), t2h(N * d);
    for (int i = 0; i < N * d; ++i) {
        x1h[i] = 0.1f * std::sin(0.01f * i); x2h[i] = 0.1f * std::cos(0.017f * i);
        t1h[i] = 0.05f * std::sin(0.03f * i); t2h[i] = 0.05f * std::cos(0.02f * i);
    }
    auto t1 = Tensor::from_cpu(backend, {N, d}, DType::F32, t1h.data());
    auto t2 = Tensor::from_cpu(backend, {N, d}, DType::F32, t2h.data());

    auto Fattn = [&](int i, const Tensor& x) {
        auto ctx = multihead_attention(Q[i]->forward(x), K[i]->forward(x), V[i]->forward(x), n_head, true, batch, seq);
        return O[i]->forward(ctx);
    };
    auto Gmlp = [&](int i, const Tensor& x) { return FC2[i]->forward(gelu(FC[i]->forward(x))); };

    auto fwd = [&](Tensor a, Tensor b) {
        for (int i = 0; i < depth; ++i) { auto y1 = add(a, Fattn(i, b)); auto y2 = add(b, Gmlp(i, y1)); a = y1; b = y2; }
        return std::make_pair(a, b);
    };

    // reference (stores all activations)
    auto x1 = Tensor::from_cpu(backend, {N, d}, DType::F32, x1h.data()); x1.set_requires_grad(true);
    auto x2 = Tensor::from_cpu(backend, {N, d}, DType::F32, x2h.data()); x2.set_requires_grad(true);
    { auto yz = fwd(x1, x2); auto loss = add(mse_loss(yz.first, t1), mse_loss(yz.second, t2)); loss.backward(); }
    auto ref_gx1 = x1.grad()->to_vector<float>();
    auto ref_gWQ = Q[0]->weight.data.grad()->to_vector<float>();
    for (int i = 0; i < depth; ++i) {
        Q[i]->weight.data.zero_grad(); K[i]->weight.data.zero_grad(); V[i]->weight.data.zero_grad();
        O[i]->weight.data.zero_grad(); FC[i]->weight.data.zero_grad(); FC2[i]->weight.data.zero_grad();
    }

    // reversible: NoGrad forward -> recover input -> recompute
    Tensor xr1, xr2;
    {
        autograd::NoGradGuard guard;
        auto a = Tensor::from_cpu(backend, {N, d}, DType::F32, x1h.data());
        auto b = Tensor::from_cpu(backend, {N, d}, DType::F32, x2h.data());
        auto yz = fwd(a, b); a = yz.first; b = yz.second;       // stores nothing
        for (int i = depth - 1; i >= 0; --i) {                   // recover input
            auto x2i = sub(b, Gmlp(i, a));
            auto x1i = sub(a, Fattn(i, x2i));
            a = x1i; b = x2i;
        }
        xr1 = a; xr2 = b;
    }
    xr1.set_requires_grad(true); xr2.set_requires_grad(true);
    { auto yz = fwd(xr1, xr2); auto loss = add(mse_loss(yz.first, t1), mse_loss(yz.second, t2)); loss.backward(); }
    auto rev_gx1 = xr1.grad()->to_vector<float>();
    auto rev_gWQ = Q[0]->weight.data.grad()->to_vector<float>();

    double ex = rel_err(rev_gx1, ref_gx1), ew = rel_err(rev_gWQ, ref_gWQ);
    printf("reversible attention block (%d blocks, %d heads, d=%d):\n", depth, n_head, d);
    printf("  grad_x  rel-err = %.3e\n", ex);
    printf("  grad_WQ rel-err = %.3e\n", ew);
    const bool ok = ex < 5e-2 && ew < 5e-2;
    printf(ok ? "PASS: reversible recompute works through attention; forward stored no activations\n"
              : "FAIL: recompute through attention diverges\n");
    return ok ? 0 : 1;
}
