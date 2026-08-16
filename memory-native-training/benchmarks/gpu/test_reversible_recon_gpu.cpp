// Witness for reversible-activation viability: does float forward->inverse reconstruct
// the input, or does rounding accumulate with depth? 12 coupling blocks:
//   forward:  y1 = x1 + F(x2);  y2 = x2 + G(y1)
//   inverse:  x2 = y2 - G(y1);  x1 = y1 - F(x2)
// F,G are deterministic Linear maps. Pass = max reconstruction error stays small with depth.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/linear.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/autograd/node.hpp>

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace motifcl;

int main() {
    auto backend = Backend::create_opencl();
    const int N = 64, half = 128, depth = 12;

    autograd::set_enabled(false);  // pure forward/inverse math, no autograd

    // deterministic F,G per block; small init so the residual stream stays stable
    std::vector<std::shared_ptr<nn::Linear>> F, G;
    for (int b = 0; b < depth; ++b) {
        F.push_back(std::make_shared<nn::Linear>(backend, half, half, false));
        G.push_back(std::make_shared<nn::Linear>(backend, half, half, false));
    }

    std::vector<float> x1h(N * half), x2h(N * half);
    for (int i = 0; i < N * half; ++i) { x1h[i] = 0.1f * std::sin(0.01f * i); x2h[i] = 0.1f * std::cos(0.017f * i); }
    auto x1 = Tensor::from_cpu(backend, {N, half}, DType::F32, x1h.data());
    auto x2 = Tensor::from_cpu(backend, {N, half}, DType::F32, x2h.data());

    // keep per-block (y1,y2) so the inverse uses the same y1 the forward produced
    auto a = x1, b = x2;
    std::vector<Tensor> y1s, y2s;
    for (int i = 0; i < depth; ++i) {
        auto y1 = add(a, F[i]->forward(b));
        auto y2 = add(b, G[i]->forward(y1));
        y1s.push_back(y1);
        y2s.push_back(y2);
        a = y1; b = y2;
    }

    // inverse from the final (a,b) back to the original input
    for (int i = depth - 1; i >= 0; --i) {
        auto y1 = y1s[i];                    // reproduced exactly below from a anyway
        auto x2r = sub(b, G[i]->forward(a)); // a holds y1 of this block
        auto x1r = sub(a, F[i]->forward(x2r));
        a = x1r; b = x2r;
    }

    auto a_h = a.to_vector<float>();
    auto b_h = b.to_vector<float>();
    double e1 = 0.0, e2 = 0.0;
    for (int i = 0; i < N * half; ++i) {
        e1 = std::max(e1, std::abs((double)a_h[i] - (double)x1h[i]));
        e2 = std::max(e2, std::abs((double)b_h[i] - (double)x2h[i]));
    }
    autograd::set_enabled(true);

    printf("reversible %d coupling blocks, half=%d:\n", depth, half);
    printf("  max reconstruction error  x1=%.3e  x2=%.3e\n", e1, e2);
    const bool ok = (e1 < 1e-3) && (e2 < 1e-3);
    printf(ok ? "PASS: float reversible reconstruction stays accurate with depth\n"
              : "REFUTED: rounding accumulates -> needs fixed-point/anchor (doc section 6)\n");
    return ok ? 0 : 1;
}
