#include <motifcl/nn/reversible.hpp>

#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/activation.hpp>   // gelu (used inside F/G Sequential chains)
#include <motifcl/ops/basic_ops.hpp>    // add, sub
#include <motifcl/tensor/tensor.hpp>

namespace motifcl::nn {

namespace {

// Pending-gradient accumulator for the two-output reversible block.
//
// The autograd engine (src/tensor/tensor.cpp:36-96) calls Node::backward with
// the upstream gradient for each output Tensor once, in topological order. A
// reversible block has two outputs (y1, y2), each carrying its own
// ReversibleBackwardNode instance distinguished by `part`. We need BOTH grads
// before we can run the inverse-recovery + recompute, so the first arrived grad
// is parked here and the second arrival triggers the actual work.
//
// Lifetime: a fresh PendingGrads is constructed for every forward pass and held
// alive by the shared_ptr in both ReversibleBackwardNode instances (one per
// output). When the second backward() runs, recompute executes; the struct is
// then dropped as the nodes go out of scope after backward completes.
struct PendingGrads {
    Tensor grad_y1;
    Tensor grad_y2;
    bool have_y1 = false;
    bool have_y2 = false;
};

struct ReversibleBackwardNode : autograd::Node {
    int part;        // 0 => grad goes to grad_y1, 1 => grad_y2
    Tensor x1, x2;   // original inputs (for inputs() topo + final grad dispatch)
    Tensor y1, y2;   // original outputs (for inverse recovery)
    Module* f;
    Module* g;
    std::shared_ptr<PendingGrads> pending;

    ReversibleBackwardNode(int part_, Tensor x1_, Tensor x2_, Tensor y1_, Tensor y2_,
                           Module* f_, Module* g_, std::shared_ptr<PendingGrads> p)
        : part(part_), x1(std::move(x1_)), x2(std::move(x2_)),
          y1(std::move(y1_)), y2(std::move(y2_)), f(f_), g(g_), pending(std::move(p)) {}

    std::vector<Tensor> inputs() const override {
        // Topological order needs the original inputs so the engine visits this
        // node after x1/x2 producers. Returning {x1, x2} is sufficient.
        return {x1, x2};
    }

    void backward(const Tensor& grad_output) override {
        if (part == 0) {
            pending->grad_y1 = grad_output;
            pending->have_y1 = true;
        } else {
            pending->grad_y2 = grad_output;
            pending->have_y2 = true;
        }
        if (!(pending->have_y1 && pending->have_y2)) {
            return;  // wait for the other output
        }
        // Both grads arrived: run inverse-recovery under NoGrad, then recompute
        // forward under grad and dispatch grads through the standard graph.
        recompute_with_grads(pending->grad_y1, pending->grad_y2);
    }

    void recompute_with_grads(const Tensor& gy1, const Tensor& gy2) {
        // Inverse recovery: G recovers x2 from y2 (G operates on y1); F
        // recovers x1 from y1 once x2 is known (F operates on x2). Done under
        // NoGrad so recovery builds no graph; the recompute pass below
        // re-enables grad on the recovered inputs.
        Tensor x1r, x2r;
        {
            autograd::NoGradGuard guard;
            auto gy = g->forward(y1);   // G(y1)
            x2r = sub(y2, gy);
            auto fx = f->forward(x2r);  // F(x2)
            x1r = sub(y1, fx);
        }
        // Mark recovered inputs as requiring grad so the recompute builds a
        // graph. Their upstream grad (into the original x1/x2 producers) is
        // dispatched in step 4.
        if (x1.requires_grad()) x1r.set_requires_grad(true);
        if (x2.requires_grad()) x2r.set_requires_grad(true);
        // The surrounding BackwardEngine::run wraps its topo loop in a
        // NoGradGuard (src/tensor/tensor.cpp:75), so the recompute forward
        // below would not attach grad_fn. Temporarily re-enable autograd for
        // the recompute forward AND the nested backward, then restore. The
        // IsolatedBackwardScope detaches the active engine so the nested
        // backward launches a fresh BackwardEngine on the freshly-built graph.
        Tensor fy1, fy2;
        {
            const bool was_enabled = autograd::is_enabled();
            autograd::set_enabled(true);
            fy1 = add(x1r, f->forward(x2r));   // y1r = x1r + F(x2r)
            fy2 = add(x2r, g->forward(fy1));   // y2r = x2r + G(y1r)
            {
                autograd::IsolatedBackwardScope scope;
                fy1.backward(gy1);
                fy2.backward(gy2);
            }
            autograd::set_enabled(was_enabled);
        }
        // Dispatch recovered-input grads into the original inputs so upstream
        // producers still receive gradient through the surrounding engine.
        if (x1.requires_grad()) {
            const auto g1 = x1r.grad();
            if (g1 && g1->valid()) x1.backward(*g1);
        }
        if (x2.requires_grad()) {
            const auto g2 = x2r.grad();
            if (g2 && g2->valid()) x2.backward(*g2);
        }
    }
};

} // namespace

ReversibleBlock::ReversibleBlock(std::shared_ptr<Module> f, std::shared_ptr<Module> g)
    : f_(std::move(f)), g_(std::move(g)) {
    MCL_CHECK(f_ != nullptr && g_ != nullptr, "ReversibleBlock requires non-null F and G modules");
}

Tensor ReversibleBlock::forward(const Tensor& /*x*/) {
    MCL_CHECK(false, "ReversibleBlock is two-input; use forward(x1, x2)");
    return {};
}

std::vector<Parameter*> ReversibleBlock::parameters() {
    std::vector<Parameter*> out;
    auto append = [&](Module* m) {
        if (!m) return;
        auto ps = m->parameters();
        out.insert(out.end(), ps.begin(), ps.end());
    };
    append(f_.get());
    append(g_.get());
    return out;
}

std::pair<Tensor, Tensor> ReversibleBlock::forward(const Tensor& x1, const Tensor& x2) {
    MCL_CHECK(x1.shape() == x2.shape(),
              "ReversibleBlock requires same-shape halves; got " + x1.shape().str() + " vs " + x2.shape().str());
    MCL_CHECK(x1.backend_ptr() == x2.backend_ptr(),
              "ReversibleBlock requires both halves on the same backend");

    Tensor y1, y2;
    {
        autograd::NoGradGuard guard;
        auto fx2 = f_->forward(x2);
        y1 = add(x1, fx2);
        auto gy1 = g_->forward(y1);
        y2 = add(x2, gy1);
    }

    if (autograd::is_enabled() && training_) {
        y1.set_requires_grad(true);
        y2.set_requires_grad(true);
        auto pending = std::make_shared<PendingGrads>();
        // One node instance per output (mirrors qkv_split's per-output
        // QKVSplitBackwardNode pattern); both share the same pending state.
        y1._set_grad_fn(std::make_shared<ReversibleBackwardNode>(
            /*part=*/0, x1, x2, y1, y2, f_.get(), g_.get(), pending));
        y2._set_grad_fn(std::make_shared<ReversibleBackwardNode>(
            /*part=*/1, x1, x2, y1, y2, f_.get(), g_.get(), pending));
    }
    return {y1, y2};
}

} // namespace motifcl::nn
