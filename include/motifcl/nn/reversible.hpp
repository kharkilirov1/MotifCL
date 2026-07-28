#pragma once

#include <memory>
#include <utility>

#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

// ReversibleBlock — memory-native pillar B (reversible activations).
//
// Two-input/two-output coupling:
//     y1 = x1 + F(x2)
//     y2 = x2 + G(y1)
// inverse (used during backward to recover inputs without storing activations):
//     x2 = y2 - G(y1)
//     x1 = y1 - F(x2)
//
// Forward runs under NoGrad: activations of F and G are not stored between
// forward and backward. Backward recovers (x1, x2) via the inverse chain
// (under NoGrad), then recomputes F and G forward with autograd enabled on the
// recovered inputs and propagates gradients through the standard graph. This
// mirrors the CounterBackwardNode recompute pattern (src/nn/compact_counter.cpp)
// and the qkv_split multi-output autograd pattern (src/ops/attention.cpp): each
// output (y1, y2) carries its own backward-node instance sharing one recompute
// driver via a part discriminator.
//
// Coupling operators F and G are any nn::Module with `forward(const Tensor&)`:
// nn::Linear, nn::Sequential(Linear, GELU, ...), nn::CounterStateLinear, etc.
// Training-only: in-backward recompute is incompatible with graph-capture
// replay / grad accumulation / DDP (same boundary as CounterStateLinear).
class ReversibleBlock : public Module {
public:
    ReversibleBlock(std::shared_ptr<Module> f, std::shared_ptr<Module> g);

    // Primary API: two inputs in, two outputs out. Non-virtual (the base
    // Module::forward contract is single-Tensor; calling it is a programming
    // error and MCL_CHECKs).
    std::pair<Tensor, Tensor> forward(const Tensor& x1, const Tensor& x2);
    Tensor forward(const Tensor& x) override;

    std::vector<Parameter*> parameters() override;

    void set_training(bool t) { training_ = t; }
    bool training() const { return training_; }

    Module& f() { return *f_; }
    Module& g() { return *g_; }

private:
    std::shared_ptr<Module> f_;
    std::shared_ptr<Module> g_;
    bool training_ = true;
};

} // namespace motifcl::nn
