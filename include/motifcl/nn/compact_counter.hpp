#pragma once

#include <cstdint>

#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

// Ternary finite-state "counter" linear layer trained as a per-synapse automaton.
// Persistent per-weight state is one uint8 (45/63 reachable states); the optimizer
// lives inside that state plus per-row scale and per-row RMS second moment. The
// weight update is fused into backward (no full weight-gradient, no Adam moments).
// Layout mirrors the validated prototype: state/grad_w are [out, in]; forward is
// y = x @ w^T via matmul_transpose_b. Eager-training only (in-backward update is a
// side effect; not graph-capture replayable).
class CounterStateLinear : public Module {
public:
    CounterStateLinear(Backend& backend, int in_features, int out_features,
                       int C = 11, float lr = 3e-3f, float lr_scale = 0.0f,
                       float init_gain = 1.0f, float rms_beta = 0.9f,
                       float rms_eps = 1e-3f, std::uint32_t seed = 12345u);

    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override { return {}; }  // self-updating

    Tensor decode_weight() const;                       // dense [out, in] f32
    void apply_update_seed(const Tensor& grad_w, std::uint32_t seed);  // reference path (dense grad_w)
    // Memory-native path: grad_w is recomputed inside the kernels from (grad_out, x);
    // no dense [out,in] weight gradient is ever allocated.
    void apply_update_backward(const Tensor& grad_out, const Tensor& x, std::uint32_t seed);
    // Memory-native grad wrt input: grad_x decoded from packed state in-kernel,
    // without materialising a dense [out,in] weight. Returns grad_x [N,in].
    Tensor backward_input_from_state(const Tensor& grad_out) const;
    std::uint32_t next_seed() { return seed_++; }

    void set_training(bool t) { training_ = t; }   // when false, forward attaches no update node
    bool training() const { return training_; }

    int in_features() const { return in_; }
    int out_features() const { return out_; }

    // Persistent state (saved/restored as buffers, not optimizer-touched).
    Tensor state;   // U8  [out, in]
    Tensor scale;   // F32 [out]
    Tensor v;       // F32 [out]

private:
    Backend* backend_ = nullptr;
    int in_ = 0;
    int out_ = 0;
    int C_ = 11;
    float lr_ = 3e-3f;
    float lr_scale_ = 0.0f;
    float rms_beta_ = 0.9f;
    float rms_eps_ = 1e-3f;
    std::uint32_t seed_ = 12345u;
    bool training_ = true;
};

} // namespace motifcl::nn
