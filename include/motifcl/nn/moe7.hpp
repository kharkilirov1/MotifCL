#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/nn/module.hpp>

namespace motifcl::nn {

// Configuration of a seven-expert routed counter-synapse FFN.
//
// The routing primitive is the runtime's existing `fog_hard_route7` op: a hard
// top-1 argmax over seven candidate tensors with a straight-through seven-way
// softmax gradient for the logits and per-row zeroing for the losing
// candidates. That is exactly top-1 MoE semantics, so no new SPIR-V kernel is
// required. Expert count is therefore fixed at seven.
struct MoE7Config {
    int dim = 288;
    int hidden_expert = 192;
    int hidden_shared = 192;
    int C = 11;
    float counter_lr = 4e-3f;
    float rms_beta = 0.9f;
    float rms_eps = 1e-3f;
    // Symmetric clamp on the balancing bias. It must exceed the router's own logit
    // scale, otherwise the bias saturates and stops adapting on long runs.
    float balance_clamp = 2.0f;
    std::uint32_t seed = 1234u;
    // Skip the losing experts' backward graphs. A loser's gradient is identically
    // zero, so its grad_w GEMM, its grad_x GEMM and its SwiGLU backward can only
    // produce zeros; the one side effect that is *not* zero is the counter state's
    // second moment, which decays (see CounterStateLinear::apply_zero_grad_update).
    // With that term applied explicitly this is bit-identical to the dense
    // backward, so it is on by default and only the measurement harness turns it
    // off. Override with MOTIFCL_MOE_SKIP_ZERO_GRAD=0/1.
    bool skip_zero_grad_experts = true;
};

// Routed FFN: seven counter-synapse experts (+ one always-on shared expert)
// selected per token by a hard top-1 router.
//
// y = route7(x @ W_r + b) + shared(x)
//
// - Expert and shared linear layers are `CounterStateLinear`: they carry no FP32
//   master weight and no optimizer moments; their 6-bit counter state is the
//   expert, and it updates inside backward (one backward == one state update).
// - `router_weight` is the only trainable FP32 parameter here.
// - `router_bias` implements aux-loss-free load balancing: it shifts the logits
//   used for *selection* but not the softmax probabilities used for the router
//   gradient. It is updated on the host, never by the optimizer.
class MoE7FFN : public Module {
public:
    static constexpr int kExperts = 7;

    MoE7FFN(Backend& backend, const MoE7Config& config);

    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override { return {&router_weight}; }

    void set_lr(float lr);
    void set_training(bool training);
    std::vector<CounterStateLinear*> counter_layers();
    // Runtime override of MoE7Config::skip_zero_grad_experts. The two settings are
    // numerically identical (see tests/test_vulkan_moe7_zero_grad.cpp), so this
    // exists to A/B them, not to change semantics.
    void set_skip_zero_grad_experts(bool skip) { skip_zero_grad_experts_ = skip; }
    bool skip_zero_grad_experts() const { return skip_zero_grad_experts_; }

    int dim() const { return config_.dim; }
    int hidden_expert() const { return config_.hidden_expert; }
    int hidden_shared() const { return config_.hidden_shared; }

    // --- load balancing (host-side bookkeeping) ---
    // Call only after the Vulkan command batch of the step has been submitted:
    // these read tensors back to the host.
    bool read_token_counts();
    void update_balance_bias(float gamma);
    // Publishes this forward's routing decision to the backward pass so the losing
    // experts can be skipped (they receive an identically zero gradient). Only
    // legal between the forward and backward batches: it is what read_token_counts
    // already computes, made explicit. Without it the backward runs the dense
    // graph for every expert -- same numbers, more work.
    bool freeze_routing() { return read_token_counts(); }
    bool routing_frozen() const { return routing_frozen_; }
    // Per-expert "this expert won at least one token in the frozen routing".
    const std::array<bool, kExperts>& route_live() const { return route_live_; }

  private:
    // Records the same update inside an open Vulkan batch; falls back to the CPU path.
    bool try_gpu_balance_bias(float gamma);

  public:
    // Switch-style load-balancing auxiliary loss, applied as an explicit router
    // gradient: L_aux = E * sum_e f_e * P_e with f frozen at the current load and
    // P_e the mean router probability. dL_aux/dlogits is computed on the host from
    // the same read-back (prepare_*), then turned into a weight gradient on the
    // device (inject_*) and accumulated into the router parameter. No new kernel.
    void prepare_balance_gradient(float aux_weight);
    void inject_balance_gradient();
    const std::array<float, kExperts>& last_load() const { return last_load_; }
    // Per-token winning expert of the most recent forward (filled by
    // read_token_counts). Exposed for routing analysis: if routing carries
    // information, different experts should prefer different tokens.
    const std::vector<int>& last_picks() const { return last_picks_; }
    // Diagnostics from the same host read-back: mean per-token logit spread
    // (max-min) and mean top1-top2 gap, over all experts.
    float last_logit_spread() const { return last_logit_spread_; }
    float last_logit_gap() const { return last_logit_gap_; }

    Parameter router_weight;             // [dim, kExperts]
    Tensor router_bias;                  // [kExperts], requires_grad == false

private:
    struct Expert {
        std::unique_ptr<CounterStateLinear> up;    // dim -> 2*hidden
        std::unique_ptr<CounterStateLinear> down;  // hidden -> dim
    };

    Tensor run_expert(Expert& expert, const Tensor& x);
    bool skip_zero_grad_experts_ = true;

    MoE7Config config_;
    std::array<Expert, kExperts> experts_;
    Expert shared_;
    Tensor last_biased_logits_;                     // [rows, 7] of the last forward
    Tensor last_input_;                             // [rows, dim] of the last forward
    std::vector<float> last_logits_host_;           // cached read-back (rows * 7)
    Tensor balance_grad_;                           // [rows, 7] dL_aux/dlogits, device
    std::array<float, kExperts> last_load_{};
    std::vector<int> last_picks_;
    std::array<bool, kExperts> route_live_{};
    bool routing_frozen_ = false;
    float last_logit_spread_ = 0.0f;
    float last_logit_gap_ = 0.0f;
    std::int64_t last_rows_ = 0;
};

} // namespace motifcl::nn
