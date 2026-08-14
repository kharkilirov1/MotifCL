#pragma once

#include <memory>
#include <vector>

#include <motifcl/nn/embedding.hpp>
#include <motifcl/nn/module.hpp>
#include <motifcl/nn/rmsnorm.hpp>
#include <motifcl/nn/transformer.hpp>

namespace motifcl::nn {

struct FogV3Config {
    int vocab_size = 8192;
    int max_seq_len = 512;
    int d_model = 320;
    int n_heads = 5;
    int n_layers = 4;
    int d_ff = 1344;
    float rms_eps = 1e-6f;
    float dropout = 0.0f; // Vulkan training currently expects zero dropout.
};

// MotifCL-native lexical backbone for FOG register_machine_v3.
// This is the stage-1 causal-LM model: the LM head is tied to the token
// embedding via matmul_transpose_b, matching the Python FOG contract without
// allocating a second vocab projection matrix.
class FogV3LexicalModel : public Module {
public:
    FogV3Config config;
    Embedding token_embedding;
    Parameter position_embedding;
    Parameter lexical_kind;
    Parameter latent_kind;
    std::vector<std::shared_ptr<ModernTransformerBlock>> blocks;
    RMSNorm final_norm;

    FogV3LexicalModel(Backend& backend, const FogV3Config& config = {});

    Tensor forward(const Tensor& token_ids) override;
    Tensor hidden(const Tensor& token_ids);
    Tensor logits_from_hidden(const Tensor& h);
    std::vector<Parameter*> parameters() override;
};

} // namespace motifcl::nn

namespace motifcl::nn {

struct FogOperatorBankOutput {
    Tensor value;
    Tensor logits;
};

// MotifCL-native seven-way finite operator grammar used by the v3 register
// machine. The learned operator parameterization is intentionally expressed in
// existing dense primitives; only BLOCK_PRODUCT and hard-ST routing need FOG
// Vulkan kernels.
class FogOperatorBankV3 : public Module {
public:
    FogOperatorBankV3(Backend& backend, int d_model = 320, int rank = 48);
    FogOperatorBankOutput forward3(const Tensor& value,
                                   const Tensor& addressed,
                                   const Tensor& control);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

private:
    int d_model_;
    int rank_;
    RMSNorm value_norm_;
    RMSNorm read_norm_;
    RMSNorm control_norm_;
    Linear route_value_;
    Linear route_read_;
    Linear route_control_;
    Linear route_out_;
    std::vector<std::unique_ptr<Linear>> state_down_;
    std::vector<std::unique_ptr<Linear>> read_down_;
    std::vector<std::unique_ptr<Linear>> control_down_;
    std::vector<std::unique_ptr<Linear>> up_;
    std::vector<Parameter> op_bias_;
};

} // namespace motifcl::nn

namespace motifcl::nn {

// Query-conditioned addressed payload read. Query/key identities are compared
// in a shared low-rank address space; payload vectors stay in the original
// d_model coordinates, so the selected value can flow directly into the FOG
// register machine without a learned V/O rotation.
class FogAddressBinderV3 : public Module {
public:
    FogAddressBinderV3(Backend& backend, int d_model = 320, int compare_rank = 320);
    Tensor forward_bind(const Tensor& query,
                        const Tensor& key_states,
                        const Tensor& value_states,
                        int64_t batch_size,
                        int64_t rows_per_batch);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

private:
    int d_model_;
    int compare_rank_;
    Linear address_map_;
    Tensor rank_norm_weight_;
};

struct FogRegisterStateV3 {
    Tensor value;
    Tensor control;
    Tensor scratch0;
    Tensor scratch1;
};

struct FogRegisterCellOutputV3 {
    FogRegisterStateV3 state;
    Tensor operator_logits;
    Tensor halt_probability;
};

// Portable typed-register cell. It preserves the core v3 contract (value,
// control, two scratch registers, shared finite operator bank, hard routing and
// HALT) while expressing the auxiliary workspace entirely with MotifCL dense
// Vulkan primitives.
class FogRegisterCellV3 : public Module {
public:
    FogRegisterCellV3(Backend& backend, int d_model = 320, int operator_rank = 48);
    FogRegisterCellOutputV3 forward_state(const FogRegisterStateV3& old,
                                          const Tensor& addressed);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

private:
    int d_model_;
    FogOperatorBankV3 operator_bank_;
    RMSNorm control_norm_;
    RMSNorm read_norm_;
    Linear control_self_;
    Linear control_read_;
    Linear scratch_self_;
    Linear scratch_read_;
    Linear scratch_control_;
    RMSNorm scratch_norm_;
    Linear halt_value_;
    Linear halt_control_;
};

// End-to-end portable FOG v3 container. The same object can be used for stage-1
// lexical pretraining and stage-2 structured recurrent machine training so
// MotifCL checkpoints keep one stable parameter ordering across stages.
class FogV3Model : public Module {
public:
    FogV3Config config;
    FogV3LexicalModel lexical;
    FogAddressBinderV3 binder;
    FogRegisterCellV3 machine;

    FogV3Model(Backend& backend, const FogV3Config& config = {});

    Tensor forward(const Tensor& token_ids) override;
    FogRegisterStateV3 initial_state(const Tensor& query_ids);
    FogRegisterCellOutputV3 structured_step(const FogRegisterStateV3& old,
                                             const Tensor& key_ids,
                                             const Tensor& value_ids,
                                             int64_t batch_size,
                                             int64_t rows_per_batch);
    Tensor direct_vocab_logits(const Tensor& value);

    std::vector<Parameter*> parameters() override;
    std::vector<Parameter*> lexical_parameters();
    std::vector<Parameter*> machine_parameters();

private:
    Tensor dmodel_norm_weight_;
};

} // namespace motifcl::nn
