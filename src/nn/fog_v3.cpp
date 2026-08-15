#include <motifcl/nn/fog_v3.hpp>

#include <unordered_set>
#include <cmath>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/matmul.hpp>

namespace motifcl::nn {
namespace {

TransformerConfig backbone_config(const FogV3Config& cfg) {
    MCL_CHECK(cfg.vocab_size > 0 && cfg.max_seq_len > 0 && cfg.d_model > 0,
              "FOG v3 config dimensions must be positive");
    MCL_CHECK(cfg.n_heads > 0 && cfg.d_model % cfg.n_heads == 0,
              "FOG v3 d_model must be divisible by n_heads");
    MCL_CHECK(cfg.n_layers > 0 && cfg.d_ff >= cfg.d_model,
              "FOG v3 invalid layer/FF geometry");
    MCL_CHECK(cfg.dropout == 0.0f,
              "FOG v3 Vulkan lexical stage currently requires dropout=0");
    TransformerConfig tc;
    tc.vocab_size = cfg.vocab_size;
    tc.block_size = cfg.max_seq_len;
    tc.n_embd = cfg.d_model;
    tc.n_head = cfg.n_heads;
    tc.n_kv_head = cfg.n_heads;
    tc.head_dim = cfg.d_model / cfg.n_heads;
    tc.v_head_dim = tc.head_dim;
    tc.n_layer = cfg.n_layers;
    tc.mlp_hidden = cfg.d_ff;
    tc.dropout = cfg.dropout;
    tc.rms_norm_eps = cfg.rms_eps;
    tc.use_rope = false;
    tc.learned_position_embeddings = true;
    tc.use_swiglu = false;
    tc.activation = TransformerActivation::GELU;
    tc.use_qkv_bias = true;
    tc.use_attention_output_bias = true;
    tc.use_mlp_bias = true;
    tc.causal = true;
    tc.norm_first = true;
    tc.split_qkv_projections = true;
    return tc;
}

} // namespace

FogV3LexicalModel::FogV3LexicalModel(Backend& backend, const FogV3Config& raw)
    : config(raw),
      token_embedding(backend, raw.vocab_size, raw.d_model),
      position_embedding(Tensor::randn(backend, {raw.max_seq_len, raw.d_model}, 0.02f), true),
      lexical_kind(Tensor::randn(backend, {raw.d_model}, 0.02f), true),
      latent_kind(Tensor::randn(backend, {raw.d_model}, 0.02f), true),
      final_norm(backend, raw.d_model, raw.rms_eps, false) {
    const auto tc = backbone_config(config);
    blocks.reserve(static_cast<std::size_t>(config.n_layers));
    for (int i = 0; i < config.n_layers; ++i) {
        blocks.push_back(std::make_shared<ModernTransformerBlock>(backend, tc));
    }
}

Tensor FogV3LexicalModel::hidden(const Tensor& token_ids) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "FOG v3 token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2,
              "FOG v3 token ids must be [T] or [B,T]");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.max_seq_len, "FOG v3 sequence exceeds max_seq_len");

    auto h = token_position_embedding(token_ids, token_embedding.weight.data, position_embedding.data)
                 .view({B * T, config.d_model});
    h = add_bias_rows(h, lexical_kind.data);
    for (auto& block : blocks) h = block->forward(h, B, T, nullptr);
    return final_norm.forward(h);
}

Tensor FogV3LexicalModel::logits_from_hidden(const Tensor& h) {
    MCL_CHECK(h.dtype() == DType::F32 && h.ndim() == 2 && h.shape()[1] == config.d_model,
              "FOG v3 hidden must be [tokens,d_model]");
    // Embedding is [vocab,d_model]; transpose-B gives h @ E^T.
    return matmul_transpose_b(h, token_embedding.weight.data);
}

Tensor FogV3LexicalModel::forward(const Tensor& token_ids) {
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    auto logits = logits_from_hidden(hidden(token_ids));
    return token_ids.ndim() == 2 ? logits.view({B, T, config.vocab_size}) : logits;
}

std::vector<Parameter*> FogV3LexicalModel::parameters() {
    std::vector<Parameter*> out = {
        &token_embedding.weight,
        &position_embedding,
        &lexical_kind,
        &latent_kind,
    };
    for (auto& block : blocks) {
        auto p = block->parameters();
        out.insert(out.end(), p.begin(), p.end());
    }
    auto fp = final_norm.parameters();
    out.insert(out.end(), fp.begin(), fp.end());
    // Protect against accidental shared-parameter duplication.
    std::vector<Parameter*> unique;
    std::unordered_set<const Parameter*> seen;
    for (auto* p : out) if (p && seen.insert(p).second) unique.push_back(p);
    return unique;
}

} // namespace motifcl::nn

#include <motifcl/ops/activation.hpp>
#include <motifcl/ops/fog.hpp>

namespace motifcl::nn {

FogOperatorBankV3::FogOperatorBankV3(Backend& backend, int d_model, int rank)
    : d_model_(d_model), rank_(rank),
      value_norm_(backend, d_model), read_norm_(backend, d_model), control_norm_(backend, d_model),
      route_value_(backend, d_model, d_model, false),
      route_read_(backend, d_model, d_model, false),
      route_control_(backend, d_model, d_model, false),
      route_out_(backend, d_model, 7, true) {
    MCL_CHECK(d_model > 0 && (d_model % 2) == 0 && rank > 0,
              "FOG operator bank requires positive even d_model and rank");
    for (int i = 0; i < 4; ++i) {
        state_down_.push_back(std::make_unique<Linear>(backend, d_model, rank, false));
        read_down_.push_back(std::make_unique<Linear>(backend, d_model, rank, false));
        control_down_.push_back(std::make_unique<Linear>(backend, d_model, rank, false));
        up_.push_back(std::make_unique<Linear>(backend, rank, d_model, false));
        op_bias_.emplace_back(Tensor::zeros(backend, {rank}), true);
    }
    // Start strongly READ-biased, matching the model-ready PyTorch v3
    // initialization contract while still allowing the router to learn other
    // legal operators through the straight-through gradient.
    auto bias = route_out_.bias.data.to_vector<float>();
    if (bias.size() == 7) {
        bias[0] = 4.0f;
        for (std::size_t i = 1; i < bias.size(); ++i) bias[i] = -2.0f;
        route_out_.bias.data = Tensor::from_cpu(backend, {7}, DType::F32, bias.data());
        route_out_.bias.data.set_requires_grad(true);
    }
}

FogOperatorBankOutput FogOperatorBankV3::forward3(const Tensor& value,
                                                  const Tensor& addressed,
                                                  const Tensor& control) {
    MCL_CHECK(value.dtype() == DType::F32 && addressed.dtype() == DType::F32 && control.dtype() == DType::F32,
              "FOG operator bank expects f32");
    MCL_CHECK(value.ndim() == 2 && value.shape() == addressed.shape() && value.shape() == control.shape(),
              "FOG operator bank expects matching [batch,d_model] tensors");
    MCL_CHECK(value.shape()[1] == d_model_, "FOG operator bank d_model mismatch");

    auto rv = route_value_.forward(value_norm_.forward(value));
    auto rr = route_read_.forward(read_norm_.forward(addressed));
    auto rc = route_control_.forward(control_norm_.forward(control));
    auto route_h = silu(add(add(rv, rr), rc));
    auto logits = route_out_.forward(route_h);

    std::array<Tensor,7> candidates;
    candidates[0] = addressed;
    candidates[1] = value;
    candidates[2] = fog_block_product(value, addressed);
    for (int o = 0; o < 4; ++o) {
        auto sv = state_down_[static_cast<std::size_t>(o)]->forward(value);
        auto sr = read_down_[static_cast<std::size_t>(o)]->forward(addressed);
        auto sc = control_down_[static_cast<std::size_t>(o)]->forward(control);
        // Same key inductive bias as Python v3: multiplicative low-rank
        // interaction + a smaller additive path + control.
        auto bilinear = mul(sv, sr);
        auto additive = scale(add(sv, sr), 0.25f);
        auto h = add_bias_rows(add(add(bilinear, additive), sc), op_bias_[static_cast<std::size_t>(o)].data);
        h = silu(h);
        auto delta = up_[static_cast<std::size_t>(o)]->forward(h);
        // Fixed initial delta scale. This avoids a scalar-tensor broadcast
        // round-trip on GCN4; a native learned scalar gate can be added after
        // the first training witness without changing the operator family.
        candidates[static_cast<std::size_t>(3 + o)] = add(value, scale(delta, 0.119202922f));
    }
    return {fog_hard_route7(logits, candidates), logits};
}

Tensor FogOperatorBankV3::forward(const Tensor&) {
    MCL_CHECK(false, "FogOperatorBankV3 requires forward3(value,addressed,control)");
    return Tensor{};
}

std::vector<Parameter*> FogOperatorBankV3::parameters() {
    std::vector<Parameter*> out;
    auto append = [&](Module& m) { auto p=m.parameters(); out.insert(out.end(),p.begin(),p.end()); };
    append(value_norm_); append(read_norm_); append(control_norm_);
    append(route_value_); append(route_read_); append(route_control_); append(route_out_);
    for (int o=0;o<4;++o) {
        append(*state_down_[static_cast<std::size_t>(o)]);
        append(*read_down_[static_cast<std::size_t>(o)]);
        append(*control_down_[static_cast<std::size_t>(o)]);
        append(*up_[static_cast<std::size_t>(o)]);
        out.push_back(&op_bias_[static_cast<std::size_t>(o)]);
    }
    return out;
}

} // namespace motifcl::nn

#include <motifcl/ops/attention.hpp>
#include <motifcl/ops/norm.hpp>

namespace motifcl::nn {
namespace {

std::vector<Parameter*> fog_unique(std::vector<Parameter*> in) {
    std::vector<Parameter*> out;
    std::unordered_set<const Parameter*> seen;
    out.reserve(in.size());
    for (auto* p : in) if (p && seen.insert(p).second) out.push_back(p);
    return out;
}

void append_params(std::vector<Parameter*>& out, Module& module) {
    auto p = module.parameters();
    out.insert(out.end(), p.begin(), p.end());
}

} // namespace

FogAddressBinderV3::FogAddressBinderV3(Backend& backend, int d_model, int compare_rank)
    : d_model_(d_model), compare_rank_(compare_rank),
      address_map_(backend, d_model, compare_rank, false),
      rank_norm_weight_(Tensor::ones(backend, {compare_rank})) {
    MCL_CHECK(d_model > 0 && compare_rank > 0,
              "FOG binder requires positive d_model/compare_rank");
}

Tensor FogAddressBinderV3::forward_bind(const Tensor& query,
                                        const Tensor& key_states,
                                        const Tensor& value_states,
                                        int64_t batch_size,
                                        int64_t rows_per_batch) {
    MCL_CHECK(query.dtype() == DType::F32 && query.ndim() == 2,
              "FOG binder query must be f32 [B,D]");
    MCL_CHECK(key_states.dtype() == DType::F32 && key_states.ndim() == 2 &&
              value_states.dtype() == DType::F32 && value_states.ndim() == 2,
              "FOG binder key/value states must be rank-2 f32");
    MCL_CHECK(query.shape()[0] == batch_size && query.shape()[1] == d_model_,
              "FOG binder query shape mismatch");
    MCL_CHECK(key_states.shape()[0] == batch_size * rows_per_batch &&
              key_states.shape()[1] == d_model_ && value_states.shape() == key_states.shape(),
              "FOG binder key/value shape mismatch");

    auto q = address_map_.forward(query);
    auto k = address_map_.forward(key_states);
    q = rmsnorm(q, rank_norm_weight_);
    k = rmsnorm(k, rank_norm_weight_);
    // GQA backward on Vulkan requires its default 1/sqrt(rank) scale. Scaling
    // q outside the attention turns RMS-normalized dot products into the same
    // ~20*cosine address logits used by the Python binding-v2/v3 path while
    // retaining the stock Vulkan backward implementation.
    const float q_scale = 20.0f / std::sqrt(static_cast<float>(compare_rank_));
    q = scale(q, q_scale);
    return grouped_query_attention(
        q, k, value_states,
        1, 1, false,
        batch_size, 1, rows_per_batch, 0, 0.0f);
}

Tensor FogAddressBinderV3::forward(const Tensor&) {
    MCL_CHECK(false, "FogAddressBinderV3 requires forward_bind(query,keys,values,B,R)");
    return Tensor{};
}

std::vector<Parameter*> FogAddressBinderV3::parameters() {
    return address_map_.parameters();
}

FogRegisterCellV3::FogRegisterCellV3(Backend& backend, int d_model, int operator_rank)
    : d_model_(d_model),
      operator_bank_(backend, d_model, operator_rank),
      control_norm_(backend, d_model), read_norm_(backend, d_model),
      control_self_(backend, d_model, d_model, false),
      control_read_(backend, d_model, d_model, false),
      scratch_self_(backend, d_model, d_model, false),
      scratch_read_(backend, d_model, d_model, false),
      scratch_control_(backend, d_model, d_model, false),
      scratch_norm_(backend, d_model),
      halt_value_(backend, d_model, 1, false),
      halt_control_(backend, d_model, 1, true) {}

FogRegisterCellOutputV3 FogRegisterCellV3::forward_state(const FogRegisterStateV3& old,
                                                         const Tensor& addressed) {
    const auto shape = old.value.shape();
    MCL_CHECK(old.value.dtype() == DType::F32 && old.value.ndim() == 2 && shape[1] == d_model_,
              "FOG register value must be [B,D]");
    MCL_CHECK(old.control.shape() == shape && old.scratch0.shape() == shape &&
              old.scratch1.shape() == shape && addressed.shape() == shape,
              "FOG register cell shape mismatch");

    auto routed = operator_bank_.forward3(old.value, addressed, old.control);

    auto control_delta = silu(add(
        control_self_.forward(control_norm_.forward(old.control)),
        control_read_.forward(read_norm_.forward(addressed))));
    auto new_control = control_norm_.forward(add(old.control, scale(control_delta, 0.1f)));

    auto update_scratch = [&](const Tensor& scratch) {
        auto h = add(add(
            scratch_self_.forward(scratch_norm_.forward(scratch)),
            scratch_read_.forward(read_norm_.forward(addressed))),
            scratch_control_.forward(control_norm_.forward(new_control)));
        return scratch_norm_.forward(add(scratch, scale(silu(h), 0.1f)));
    };
    auto s0 = update_scratch(old.scratch0);
    auto s1 = update_scratch(old.scratch1);

    auto halt_logits = add(
        halt_value_.forward(routed.value),
        halt_control_.forward(new_control));
    auto halt = sigmoid(halt_logits);
    return {{routed.value, new_control, s0, s1}, routed.logits, halt};
}

Tensor FogRegisterCellV3::forward(const Tensor&) {
    MCL_CHECK(false, "FogRegisterCellV3 requires forward_state(registers,addressed)");
    return Tensor{};
}

std::vector<Parameter*> FogRegisterCellV3::parameters() {
    std::vector<Parameter*> out;
    append_params(out, operator_bank_);
    append_params(out, control_norm_); append_params(out, read_norm_);
    append_params(out, control_self_); append_params(out, control_read_);
    append_params(out, scratch_self_); append_params(out, scratch_read_);
    append_params(out, scratch_control_); append_params(out, scratch_norm_);
    append_params(out, halt_value_); append_params(out, halt_control_);
    return fog_unique(std::move(out));
}

FogV3Model::FogV3Model(Backend& backend, const FogV3Config& raw)
    : config(raw), lexical(backend, raw),
      binder(backend, raw.d_model, raw.d_model),
      machine(backend, raw.d_model, 48),
      dmodel_norm_weight_(Tensor::ones(backend, {raw.d_model})) {}

Tensor FogV3Model::forward(const Tensor& token_ids) {
    return lexical.forward(token_ids);
}

FogRegisterStateV3 FogV3Model::initial_state(const Tensor& query_ids) {
    MCL_CHECK(query_ids.dtype() == DType::I32 && query_ids.ndim() == 1,
              "FOG initial query ids must be [B] i32");
    const int64_t B = query_ids.shape()[0];
    auto q = lexical.token_embedding.forward(query_ids).view({B, config.d_model});
    auto zero = Tensor::zeros(q.backend(), {B, config.d_model}, DType::F32);
    // Value starts as the exact query code. Control starts from the same
    // identity; scratch registers are empty. This matches the corrected Python
    // v3 semantics where computation is possible on tick one.
    return {q, q, zero, zero};
}

FogRegisterCellOutputV3 FogV3Model::structured_step(const FogRegisterStateV3& old,
                                                    const Tensor& key_ids,
                                                    const Tensor& value_ids,
                                                    int64_t batch_size,
                                                    int64_t rows_per_batch) {
    MCL_CHECK(key_ids.dtype() == DType::I32 && value_ids.dtype() == DType::I32,
              "FOG structured key/value ids must be i32");
    MCL_CHECK(key_ids.numel() == batch_size * rows_per_batch && value_ids.numel() == key_ids.numel(),
              "FOG structured key/value count mismatch");
    auto keys = lexical.token_embedding.forward(key_ids).view({batch_size * rows_per_batch, config.d_model});
    auto values = lexical.token_embedding.forward(value_ids).view({batch_size * rows_per_batch, config.d_model});
    auto addressed = binder.forward_bind(old.value, keys, values, batch_size, rows_per_batch);
    return machine.forward_state(old, addressed);
}

Tensor FogV3Model::direct_vocab_logits(const Tensor& value) {
    MCL_CHECK(value.dtype() == DType::F32 && value.ndim() == 2 && value.shape()[1] == config.d_model,
              "FOG direct readout expects [B,D]");
    auto state = rmsnorm(value, dmodel_norm_weight_);
    auto codebook = rmsnorm(lexical.token_embedding.weight.data, dmodel_norm_weight_);
    // RMSNorm gives both rows norm sqrt(D), hence divide D and multiply by the
    // Python cosine-head temperature 20.
    return scale(matmul_transpose_b(state, codebook), 20.0f / static_cast<float>(config.d_model));
}

std::vector<Parameter*> FogV3Model::lexical_parameters() {
    return lexical.parameters();
}

std::vector<Parameter*> FogV3Model::machine_parameters() {
    std::vector<Parameter*> out;
    append_params(out, binder);
    append_params(out, machine);
    return fog_unique(std::move(out));
}

std::vector<Parameter*> FogV3Model::parameters() {
    auto out = lexical.parameters();
    auto mp = machine_parameters();
    out.insert(out.end(), mp.begin(), mp.end());
    return fog_unique(std::move(out));
}

} // namespace motifcl::nn
