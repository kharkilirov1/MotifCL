#include <motifcl/nn/transformer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/ops/attention.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/fused_transformer.hpp>
#include <motifcl/ops/indexing.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/ops/quant.hpp>

namespace motifcl::nn {

Tensor MLP::forward(const Tensor& x) {
    auto h = motifcl::matmul(x, fc1.weight.data);
    h = fc1.has_bias() ? motifcl::add_bias_gelu_rows(h, fc1.bias.data) : motifcl::gelu(h);
    return fc2.forward(h);
}

std::vector<Parameter*> MLP::parameters() {
    auto p1 = fc1.parameters();
    auto p2 = fc2.parameters();
    p1.insert(p1.end(), p2.begin(), p2.end());
    return p1;
}

TransformerBlock::TransformerBlock(Backend& backend, int n_embd, int n_head, int mlp_hidden)
    : norm1_(backend, n_embd), attn_(backend, n_embd, n_head), norm2_(backend, n_embd), mlp_(backend, n_embd, mlp_hidden, n_embd) {}

Tensor TransformerBlock::forward(const Tensor& x) {
    return forward(x, 1, x.shape()[0]);
}

Tensor TransformerBlock::forward(const Tensor& x, int64_t batch_size, int64_t seq_len) {
    MCL_CHECK(x.ndim() == 2, "TransformerBlock expects flattened [tokens, channels] input");
    MCL_CHECK(batch_size > 0 && seq_len > 0 && batch_size * seq_len == x.shape()[0], "TransformerBlock invalid batch/sequence dimensions");
    auto h = add(x, attn_.forward(norm1_.forward(x), batch_size, seq_len, true));
    return add(h, mlp_.forward(norm2_.forward(h)));
}

std::vector<Parameter*> TransformerBlock::parameters() {
    std::vector<Parameter*> result;
    for (auto* module : {static_cast<Module*>(&norm1_), static_cast<Module*>(&attn_), static_cast<Module*>(&norm2_), static_cast<Module*>(&mlp_)}) {
        auto p = module->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    return result;
}

GPTModel::GPTModel(Backend& backend, int vocab_size, int block_size, int n_embd, int n_head, int n_layer, int mlp_hidden)
    : token_embedding(Tensor::randn(backend, {vocab_size, n_embd}, 0.02f)),
      position_embedding(Tensor::randn(backend, {block_size, n_embd}, 0.02f)),
      final_norm(backend, n_embd),
      lm_head(Tensor::randn(backend, {n_embd, vocab_size}, 0.02f)),
      vocab_size_(vocab_size),
      block_size_(block_size),
      n_embd_(n_embd) {
    blocks.reserve(static_cast<std::size_t>(n_layer));
    for (int i = 0; i < n_layer; ++i) {
        blocks.push_back(std::make_shared<TransformerBlock>(backend, n_embd, n_head, mlp_hidden));
    }
}

Tensor GPTModel::forward(const Tensor& token_ids) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "GPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "GPTModel token ids must be [T] or [B,T]");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T <= block_size_, "sequence length exceeds GPTModel block_size");

    Tensor h = token_position_embedding(token_ids, token_embedding.data, position_embedding.data); // [B*T, C]
    for (auto& block : blocks) h = block->forward(h, B, T);
    h = final_norm.forward(h);
    Tensor logits = matmul(h, lm_head.data); // [B*T, V]
    if (token_ids.ndim() == 2) return logits.view({B, T, vocab_size_});
    return logits;
}

std::vector<Parameter*> GPTModel::parameters() {
    std::vector<Parameter*> result{&token_embedding, &position_embedding};
    for (auto& block : blocks) {
        auto p = block->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    auto nf = final_norm.parameters();
    result.insert(result.end(), nf.begin(), nf.end());
    result.push_back(&lm_head);
    return result;
}

namespace {
TransformerConfig normalize_config(TransformerConfig cfg) {
    MCL_CHECK(cfg.vocab_size > 0, "TransformerConfig vocab_size must be positive");
    MCL_CHECK(cfg.block_size > 0, "TransformerConfig block_size must be positive");
    MCL_CHECK(cfg.n_embd > 0, "TransformerConfig n_embd must be positive");
    MCL_CHECK(cfg.n_head > 0, "TransformerConfig n_head must be positive");
    MCL_CHECK(cfg.n_layer > 0, "TransformerConfig n_layer must be positive");
    MCL_CHECK(cfg.n_embd % cfg.n_head == 0, "TransformerConfig n_embd must be divisible by n_head");
    if (cfg.n_kv_head <= 0) cfg.n_kv_head = cfg.n_head;
    MCL_CHECK(cfg.n_head % cfg.n_kv_head == 0, "TransformerConfig n_head must be divisible by n_kv_head");
    const int head_dim = cfg.n_embd / cfg.n_head;
    if (cfg.rotary_dim <= 0) cfg.rotary_dim = head_dim;
    MCL_CHECK(cfg.rotary_dim <= head_dim && cfg.rotary_dim % 2 == 0, "TransformerConfig rotary_dim must be even and <= head_dim");
    if (cfg.mlp_hidden <= 0) cfg.mlp_hidden = cfg.use_swiglu ? (cfg.n_embd * 8 / 3) : (cfg.n_embd * 4);
    MCL_CHECK(cfg.dropout >= 0.0f && cfg.dropout < 1.0f, "TransformerConfig dropout must be in [0, 1)");
    return cfg;
}

bool env_enabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) return false;
    return std::string(value) != "0" && std::string(value) != "false" && std::string(value) != "FALSE";
}

bool can_use_fused_swiglu_mlp_residual(const ModernMLP& mlp) {
    return env_enabled("MOTIFCL_ENABLE_FUSED_MLP_BACKWARD") &&
           mlp.use_swiglu &&
           mlp.dropout_p == 0.0f &&
           !mlp.gate_up_proj.has_bias() &&
           !mlp.down_proj.has_bias();
}

Tensor fused_or_eager_mlp_residual(const Tensor& h, RMSNorm& norm, ModernMLP& mlp) {
    if (can_use_fused_swiglu_mlp_residual(mlp)) {
        return motifcl::fused_swiglu_mlp_rmsnorm_residual(
            h,
            norm.weight.data,
            mlp.gate_up_proj.weight.data,
            mlp.down_proj.weight.data,
            norm.eps);
    }
    return motifcl::add(h, mlp.forward(norm.forward(h)));
}
}

KVCache::KVCache(Backend& backend, int64_t batch, int64_t max_seq, int n_kv, int head_dim_value)
    : k(Tensor::zeros(backend, {batch * max_seq, n_kv * head_dim_value}, DType::F32)),
      v(Tensor::zeros(backend, {batch * max_seq, n_kv * head_dim_value}, DType::F32)),
      batch_size(batch),
      max_seq_len(max_seq),
      n_kv_head(n_kv),
      head_dim(head_dim_value) {
    MCL_CHECK(batch > 0 && max_seq > 0 && n_kv > 0 && head_dim_value > 0, "KVCache invalid dimensions");
}

ModernMLP::ModernMLP(Backend& backend, int n_embd, int hidden, bool swiglu_enabled, bool use_bias, float dropout)
    : gate_up_proj(backend, n_embd, swiglu_enabled ? hidden * 2 : hidden, use_bias),
      down_proj(backend, hidden, n_embd, use_bias),
      use_swiglu(swiglu_enabled),
      dropout_p(dropout) {}

Tensor ModernMLP::forward(const Tensor& x) {
    auto hidden = gate_up_proj.forward(x);
    hidden = use_swiglu ? motifcl::swiglu(hidden) : motifcl::gelu(hidden);
    auto out = down_proj.forward(hidden);
    return dropout_p > 0.0f ? motifcl::dropout(out, dropout_p, true) : out;
}

std::vector<Parameter*> ModernMLP::parameters() {
    auto p1 = gate_up_proj.parameters();
    auto p2 = down_proj.parameters();
    p1.insert(p1.end(), p2.begin(), p2.end());
    return p1;
}

void ModernMLP::enable_quantized_inference(DType qdtype) {
    gate_up_proj.enable_quantized_inference(qdtype);
    down_proj.enable_quantized_inference(qdtype);
}

void ModernMLP::disable_quantized_inference() {
    gate_up_proj.disable_quantized_inference();
    down_proj.disable_quantized_inference();
}

bool ModernMLP::quantized_inference_enabled() const {
    return gate_up_proj.quantized_inference_enabled() && down_proj.quantized_inference_enabled();
}

DType ModernMLP::quantized_weight_dtype() const {
    return quantized_inference_enabled() ? gate_up_proj.quantized_weight_dtype() : DType::F32;
}

ModernSelfAttention::ModernSelfAttention(Backend& backend, const TransformerConfig& raw_config)
    : qkv_proj_(backend,
                normalize_config(raw_config).n_embd,
                normalize_config(raw_config).n_embd + 2 * normalize_config(raw_config).n_kv_head * (normalize_config(raw_config).n_embd / normalize_config(raw_config).n_head),
                normalize_config(raw_config).use_qkv_bias),
      o_proj_(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).n_embd, normalize_config(raw_config).use_qkv_bias) {
    const auto cfg = normalize_config(raw_config);
    n_embd_ = cfg.n_embd;
    n_head_ = cfg.n_head;
    n_kv_head_ = cfg.n_kv_head;
    head_dim_ = cfg.n_embd / cfg.n_head;
    q_dim_ = cfg.n_head * head_dim_;
    kv_dim_ = cfg.n_kv_head * head_dim_;
    use_rope_ = cfg.use_rope;
    rope_theta_ = cfg.rope_theta;
    rotary_dim_ = cfg.rotary_dim;
    dropout_p_ = cfg.dropout;
}

Tensor ModernSelfAttention::forward(const Tensor& x) {
    return forward(x, 1, x.shape()[0], true);
}

Tensor ModernSelfAttention::forward(const Tensor& x, int64_t batch_size, int64_t seq_len, bool causal) {
    MCL_CHECK(x.ndim() == 2 && x.shape()[1] == n_embd_, "ModernSelfAttention expects [B*T, n_embd]");
    auto packed = qkv_proj_.forward(x);
    auto split = motifcl::qkv_split(packed, q_dim_, kv_dim_);
    auto q = use_rope_ ? motifcl::rope(split.q, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_, 0) : split.q;
    auto k = use_rope_ ? motifcl::rope(split.k, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_, 0) : split.k;
    auto context = motifcl::grouped_query_attention(q, k, split.v, n_head_, n_kv_head_, causal, batch_size, seq_len, seq_len, 0);
    auto out = o_proj_.forward(context);
    return dropout_p_ > 0.0f ? motifcl::dropout(out, dropout_p_, true) : out;
}

Tensor ModernSelfAttention::forward_masked(const Tensor& x, const Tensor& mask, int64_t batch_size, int64_t seq_len, bool causal) {
    MCL_CHECK(x.ndim() == 2 && x.shape()[1] == n_embd_, "ModernSelfAttention expects [B*T, n_embd]");
    auto packed = qkv_proj_.forward(x);
    auto split = motifcl::qkv_split(packed, q_dim_, kv_dim_);
    auto q = use_rope_ ? motifcl::rope(split.q, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_, 0) : split.q;
    auto k = use_rope_ ? motifcl::rope(split.k, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_, 0) : split.k;
    auto context = motifcl::grouped_query_attention_masked(q, k, split.v, mask, n_head_, n_kv_head_,
                                                           causal, batch_size, seq_len, seq_len, 0, false);
    auto out = o_proj_.forward(context);
    return dropout_p_ > 0.0f ? motifcl::dropout(out, dropout_p_, true) : out;
}

Tensor ModernSelfAttention::forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len) {
    MCL_CHECK(cache.k.valid() && cache.v.valid(), "ModernSelfAttention cache is not initialized");
    MCL_CHECK(batch_size == cache.batch_size && n_kv_head_ == cache.n_kv_head && head_dim_ == cache.head_dim, "ModernSelfAttention cache shape mismatch");
    MCL_CHECK(cache.length + seq_len <= cache.max_seq_len, "ModernSelfAttention KV cache capacity exceeded");
    auto packed = qkv_proj_.forward(x);
    auto split = motifcl::qkv_split(packed, q_dim_, kv_dim_);
    const int64_t offset = cache.length;
    auto q = use_rope_ ? motifcl::rope(split.q, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_, offset) : split.q;
    auto k_new = use_rope_ ? motifcl::rope(split.k, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_, offset) : split.k;
    motifcl::kv_cache_append(k_new, split.v, cache.k, cache.v, batch_size, seq_len, cache.max_seq_len, offset);
    cache.length += seq_len;
    auto context = motifcl::grouped_query_attention(q, cache.k, cache.v, n_head_, n_kv_head_, true,
                                                    batch_size, seq_len, cache.max_seq_len, offset);
    return o_proj_.forward(context);
}

Tensor ModernSelfAttention::forward_with_cache_masked(const Tensor& x, const Tensor& mask, KVCache& cache, int64_t batch_size, int64_t seq_len) {
    MCL_CHECK(cache.k.valid() && cache.v.valid(), "ModernSelfAttention cache is not initialized");
    MCL_CHECK(batch_size == cache.batch_size && n_kv_head_ == cache.n_kv_head && head_dim_ == cache.head_dim, "ModernSelfAttention cache shape mismatch");
    MCL_CHECK(cache.length + seq_len <= cache.max_seq_len, "ModernSelfAttention KV cache capacity exceeded");
    auto packed = qkv_proj_.forward(x);
    auto split = motifcl::qkv_split(packed, q_dim_, kv_dim_);
    const int64_t offset = cache.length;
    auto q = use_rope_ ? motifcl::rope(split.q, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_, offset) : split.q;
    auto k_new = use_rope_ ? motifcl::rope(split.k, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_, offset) : split.k;
    motifcl::kv_cache_append(k_new, split.v, cache.k, cache.v, batch_size, seq_len, cache.max_seq_len, offset);
    cache.length += seq_len;
    auto context = motifcl::grouped_query_attention_masked(q, cache.k, cache.v, mask, n_head_, n_kv_head_, true,
                                                           batch_size, seq_len, cache.max_seq_len, offset, false);
    return o_proj_.forward(context);
}

Tensor ModernSelfAttention::forward_with_cache_positions_masked(const Tensor& x, const Tensor& positions, const Tensor& mask,
                                                                KVCache& cache, int64_t batch_size, int64_t seq_len,
                                                                int64_t cache_length_after, bool causal) {
    MCL_CHECK(cache.k.valid() && cache.v.valid(), "ModernSelfAttention cache is not initialized");
    MCL_CHECK(batch_size == cache.batch_size && n_kv_head_ == cache.n_kv_head && head_dim_ == cache.head_dim, "ModernSelfAttention cache shape mismatch");
    MCL_CHECK(cache_length_after >= cache.length && cache_length_after <= cache.max_seq_len, "ModernSelfAttention positioned KV cache length out of range");
    auto packed = qkv_proj_.forward(x);
    auto split = motifcl::qkv_split(packed, q_dim_, kv_dim_);
    auto q = use_rope_ ? motifcl::rope_positions(split.q, positions, n_head_, batch_size, seq_len, rope_theta_, rotary_dim_) : split.q;
    auto k_new = use_rope_ ? motifcl::rope_positions(split.k, positions, n_kv_head_, batch_size, seq_len, rope_theta_, rotary_dim_) : split.k;
    motifcl::kv_cache_append_positions(k_new, split.v, positions, cache.k, cache.v, batch_size, seq_len, cache.max_seq_len);
    cache.length = cache_length_after;
    auto context = motifcl::grouped_query_attention_masked(q, cache.k, cache.v, mask, n_head_, n_kv_head_, causal,
                                                           batch_size, seq_len, cache.max_seq_len, 0, false);
    return o_proj_.forward(context);
}

std::vector<Parameter*> ModernSelfAttention::parameters() {
    auto p1 = qkv_proj_.parameters();
    auto p2 = o_proj_.parameters();
    p1.insert(p1.end(), p2.begin(), p2.end());
    return p1;
}

void ModernSelfAttention::enable_quantized_inference(DType qdtype) {
    qkv_proj_.enable_quantized_inference(qdtype);
    o_proj_.enable_quantized_inference(qdtype);
}

void ModernSelfAttention::disable_quantized_inference() {
    qkv_proj_.disable_quantized_inference();
    o_proj_.disable_quantized_inference();
}

bool ModernSelfAttention::quantized_inference_enabled() const {
    return qkv_proj_.quantized_inference_enabled() && o_proj_.quantized_inference_enabled();
}

DType ModernSelfAttention::quantized_weight_dtype() const {
    return quantized_inference_enabled() ? qkv_proj_.quantized_weight_dtype() : DType::F32;
}

ModernTransformerBlock::ModernTransformerBlock(Backend& backend, const TransformerConfig& raw_config)
    : norm1_(backend, normalize_config(raw_config).n_embd),
      attn_(backend, normalize_config(raw_config)),
      norm2_(backend, normalize_config(raw_config).n_embd),
      mlp_(backend, normalize_config(raw_config).n_embd, normalize_config(raw_config).mlp_hidden,
           normalize_config(raw_config).use_swiglu, normalize_config(raw_config).use_qkv_bias, normalize_config(raw_config).dropout) {
    const auto cfg = normalize_config(raw_config);
    dropout_p_ = cfg.dropout;
    causal_ = cfg.causal;
}

Tensor ModernTransformerBlock::forward(const Tensor& x) {
    return forward(x, 1, x.shape()[0]);
}

Tensor ModernTransformerBlock::forward(const Tensor& x, int64_t batch_size, int64_t seq_len) {
    auto attn_out = attn_.forward(norm1_.forward(x), batch_size, seq_len, causal_);
    auto h = add(x, attn_out);
    return fused_or_eager_mlp_residual(h, norm2_, mlp_);
}

Tensor ModernTransformerBlock::forward_masked(const Tensor& x, const Tensor& mask, int64_t batch_size, int64_t seq_len) {
    auto attn_out = attn_.forward_masked(norm1_.forward(x), mask, batch_size, seq_len, causal_);
    auto h = add(x, attn_out);
    return fused_or_eager_mlp_residual(h, norm2_, mlp_);
}

Tensor ModernTransformerBlock::forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len) {
    auto attn_out = attn_.forward_with_cache(norm1_.forward(x), cache, batch_size, seq_len);
    auto h = add(x, attn_out);
    return fused_or_eager_mlp_residual(h, norm2_, mlp_);
}

Tensor ModernTransformerBlock::forward_with_cache_masked(const Tensor& x, const Tensor& mask, KVCache& cache, int64_t batch_size, int64_t seq_len) {
    auto attn_out = attn_.forward_with_cache_masked(norm1_.forward(x), mask, cache, batch_size, seq_len);
    auto h = add(x, attn_out);
    return fused_or_eager_mlp_residual(h, norm2_, mlp_);
}

Tensor ModernTransformerBlock::forward_with_cache_positions_masked(const Tensor& x, const Tensor& positions, const Tensor& mask,
                                                                   KVCache& cache, int64_t batch_size, int64_t seq_len,
                                                                   int64_t cache_length_after, bool causal) {
    auto attn_out = attn_.forward_with_cache_positions_masked(norm1_.forward(x), positions, mask, cache,
                                                              batch_size, seq_len, cache_length_after, causal);
    auto h = add(x, attn_out);
    return fused_or_eager_mlp_residual(h, norm2_, mlp_);
}

std::vector<Parameter*> ModernTransformerBlock::parameters() {
    std::vector<Parameter*> result;
    for (auto* module : {static_cast<Module*>(&norm1_), static_cast<Module*>(&attn_), static_cast<Module*>(&norm2_), static_cast<Module*>(&mlp_)}) {
        auto p = module->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    return result;
}

void ModernTransformerBlock::enable_quantized_inference(DType qdtype) {
    attn_.enable_quantized_inference(qdtype);
    mlp_.enable_quantized_inference(qdtype);
}

void ModernTransformerBlock::disable_quantized_inference() {
    attn_.disable_quantized_inference();
    mlp_.disable_quantized_inference();
}

bool ModernTransformerBlock::quantized_inference_enabled() const {
    return attn_.quantized_inference_enabled() && mlp_.quantized_inference_enabled();
}

DType ModernTransformerBlock::quantized_weight_dtype() const {
    return quantized_inference_enabled() ? attn_.quantized_weight_dtype() : DType::F32;
}

ModernGPTModel::ModernGPTModel(Backend& backend, const TransformerConfig& raw_config)
    : config(normalize_config(raw_config)),
      token_embedding(backend, normalize_config(raw_config).vocab_size, normalize_config(raw_config).n_embd),
      position_embedding(Tensor::randn(backend, {normalize_config(raw_config).block_size, normalize_config(raw_config).n_embd}, 0.02f),
                         normalize_config(raw_config).learned_position_embeddings),
      final_norm(backend, normalize_config(raw_config).n_embd),
      lm_head(Tensor::randn(backend, {normalize_config(raw_config).n_embd, normalize_config(raw_config).vocab_size}, 0.02f)) {
    use_positions_ = config.learned_position_embeddings && !config.use_rope;
    blocks.reserve(static_cast<std::size_t>(config.n_layer));
    for (int i = 0; i < config.n_layer; ++i) {
        blocks.push_back(std::make_shared<ModernTransformerBlock>(backend, config));
    }
}

Tensor ModernGPTModel::forward(const Tensor& token_ids) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T <= config.block_size, "ModernGPTModel sequence length exceeds block_size");
    Tensor h = use_positions_
        ? token_position_embedding(token_ids, token_embedding.weight.data, position_embedding.data)
        : token_embedding.forward(token_ids).view({B * T, config.n_embd});
    for (auto& block : blocks) h = block->forward(h, B, T);
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor ModernGPTModel::forward_masked(const Tensor& token_ids, const Tensor& mask) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T <= config.block_size, "ModernGPTModel sequence length exceeds block_size");
    Tensor h = use_positions_
        ? token_position_embedding(token_ids, token_embedding.weight.data, position_embedding.data)
        : token_embedding.forward(token_ids).view({B * T, config.n_embd});
    for (auto& block : blocks) h = block->forward_masked(h, mask, B, T);
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor ModernGPTModel::forward_with_cache(const Tensor& token_ids, std::vector<KVCache>& caches) {
    std::vector<KVCache*> ptrs;
    ptrs.reserve(caches.size());
    for (auto& cache : caches) ptrs.push_back(&cache);
    return forward_with_cache(token_ids, ptrs);
}

Tensor ModernGPTModel::forward_with_cache(const Tensor& token_ids, std::vector<KVCache*>& caches) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(caches.size() == blocks.size(), "ModernGPTModel KV cache count must match layer count");
    MCL_CHECK(!use_positions_, "ModernGPTModel cached inference currently expects RoPE/token-only positions");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "ModernGPTModel invalid cached sequence length");
    if (!caches.empty()) {
        MCL_CHECK(caches[0] != nullptr, "ModernGPTModel null KV cache");
        const int64_t start = caches[0]->length;
        MCL_CHECK(start + T <= config.block_size, "ModernGPTModel KV cache exceeds block_size");
        for (auto* cache : caches) {
            MCL_CHECK(cache != nullptr, "ModernGPTModel null KV cache");
            MCL_CHECK(cache->length == start, "ModernGPTModel KV caches must have equal current length");
        }
    }
    Tensor h = token_embedding.forward(token_ids).view({B * T, config.n_embd});
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        h = blocks[i]->forward_with_cache(h, *caches[i], B, T);
    }
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor ModernGPTModel::forward_with_cache_masked(const Tensor& token_ids, const Tensor& mask, std::vector<KVCache>& caches) {
    std::vector<KVCache*> ptrs;
    ptrs.reserve(caches.size());
    for (auto& cache : caches) ptrs.push_back(&cache);
    return forward_with_cache_masked(token_ids, mask, ptrs);
}

Tensor ModernGPTModel::forward_with_cache_masked(const Tensor& token_ids, const Tensor& mask, std::vector<KVCache*>& caches) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(caches.size() == blocks.size(), "ModernGPTModel KV cache count must match layer count");
    MCL_CHECK(!use_positions_, "ModernGPTModel cached inference currently expects RoPE/token-only positions");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "ModernGPTModel invalid cached sequence length");
    if (!caches.empty()) {
        MCL_CHECK(caches[0] != nullptr, "ModernGPTModel null KV cache");
        const int64_t start = caches[0]->length;
        MCL_CHECK(start + T <= config.block_size, "ModernGPTModel KV cache exceeds block_size");
        for (auto* cache : caches) {
            MCL_CHECK(cache != nullptr, "ModernGPTModel null KV cache");
            MCL_CHECK(cache->length == start, "ModernGPTModel KV caches must have equal current length");
        }
    }
    Tensor h = token_embedding.forward(token_ids).view({B * T, config.n_embd});
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        h = blocks[i]->forward_with_cache_masked(h, mask, *caches[i], B, T);
    }
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

Tensor ModernGPTModel::forward_with_cache_positions_masked(const Tensor& token_ids, const Tensor& positions, const Tensor& mask,
                                                              std::vector<KVCache>& caches, int64_t cache_length_after,
                                                              bool causal) {
    std::vector<KVCache*> ptrs;
    ptrs.reserve(caches.size());
    for (auto& cache : caches) ptrs.push_back(&cache);
    return forward_with_cache_positions_masked(token_ids, positions, mask, ptrs, cache_length_after, causal);
}

Tensor ModernGPTModel::forward_with_cache_positions_masked(const Tensor& token_ids, const Tensor& positions, const Tensor& mask,
                                                              std::vector<KVCache*>& caches, int64_t cache_length_after,
                                                              bool causal) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "ModernGPTModel token ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "ModernGPTModel token ids must be [T] or [B,T]");
    MCL_CHECK(positions.dtype() == DType::I32 && positions.ndim() == 2, "ModernGPTModel positioned cache expects i32 [B,T] positions");
    MCL_CHECK(caches.size() == blocks.size(), "ModernGPTModel KV cache count must match layer count");
    MCL_CHECK(!use_positions_, "ModernGPTModel cached inference currently expects RoPE/token-only positions");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    MCL_CHECK(T > 0 && T <= config.block_size, "ModernGPTModel invalid cached sequence length");
    MCL_CHECK(positions.shape()[0] == B && positions.shape()[1] == T, "ModernGPTModel positions shape mismatch");
    MCL_CHECK(positions.backend_ptr() == token_ids.backend_ptr(), "ModernGPTModel positions must share backend with token ids");
    if (!caches.empty()) {
        MCL_CHECK(caches[0] != nullptr, "ModernGPTModel null KV cache");
        MCL_CHECK(cache_length_after >= caches[0]->length && cache_length_after <= config.block_size, "ModernGPTModel positioned KV cache exceeds block_size");
        for (auto* cache : caches) {
            MCL_CHECK(cache != nullptr, "ModernGPTModel null KV cache");
            MCL_CHECK(cache->length == caches[0]->length, "ModernGPTModel KV caches must have equal current length");
        }
    }
    Tensor h = token_embedding.forward(token_ids).view({B * T, config.n_embd});
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        h = blocks[i]->forward_with_cache_positions_masked(h, positions, mask, *caches[i], B, T, cache_length_after, causal);
    }
    h = final_norm.forward(h);
    Tensor logits = project_logits(h);
    if (token_ids.ndim() == 2) return logits.view({B, T, config.vocab_size});
    return logits;
}

std::vector<Parameter*> ModernGPTModel::parameters() {
    auto result = token_embedding.parameters();
    if (use_positions_) result.push_back(&position_embedding);
    for (auto& block : blocks) {
        auto p = block->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    auto nf = final_norm.parameters();
    result.insert(result.end(), nf.begin(), nf.end());
    result.push_back(&lm_head);
    return result;
}

std::vector<KVCache> ModernGPTModel::create_kv_cache(Backend& backend, int64_t batch_size) const {
    std::vector<KVCache> caches;
    caches.reserve(static_cast<std::size_t>(config.n_layer));
    const int head_dim = config.n_embd / config.n_head;
    for (int i = 0; i < config.n_layer; ++i) {
        caches.emplace_back(backend, batch_size, config.block_size, config.n_kv_head, head_dim);
    }
    return caches;
}

void ModernGPTModel::enable_quantized_inference(DType qdtype) {
    MCL_CHECK(qdtype == DType::Q8_0 || qdtype == DType::Q4_0,
              "ModernGPTModel quantized inference expects qdtype Q8_0 or Q4_0");
    for (auto& block : blocks) block->enable_quantized_inference(qdtype);
    quantized_weight_dtype_ = qdtype;
    quantized_lm_head_ = qdtype == DType::Q4_0
        ? quantize_q4_symmetric_cols(lm_head.data)
        : quantize_q8_symmetric_cols(lm_head.data);
}

void ModernGPTModel::enable_quantized_inference(const QuantizationPolicy& policy) {
    MCL_CHECK(policy.default_dtype == DType::Q8_0 || policy.default_dtype == DType::Q4_0,
              "ModernGPTModel quantization policy default dtype must be Q8_0 or Q4_0");
    MCL_CHECK(policy.lm_head_dtype == DType::Q8_0 || policy.lm_head_dtype == DType::Q4_0,
              "ModernGPTModel quantization policy lm_head dtype must be Q8_0 or Q4_0");
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        const bool use_q8 = std::find(policy.q8_layers.begin(), policy.q8_layers.end(), static_cast<int>(i)) != policy.q8_layers.end();
        blocks[i]->enable_quantized_inference(use_q8 ? DType::Q8_0 : policy.default_dtype);
    }
    quantized_weight_dtype_ = policy.default_dtype;
    quantized_lm_head_ = policy.lm_head_dtype == DType::Q4_0
        ? quantize_q4_symmetric_cols(lm_head.data)
        : quantize_q8_symmetric_cols(lm_head.data);
}

void ModernGPTModel::set_quantized_lm_head(const Tensor& weight) {
    MCL_CHECK(weight.valid() && (weight.dtype() == DType::Q8_0 || weight.dtype() == DType::Q4_0),
              "ModernGPTModel quantized lm_head must be Q8_0 or Q4_0");
    MCL_CHECK(weight.ndim() == 2 && weight.shape()[0] == config.n_embd && weight.shape()[1] == config.vocab_size,
              "ModernGPTModel quantized lm_head shape mismatch");
    quantized_lm_head_ = weight;
    quantized_weight_dtype_ = weight.dtype();
}

void ModernGPTModel::disable_quantized_inference() {
    for (auto& block : blocks) block->disable_quantized_inference();
    quantized_lm_head_ = Tensor{};
    quantized_weight_dtype_ = DType::F32;
}

Tensor ModernGPTModel::project_logits(const Tensor& h) {
    if (quantized_lm_head_.valid() && !autograd::is_enabled()) {
        auto hq = quantize_q8_symmetric_rows(h);
        return matmul(hq, quantized_lm_head_);
    }
    return matmul(h, lm_head.data);
}

} // namespace motifcl::nn
