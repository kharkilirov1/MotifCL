#include <motifcl/nn/transformer.hpp>

#include <cmath>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/activation.hpp>
#include <motifcl/ops/attention.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/indexing.hpp>
#include <motifcl/ops/matmul.hpp>

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

std::vector<Parameter*> ModernSelfAttention::parameters() {
    auto p1 = qkv_proj_.parameters();
    auto p2 = o_proj_.parameters();
    p1.insert(p1.end(), p2.begin(), p2.end());
    return p1;
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
    auto mlp_out = mlp_.forward(norm2_.forward(h));
    return add(h, mlp_out);
}

Tensor ModernTransformerBlock::forward_masked(const Tensor& x, const Tensor& mask, int64_t batch_size, int64_t seq_len) {
    auto attn_out = attn_.forward_masked(norm1_.forward(x), mask, batch_size, seq_len, causal_);
    auto h = add(x, attn_out);
    return add(h, mlp_.forward(norm2_.forward(h)));
}

Tensor ModernTransformerBlock::forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len) {
    auto attn_out = attn_.forward_with_cache(norm1_.forward(x), cache, batch_size, seq_len);
    auto h = add(x, attn_out);
    return add(h, mlp_.forward(norm2_.forward(h)));
}

Tensor ModernTransformerBlock::forward_with_cache_masked(const Tensor& x, const Tensor& mask, KVCache& cache, int64_t batch_size, int64_t seq_len) {
    auto attn_out = attn_.forward_with_cache_masked(norm1_.forward(x), mask, cache, batch_size, seq_len);
    auto h = add(x, attn_out);
    return add(h, mlp_.forward(norm2_.forward(h)));
}

std::vector<Parameter*> ModernTransformerBlock::parameters() {
    std::vector<Parameter*> result;
    for (auto* module : {static_cast<Module*>(&norm1_), static_cast<Module*>(&attn_), static_cast<Module*>(&norm2_), static_cast<Module*>(&mlp_)}) {
        auto p = module->parameters();
        result.insert(result.end(), p.begin(), p.end());
    }
    return result;
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
    Tensor logits = matmul(h, lm_head.data);
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
    Tensor logits = matmul(h, lm_head.data);
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
    Tensor logits = matmul(h, lm_head.data);
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
    Tensor logits = matmul(h, lm_head.data);
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

} // namespace motifcl::nn
