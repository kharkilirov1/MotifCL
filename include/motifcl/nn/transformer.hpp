#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <motifcl/core/dtype.hpp>
#include <motifcl/nn/attention.hpp>
#include <motifcl/nn/embedding.hpp>
#include <motifcl/nn/mlp.hpp>
#include <motifcl/nn/rmsnorm.hpp>

namespace motifcl::nn {

class TransformerBlock : public Module {
public:
    TransformerBlock(Backend& backend, int n_embd, int n_head, int mlp_hidden);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, int64_t batch_size, int64_t seq_len);
    std::vector<Parameter*> parameters() override;

private:
    RMSNorm norm1_;
    SelfAttention attn_;
    RMSNorm norm2_;
    MLP mlp_;
};

class GPTModel : public Module {
public:
    Parameter token_embedding;
    Parameter position_embedding;
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    RMSNorm final_norm;
    Parameter lm_head;

    GPTModel(Backend& backend, int vocab_size, int block_size, int n_embd, int n_head, int n_layer, int mlp_hidden);
    Tensor forward(const Tensor& token_ids) override;
    std::vector<Parameter*> parameters() override;

    int vocab_size() const { return vocab_size_; }
    int block_size() const { return block_size_; }
    int n_embd() const { return n_embd_; }

private:
    int vocab_size_ = 0;
    int block_size_ = 0;
    int n_embd_ = 0;
};

struct TransformerConfig {
    int vocab_size = 0;
    int block_size = 0;
    int n_embd = 0;
    int n_head = 0;
    int n_kv_head = 0;
    int n_layer = 0;
    int mlp_hidden = 0;
    float dropout = 0.0f;
    bool use_rope = true;
    bool use_swiglu = true;
    bool use_qkv_bias = false;
    bool causal = true;
    bool learned_position_embeddings = false;
    float rope_theta = 10000.0f;
    int rotary_dim = 0;
};

class KVCache {
public:
    Tensor k;
    Tensor v;
    int64_t batch_size = 0;
    int64_t max_seq_len = 0;
    int64_t length = 0;
    int n_kv_head = 0;
    int head_dim = 0;

    KVCache() = default;
    KVCache(Backend& backend, int64_t batch, int64_t max_seq, int n_kv_head, int head_dim);
    void reset() { length = 0; }
};

class ModernMLP : public Module {
public:
    Linear gate_up_proj;
    Linear down_proj;
    bool use_swiglu = true;
    float dropout_p = 0.0f;

    ModernMLP(Backend& backend, int n_embd, int hidden, bool use_swiglu = true, bool use_bias = false, float dropout = 0.0f);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

    void enable_quantized_inference(DType qdtype = DType::Q4_0);
    void disable_quantized_inference();
    bool quantized_inference_enabled() const;
    DType quantized_weight_dtype() const;
};

class ModernSelfAttention : public Module {
public:
    ModernSelfAttention(Backend& backend, const TransformerConfig& config);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, int64_t batch_size, int64_t seq_len, bool causal = true);
    Tensor forward_masked(const Tensor& x, const Tensor& mask, int64_t batch_size, int64_t seq_len, bool causal = true);
    Tensor forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len);
    Tensor forward_with_cache_masked(const Tensor& x, const Tensor& mask, KVCache& cache, int64_t batch_size, int64_t seq_len);
    Tensor forward_with_cache_positions_masked(const Tensor& x, const Tensor& positions, const Tensor& mask,
                                               KVCache& cache, int64_t batch_size, int64_t seq_len,
                                               int64_t cache_length_after, bool causal = false);
    std::vector<Parameter*> parameters() override;

    int n_head() const { return n_head_; }
    int n_kv_head() const { return n_kv_head_; }
    int head_dim() const { return head_dim_; }
    Linear& qkv_proj() { return qkv_proj_; }
    const Linear& qkv_proj() const { return qkv_proj_; }
    Linear& o_proj() { return o_proj_; }
    const Linear& o_proj() const { return o_proj_; }
    void enable_quantized_inference(DType qdtype = DType::Q4_0);
    void disable_quantized_inference();
    bool quantized_inference_enabled() const;
    DType quantized_weight_dtype() const;

private:
    Linear qkv_proj_;
    Linear o_proj_;
    int n_embd_ = 0;
    int n_head_ = 0;
    int n_kv_head_ = 0;
    int head_dim_ = 0;
    int q_dim_ = 0;
    int kv_dim_ = 0;
    bool use_rope_ = true;
    float rope_theta_ = 10000.0f;
    int rotary_dim_ = 0;
    float dropout_p_ = 0.0f;
};

class ModernTransformerBlock : public Module {
public:
    ModernTransformerBlock(Backend& backend, const TransformerConfig& config);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, int64_t batch_size, int64_t seq_len);
    Tensor forward_masked(const Tensor& x, const Tensor& mask, int64_t batch_size, int64_t seq_len);
    Tensor forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len);
    Tensor forward_with_cache_masked(const Tensor& x, const Tensor& mask, KVCache& cache, int64_t batch_size, int64_t seq_len);
    Tensor forward_with_cache_positions_masked(const Tensor& x, const Tensor& positions, const Tensor& mask,
                                               KVCache& cache, int64_t batch_size, int64_t seq_len,
                                               int64_t cache_length_after, bool causal = false);
    std::vector<Parameter*> parameters() override;
    void enable_quantized_inference(DType qdtype = DType::Q4_0);
    void disable_quantized_inference();
    bool quantized_inference_enabled() const;
    DType quantized_weight_dtype() const;
    RMSNorm& norm1() { return norm1_; }
    const RMSNorm& norm1() const { return norm1_; }
    ModernSelfAttention& attention() { return attn_; }
    const ModernSelfAttention& attention() const { return attn_; }
    RMSNorm& norm2() { return norm2_; }
    const RMSNorm& norm2() const { return norm2_; }
    ModernMLP& mlp() { return mlp_; }
    const ModernMLP& mlp() const { return mlp_; }

private:
    RMSNorm norm1_;
    ModernSelfAttention attn_;
    RMSNorm norm2_;
    ModernMLP mlp_;
    float dropout_p_ = 0.0f;
    bool causal_ = true;
};

struct QuantizationPolicy {
    DType default_dtype = DType::Q4_0;
    DType lm_head_dtype = DType::Q4_0;
    std::vector<int> q8_layers;
};

class ModernGPTModel : public Module {
public:
    TransformerConfig config;
    Embedding token_embedding;
    Parameter position_embedding;
    std::vector<std::shared_ptr<ModernTransformerBlock>> blocks;
    RMSNorm final_norm;
    Parameter lm_head;

    ModernGPTModel(Backend& backend, const TransformerConfig& config);
    Tensor forward(const Tensor& token_ids) override;
    Tensor forward_masked(const Tensor& token_ids, const Tensor& mask);
    Tensor forward_with_cache(const Tensor& token_ids, std::vector<KVCache>& caches);
    Tensor forward_with_cache(const Tensor& token_ids, std::vector<KVCache*>& caches);
    Tensor forward_with_cache_masked(const Tensor& token_ids, const Tensor& mask, std::vector<KVCache>& caches);
    Tensor forward_with_cache_masked(const Tensor& token_ids, const Tensor& mask, std::vector<KVCache*>& caches);
    Tensor forward_with_cache_positions_masked(const Tensor& token_ids, const Tensor& positions, const Tensor& mask,
                                               std::vector<KVCache>& caches, int64_t cache_length_after,
                                               bool causal = false);
    Tensor forward_with_cache_positions_masked(const Tensor& token_ids, const Tensor& positions, const Tensor& mask,
                                               std::vector<KVCache*>& caches, int64_t cache_length_after,
                                               bool causal = false);
    std::vector<Parameter*> parameters() override;
    std::vector<KVCache> create_kv_cache(Backend& backend, int64_t batch_size) const;
    void enable_quantized_inference(DType qdtype = DType::Q4_0);
    void enable_quantized_inference(const QuantizationPolicy& policy);
    void set_quantized_lm_head(const Tensor& weight);
    void disable_quantized_inference();
    bool quantized_inference_enabled() const { return quantized_lm_head_.valid(); }
    DType quantized_weight_dtype() const { return quantized_weight_dtype_; }
    const Tensor& quantized_lm_head() const { return quantized_lm_head_; }

private:
    bool use_positions_ = false;
    Tensor quantized_lm_head_;
    DType quantized_weight_dtype_ = DType::F32;
    Tensor project_logits(const Tensor& h);
};

} // namespace motifcl::nn
