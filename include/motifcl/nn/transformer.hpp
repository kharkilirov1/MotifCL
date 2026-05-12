#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <motifcl/core/dtype.hpp>
#include <motifcl/nn/attention.hpp>
#include <motifcl/nn/embedding.hpp>
#include <motifcl/nn/mlp.hpp>
#include <motifcl/nn/rmsnorm.hpp>
#include <motifcl/ops/attention.hpp>

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
    int head_dim = 0;
    int n_layer = 0;
    int mlp_hidden = 0;
    float dropout = 0.0f;
    float rms_norm_eps = 1e-6f;
    bool use_rope = true;
    bool use_swiglu = true;
    bool use_qkv_bias = false;
    bool causal = true;
    bool learned_position_embeddings = false;
    float rope_theta = 10000.0f;
    int rotary_dim = 0;
    int sliding_window = 0;
    float embedding_scale = 1.0f;
    int per_layer_input_dim = 0;
    int per_layer_input_vocab_size = 0;
    bool use_per_layer_inputs = false;
    std::vector<int> layer_head_dims;
    std::vector<int> layer_mlp_hiddens;
    std::vector<float> layer_rope_thetas;
    bool split_qkv_projections = false;
    bool split_mlp_projections = false;
    bool skip_weight_init = false;
    bool use_qk_norm = false;
    bool use_post_attention_norm = false;
    bool use_post_ffw_norm = false;
    bool use_layer_output_scale = false;
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

class PagedKVCache {
public:
    Tensor k_pages;
    Tensor v_pages;
    Tensor page_table;
    int64_t batch_size = 0;
    int64_t max_seq_len = 0;
    int64_t page_size = 0;
    int64_t page_count = 0;
    int64_t length = 0;
    int n_kv_head = 0;
    int head_dim = 0;

    PagedKVCache() = default;
    PagedKVCache(Backend& backend, int64_t batch, int64_t max_seq, int64_t page_size,
                 int n_kv_head, int head_dim);
    int64_t capacity() const { return page_count * page_size; }
    int64_t tokens_seen = 0;
    void reset() { length = 0; tokens_seen = 0; }
};

class DeltaStateCache {
public:
    Tensor state;
    int64_t batch_size = 0;
    int num_heads = 0;
    int head_dim = 0;
    int state_dim = 0;

    DeltaStateCache() = default;
    DeltaStateCache(Backend& backend, int64_t batch, int num_heads, int head_dim, int state_dim);
    void zero();
};

class ModernMLP : public Module {
public:
    Linear gate_up_proj;
    Linear down_proj;
    bool use_swiglu = true;
    float dropout_p = 0.0f;

    ModernMLP(Backend& backend, int n_embd, int hidden, bool use_swiglu = true,
              bool use_bias = false, float dropout = 0.0f, bool skip_weight_init = false);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;

    void enable_quantized_inference(DType qdtype = DType::Q4_0);
    void enable_split_projections(bool enabled = true) { use_split_projections_ = enabled; }
    void disable_quantized_inference();
    bool quantized_inference_enabled() const;
    DType quantized_weight_dtype() const;
    bool split_projections_enabled() const { return use_split_projections_; }
    Linear& gate_proj() { return gate_proj_; }
    const Linear& gate_proj() const { return gate_proj_; }
    Linear& up_proj() { return up_proj_; }
    const Linear& up_proj() const { return up_proj_; }

private:
    Linear gate_proj_;
    Linear up_proj_;
    bool use_split_projections_ = false;
};

class MoEFFN : public Module {
public:
    Parameter router_weight;
    Parameter expert_gate_weight;
    Parameter expert_up_weight;
    Parameter expert_down_weight;
    int num_experts = 0;
    int experts_per_token = 0;
    int in_features = 0;
    int hidden_features = 0;

    MoEFFN(Backend& backend, int n_embd, int hidden, int experts, int top_k);
    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override;
};

class GatedDeltaNetLayer : public Module {
public:
    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear gate_proj;
    Linear o_proj;

    GatedDeltaNetLayer(Backend& backend, const TransformerConfig& config, int state_dim = 0);
    Tensor forward(const Tensor& x) override;
    Tensor forward_with_state(const Tensor& x, DeltaStateCache& state, int64_t batch_size, int64_t seq_len);
    std::vector<Parameter*> parameters() override;
    int n_head() const { return n_head_; }
    int head_dim() const { return head_dim_; }
    int state_dim() const { return state_dim_; }
    float decay() const { return decay_; }
    void set_decay(float value) { decay_ = value; }

private:
    int n_embd_ = 0;
    int n_head_ = 0;
    int head_dim_ = 0;
    int state_dim_ = 0;
    float decay_ = 1.0f;
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
    Tensor forward_with_cache(const Tensor& x, PagedKVCache& cache, int64_t batch_size, int64_t seq_len);
    std::vector<Parameter*> parameters() override;

    int n_head() const { return n_head_; }
    int n_kv_head() const { return n_kv_head_; }
    int head_dim() const { return head_dim_; }
    int attention_window() const { return attention_window_; }
    void set_attention_window(int window);
    bool qk_norm_enabled() const { return use_qk_norm_; }
    void enable_split_projections(bool enabled = true) { use_split_projections_ = enabled; }
    bool split_projections_enabled() const { return use_split_projections_; }
    Linear& qkv_proj() { return qkv_proj_; }
    const Linear& qkv_proj() const { return qkv_proj_; }
    Linear& q_proj() { return q_proj_; }
    const Linear& q_proj() const { return q_proj_; }
    Linear& k_proj() { return k_proj_; }
    const Linear& k_proj() const { return k_proj_; }
    Linear& v_proj() { return v_proj_; }
    const Linear& v_proj() const { return v_proj_; }
    Linear& o_proj() { return o_proj_; }
    const Linear& o_proj() const { return o_proj_; }
    RMSNorm& q_norm() { return q_norm_; }
    const RMSNorm& q_norm() const { return q_norm_; }
    RMSNorm& k_norm() { return k_norm_; }
    const RMSNorm& k_norm() const { return k_norm_; }
    void enable_quantized_inference(DType qdtype = DType::Q4_0);
    void disable_quantized_inference();
    bool quantized_inference_enabled() const;
    DType quantized_weight_dtype() const;

private:
    Linear qkv_proj_;
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear o_proj_;
    RMSNorm q_norm_;
    RMSNorm k_norm_;
    QKV project_qkv(const Tensor& x);
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
    int attention_window_ = 0;
    bool use_split_projections_ = false;
    bool use_qk_norm_ = false;
};

class GatedAttentionLayer : public Module {
public:
    ModernSelfAttention attention;
    Linear gate_proj;

    GatedAttentionLayer(Backend& backend, const TransformerConfig& config);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, int64_t batch_size, int64_t seq_len, bool causal = true);
    Tensor forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len);
    Tensor forward_with_cache(const Tensor& x, PagedKVCache& cache, int64_t batch_size, int64_t seq_len);
    std::vector<Parameter*> parameters() override;
};

enum class HybridAttentionKind {
    FullAttention,
    SlidingAttention,
    GatedAttention,
    GatedDeltaNet,
};

enum class HybridFFNKind {
    SwiGLUFFN,
    MoEFFN,
};

struct HybridLayerConfig {
    HybridAttentionKind attention = HybridAttentionKind::FullAttention;
    HybridFFNKind ffn = HybridFFNKind::SwiGLUFFN;
    int sliding_window = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    int delta_state_dim = 0;
};

class HybridTransformerBlock : public Module {
public:
    HybridLayerConfig layer_config;

    HybridTransformerBlock(Backend& backend, const TransformerConfig& config, const HybridLayerConfig& layer_config);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, int64_t batch_size, int64_t seq_len);
    Tensor forward_with_cache(const Tensor& x,
                              KVCache* kv_cache,
                              PagedKVCache* paged_kv_cache,
                              DeltaStateCache* delta_state,
                              bool use_paged_kv,
                              int64_t batch_size,
                              int64_t seq_len);
    std::vector<Parameter*> parameters() override;

    bool uses_kv_cache() const;
    bool uses_state_cache() const;

    RMSNorm& norm1() { return norm1_; }
    const RMSNorm& norm1() const { return norm1_; }
    RMSNorm& norm2() { return norm2_; }
    const RMSNorm& norm2() const { return norm2_; }
    ModernSelfAttention* attention() { return attention_.get(); }
    const ModernSelfAttention* attention() const { return attention_.get(); }
    GatedAttentionLayer* gated_attention() { return gated_attention_.get(); }
    const GatedAttentionLayer* gated_attention() const { return gated_attention_.get(); }
    GatedDeltaNetLayer* delta() { return delta_.get(); }
    const GatedDeltaNetLayer* delta() const { return delta_.get(); }
    ModernMLP* mlp() { return mlp_.get(); }
    const ModernMLP* mlp() const { return mlp_.get(); }
    MoEFFN* moe() { return moe_.get(); }
    const MoEFFN* moe() const { return moe_.get(); }

private:
    RMSNorm norm1_;
    std::unique_ptr<ModernSelfAttention> attention_;
    std::unique_ptr<GatedAttentionLayer> gated_attention_;
    std::unique_ptr<GatedDeltaNetLayer> delta_;
    RMSNorm norm2_;
    std::unique_ptr<ModernMLP> mlp_;
    std::unique_ptr<MoEFFN> moe_;
    bool causal_ = true;
};

class ModernTransformerBlock : public Module {
public:
    ModernTransformerBlock(Backend& backend, const TransformerConfig& config);
    Tensor forward(const Tensor& x) override;
    Tensor forward(const Tensor& x, int64_t batch_size, int64_t seq_len, const Tensor* per_layer_input = nullptr);
    Tensor forward_masked(const Tensor& x, const Tensor& mask, int64_t batch_size, int64_t seq_len,
                          const Tensor* per_layer_input = nullptr);
    Tensor forward_with_cache(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len,
                              const Tensor* per_layer_input = nullptr);
    Tensor forward_with_cache_packed_per_layer_input(const Tensor& x, KVCache& cache, int64_t batch_size, int64_t seq_len,
                                                     const Tensor* packed_per_layer_input,
                                                     int layer_index,
                                                     int64_t token_count);
    Tensor forward_with_cache_masked(const Tensor& x, const Tensor& mask, KVCache& cache, int64_t batch_size, int64_t seq_len,
                                     const Tensor* per_layer_input = nullptr);
    Tensor forward_with_cache_positions_masked(const Tensor& x, const Tensor& positions, const Tensor& mask,
                                               KVCache& cache, int64_t batch_size, int64_t seq_len,
                                               int64_t cache_length_after, bool causal = false,
                                               const Tensor* per_layer_input = nullptr);
    Tensor forward_with_cache(const Tensor& x, PagedKVCache& cache, int64_t batch_size, int64_t seq_len,
                              const Tensor* per_layer_input = nullptr);
    Tensor forward_with_cache_packed_per_layer_input(const Tensor& x, PagedKVCache& cache, int64_t batch_size, int64_t seq_len,
                                                     const Tensor* packed_per_layer_input,
                                                     int layer_index,
                                                     int64_t token_count);
    std::vector<Parameter*> parameters() override;
    void enable_quantized_inference(DType qdtype = DType::Q4_0);
    void disable_quantized_inference();
    bool quantized_inference_enabled() const;
    DType quantized_weight_dtype() const;
    int attention_window() const { return attn_.attention_window(); }
    void set_attention_window(int window) { attn_.set_attention_window(window); }
    RMSNorm& norm1() { return norm1_; }
    const RMSNorm& norm1() const { return norm1_; }
    ModernSelfAttention& attention() { return attn_; }
    const ModernSelfAttention& attention() const { return attn_; }
    RMSNorm& norm2() { return norm2_; }
    const RMSNorm& norm2() const { return norm2_; }
    RMSNorm& post_attention_norm() { return post_attention_norm_; }
    const RMSNorm& post_attention_norm() const { return post_attention_norm_; }
    RMSNorm& post_ffw_norm() { return post_ffw_norm_; }
    const RMSNorm& post_ffw_norm() const { return post_ffw_norm_; }
    Parameter& layer_output_scale() { return layer_output_scale_; }
    const Parameter& layer_output_scale() const { return layer_output_scale_; }
    bool per_layer_input_enabled() const { return use_per_layer_input_; }
    Linear* per_layer_input_gate() { return per_layer_input_gate_.get(); }
    const Linear* per_layer_input_gate() const { return per_layer_input_gate_.get(); }
    Linear* per_layer_projection() { return per_layer_projection_.get(); }
    const Linear* per_layer_projection() const { return per_layer_projection_.get(); }
    RMSNorm* post_per_layer_norm() { return post_per_layer_norm_.get(); }
    const RMSNorm* post_per_layer_norm() const { return post_per_layer_norm_.get(); }
    ModernMLP& mlp() { return mlp_; }
    const ModernMLP& mlp() const { return mlp_; }

private:
    RMSNorm norm1_;
    ModernSelfAttention attn_;
    RMSNorm norm2_;
    RMSNorm post_attention_norm_;
    RMSNorm post_ffw_norm_;
    Parameter layer_output_scale_;
    std::unique_ptr<Linear> per_layer_input_gate_;
    std::unique_ptr<Linear> per_layer_projection_;
    std::unique_ptr<RMSNorm> post_per_layer_norm_;
    ModernMLP mlp_;
    float dropout_p_ = 0.0f;
    bool causal_ = true;
    bool use_post_attention_norm_ = false;
    bool use_post_ffw_norm_ = false;
    bool use_layer_output_scale_ = false;
    bool use_per_layer_input_ = false;

    Tensor apply_attention_residual(const Tensor& x, const Tensor& attn_out);
    Tensor apply_ffn_residual(const Tensor& h, const Tensor& mlp_out);
    Tensor apply_per_layer_input(const Tensor& h, const Tensor* per_layer_input);
    Tensor apply_packed_per_layer_input(const Tensor& h, const Tensor* packed_per_layer_input,
                                        int layer_index, int64_t token_count);
    Tensor apply_packed_per_layer_input_and_scale(const Tensor& h, const Tensor* packed_per_layer_input,
                                                  int layer_index, int64_t token_count);
    Tensor forward_with_cache_packed_per_layer_input_eager(const Tensor& x, KVCache& cache,
                                                           int64_t batch_size, int64_t seq_len,
                                                           const Tensor* packed_per_layer_input,
                                                           int layer_index,
                                                           int64_t token_count);
    Tensor forward_with_cache_packed_per_layer_input_replay(const Tensor& x, KVCache& cache,
                                                            int64_t batch_size, int64_t seq_len,
                                                            const Tensor* packed_per_layer_input,
                                                            int layer_index,
                                                            int64_t token_count);
    Tensor decode_tail_after_attention_packed(const Tensor& h, const Tensor* packed_per_layer_input,
                                              int layer_index, int64_t token_count);
    Tensor decode_tail_after_attention_packed_replay(const Tensor& h, const Tensor* packed_per_layer_input,
                                                     int layer_index, int64_t token_count);
    Tensor apply_layer_output_scale(const Tensor& h);
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
    std::unique_ptr<Embedding> per_layer_token_embedding;
    std::unique_ptr<Linear> per_layer_model_projection;
    std::unique_ptr<RMSNorm> per_layer_projection_norm;

    ModernGPTModel(Backend& backend, const TransformerConfig& config);
    Tensor forward(const Tensor& token_ids) override;
    Tensor forward_masked(const Tensor& token_ids, const Tensor& mask);
    Tensor forward_with_cache(const Tensor& token_ids, std::vector<KVCache>& caches);
    Tensor forward_with_cache(const Tensor& token_ids, std::vector<KVCache*>& caches);
    Tensor forward_with_cache(const Tensor& token_ids, std::vector<PagedKVCache>& caches);
    Tensor forward_with_cache(const Tensor& token_ids, std::vector<PagedKVCache*>& caches);
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
    std::vector<PagedKVCache> create_paged_kv_cache(Backend& backend, int64_t batch_size,
                                                    int64_t page_size = 256) const;
    std::vector<DeltaStateCache> create_delta_state_cache(Backend& backend, int64_t batch_size,
                                                          int state_dim = 0) const;
    void set_layer_attention_window(int layer, int window);
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
    Tensor input_embeddings(const Tensor& token_ids, int64_t batch_size, int64_t seq_len);
    Tensor compute_per_layer_inputs(const Tensor& token_ids,
                                    const Tensor& inputs_embeds,
                                    int64_t batch_size,
                                    int64_t seq_len);
    Tensor per_layer_input_slice(const Tensor& packed, int layer, int64_t token_count) const;
    Tensor project_logits(const Tensor& h);
};

class HybridRuntimeCache {
public:
    std::vector<KVCache> kv_caches;
    std::vector<PagedKVCache> paged_kv_caches;
    std::vector<DeltaStateCache> delta_states;
    bool use_paged_kv = false;
    int64_t batch_size = 0;
    int64_t page_size = 0;

    void reset();
};

class HybridGPTModel : public Module {
public:
    TransformerConfig config;
    std::vector<HybridLayerConfig> layer_configs;
    Embedding token_embedding;
    Parameter position_embedding;
    std::vector<std::shared_ptr<HybridTransformerBlock>> blocks;
    RMSNorm final_norm;
    Parameter lm_head;

    HybridGPTModel(Backend& backend, const TransformerConfig& config,
                   const std::vector<HybridLayerConfig>& layer_configs = {});
    Tensor forward(const Tensor& token_ids) override;
    Tensor forward_with_cache(const Tensor& token_ids, HybridRuntimeCache& cache);
    std::vector<Parameter*> parameters() override;
    HybridRuntimeCache create_runtime_cache(Backend& backend, int64_t batch_size,
                                            bool use_paged_kv = false,
                                            int64_t page_size = 256) const;
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
