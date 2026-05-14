#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <motifcl/core/dtype.hpp>
#include <motifcl/nn/linear.hpp>
#include <motifcl/nn/transformer.hpp>
#include <motifcl/tensor/tensor.hpp>

namespace motifcl {

struct SafeTensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::uint64_t data_begin = 0;
    std::uint64_t data_end = 0;
};

class SafeTensorsFile {
public:
    SafeTensorsFile() = default;

    static SafeTensorsFile open(const std::string& path);

    const std::string& path() const { return path_; }
    std::vector<std::string> tensor_names() const;
    bool contains(const std::string& name) const;
    const SafeTensorInfo& tensor_info(const std::string& name) const;
    Tensor load_tensor(Backend& backend, const std::string& name, bool force_f32 = true) const;
    std::vector<float> load_f32_vector(const std::string& name) const;

private:
    std::string path_;
    std::string header_;
    std::uint64_t data_start_ = 0;
    std::unordered_map<std::string, SafeTensorInfo> infos_;
};

std::unordered_map<std::string, Tensor> load_safetensors(Backend& backend,
                                                         const std::vector<std::string>& paths,
                                                         bool force_f32 = true);

namespace nn {

struct GemmaConfig {
    int vocab_size = 0;
    int max_position_embeddings = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;
    float attention_dropout = 0.0f;
    bool attention_bias = false;
    bool attention_k_eq_v = false;
    bool tie_word_embeddings = false;
    int bos_token_id = 1;
    int eos_token_id = 2;
    int pad_token_id = 0;
    int sliding_window = 0;
};

GemmaConfig load_gemma_config_json(const std::string& path);
TransformerConfig to_transformer_config(const GemmaConfig& cfg);
ModernGPTModel make_gemma_model(Backend& backend, const GemmaConfig& cfg);

struct GemmaWeightName {
    std::string hf_name;
    std::string internal_name;
    std::string kind;
    int layer = -1;
};

GemmaWeightName map_gemma_hf_weight_name(const std::string& hf_name);
std::vector<std::string> expected_gemma_hf_weight_names(const GemmaConfig& cfg, bool include_lm_head = true);

struct GemmaWeightLoadReport {
    int loaded_tensors = 0;
    std::vector<std::string> applied;
    std::vector<std::string> missing;
    std::vector<std::string> unexpected;
};

GemmaWeightLoadReport load_gemma_hf_weights(Backend& backend,
                                            ModernGPTModel& model,
                                            const std::vector<std::string>& safetensors_paths,
                                            const GemmaConfig& cfg,
                                            bool strict = false,
                                            bool trainable = false);

class GemmaTokenizer {
public:
    GemmaTokenizer() = default;

    static GemmaTokenizer byte_fallback(int vocab_size = 256, int bos_token_id = 1, int eos_token_id = 2);
    static GemmaTokenizer from_tokens(const std::vector<std::string>& tokens,
                                      int bos_token_id = 1,
                                      int eos_token_id = 2,
                                      const std::string& tokenizer_model_type = "GGUF");
    static GemmaTokenizer load_vocab(const std::string& path, int bos_token_id = 1, int eos_token_id = 2);

    std::vector<std::int32_t> encode(const std::string& text, bool add_bos = false, bool add_eos = false) const;
    std::string decode(const std::vector<std::int32_t>& ids, bool skip_special = true) const;

    int vocab_size() const { return vocab_size_; }
    int bos_token_id() const { return bos_token_id_; }
    int eos_token_id() const { return eos_token_id_; }
    bool is_byte_fallback() const { return byte_fallback_; }
    const std::string& tokenizer_model_type() const { return tokenizer_model_type_; }

private:
    int vocab_size_ = 256;
    int bos_token_id_ = 1;
    int eos_token_id_ = 2;
    bool byte_fallback_ = true;
    bool sentencepiece_style_ = false;
    bool sp_add_dummy_prefix_ = true;
    bool sp_remove_extra_whitespaces_ = true;
    bool sp_escape_whitespaces_ = true;
    std::string tokenizer_model_type_ = "byte";
    std::string sp_normalizer_name_ = "nmt_nfkc";
    std::unordered_map<std::string, std::int32_t> token_to_id_;
    std::unordered_map<std::int32_t, std::string> id_to_token_;
    std::unordered_map<std::string, int> bpe_merge_rank_;
};

class QuantizedLinear : public Module {
public:
    QuantizedLinear() = default;
    explicit QuantizedLinear(const Tensor& f32_weight, DType qdtype = DType::Q4_0);
    QuantizedLinear(const Tensor& f32_weight, const Tensor& bias, DType qdtype = DType::Q4_0);

    static QuantizedLinear from_linear(const Linear& linear, DType qdtype = DType::Q4_0);

    Tensor forward(const Tensor& x) override;
    std::vector<Parameter*> parameters() override { return {}; }

    const Tensor& quantized_weight() const { return weight_; }
    const Tensor& bias() const { return bias_; }
    bool has_bias() const { return use_bias_; }
    DType weight_dtype() const { return weight_dtype_; }
    int in_features() const { return in_features_; }
    int out_features() const { return out_features_; }

private:
    Tensor weight_;
    Tensor bias_;
    DType weight_dtype_ = DType::Q4_0;
    bool use_bias_ = false;
    int in_features_ = 0;
    int out_features_ = 0;
};

struct GenerateOptions {
    int max_new_tokens = 32;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    int bos_token_id = 1;
    int eos_token_id = 2;
    int pad_token_id = 0;
    bool add_bos = false;
    bool prefill_prompt = true;
    bool adaptive_prefill = true;
    int adaptive_prefill_max_tokens = 24;
    bool gpu_greedy_sampling = true;
    bool use_paged_kv_cache = false;
    int kv_page_size = 256;
    std::uint32_t seed = 1234;
};

bool should_use_streaming_prefill(const ModernGPTModel& model,
                                  std::size_t prompt_token_count,
                                  const GenerateOptions& options = {});

std::vector<std::int32_t> generate(Backend& backend,
                                   ModernGPTModel& model,
                                   const std::vector<std::int32_t>& prompt_tokens,
                                   const GenerateOptions& options = {});

std::string generate_text(Backend& backend,
                          ModernGPTModel& model,
                          const GemmaTokenizer& tokenizer,
                          const std::string& prompt,
                          const GenerateOptions& options = {});

std::vector<std::vector<std::int32_t>> generate_batch(
    Backend& backend,
    ModernGPTModel& model,
    const std::vector<std::vector<std::int32_t>>& prompt_tokens,
    const GenerateOptions& options = {});

std::vector<std::string> generate_batch_text(Backend& backend,
                                             ModernGPTModel& model,
                                             const GemmaTokenizer& tokenizer,
                                             const std::vector<std::string>& prompts,
                                             const GenerateOptions& options = {});

} // namespace nn
} // namespace motifcl
