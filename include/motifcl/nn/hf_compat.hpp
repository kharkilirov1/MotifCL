#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <motifcl/nn/gemma.hpp>
#include <motifcl/nn/transformer.hpp>

namespace motifcl::nn {

enum class HFArchitecture {
    Auto,
    GenericDecoder,
    Gemma,
    Gemma2,
    Gemma3,
    Gemma4,
    Llama,
    Mistral,
    Qwen2,
    Qwen3,
    Qwen35,
    Phi3,
    Phi4,
    Mixtral,
    DeepSeek,
    Falcon,
    GPTNeoX,
    Mamba,
};

std::string hf_architecture_name(HFArchitecture architecture);
HFArchitecture parse_hf_architecture(const std::string& value);

enum class HFChatTemplateKind {
    Auto,
    None,
    Generic,
    ChatML,
    Llama2,
    Llama3,
    Mistral,
    Gemma,
};

enum class HFModelFormat {
    Unknown,
    HuggingFaceDirectory,
    HuggingFaceConfig,
    GGUF,
};

enum class ModernLayerKind {
    FullAttention,
    SlidingAttention,
    GatedDeltaNet,
    GatedAttention,
    MoEFFN,
    VisionProjector,
    AudioProjector,
    SwiGLUFFN,
};

std::string modern_layer_kind_name(ModernLayerKind kind);
ModernLayerKind parse_modern_layer_kind(const std::string& value);

struct LayerSpec {
    int graph_index = 0;
    int transformer_layer = -1;
    ModernLayerKind kind = ModernLayerKind::FullAttention;
    int sliding_window = 0;
    int num_experts = 0;
    int experts_per_token = 0;
    bool uses_kv_cache = false;
    bool uses_state_cache = false;
    bool consumes_per_layer_input = false;
    std::string name;
};

struct ModernModelSpec {
    HFArchitecture architecture = HFArchitecture::GenericDecoder;
    std::string architecture_name = "generic_decoder";
    TransformerConfig transformer;
    std::vector<LayerSpec> layers;
    bool text_core = true;
    bool per_layer_inputs = false;
    bool has_moe = false;
    bool has_vision_projector = false;
    bool has_audio_projector = false;
    bool has_recurrent_state = false;
    std::vector<std::string> blockers;
};

struct LongContextRuntimeSpec {
    int max_context = 0;
    int page_size = 256;
    int sliding_window = 0;
    std::vector<int> kv_cache_layers;
    std::vector<int> sliding_window_layers;
    std::vector<int> state_cache_layers;
    bool needs_paged_kv = false;
    bool needs_sliding_window_cache = false;
    bool needs_state_cache = false;
};

struct HFArchitectureInfo {
    HFArchitecture architecture = HFArchitecture::GenericDecoder;
    std::string name;
    std::vector<std::string> aliases;
    bool decoder_only = true;
    bool supports_hf_safetensors = false;
    bool supports_gguf = false;
    bool supports_chat_template = false;
    std::vector<std::string> blockers;
    std::string notes;
};

struct HFChatMessage {
    std::string role;
    std::string content;
};

struct HFTransformerConfig {
    HFArchitecture architecture = HFArchitecture::GenericDecoder;
    std::string architecture_name = "generic";
    TransformerConfig transformer;
    float rms_norm_eps = 1e-6f;
    bool tie_word_embeddings = false;
    bool attention_bias = false;
    int bos_token_id = 1;
    int eos_token_id = 2;
    int pad_token_id = 0;
    int sliding_window = 0;
    bool attention_k_eq_v = false;
    std::vector<std::string> layer_types;
    std::vector<int> local_attention_layers;
    std::vector<int> global_attention_layers;
    bool per_layer_inputs = false;
    bool has_moe = false;
    int num_experts = 0;
    int experts_per_token = 0;
    bool has_vision_projector = false;
    bool has_audio_projector = false;
    bool has_gated_delta_net = false;
    bool has_gated_attention = false;
};

struct HFModelProbe {
    std::string source_path;
    HFModelFormat format = HFModelFormat::Unknown;
    HFTransformerConfig config;
    bool has_config = false;
    bool has_tokenizer = false;
    bool has_weights = false;
    std::size_t weight_files = 0;
    std::size_t gguf_tensors = 0;
    std::size_t gguf_quantized_tensors = 0;
    bool can_instantiate = false;
    bool can_load_weights = false;
    bool can_generate_text = false;
    std::vector<std::string> warnings;
    std::vector<std::string> blockers;
};

using HFWeightName = GemmaWeightName;
using HFWeightLoadReport = GemmaWeightLoadReport;
using HFTokenizer = GemmaTokenizer;

HFTransformerConfig load_hf_transformer_config_json(const std::string& path,
                                                    HFArchitecture architecture = HFArchitecture::Auto);
HFTransformerConfig load_hf_transformer_config_gguf(const std::string& path,
                                                    HFArchitecture architecture = HFArchitecture::Auto);

const HFArchitectureInfo& hf_architecture_info(HFArchitecture architecture);
std::vector<HFArchitectureInfo> hf_architecture_registry();
bool hf_architecture_supported_now(HFArchitecture architecture, HFModelFormat format = HFModelFormat::Unknown);
std::vector<std::string> hf_architecture_blockers(HFArchitecture architecture);
std::string hf_model_format_name(HFModelFormat format);
std::string hf_chat_template_name(HFChatTemplateKind kind);
HFChatTemplateKind parse_hf_chat_template_kind(const std::string& value);
ModernModelSpec modern_model_spec_from_config(const HFTransformerConfig& cfg);
ModernModelSpec load_modern_model_spec_json(const std::string& path,
                                            HFArchitecture architecture = HFArchitecture::Auto);
bool modern_model_spec_runnable_by_modern_gpt(const ModernModelSpec& spec);
bool modern_model_spec_runnable_by_hybrid(const ModernModelSpec& spec);
std::vector<std::string> modern_model_spec_blockers(const ModernModelSpec& spec);
LongContextRuntimeSpec long_context_runtime_spec_from_model_spec(const ModernModelSpec& spec,
                                                                 int page_size = 256);
std::string format_modern_model_spec(const ModernModelSpec& spec);
HFChatTemplateKind infer_hf_chat_template_kind(HFArchitecture architecture,
                                               const std::string& model_dir_or_tokenizer_config = "");
std::string apply_hf_chat_template(const std::vector<HFChatMessage>& messages,
                                   HFArchitecture architecture = HFArchitecture::GenericDecoder,
                                   HFChatTemplateKind kind = HFChatTemplateKind::Auto,
                                   bool add_generation_prompt = true);
HFModelProbe probe_hf_transformer_model(const std::string& path,
                                        HFArchitecture architecture = HFArchitecture::Auto);
std::string format_hf_model_probe(const HFModelProbe& probe);

GemmaConfig to_gemma_compatible_config(const HFTransformerConfig& cfg);
ModernGPTModel make_hf_transformer_model(Backend& backend, const HFTransformerConfig& cfg);
std::vector<HybridLayerConfig> hybrid_layer_configs_from_model_spec(const ModernModelSpec& spec);
HybridGPTModel make_hf_hybrid_transformer_model(Backend& backend, const HFTransformerConfig& cfg);

HFWeightName map_hf_transformer_weight_name(HFArchitecture architecture, const std::string& hf_name);
HFWeightName map_gguf_transformer_weight_name(HFArchitecture architecture, const std::string& gguf_name);
std::vector<std::string> expected_hf_transformer_weight_names(const HFTransformerConfig& cfg,
                                                              bool include_lm_head = true);
std::vector<std::string> expected_gguf_transformer_weight_names(const HFTransformerConfig& cfg,
                                                                bool include_lm_head = true);

HFWeightLoadReport load_hf_transformer_weights(Backend& backend,
                                               ModernGPTModel& model,
                                               const std::vector<std::string>& safetensors_paths,
                                               const HFTransformerConfig& cfg,
                                               bool strict = false,
                                               bool trainable = false);
HFWeightLoadReport load_hf_transformer_gguf_weights(Backend& backend,
                                                    ModernGPTModel& model,
                                                    const std::string& gguf_path,
                                                    const HFTransformerConfig& cfg,
                                                    bool strict = false,
                                                    bool trainable = false);
HFWeightLoadReport load_hf_hybrid_transformer_weights(Backend& backend,
                                                      HybridGPTModel& model,
                                                      const std::vector<std::string>& safetensors_paths,
                                                      const HFTransformerConfig& cfg,
                                                      bool strict = false,
                                                      bool trainable = false);

void enable_hf_transformer_quantized_inference(ModernGPTModel& model, DType qdtype = DType::Q4_0);
void disable_hf_transformer_quantized_inference(ModernGPTModel& model);

HFTokenizer load_hf_tokenizer(const std::string& model_dir_or_vocab_path, const HFTransformerConfig& cfg);
HFTokenizer load_hf_tokenizer_gguf(const std::string& path, const HFTransformerConfig& cfg);

std::string generate_hf_text(Backend& backend,
                             ModernGPTModel& model,
                             const HFTokenizer& tokenizer,
                             const std::string& prompt,
                             const GenerateOptions& options = {});
std::string generate_hf_hybrid_text(Backend& backend,
                                    HybridGPTModel& model,
                                    const HFTokenizer& tokenizer,
                                    const std::string& prompt,
                                    const GenerateOptions& options = {});

std::vector<std::string> generate_hf_batch_text(Backend& backend,
                                                ModernGPTModel& model,
                                                const HFTokenizer& tokenizer,
                                                const std::vector<std::string>& prompts,
                                                const GenerateOptions& options = {});

} // namespace motifcl::nn
