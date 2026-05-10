#pragma once

#include <string>
#include <vector>

#include <motifcl/nn/gemma.hpp>
#include <motifcl/nn/transformer.hpp>

namespace motifcl::nn {

enum class HFArchitecture {
    Auto,
    GenericDecoder,
    Gemma,
    Llama,
    Mistral,
    Qwen2,
};

std::string hf_architecture_name(HFArchitecture architecture);
HFArchitecture parse_hf_architecture(const std::string& value);

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
};

using HFWeightName = GemmaWeightName;
using HFWeightLoadReport = GemmaWeightLoadReport;
using HFTokenizer = GemmaTokenizer;

HFTransformerConfig load_hf_transformer_config_json(const std::string& path,
                                                    HFArchitecture architecture = HFArchitecture::Auto);

GemmaConfig to_gemma_compatible_config(const HFTransformerConfig& cfg);
ModernGPTModel make_hf_transformer_model(Backend& backend, const HFTransformerConfig& cfg);

HFWeightName map_hf_transformer_weight_name(HFArchitecture architecture, const std::string& hf_name);
std::vector<std::string> expected_hf_transformer_weight_names(const HFTransformerConfig& cfg,
                                                              bool include_lm_head = true);

HFWeightLoadReport load_hf_transformer_weights(Backend& backend,
                                               ModernGPTModel& model,
                                               const std::vector<std::string>& safetensors_paths,
                                               const HFTransformerConfig& cfg,
                                               bool strict = false,
                                               bool trainable = false);

void enable_hf_transformer_quantized_inference(ModernGPTModel& model, DType qdtype = DType::Q4_0);
void disable_hf_transformer_quantized_inference(ModernGPTModel& model);

HFTokenizer load_hf_tokenizer(const std::string& model_dir_or_vocab_path, const HFTransformerConfig& cfg);

std::string generate_hf_text(Backend& backend,
                             ModernGPTModel& model,
                             const HFTokenizer& tokenizer,
                             const std::string& prompt,
                             const GenerateOptions& options = {});

std::vector<std::string> generate_hf_batch_text(Backend& backend,
                                                ModernGPTModel& model,
                                                const HFTokenizer& tokenizer,
                                                const std::vector<std::string>& prompts,
                                                const GenerateOptions& options = {});

} // namespace motifcl::nn
