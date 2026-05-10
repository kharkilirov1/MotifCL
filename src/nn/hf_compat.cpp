#include <motifcl/nn/hf_compat.hpp>

#include <motifcl/core/error.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace motifcl::nn {
namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open HF config/tokenizer file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string json_unescape(std::string value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '\\' && i + 1 < value.size()) {
            const char c = value[++i];
            switch (c) {
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case '\\': out.push_back('\\'); break;
            case '"': out.push_back('"'); break;
            default: out.push_back(c); break;
            }
        } else {
            out.push_back(value[i]);
        }
    }
    return out;
}

std::string json_string_or(const std::string& text, const std::string& key, const std::string& fallback) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return fallback;
    return json_unescape(m[1].str());
}

std::string first_architecture_name(const std::string& text) {
    const auto direct = json_string_or(text, "model_type", "");
    if (!direct.empty()) return direct;

    const auto pos = text.find("\"architectures\"");
    if (pos == std::string::npos) return {};
    const auto bracket = text.find('[', pos);
    if (bracket == std::string::npos) return {};
    const auto end = text.find(']', bracket);
    if (end == std::string::npos) return {};
    const std::string array_text = text.substr(bracket + 1, end - bracket - 1);
    std::smatch m;
    const std::regex string_re("\"((?:\\\\.|[^\"\\\\])*)\"");
    if (!std::regex_search(array_text, m, string_re)) return {};
    return json_unescape(m[1].str());
}

bool architecture_uses_llama_like_weights(HFArchitecture architecture) {
    return architecture == HFArchitecture::GenericDecoder ||
           architecture == HFArchitecture::Gemma ||
           architecture == HFArchitecture::Llama ||
           architecture == HFArchitecture::Mistral ||
           architecture == HFArchitecture::Qwen2;
}

std::filesystem::path resolve_tokenizer_path(const std::string& model_dir_or_vocab_path) {
    namespace fs = std::filesystem;
    fs::path p(model_dir_or_vocab_path);
    if (!fs::is_directory(p)) return p;

    const fs::path tokenizer_model = p / "tokenizer.model";
    if (fs::exists(tokenizer_model)) return tokenizer_model;
    const fs::path vocab_json = p / "vocab.json";
    if (fs::exists(vocab_json)) return vocab_json;
    const fs::path simple_vocab = p / "vocab.txt";
    if (fs::exists(simple_vocab)) return simple_vocab;
    const fs::path tokenizer_json = p / "tokenizer.json";
    if (fs::exists(tokenizer_json)) return tokenizer_json;
    return {};
}

} // namespace

std::string hf_architecture_name(HFArchitecture architecture) {
    switch (architecture) {
    case HFArchitecture::Auto: return "auto";
    case HFArchitecture::GenericDecoder: return "generic_decoder";
    case HFArchitecture::Gemma: return "gemma";
    case HFArchitecture::Llama: return "llama";
    case HFArchitecture::Mistral: return "mistral";
    case HFArchitecture::Qwen2: return "qwen2";
    }
    return "generic_decoder";
}

HFArchitecture parse_hf_architecture(const std::string& value) {
    const auto v = lower_ascii(value);
    if (v.empty() || v == "auto") return HFArchitecture::Auto;
    if (v.find("gemma") != std::string::npos) return HFArchitecture::Gemma;
    if (v.find("mistral") != std::string::npos) return HFArchitecture::Mistral;
    if (v.find("qwen2") != std::string::npos || v.find("qwen") != std::string::npos) return HFArchitecture::Qwen2;
    if (v.find("llama") != std::string::npos || v.find("llm") != std::string::npos) return HFArchitecture::Llama;
    if (v == "generic" || v == "generic_decoder" || v == "decoder") return HFArchitecture::GenericDecoder;
    return HFArchitecture::GenericDecoder;
}

HFTransformerConfig load_hf_transformer_config_json(const std::string& path, HFArchitecture architecture) {
    const auto text = read_text_file(path);
    HFTransformerConfig out;
    out.architecture = architecture == HFArchitecture::Auto ? parse_hf_architecture(first_architecture_name(text)) : architecture;
    if (out.architecture == HFArchitecture::Auto) out.architecture = HFArchitecture::GenericDecoder;
    out.architecture_name = hf_architecture_name(out.architecture);

    const auto gemma_like = load_gemma_config_json(path);
    out.transformer = to_transformer_config(gemma_like);
    out.rms_norm_eps = gemma_like.rms_norm_eps;
    out.tie_word_embeddings = gemma_like.tie_word_embeddings;
    out.attention_bias = gemma_like.attention_bias;
    out.bos_token_id = gemma_like.bos_token_id;
    out.eos_token_id = gemma_like.eos_token_id;
    out.pad_token_id = gemma_like.pad_token_id;
    out.sliding_window = gemma_like.sliding_window;
    return out;
}

GemmaConfig to_gemma_compatible_config(const HFTransformerConfig& cfg) {
    GemmaConfig out;
    out.vocab_size = cfg.transformer.vocab_size;
    out.max_position_embeddings = cfg.transformer.block_size;
    out.hidden_size = cfg.transformer.n_embd;
    out.intermediate_size = cfg.transformer.mlp_hidden;
    out.num_hidden_layers = cfg.transformer.n_layer;
    out.num_attention_heads = cfg.transformer.n_head;
    out.num_key_value_heads = cfg.transformer.n_kv_head > 0 ? cfg.transformer.n_kv_head : cfg.transformer.n_head;
    out.head_dim = cfg.transformer.n_head > 0 ? cfg.transformer.n_embd / cfg.transformer.n_head : 0;
    out.rms_norm_eps = cfg.rms_norm_eps;
    out.rope_theta = cfg.transformer.rope_theta;
    out.attention_dropout = cfg.transformer.dropout;
    out.attention_bias = cfg.attention_bias || cfg.transformer.use_qkv_bias;
    out.tie_word_embeddings = cfg.tie_word_embeddings;
    out.bos_token_id = cfg.bos_token_id;
    out.eos_token_id = cfg.eos_token_id;
    out.pad_token_id = cfg.pad_token_id;
    out.sliding_window = cfg.sliding_window;
    return out;
}

ModernGPTModel make_hf_transformer_model(Backend& backend, const HFTransformerConfig& cfg) {
    ModernGPTModel model(backend, cfg.transformer);
    model.final_norm.eps = cfg.rms_norm_eps;
    for (auto& block : model.blocks) {
        block->norm1().eps = cfg.rms_norm_eps;
        block->norm2().eps = cfg.rms_norm_eps;
    }
    return model;
}

HFWeightName map_hf_transformer_weight_name(HFArchitecture architecture, const std::string& hf_name) {
    MCL_CHECK(architecture_uses_llama_like_weights(architecture),
              "HF architecture weight mapping is not implemented: " + hf_architecture_name(architecture));
    return map_gemma_hf_weight_name(hf_name);
}

std::vector<std::string> expected_hf_transformer_weight_names(const HFTransformerConfig& cfg, bool include_lm_head) {
    MCL_CHECK(architecture_uses_llama_like_weights(cfg.architecture),
              "HF architecture expected weights are not implemented: " + cfg.architecture_name);
    return expected_gemma_hf_weight_names(to_gemma_compatible_config(cfg), include_lm_head);
}

HFWeightLoadReport load_hf_transformer_weights(Backend& backend,
                                               ModernGPTModel& model,
                                               const std::vector<std::string>& safetensors_paths,
                                               const HFTransformerConfig& cfg,
                                               bool strict,
                                               bool trainable) {
    MCL_CHECK(architecture_uses_llama_like_weights(cfg.architecture),
              "HF architecture weight loading is not implemented: " + cfg.architecture_name);
    return load_gemma_hf_weights(backend, model, safetensors_paths, to_gemma_compatible_config(cfg), strict, trainable);
}

void enable_hf_transformer_quantized_inference(ModernGPTModel& model, DType qdtype) {
    model.enable_quantized_inference(qdtype);
}

void disable_hf_transformer_quantized_inference(ModernGPTModel& model) {
    model.disable_quantized_inference();
}

HFTokenizer load_hf_tokenizer(const std::string& model_dir_or_vocab_path, const HFTransformerConfig& cfg) {
    const auto tokenizer_path = resolve_tokenizer_path(model_dir_or_vocab_path);
    if (!tokenizer_path.empty() && std::filesystem::exists(tokenizer_path)) {
        return HFTokenizer::load_vocab(tokenizer_path.string(), cfg.bos_token_id, cfg.eos_token_id);
    }
    return HFTokenizer::byte_fallback(cfg.transformer.vocab_size, cfg.bos_token_id, cfg.eos_token_id);
}

std::string generate_hf_text(Backend& backend,
                             ModernGPTModel& model,
                             const HFTokenizer& tokenizer,
                             const std::string& prompt,
                             const GenerateOptions& options) {
    return generate_text(backend, model, tokenizer, prompt, options);
}

std::vector<std::string> generate_hf_batch_text(Backend& backend,
                                                ModernGPTModel& model,
                                                const HFTokenizer& tokenizer,
                                                const std::vector<std::string>& prompts,
                                                const GenerateOptions& options) {
    return generate_batch_text(backend, model, tokenizer, prompts, options);
}

} // namespace motifcl::nn
