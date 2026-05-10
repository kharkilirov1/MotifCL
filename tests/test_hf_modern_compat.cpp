#include <motifcl/motifcl.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool cond, const std::string& message) {
    if (!cond) throw std::runtime_error(message);
}

bool has_name(const std::vector<std::string>& names, const std::string& value) {
    for (const auto& name : names) {
        if (name == value) return true;
    }
    return false;
}

} // namespace

int main() {
    const auto dir = std::filesystem::current_path() / "hf_modern_compat_test";
    std::filesystem::create_directories(dir);
    const auto config_path = dir / "config.json";
    {
        std::ofstream out(config_path);
        out << R"json({
  "model_type": "qwen2",
  "vocab_size": 256,
  "max_position_embeddings": 8,
  "hidden_size": 16,
  "intermediate_size": 32,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "rms_norm_eps": 0.00001,
  "rope_theta": 1000000.0,
  "attention_bias": true,
  "tie_word_embeddings": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
})json";
    }

    auto cfg = motifcl::nn::load_hf_transformer_config_json(config_path.string());
    require(cfg.architecture == motifcl::nn::HFArchitecture::Qwen2, "HF architecture auto-detect failed");
    require(cfg.transformer.vocab_size == 256 && cfg.transformer.n_embd == 16, "HF config mapping failed");
    require(cfg.transformer.n_kv_head == 2 && cfg.transformer.use_qkv_bias, "HF GQA/bias mapping failed");
    require(motifcl::nn::hf_architecture_name(cfg.architecture) == "qwen2", "HF architecture name failed");

    auto names = motifcl::nn::expected_hf_transformer_weight_names(cfg);
    require(has_name(names, "model.layers.0.self_attn.q_proj.weight"), "HF expected attention weight missing");
    require(has_name(names, "model.layers.0.self_attn.q_proj.bias"), "HF expected attention bias missing");
    auto mapped = motifcl::nn::map_hf_transformer_weight_name(cfg.architecture, "model.layers.0.mlp.gate_proj.weight");
    require(mapped.kind == "gate_proj" && mapped.layer == 0, "HF weight mapper failed");

    auto backend = motifcl::Backend::create_opencl();
    auto model = motifcl::nn::make_hf_transformer_model(backend, cfg);
    std::vector<int32_t> token_values{65, 66, 67, 68};
    auto tokens = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::I32, token_values.data());
    auto logits = model.forward(tokens);
    require(logits.shape() == motifcl::Shape({1, 4, 256}), "HF ModernGPTModel forward shape failed");

    auto tokenizer = motifcl::nn::load_hf_tokenizer(dir.string(), cfg);
    auto encoded = tokenizer.encode("AB", false, false);
    require(encoded == std::vector<int32_t>({65, 66}), "HF tokenizer fallback failed");
    motifcl::nn::GenerateOptions opts;
    opts.max_new_tokens = 0;
    require(motifcl::nn::generate_hf_text(backend, model, tokenizer, "AB", opts) == "AB",
            "HF generate_hf_text prompt-only smoke failed");
    opts.prefill_prompt = false;
    require(motifcl::nn::generate_hf_text(backend, model, tokenizer, "AB", opts) == "AB",
            "HF generate_hf_text token-by-token smoke failed");

    motifcl::nn::enable_hf_transformer_quantized_inference(model, motifcl::DType::Q8_0);
    require(model.quantized_inference_enabled() && model.quantized_weight_dtype() == motifcl::DType::Q8_0,
            "HF Q8 quantized inference enable failed");
    opts.prefill_prompt = true;
    require(motifcl::nn::generate_hf_text(backend, model, tokenizer, "AB", opts) == "AB",
            "HF Q8 quantized generate smoke failed");
    motifcl::nn::enable_hf_transformer_quantized_inference(model, motifcl::DType::Q4_0);
    require(model.quantized_inference_enabled() && model.quantized_weight_dtype() == motifcl::DType::Q4_0,
            "HF Q4 quantized inference enable failed");
    require(motifcl::nn::generate_hf_text(backend, model, tokenizer, "AB", opts) == "AB",
            "HF Q4 quantized generate smoke failed");
    motifcl::nn::disable_hf_transformer_quantized_inference(model);
    require(!model.quantized_inference_enabled(), "HF quantized inference disable failed");

    std::filesystem::remove_all(dir);
    return 0;
}
