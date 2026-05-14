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

int count_kind(const motifcl::nn::ModernModelSpec& spec, motifcl::nn::ModernLayerKind kind) {
    int count = 0;
    for (const auto& layer : spec.layers) {
        if (layer.kind == kind) ++count;
    }
    return count;
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

    require(motifcl::nn::parse_hf_architecture("qwen3.5") == motifcl::nn::HFArchitecture::Qwen35,
            "HF Qwen3.5 architecture detect failed");
    require(motifcl::nn::hf_architecture_supported_now(motifcl::nn::HFArchitecture::Qwen35),
            "HF Qwen3.5 safetensors runner should be supported");
    require(motifcl::nn::hf_architecture_blockers(motifcl::nn::HFArchitecture::Qwen35).empty(),
            "HF Qwen3.5 architecture blockers should be cleared for HybridGPTModel");
    auto qwen35_delta_map = motifcl::nn::map_hf_transformer_weight_name(
        motifcl::nn::HFArchitecture::Qwen35, "model.layers.0.linear_attn.q_proj.weight");
    require(qwen35_delta_map.kind == "delta_q_proj" && qwen35_delta_map.layer == 0,
            "Qwen3.5 DeltaNet weight mapper failed");
    auto qwen35_moe_map = motifcl::nn::map_hf_transformer_weight_name(
        motifcl::nn::HFArchitecture::Qwen35, "model.layers.0.mlp.experts.2.down_proj.weight");
    require(qwen35_moe_map.kind == "moe_expert_down_proj" && qwen35_moe_map.layer == 0,
            "Qwen3.5 MoE weight mapper failed");

    const auto gemma4_config_path = dir / "gemma4_config.json";
    {
        std::ofstream out(gemma4_config_path);
        out << R"json({
  "model_type": "gemma4",
  "vocab_size": 128,
  "max_position_embeddings": 16,
  "hidden_size": 16,
  "intermediate_size": 32,
  "num_hidden_layers": 4,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "hidden_activation": "gelu_pytorch_tanh",
  "rms_norm_eps": 0.000001,
  "rope_theta": 10000.0,
  "sliding_window": 3,
  "local_attention_layers": [0, 1, 2],
  "global_attention_layers": [3],
  "tie_word_embeddings": true,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
})json";
    }
    auto gemma4_cfg = motifcl::nn::load_hf_transformer_config_json(gemma4_config_path.string());
    require(gemma4_cfg.architecture == motifcl::nn::HFArchitecture::Gemma4,
            "Gemma4 architecture auto-detect failed");
    require(gemma4_cfg.transformer.sliding_window == 3 && !gemma4_cfg.per_layer_inputs,
            "Gemma4 text-core config features missing");
    require(!gemma4_cfg.transformer.use_swiglu && gemma4_cfg.transformer.split_mlp_projections,
            "Gemma4 must route through GeGLU split MLP, not SwiGLU packed MLP");
    require(gemma4_cfg.transformer.rope_split_half,
            "Gemma4 must use HF split-half RoPE layout");
    auto gemma4_spec = motifcl::nn::modern_model_spec_from_config(gemma4_cfg);
    require(motifcl::nn::modern_model_spec_runnable_by_modern_gpt(gemma4_spec),
            "Gemma4 dense text-core spec should be ModernGPTModel-runnable");
    require(count_kind(gemma4_spec, motifcl::nn::ModernLayerKind::SlidingAttention) == 3,
            "Gemma4 local attention layer spec failed");
    require(count_kind(gemma4_spec, motifcl::nn::ModernLayerKind::FullAttention) == 1,
            "Gemma4 global attention layer spec failed");
    require(count_kind(gemma4_spec, motifcl::nn::ModernLayerKind::SwiGLUFFN) == 4,
            "Gemma4 SwiGLU FFN layer spec failed");
    auto runtime_spec = motifcl::nn::long_context_runtime_spec_from_model_spec(gemma4_spec, 4);
    require(runtime_spec.needs_paged_kv && runtime_spec.needs_sliding_window_cache,
            "Gemma4 long-context runtime spec failed");
    require(runtime_spec.kv_cache_layers.size() == 4 && runtime_spec.sliding_window_layers.size() == 3,
            "Gemma4 KV/sliding layer lists failed");
    auto gemma4_mapped = motifcl::nn::map_hf_transformer_weight_name(
        gemma4_cfg.architecture, "model.layers.0.self_attn.q_proj.weight");
    require(gemma4_mapped.kind == "q_proj" && gemma4_mapped.layer == 0,
            "Gemma4 text-core weight mapper failed");

    const auto qwen35_config_path = dir / "qwen35_config.json";
    {
        std::ofstream out(qwen35_config_path);
        out << R"json({
  "model_type": "qwen3_5",
  "vision_config": {"hidden_size": 8},
  "text_config": {
    "model_type": "qwen3_5_text",
    "vocab_size": 128,
    "max_position_embeddings": 16,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "layer_types": ["gated_delta_net", "gated_delta_net", "gated_attention"],
    "num_experts": 4,
    "num_experts_per_tok": 2
  }
})json";
    }
    auto qwen35_cfg = motifcl::nn::load_hf_transformer_config_json(qwen35_config_path.string());
    auto qwen35_spec = motifcl::nn::modern_model_spec_from_config(qwen35_cfg);
    require(qwen35_cfg.architecture == motifcl::nn::HFArchitecture::Qwen35 &&
                qwen35_cfg.has_gated_delta_net && qwen35_cfg.has_moe,
            "Qwen3.5 hybrid config parse failed");
    require(!qwen35_cfg.has_vision_projector && qwen35_spec.text_core,
            "Qwen3.5 text_config wrapper should route text core without vision projector");
    require(!motifcl::nn::modern_model_spec_runnable_by_modern_gpt(qwen35_spec),
            "Qwen3.5 hybrid spec should not claim ModernGPTModel execution");
    require(motifcl::nn::modern_model_spec_runnable_by_hybrid(qwen35_spec),
            "Qwen3.5 hybrid spec should be executable by HybridGPTModel");
    require(count_kind(qwen35_spec, motifcl::nn::ModernLayerKind::GatedDeltaNet) == 2 &&
                count_kind(qwen35_spec, motifcl::nn::ModernLayerKind::GatedAttention) == 1 &&
                count_kind(qwen35_spec, motifcl::nn::ModernLayerKind::MoEFFN) == 3,
            "Qwen3.5 graph layer kinds failed");
    auto qwen35_runtime = motifcl::nn::long_context_runtime_spec_from_model_spec(qwen35_spec, 4);
    require(qwen35_runtime.needs_state_cache && qwen35_runtime.state_cache_layers.size() == 2,
            "Qwen3.5 state-cache runtime spec failed");

    const auto mixtral_config_path = dir / "mixtral_config.json";
    {
        std::ofstream out(mixtral_config_path);
        out << R"json({
  "model_type": "mixtral",
  "vocab_size": 128,
  "max_position_embeddings": 16,
  "hidden_size": 16,
  "intermediate_size": 32,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "num_local_experts": 4,
  "num_experts_per_tok": 2
})json";
    }
    auto mixtral_cfg = motifcl::nn::load_hf_transformer_config_json(mixtral_config_path.string());
    auto mixtral_spec = motifcl::nn::modern_model_spec_from_config(mixtral_cfg);
    require(mixtral_cfg.architecture == motifcl::nn::HFArchitecture::Mixtral &&
                mixtral_cfg.has_moe && mixtral_cfg.num_experts == 4,
            "Mixtral MoE config parse failed");
    require(!motifcl::nn::modern_model_spec_runnable_by_modern_gpt(mixtral_spec) &&
                motifcl::nn::modern_model_spec_runnable_by_hybrid(mixtral_spec),
            "Mixtral MoE graph should route through HybridGPTModel");
    require(count_kind(mixtral_spec, motifcl::nn::ModernLayerKind::FullAttention) == 2 &&
                count_kind(mixtral_spec, motifcl::nn::ModernLayerKind::MoEFFN) == 2,
            "Mixtral graph layer kinds failed");

    std::vector<motifcl::nn::HFChatMessage> messages{{"system", "Be concise"}, {"user", "Hello"}};
    auto chatml = motifcl::nn::apply_hf_chat_template(messages, motifcl::nn::HFArchitecture::Qwen2,
                                                      motifcl::nn::HFChatTemplateKind::Auto, true);
    require(chatml.find("<|im_start|>system") != std::string::npos &&
                chatml.find("<|im_start|>assistant") != std::string::npos,
            "HF ChatML template failed");
    auto llama_chat = motifcl::nn::apply_hf_chat_template(messages, motifcl::nn::HFArchitecture::Llama,
                                                         motifcl::nn::HFChatTemplateKind::Auto, true);
    require(llama_chat.find("<|start_header_id|>user") != std::string::npos,
            "HF Llama3 template failed");
    auto gemma4_chat = motifcl::nn::apply_hf_chat_template(
        {{"user", " привет "}},
        motifcl::nn::HFArchitecture::Gemma4,
        motifcl::nn::HFChatTemplateKind::Gemma,
        true);
    require(gemma4_chat == "<bos><|turn>user\nпривет<turn|>\n<|turn>model\n",
            "Gemma4 chat template control tokens failed");
    {
        std::ofstream out(dir / "tokenizer_config.json");
        out << R"json({"chat_template":"{% for message in messages %}<start_of_turn>{{ message['role'] }}\n{{ message['content'] }}<end_of_turn>{% endfor %}"})json";
    }
    require(motifcl::nn::infer_hf_chat_template_kind(motifcl::nn::HFArchitecture::Qwen2, dir.string()) ==
                motifcl::nn::HFChatTemplateKind::Gemma,
            "HF tokenizer_config chat template inference failed");
    auto probe = motifcl::nn::probe_hf_transformer_model(dir.string());
    require(probe.has_config && probe.config.architecture == motifcl::nn::HFArchitecture::Qwen2,
            "HF model probe config failed");
    require(probe.can_instantiate && !probe.can_load_weights && !probe.warnings.empty(),
            "HF model probe readiness failed");

    auto backend = motifcl::Backend::create_opencl();
    auto qwen35_hybrid = motifcl::nn::make_hf_hybrid_transformer_model(backend, qwen35_cfg);
    std::vector<int32_t> qwen35_token_values{65, 66, 67};
    auto qwen35_tokens = motifcl::Tensor::from_cpu(backend, {1, 3}, motifcl::DType::I32, qwen35_token_values.data());
    auto qwen35_logits = qwen35_hybrid.forward(qwen35_tokens);
    require(qwen35_logits.shape() == motifcl::Shape({1, 3, 128}),
            "Qwen3.5 HybridGPTModel forward shape failed");
    auto qwen35_cache = qwen35_hybrid.create_runtime_cache(backend, 1, true, 4);
    auto qwen35_cached_logits = qwen35_hybrid.forward_with_cache(qwen35_tokens, qwen35_cache);
    require(qwen35_cached_logits.shape() == motifcl::Shape({1, 3, 128}),
            "Qwen3.5 HybridGPTModel cached forward shape failed");
    auto qwen35_tokenizer = motifcl::nn::load_hf_tokenizer(dir.string(), qwen35_cfg);
    motifcl::nn::GenerateOptions qwen35_opts;
    qwen35_opts.max_new_tokens = 0;
    require(motifcl::nn::generate_hf_hybrid_text(backend, qwen35_hybrid, qwen35_tokenizer, "AB", qwen35_opts) == "AB",
            "Qwen3.5 HybridGPTModel prompt-only generation failed");

    auto model = motifcl::nn::make_hf_transformer_model(backend, cfg);
    std::vector<int32_t> token_values{65, 66, 67, 68};
    auto tokens = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::I32, token_values.data());
    auto logits = model.forward(tokens);
    require(logits.shape() == motifcl::Shape({1, 4, 256}), "HF ModernGPTModel forward shape failed");

    auto gemma4_model = motifcl::nn::make_hf_transformer_model(backend, gemma4_cfg);
    require(gemma4_model.blocks[0]->attention_window() == 3 &&
                gemma4_model.blocks[3]->attention_window() == 0,
            "Gemma4 per-layer local/global attention windows failed");
    require(gemma4_model.blocks[0]->attention().rope_split_half_enabled() &&
                gemma4_model.blocks[3]->attention().rope_split_half_enabled(),
            "Gemma4 runtime split-half RoPE wiring failed");
    require(!gemma4_model.blocks[0]->mlp().use_swiglu &&
                gemma4_model.blocks[0]->mlp().split_projections_enabled(),
            "Gemma4 runtime MLP activation wiring failed");
    std::vector<int32_t> gemma4_token_values{65, 66, 67, 68};
    auto gemma4_tokens = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::I32, gemma4_token_values.data());
    auto gemma4_logits = gemma4_model.forward(gemma4_tokens);
    require(gemma4_logits.shape() == motifcl::Shape({1, 4, 128}),
            "Gemma4 dense text-only forward smoke failed");

    auto tokenizer = motifcl::nn::load_hf_tokenizer(dir.string(), cfg);
    auto encoded = tokenizer.encode("AB", false, false);
    require(encoded == std::vector<int32_t>({65, 66}), "HF tokenizer fallback failed");
    auto sp_tokenizer = motifcl::nn::GemmaTokenizer::from_tokens(
        {"<unk>", "\xE2\x96\x81Hello", "\xE2\x96\x81world", "!", "?"},
        0,
        0,
        "SentencePiece");
    auto sp_encoded = sp_tokenizer.encode("  Hello\tworld!", false, false);
    require(sp_encoded == std::vector<int32_t>({1, 2, 3}),
            "SentencePiece whitespace normalizer parity failed");
    auto sp_fullwidth = sp_tokenizer.encode("\xEF\xBC\x9F", false, false);
    require(sp_fullwidth == std::vector<int32_t>({4}),
            "SentencePiece NFKC-lite fullwidth normalizer failed");
    std::vector<std::string> gemma4_vocab(10629);
    for (std::size_t i = 0; i < gemma4_vocab.size(); ++i) {
        gemma4_vocab[i] = "<unused_test_" + std::to_string(i) + ">";
    }
    const std::string p = "\xD0\xBF";
    const std::string r = "\xD1\x80";
    const std::string i_cyr = "\xD0\xB8";
    const std::string v = "\xD0\xB2";
    const std::string e = "\xD0\xB5";
    const std::string t = "\xD1\x82";
    gemma4_vocab[2] = "<bos>";
    gemma4_vocab[105] = "<|turn>";
    gemma4_vocab[106] = "<turn|>";
    gemma4_vocab[107] = "\n";
    gemma4_vocab[2364] = "user";
    gemma4_vocab[4368] = "model";
    gemma4_vocab[8362] = v + e + t;
    gemma4_vocab[10628] = p + r + i_cyr;
    auto gemma4_tokenizer = motifcl::nn::GemmaTokenizer::from_tokens(
        gemma4_vocab,
        2,
        106,
        "gemma4",
        {"u s", "us e", "use r", "m o", "mo d", "mod e", "mode l",
         p + " " + r, p + r + " " + i_cyr, v + " " + e, v + e + " " + t},
        false);
    auto gemma4_ids = gemma4_tokenizer.encode("<bos><|turn>user\nпривет<turn|>\n<|turn>model\n", false, false);
    require(gemma4_ids == std::vector<int32_t>({2, 105, 2364, 107, 10628, 8362, 106, 107, 105, 4368, 107}),
            "Gemma4 BPE/control-token encoding failed");
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
