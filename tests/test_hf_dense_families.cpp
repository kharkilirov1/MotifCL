#include <motifcl/motifcl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_utils.hpp"

namespace {

void require(bool cond, const std::string& message) {
    if (!cond) throw std::runtime_error(message);
}

struct FakeTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> values;
};

struct FamilyCase {
    std::string name;
    std::string model_type;
    motifcl::nn::HFArchitecture architecture;
    bool fused_qkv = false;
    bool fused_mlp = false;
    bool qk_norm = false;
    bool tied = false;
    bool glm_names = false;
    int kv_heads = 2;
    std::string extra_json;
};

std::uint64_t numel(const std::vector<int64_t>& shape) {
    std::uint64_t value = 1;
    for (const auto dim : shape) value *= static_cast<std::uint64_t>(dim);
    return value;
}

void write_le_u64(std::ostream& out, std::uint64_t value) {
    char bytes[8];
    for (int i = 0; i < 8; ++i) bytes[i] = static_cast<char>((value >> (8 * i)) & 0xffu);
    out.write(bytes, sizeof(bytes));
}

void write_safetensors(const std::filesystem::path& path, const std::vector<FakeTensor>& tensors) {
    std::string header = "{";
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& tensor = tensors[i];
        require(numel(tensor.shape) == tensor.values.size(), "synthetic safetensors shape mismatch");
        if (i > 0) header += ",";
        const auto begin = offset;
        const auto end = begin + tensor.values.size() * sizeof(float);
        offset = end;
        header += "\"" + tensor.name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t d = 0; d < tensor.shape.size(); ++d) {
            if (d > 0) header += ",";
            header += std::to_string(tensor.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
    }
    header += "}";

    std::ofstream out(path, std::ios::binary);
    require(out.good(), "failed to create synthetic safetensors");
    write_le_u64(out, static_cast<std::uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& tensor : tensors) {
        out.write(reinterpret_cast<const char*>(tensor.values.data()),
                  static_cast<std::streamsize>(tensor.values.size() * sizeof(float)));
    }
    require(out.good(), "failed to write synthetic safetensors");
}

std::vector<float> values(std::size_t size, float phase = 0.0f) {
    std::vector<float> out(size);
    for (std::size_t i = 0; i < size; ++i) {
        out[i] = 0.025f * std::sin(static_cast<float>(i + 1) * 0.37f + phase);
    }
    return out;
}

std::vector<float> ones(std::size_t size) {
    return std::vector<float>(size, 1.0f);
}

std::vector<float> zeros(std::size_t size) {
    return std::vector<float>(size, 0.0f);
}

void add_tensor(std::vector<FakeTensor>& tensors,
                const std::string& name,
                std::vector<int64_t> shape,
                std::vector<float> data) {
    tensors.push_back({name, std::move(shape), std::move(data)});
}

std::string standard_config(const FamilyCase& test) {
    std::string json = "{\n";
    json += "  \"model_type\": \"" + test.model_type + "\",\n";
    json += "  \"vocab_size\": 32,\n";
    json += "  \"max_position_embeddings\": 8,\n";
    json += "  \"hidden_size\": 16,\n";
    json += "  \"intermediate_size\": 24,\n";
    json += "  \"num_hidden_layers\": 1,\n";
    json += "  \"num_attention_heads\": 4,\n";
    json += "  \"num_key_value_heads\": " + std::to_string(test.kv_heads) + ",\n";
    json += "  \"head_dim\": 4,\n";
    json += "  \"rms_norm_eps\": 0.00001,\n";
    json += "  \"hidden_act\": \"silu\",\n";
    json += "  \"tie_word_embeddings\": " + std::string(test.tied ? "true" : "false");
    if (!test.extra_json.empty()) json += ",\n" + test.extra_json;
    json += "\n}\n";
    return json;
}

std::string glm_config(const FamilyCase& test) {
    return "{\n"
           "  \"model_type\": \"" + test.model_type + "\",\n"
           "  \"padded_vocab_size\": 32,\n"
           "  \"seq_length\": 8,\n"
           "  \"hidden_size\": 16,\n"
           "  \"ffn_hidden_size\": 24,\n"
           "  \"num_layers\": 1,\n"
           "  \"num_attention_heads\": 4,\n"
           "  \"multi_query_group_num\": 1,\n"
           "  \"kv_channels\": 4,\n"
           "  \"rmsnorm\": true,\n"
           "  \"layernorm_epsilon\": 0.00001,\n"
           "  \"add_qkv_bias\": true,\n"
           "  \"add_bias_linear\": false,\n"
           "  \"hidden_act\": \"silu\"\n"
           "}\n";
}

std::vector<FakeTensor> family_weights(const FamilyCase& test,
                                       const motifcl::nn::HFTransformerConfig& cfg) {
    const int hidden = cfg.transformer.n_embd;
    const int vocab = cfg.transformer.vocab_size;
    const int head_dim = cfg.transformer.head_dim;
    const int q_dim = cfg.transformer.n_head * head_dim;
    const int kv_dim = cfg.transformer.n_kv_head * head_dim;
    const int v_dim = cfg.transformer.n_kv_head * cfg.transformer.v_head_dim;
    const int mlp = cfg.transformer.mlp_hidden;
    const bool glm = test.glm_names;
    const std::string p = glm ? "transformer.encoder.layers.0." : "model.layers.0.";
    std::vector<FakeTensor> tensors;

    add_tensor(tensors,
               glm ? "transformer.embedding.word_embeddings.weight" : "model.embed_tokens.weight",
               {vocab, hidden}, values(static_cast<std::size_t>(vocab * hidden), 0.1f));
    add_tensor(tensors,
               glm ? "transformer.encoder.final_layernorm.weight" : "model.norm.weight",
               {hidden}, ones(hidden));
    if (cfg.layer_norm) {
        add_tensor(tensors,
                   glm ? "transformer.encoder.final_layernorm.bias" : "model.norm.bias",
                   {hidden}, zeros(hidden));
    }
    if (!cfg.tie_word_embeddings) {
        add_tensor(tensors,
                   glm ? "transformer.output_layer.weight" : "lm_head.weight",
                   {vocab, hidden}, values(static_cast<std::size_t>(vocab * hidden), 0.2f));
    }

    add_tensor(tensors, p + "input_layernorm.weight", {hidden}, ones(hidden));
    add_tensor(tensors, p + "post_attention_layernorm.weight", {hidden}, ones(hidden));
    if (cfg.layer_norm) {
        add_tensor(tensors, p + "input_layernorm.bias", {hidden}, zeros(hidden));
        add_tensor(tensors, p + "post_attention_layernorm.bias", {hidden}, zeros(hidden));
    }

    if (test.fused_qkv) {
        add_tensor(tensors,
                   p + (glm ? "self_attention.query_key_value.weight" : "self_attn.qkv_proj.weight"),
                   {q_dim + kv_dim + v_dim, hidden},
                   values(static_cast<std::size_t>((q_dim + kv_dim + v_dim) * hidden), 0.3f));
        if (cfg.attention_bias) {
            add_tensor(tensors,
                       p + (glm ? "self_attention.query_key_value.bias" : "self_attn.qkv_proj.bias"),
                       {q_dim + kv_dim + v_dim}, zeros(q_dim + kv_dim + v_dim));
        }
    } else {
        add_tensor(tensors, p + "self_attn.q_proj.weight", {q_dim, hidden},
                   values(static_cast<std::size_t>(q_dim * hidden), 0.4f));
        add_tensor(tensors, p + "self_attn.k_proj.weight", {kv_dim, hidden},
                   values(static_cast<std::size_t>(kv_dim * hidden), 0.5f));
        add_tensor(tensors, p + "self_attn.v_proj.weight", {v_dim, hidden},
                   values(static_cast<std::size_t>(v_dim * hidden), 0.6f));
        if (cfg.attention_bias) {
            add_tensor(tensors, p + "self_attn.q_proj.bias", {q_dim}, zeros(q_dim));
            add_tensor(tensors, p + "self_attn.k_proj.bias", {kv_dim}, zeros(kv_dim));
            add_tensor(tensors, p + "self_attn.v_proj.bias", {v_dim}, zeros(v_dim));
        }
    }
    add_tensor(tensors,
               p + (glm ? "self_attention.dense.weight" : "self_attn.o_proj.weight"),
               {hidden, cfg.transformer.n_head * cfg.transformer.v_head_dim},
               values(static_cast<std::size_t>(hidden * cfg.transformer.n_head *
                                               cfg.transformer.v_head_dim),
                      0.7f));
    if (cfg.attention_output_bias) {
        add_tensor(tensors,
                   p + (glm ? "self_attention.dense.bias" : "self_attn.o_proj.bias"),
                   {hidden}, zeros(hidden));
    }
    if (test.qk_norm) {
        add_tensor(tensors, p + "self_attn.q_norm.weight", {head_dim}, ones(head_dim));
        add_tensor(tensors, p + "self_attn.k_norm.weight", {head_dim}, ones(head_dim));
    }

    if (test.fused_mlp) {
        add_tensor(tensors,
                   p + (glm ? "mlp.dense_h_to_4h.weight" : "mlp.gate_up_proj.weight"),
                   {2 * mlp, hidden}, values(static_cast<std::size_t>(2 * mlp * hidden), 0.8f));
        if (cfg.mlp_bias) {
            add_tensor(tensors,
                       p + (glm ? "mlp.dense_h_to_4h.bias" : "mlp.gate_up_proj.bias"),
                       {2 * mlp}, zeros(2 * mlp));
        }
    } else {
        add_tensor(tensors, p + "mlp.gate_proj.weight", {mlp, hidden},
                   values(static_cast<std::size_t>(mlp * hidden), 0.9f));
        add_tensor(tensors, p + "mlp.up_proj.weight", {mlp, hidden},
                   values(static_cast<std::size_t>(mlp * hidden), 1.0f));
        if (cfg.mlp_bias) {
            add_tensor(tensors, p + "mlp.gate_proj.bias", {mlp}, zeros(mlp));
            add_tensor(tensors, p + "mlp.up_proj.bias", {mlp}, zeros(mlp));
        }
    }
    add_tensor(tensors,
               p + (glm ? "mlp.dense_4h_to_h.weight" : "mlp.down_proj.weight"),
               {hidden, mlp}, values(static_cast<std::size_t>(hidden * mlp), 1.1f));
    if (cfg.mlp_bias) {
        add_tensor(tensors,
                   p + (glm ? "mlp.dense_4h_to_h.bias" : "mlp.down_proj.bias"),
                   {hidden}, zeros(hidden));
    }
    return tensors;
}

void check_finite_logits(const motifcl::Tensor& logits, const std::string& family) {
    require(logits.shape() == motifcl::Shape({1, 2, 32}), family + " forward shape mismatch");
    for (const auto value : logits.to_vector<float>()) {
        require(std::isfinite(value), family + " forward produced non-finite logits");
    }
}

void check_moe_rejection(const std::filesystem::path& dir,
                         const std::string& name,
                         const std::string& model_type,
                         const std::string& expert_key) {
    const auto path = dir / (name + "_moe_config.json");
    {
        std::ofstream out(path);
        out << "{\n"
               "  \"model_type\": \"" << model_type << "\",\n"
               "  \"vocab_size\": 32,\n"
               "  \"max_position_embeddings\": 8,\n"
               "  \"hidden_size\": 16,\n"
               "  \"intermediate_size\": 24,\n"
               "  \"num_hidden_layers\": 1,\n"
               "  \"num_attention_heads\": 4,\n"
               "  \"num_key_value_heads\": 2,\n"
               "  \"" << expert_key << "\": 4,\n"
               "  \"num_experts_per_tok\": 2\n"
               "}\n";
    }
    bool rejected = false;
    try {
        (void)motifcl::nn::load_hf_transformer_config_json(path.string());
    } catch (const std::exception& e) {
        const std::string message = e.what();
        rejected = message.find("unsupported MoE model family") != std::string::npos &&
                   message.find("router/expert") != std::string::npos;
    }
    require(rejected, name + " MoE config was not rejected clearly");
}

} // namespace

int main() {
    const auto dir = std::filesystem::current_path() / "hf_dense_families_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    const std::vector<FamilyCase> families{
        {"llama3", "llama", motifcl::nn::HFArchitecture::Llama,
         false, false, false, false, false, 2,
         "  \"rope_scaling\": {\"rope_type\":\"llama3\",\"factor\":8.0,"
         "\"original_max_position_embeddings\":8,\"low_freq_factor\":1.0,\"high_freq_factor\":4.0}"},
        {"llama4_dense", "llama4_text", motifcl::nn::HFArchitecture::Llama4,
         false, false, true, false, false, 2,
         "  \"use_qk_norm\": true"},
        {"mistral", "mistral", motifcl::nn::HFArchitecture::Mistral,
         false, false, false, false, false, 2,
         "  \"sliding_window\": 4,\n"
         "  \"rope_scaling\": {\"type\":\"yarn\",\"factor\":4.0,"
         "\"original_max_position_embeddings\":8,\"beta_fast\":32,\"beta_slow\":1}"},
        {"ministral_mqa", "ministral", motifcl::nn::HFArchitecture::Ministral,
         false, false, false, false, false, 1,
         "  \"rope_scaling\": {\"type\":\"linear\",\"factor\":2.0}"},
        {"phi4", "phi3", motifcl::nn::HFArchitecture::Phi4,
         true, true, false, false, false, 2,
         "  \"architectures\": [\"Phi3ForCausalLM\"],\n"
         "  \"partial_rotary_factor\": 0.5,\n"
         "  \"attention_bias\": true"},
        {"qwen3", "qwen3", motifcl::nn::HFArchitecture::Qwen3,
         false, false, true, false, false, 2,
         "  \"use_qk_norm\": true"},
        {"deepseek_llm", "deepseek-llm", motifcl::nn::HFArchitecture::DeepSeek,
         false, false, false, false, false, 2, ""},
        {"deepseek_v2_dense", "deepseek_v2", motifcl::nn::HFArchitecture::DeepSeek,
         false, false, false, false, false, 2,
         "  \"rope_scaling\": {\"type\":\"yarn\",\"factor\":2.0,"
         "\"original_max_position_embeddings\":8}"},
        {"glm4", "chatglm4", motifcl::nn::HFArchitecture::GLM4,
         true, true, false, false, true, 1, ""},
        {"smollm2_tied", "smollm2", motifcl::nn::HFArchitecture::SmolLM,
         false, false, false, true, false, 2, ""},
    };

    std::vector<motifcl::nn::HFTransformerConfig> configs;
    configs.reserve(families.size());
    for (const auto& family : families) {
        const auto case_dir = dir / family.name;
        std::filesystem::create_directories(case_dir);
        const auto config_path = case_dir / "config.json";
        {
            std::ofstream out(config_path);
            out << (family.glm_names ? glm_config(family) : standard_config(family));
        }
        auto cfg = motifcl::nn::load_hf_transformer_config_json(config_path.string());
        require(cfg.architecture == family.architecture, family.name + " adapter route mismatch");
        require(cfg.transformer.n_kv_head == family.kv_heads, family.name + " GQA/MQA mapping mismatch");
        require(cfg.fused_qkv_weights == family.fused_qkv, family.name + " QKV layout mapping mismatch");
        require(cfg.fused_mlp_weights == family.fused_mlp, family.name + " MLP layout mapping mismatch");
        require(cfg.transformer.use_qk_norm == family.qk_norm, family.name + " QK-norm mapping mismatch");
        require(cfg.tie_word_embeddings == family.tied, family.name + " tied embedding mapping mismatch");
        if (family.name == "llama3") {
            require(cfg.transformer.rope_scaling_type == motifcl::nn::RopeScalingType::Llama3,
                    "Llama3 rope_scaling mapping failed");
        }
        if (family.name == "mistral" || family.name == "deepseek_v2_dense") {
            require(cfg.transformer.rope_scaling_type == motifcl::nn::RopeScalingType::YaRN,
                    family.name + " YaRN mapping failed");
        }
        if (family.name == "phi4") {
            require(cfg.transformer.rotary_dim == 2, "Phi-4 partial rotary mapping failed");
        }
        if (family.name == "ministral_mqa" || family.name == "glm4") {
            require(cfg.transformer.n_kv_head == 1, family.name + " MQA mapping failed");
        }
        configs.push_back(cfg);

        const auto weights_path = case_dir / "model.safetensors";
        write_safetensors(weights_path, family_weights(family, cfg));
    }

    const auto generic_dir = dir / "generic";
    std::filesystem::create_directories(generic_dir);
    const auto generic_config_path = generic_dir / "config.json";
    {
        std::ofstream out(generic_config_path);
        out << "{\n"
               "  \"model_type\": \"acme_dense_decoder\",\n"
               "  \"vocab_size\": 32,\n"
               "  \"context_length\": 8,\n"
               "  \"d_model\": 16,\n"
               "  \"ffn_hidden_size\": 24,\n"
               "  \"n_layers\": 1,\n"
               "  \"n_heads\": 4,\n"
               "  \"kv_heads\": 1,\n"
               "  \"head_dim\": 4,\n"
               "  \"layer_norm_eps\": 0.00001,\n"
               "  \"norm_type\": \"layer_norm\",\n"
               "  \"norm_first\": false,\n"
               "  \"hidden_activation\": \"gelu_pytorch_tanh\",\n"
               "  \"fused_mlp\": true,\n"
               "  \"qkv_bias\": true,\n"
               "  \"attention_output_bias\": true,\n"
               "  \"mlp_bias\": true,\n"
               "  \"attention_softcap\": 0.000001,\n"
               "  \"final_logit_softcap\": 30.0,\n"
               "  \"rope_scaling\": {\"type\":\"dynamic\",\"factor\":2.0,"
               "\"original_max_position_embeddings\":4},\n"
               "  \"sliding_window_size\": 4,\n"
               "  \"tie_word_embeddings\": false\n"
               "}\n";
    }
    FamilyCase generic_case{"generic_layernorm", "acme_dense_decoder",
                            motifcl::nn::HFArchitecture::GenericDecoder,
                            false, true, false, false, false, 1, ""};
    auto generic_cfg = motifcl::nn::load_hf_transformer_config_json(generic_config_path.string());
    require(generic_cfg.architecture == motifcl::nn::HFArchitecture::GenericDecoder,
            "generic decoder fallback route failed");
    require(generic_cfg.layer_norm && generic_cfg.transformer.use_layer_norm,
            "generic LayerNorm config mapping failed");
    require(!generic_cfg.norm_first && !generic_cfg.transformer.norm_first,
            "generic post-norm placement mapping failed");
    require(generic_cfg.transformer.activation == motifcl::nn::TransformerActivation::GELUTanh,
            "generic GELU-tanh activation mapping failed");
    require(generic_cfg.fused_mlp_weights,
            "generic fused gate/up config mapping failed");
    require(generic_cfg.transformer.rope_scaling_type == motifcl::nn::RopeScalingType::Dynamic,
            "generic dynamic RoPE mapping failed");
    require(generic_cfg.transformer.n_kv_head == 1 && generic_cfg.transformer.sliding_window == 4,
            "generic MQA/sliding-window mapping failed");
    require(generic_cfg.transformer.attention_softcap == 0.000001f &&
                generic_cfg.transformer.final_logit_softcap == 30.0f,
            "generic attention/final softcap config mapping failed");
    auto generic_weights = family_weights(generic_case, generic_cfg);
    for (auto& tensor : generic_weights) {
        if (tensor.name.find("self_attn.q_proj.weight") != std::string::npos ||
            tensor.name.find("self_attn.k_proj.weight") != std::string::npos) {
            for (auto& value : tensor.values) value *= 20.0f;
        }
        if (tensor.name.find("mlp.gate_up_proj.weight") != std::string::npos) {
            for (auto& value : tensor.values) value *= 20.0f;
        }
    }
    write_safetensors(generic_dir / "model.safetensors", generic_weights);

    auto phi_map = motifcl::nn::map_hf_transformer_weight_name(
        motifcl::nn::HFArchitecture::Phi4, "model.layers.0.self_attn.qkv_proj.weight");
    require(phi_map.kind == "qkv_proj" && phi_map.layer == 0, "Phi-4 fused QKV mapper failed");
    auto glm_map = motifcl::nn::map_hf_transformer_weight_name(
        motifcl::nn::HFArchitecture::GLM4,
        "transformer.encoder.layers.0.self_attention.query_key_value.weight");
    require(glm_map.kind == "qkv_proj" && glm_map.layer == 0, "GLM-4 QKV mapper failed");

    check_moe_rejection(dir, "mixtral", "mixtral", "num_local_experts");
    check_moe_rejection(dir, "deepseek", "deepseek_v2", "n_routed_experts");
    check_moe_rejection(dir, "qwen3", "qwen3_moe", "num_experts");
    const auto mla_config_path = dir / "deepseek_mla_config.json";
    {
        std::ofstream out(mla_config_path);
        out << "{\n"
               "  \"model_type\": \"deepseek_v2\",\n"
               "  \"vocab_size\": 32,\n"
               "  \"max_position_embeddings\": 8,\n"
               "  \"hidden_size\": 16,\n"
               "  \"intermediate_size\": 24,\n"
               "  \"num_hidden_layers\": 1,\n"
               "  \"num_attention_heads\": 4,\n"
               "  \"num_key_value_heads\": 2,\n"
               "  \"kv_lora_rank\": 4,\n"
               "  \"qk_nope_head_dim\": 2,\n"
               "  \"qk_rope_head_dim\": 2\n"
               "}\n";
    }
    bool mla_rejected = false;
    try {
        (void)motifcl::nn::load_hf_transformer_config_json(mla_config_path.string());
    } catch (const std::exception& e) {
        mla_rejected =
            std::string(e.what()).find("unsupported DeepSeek MLA attention") != std::string::npos;
    }
    require(mla_rejected, "DeepSeek MLA config was not rejected before weight loading");

    motifcl::Backend backend;
    try {
        backend = motifcl::Backend::create_opencl();
    } catch (const std::exception& e) {
        if (!motifcl_test::is_opencl_unavailable(e)) throw;
        try {
            backend = motifcl::Backend::create_vulkan();
        } catch (const std::exception& vk_error) {
            std::cerr << "Skipping dense-family runtime checks: OpenCL=" << e.what()
                      << "; Vulkan=" << vk_error.what() << '\n';
            std::filesystem::remove_all(dir);
            return 77;
        }
    }

    std::vector<int32_t> token_values{1, 2};
    auto tokens = motifcl::Tensor::from_cpu(backend, {1, 2}, motifcl::DType::I32, token_values.data());
    for (std::size_t i = 0; i < families.size(); ++i) {
        auto model = motifcl::nn::make_hf_transformer_model(backend, configs[i]);
        const auto weights_path = dir / families[i].name / "model.safetensors";
        const auto report = motifcl::nn::load_hf_transformer_weights(
            backend, model, {weights_path.string()}, configs[i], true, false);
        require(report.missing.empty(), families[i].name + " strict weight load reported missing tensors");
        const auto logits = model.forward(tokens);
        check_finite_logits(logits, families[i].name);
        if (families[i].name == "llama3" || families[i].name == "mistral") {
            auto unscaled_cfg = configs[i];
            unscaled_cfg.transformer.rope_scaling_type = motifcl::nn::RopeScalingType::None;
            unscaled_cfg.transformer.attention_scale = 0.0f;
            auto unscaled_model = motifcl::nn::make_hf_transformer_model(backend, unscaled_cfg);
            (void)motifcl::nn::load_hf_transformer_weights(
                backend, unscaled_model, {weights_path.string()}, unscaled_cfg, true, false);
            const auto scaled_values = logits.to_vector<float>();
            const auto unscaled_values = unscaled_model.forward(tokens).to_vector<float>();
            float max_delta = 0.0f;
            for (std::size_t j = 0; j < scaled_values.size(); ++j) {
                max_delta = std::max(max_delta, std::abs(scaled_values[j] - unscaled_values[j]));
            }
            require(max_delta > 1e-8f, families[i].name + " RoPE scaling path did not affect logits");
        }
    }

    auto generic_model = motifcl::nn::make_hf_transformer_model(backend, generic_cfg);
    const auto generic_report = motifcl::nn::load_hf_transformer_weights(
        backend, generic_model, {(generic_dir / "model.safetensors").string()},
        generic_cfg, true, false);
    require(generic_report.missing.empty(), "generic strict weight load reported missing tensors");
    const auto generic_logits = generic_model.forward(tokens);
    check_finite_logits(generic_logits, "generic_layernorm");
    auto uncapped_cfg = generic_cfg;
    uncapped_cfg.transformer.attention_softcap = 0.0f;
    auto uncapped_model = motifcl::nn::make_hf_transformer_model(backend, uncapped_cfg);
    (void)motifcl::nn::load_hf_transformer_weights(
        backend, uncapped_model, {(generic_dir / "model.safetensors").string()},
        uncapped_cfg, true, false);
    const auto capped_values = generic_logits.to_vector<float>();
    const auto uncapped_values = uncapped_model.forward(tokens).to_vector<float>();
    float max_softcap_delta = 0.0f;
    for (std::size_t i = 0; i < capped_values.size(); ++i) {
        max_softcap_delta =
            std::max(max_softcap_delta, std::abs(capped_values[i] - uncapped_values[i]));
    }
    require(max_softcap_delta > 1e-8f,
            "generic attention softcap graph switch did not affect logits; max delta=" +
                std::to_string(max_softcap_delta));

    std::vector<int32_t> long_token_values{1, 2, 3, 4, 5, 6};
    auto long_tokens = motifcl::Tensor::from_cpu(
        backend, {1, 6}, motifcl::DType::I32, long_token_values.data());
    auto dynamic_cfg = generic_cfg;
    dynamic_cfg.transformer.attention_softcap = 0.0f;
    auto dynamic_model = motifcl::nn::make_hf_transformer_model(backend, dynamic_cfg);
    (void)motifcl::nn::load_hf_transformer_weights(
        backend, dynamic_model, {(generic_dir / "model.safetensors").string()},
        dynamic_cfg, true, false);
    auto no_dynamic_cfg = dynamic_cfg;
    no_dynamic_cfg.transformer.rope_scaling_type = motifcl::nn::RopeScalingType::None;
    auto no_dynamic_model = motifcl::nn::make_hf_transformer_model(backend, no_dynamic_cfg);
    (void)motifcl::nn::load_hf_transformer_weights(
        backend, no_dynamic_model, {(generic_dir / "model.safetensors").string()},
        no_dynamic_cfg, true, false);
    const auto dynamic_values = dynamic_model.forward(long_tokens).to_vector<float>();
    const auto no_dynamic_values = no_dynamic_model.forward(long_tokens).to_vector<float>();
    float max_dynamic_delta = 0.0f;
    for (std::size_t i = 0; i < dynamic_values.size(); ++i) {
        max_dynamic_delta =
            std::max(max_dynamic_delta, std::abs(dynamic_values[i] - no_dynamic_values[i]));
    }
    require(max_dynamic_delta > 1e-8f,
            "generic dynamic RoPE path did not affect long-context logits");

    auto exact_gelu_cfg = dynamic_cfg;
    exact_gelu_cfg.transformer.activation = motifcl::nn::TransformerActivation::GELU;
    auto exact_gelu_model = motifcl::nn::make_hf_transformer_model(backend, exact_gelu_cfg);
    (void)motifcl::nn::load_hf_transformer_weights(
        backend, exact_gelu_model, {(generic_dir / "model.safetensors").string()},
        exact_gelu_cfg, true, false);
    const auto exact_gelu_values = exact_gelu_model.forward(long_tokens).to_vector<float>();
    float max_activation_delta = 0.0f;
    for (std::size_t i = 0; i < dynamic_values.size(); ++i) {
        max_activation_delta = std::max(
            max_activation_delta, std::abs(dynamic_values[i] - exact_gelu_values[i]));
    }
    require(max_activation_delta > 1e-8f,
            "generic GELU vs GELU-tanh graph switch did not affect logits");

    std::filesystem::remove_all(dir);
    return 0;
}
