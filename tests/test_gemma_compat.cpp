#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>

#include "test_utils.hpp"

namespace {

bool near(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps;
}

void require(bool cond, const std::string& message) {
    if (!cond) throw std::runtime_error(message);
}

void write_le_u64(std::ostream& out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        const char byte = static_cast<char>((value >> (8 * i)) & 0xffu);
        out.write(&byte, 1);
    }
}

void append_varint(std::string& out, std::uint64_t value) {
    while (value >= 0x80u) {
        out.push_back(static_cast<char>((value & 0x7fu) | 0x80u));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value));
}

void append_length_field(std::string& out, int field, const std::string& payload) {
    append_varint(out, static_cast<std::uint64_t>((field << 3) | 2));
    append_varint(out, static_cast<std::uint64_t>(payload.size()));
    out += payload;
}

std::string sentencepiece_piece_message(const std::string& piece) {
    std::string msg;
    append_length_field(msg, 1, piece);
    append_varint(msg, static_cast<std::uint64_t>((3 << 3) | 0));
    append_varint(msg, 1); // NORMAL
    return msg;
}

void write_sentencepiece_model(const std::filesystem::path& path, const std::vector<std::string>& pieces) {
    std::string data;
    for (const auto& piece : pieces) append_length_field(data, 1, sentencepiece_piece_message(piece));
    std::string trainer;
    append_varint(trainer, static_cast<std::uint64_t>((3 << 3) | 0));
    append_varint(trainer, 1); // UNIGRAM
    append_length_field(data, 2, trainer);
    std::ofstream out(path, std::ios::binary);
    require(out.good(), "failed to open fake sentencepiece model");
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    require(out.good(), "failed to write fake sentencepiece model");
}

struct FakeTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> values;
};

std::uint64_t numel(const std::vector<int64_t>& shape) {
    std::uint64_t n = 1;
    for (auto dim : shape) n *= static_cast<std::uint64_t>(dim);
    return n;
}

void write_safetensors(const std::filesystem::path& path, const std::vector<FakeTensor>& tensors) {
    std::string header = "{";
    std::uint64_t offset = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& t = tensors[i];
        require(numel(t.shape) == t.values.size(), "fake tensor shape mismatch");
        if (i > 0) header += ",";
        const auto begin = offset;
        const auto end = begin + static_cast<std::uint64_t>(t.values.size() * sizeof(float));
        offset = end;
        header += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (std::size_t d = 0; d < t.shape.size(); ++d) {
            if (d > 0) header += ",";
            header += std::to_string(t.shape[d]);
        }
        header += "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
    }
    header += "}";

    std::ofstream out(path, std::ios::binary);
    require(out.good(), "failed to open fake safetensors");
    write_le_u64(out, static_cast<std::uint64_t>(header.size()));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto& t : tensors) {
        out.write(reinterpret_cast<const char*>(t.values.data()), static_cast<std::streamsize>(t.values.size() * sizeof(float)));
    }
    require(out.good(), "failed to write fake safetensors");
}

std::vector<float> seq(std::size_t n, float scale = 0.01f) {
    std::vector<float> values(n);
    for (std::size_t i = 0; i < n; ++i) values[i] = static_cast<float>(i + 1) * scale;
    return values;
}

} // namespace

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();

        const auto config_path = std::filesystem::current_path() / "gemma_config_test.json";
        {
            std::ofstream cfg(config_path);
            cfg << R"({
                "vocab_size": 32,
                "max_position_embeddings": 8,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "rms_norm_eps": 0.000001,
                "rope_theta": 10000.0,
                "tie_word_embeddings": false
            })";
        }
        auto cfg = motifcl::nn::load_gemma_config_json(config_path.string());
        std::filesystem::remove(config_path);
        require(cfg.hidden_size == 16 && cfg.num_key_value_heads == 2, "GemmaConfig parse failed");
        auto tcfg = motifcl::nn::to_transformer_config(cfg);
        require(tcfg.n_embd == 16 && tcfg.use_rope && tcfg.use_swiglu, "GemmaConfig conversion failed");

        const auto single_path = std::filesystem::current_path() / "single_test.safetensors";
        write_safetensors(single_path, {{"x", {2, 2}, {1, 2, 3, 4}}});
        auto sf = motifcl::SafeTensorsFile::open(single_path.string());
        require(sf.contains("x") && sf.tensor_info("x").dtype == "F32", "SafeTensors metadata failed");
        auto x_loaded = sf.load_tensor(backend, "x").to_vector<float>();
        require(x_loaded == std::vector<float>({1, 2, 3, 4}), "SafeTensors tensor load failed");
        std::filesystem::remove(single_path);

        auto tok = motifcl::nn::GemmaTokenizer::byte_fallback(512, 1, 2);
        auto ids = tok.encode("abc", true, true);
        require(tok.decode(ids, true) == "abc", "GemmaTokenizer byte fallback failed");

        const auto bpe_tok_path = std::filesystem::current_path() / "tokenizer_bpe_test.json";
        {
            std::ofstream out(bpe_tok_path);
            out << R"json({
              "model": {
                "type": "BPE",
                "vocab": {"A": 10, "B": 11, "C": 12, "AB": 13, "ABC": 14},
                "merges": ["A B"]
              }
            })json";
        }
        auto bpe_tok = motifcl::nn::GemmaTokenizer::load_vocab(bpe_tok_path.string(), 1, 2);
        require(bpe_tok.tokenizer_model_type() == "BPE", "BPE tokenizer type parse failed");
        require(bpe_tok.encode("ABC", false, false) == std::vector<int32_t>({13, 12}),
                "BPE tokenizer merges/ranks failed");
        std::filesystem::remove(bpe_tok_path);

        const auto sp_tok_path = std::filesystem::current_path() / "tokenizer_sp_test.json";
        {
            std::ofstream out(sp_tok_path);
            out << R"json({
              "model": {
                "type": "Unigram",
                "vocab": [["<unk>", 0.0], ["▁hello", -1.0], ["▁world", -1.0]]
              },
              "added_tokens": [{"id": 98, "content": "<s>"}, {"id": 99, "content": "</s>"}]
            })json";
        }
        auto sp_tok = motifcl::nn::GemmaTokenizer::load_vocab(sp_tok_path.string(), 1, 2);
        auto sp_ids = sp_tok.encode("hello world", false, false);
        require(sp_ids == std::vector<int32_t>({1, 2}), "SentencePiece-style tokenizer encode failed");
        require(sp_tok.decode(sp_ids, false) == "hello world", "SentencePiece-style tokenizer decode failed");
        std::filesystem::remove(sp_tok_path);

        const auto sp_model_path = std::filesystem::current_path() / "tokenizer_sp_test.model";
        write_sentencepiece_model(sp_model_path, {"<unk>", "▁hello", "▁world"});
        auto sp_model_tok = motifcl::nn::GemmaTokenizer::load_vocab(sp_model_path.string(), 1, 2);
        require(sp_model_tok.tokenizer_model_type() == "Unigram", "binary SentencePiece model type parse failed");
        auto sp_model_ids = sp_model_tok.encode("hello world", false, false);
        require(sp_model_ids == std::vector<int32_t>({1, 2}), "binary SentencePiece tokenizer encode failed");
        require(sp_model_tok.decode(sp_model_ids, false) == "hello world", "binary SentencePiece tokenizer decode failed");
        std::filesystem::remove(sp_model_path);

        auto mapped = motifcl::nn::map_gemma_hf_weight_name("model.layers.3.self_attn.q_proj.weight");
        require(mapped.layer == 3 && mapped.kind == "q_proj", "Gemma HF mapper failed");

        auto q_weight = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32,
            std::vector<float>{1, 2, 3, 4, 5, 6}.data());
        auto q_bias = motifcl::Tensor::from_cpu(backend, {2}, motifcl::DType::F32,
            std::vector<float>{0.5f, -0.5f}.data());
        motifcl::nn::QuantizedLinear qlinear(q_weight, q_bias, motifcl::DType::Q4_0);
        auto q_input = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32,
            std::vector<float>{1, 0, 0, 0, 1, 1}.data());
        auto q_out = qlinear.forward(q_input).to_vector<float>();
        require(q_out.size() == 4 && near(q_out[0], 1.5f, 0.8f) && near(q_out[3], 9.5f, 0.8f),
                "QuantizedLinear output mismatch");

        motifcl::nn::GemmaConfig tiny;
        tiny.vocab_size = 8;
        tiny.max_position_embeddings = 8;
        tiny.hidden_size = 4;
        tiny.intermediate_size = 6;
        tiny.num_hidden_layers = 1;
        tiny.num_attention_heads = 2;
        tiny.num_key_value_heads = 1;
        tiny.head_dim = 2;
        tiny.tie_word_embeddings = false;

        const auto weights_path = std::filesystem::current_path() / "gemma_weights_test.safetensors";
        write_safetensors(weights_path, {
            {"model.embed_tokens.weight", {8, 4}, seq(32)},
            {"model.norm.weight", {4}, {1, 1, 1, 1}},
            {"lm_head.weight", {8, 4}, seq(32, 0.02f)},
            {"model.layers.0.input_layernorm.weight", {4}, {1, 1, 1, 1}},
            {"model.layers.0.post_attention_layernorm.weight", {4}, {1, 1, 1, 1}},
            {"model.layers.0.self_attn.q_proj.weight", {4, 4}, seq(16, 0.03f)},
            {"model.layers.0.self_attn.k_proj.weight", {2, 4}, seq(8, 0.04f)},
            {"model.layers.0.self_attn.v_proj.weight", {2, 4}, seq(8, 0.05f)},
            {"model.layers.0.self_attn.o_proj.weight", {4, 4}, seq(16, 0.06f)},
            {"model.layers.0.mlp.gate_proj.weight", {6, 4}, seq(24, 0.07f)},
            {"model.layers.0.mlp.up_proj.weight", {6, 4}, seq(24, 0.08f)},
            {"model.layers.0.mlp.down_proj.weight", {4, 6}, seq(24, 0.09f)}
        });
        auto model = motifcl::nn::make_gemma_model(backend, tiny);
        auto report = motifcl::nn::load_gemma_hf_weights(backend, model, {weights_path.string()}, tiny, true, false);
        require(report.missing.empty() && report.loaded_tensors >= 12, "Gemma HF weight load report failed");
        auto embed_host = model.token_embedding.weight.data.to_vector<float>();
        require(near(embed_host[0], 0.01f) && near(embed_host[31], 0.32f), "Gemma embedding load failed");
        std::filesystem::remove(weights_path);

        auto gen_model = motifcl::nn::make_gemma_model(backend, cfg);
        motifcl::nn::GenerateOptions options;
        options.max_new_tokens = 2;
        options.eos_token_id = -1;
        options.gpu_greedy_sampling = true;
        auto generated = motifcl::nn::generate(backend, gen_model, {1, 2}, options);
        require(generated.size() == 4, "Gemma generate length mismatch");
        auto batch_generated = motifcl::nn::generate_batch(backend, gen_model, {{1, 2}, {2, 3}}, options);
        require(batch_generated.size() == 2 && batch_generated[0].size() == 4 && batch_generated[1].size() == 4,
                "Gemma batch generate length mismatch");
        auto variable_batch = motifcl::nn::generate_batch(backend, gen_model, {{1, 2}, {2, 3, 4}}, options);
        require(variable_batch.size() == 2 && variable_batch[0].size() == 4 && variable_batch[1].size() == 5,
                "Gemma variable-length fused batch generate length mismatch");
        options.temperature = 1.0f;
        options.top_k = 1;
        auto sampled_batch = motifcl::nn::generate_batch(backend, gen_model, {{1, 2}, {2, 3, 4}}, options);
        require(sampled_batch.size() == 2 && sampled_batch[0].size() == 4 && sampled_batch[1].size() == 5,
                "Gemma GPU top-k batch sampling length mismatch");
        options.top_k = 0;
        options.top_p = 0.01f;
        auto topp_batch = motifcl::nn::generate_batch(backend, gen_model, {{1, 2}, {2, 3, 4}}, options);
        require(topp_batch.size() == 2 && topp_batch[0].size() == 4 && topp_batch[1].size() == 5,
                "Gemma GPU top-p batch sampling length mismatch");

        gen_model.enable_quantized_inference(motifcl::DType::Q4_0);
        const auto q_ckpt_dir = std::filesystem::current_path() / "quant_transformer_ckpt";
        motifcl::save_quantized_transformer_checkpoint(gen_model, q_ckpt_dir.string(), motifcl::DType::Q4_0);
        auto loaded_quant_model = motifcl::nn::make_gemma_model(backend, cfg);
        motifcl::load_quantized_transformer_checkpoint(loaded_quant_model, backend, q_ckpt_dir.string());
        require(loaded_quant_model.quantized_inference_enabled() &&
                    loaded_quant_model.quantized_weight_dtype() == motifcl::DType::Q4_0,
                "Quantized transformer checkpoint load failed");
        options.temperature = 0.0f;
        options.top_k = 0;
        options.top_p = 1.0f;
        auto loaded_quant_generated = motifcl::nn::generate_batch(backend, loaded_quant_model, {{1, 2}, {2, 3}}, options);
        require(loaded_quant_generated.size() == 2 && loaded_quant_generated[0].size() == 4,
                "Quantized transformer checkpoint generate failed");
        std::filesystem::remove_all(q_ckpt_dir);

        motifcl::nn::QuantizationPolicy qpolicy;
        qpolicy.default_dtype = motifcl::DType::Q4_0;
        qpolicy.lm_head_dtype = motifcl::DType::Q8_0;
        qpolicy.q8_layers = {0};
        const auto mixed_q_ckpt_dir = std::filesystem::current_path() / "quant_transformer_policy_ckpt";
        motifcl::save_quantized_transformer_checkpoint(gen_model, mixed_q_ckpt_dir.string(), qpolicy);
        auto loaded_policy_model = motifcl::nn::make_gemma_model(backend, cfg);
        motifcl::load_quantized_transformer_checkpoint(loaded_policy_model, backend, mixed_q_ckpt_dir.string());
        require(loaded_policy_model.blocks[0]->attention().qkv_proj().quantized_weight_dtype() == motifcl::DType::Q8_0 &&
                    loaded_policy_model.blocks[0]->mlp().down_proj.quantized_weight_dtype() == motifcl::DType::Q8_0 &&
                    loaded_policy_model.quantized_lm_head().dtype() == motifcl::DType::Q8_0,
                "QuantizationPolicy mixed Q8 layer/lm_head checkpoint failed");
        std::filesystem::remove_all(mixed_q_ckpt_dir);
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
