#include <motifcl/motifcl.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void require(bool cond, const std::string& message) {
    if (!cond) throw std::runtime_error(message);
}

template <typename T>
void write_le(std::ostream& out, T value) {
    if constexpr (std::is_same<T, float>::value) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        write_le<std::uint32_t>(out, bits);
    } else {
        using U = typename std::make_unsigned<T>::type;
        U u = static_cast<U>(value);
        for (std::size_t i = 0; i < sizeof(T); ++i) {
            const auto byte = static_cast<char>((u >> (8 * i)) & 0xffu);
            out.write(&byte, 1);
        }
    }
}

void write_string(std::ostream& out, const std::string& value) {
    write_le<std::uint64_t>(out, static_cast<std::uint64_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void write_metadata_string(std::ostream& out, const std::string& key, const std::string& value) {
    write_string(out, key);
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(motifcl::gguf::MetadataType::String));
    write_string(out, value);
}

void write_metadata_u32(std::ostream& out, const std::string& key, std::uint32_t value) {
    write_string(out, key);
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(motifcl::gguf::MetadataType::UInt32));
    write_le<std::uint32_t>(out, value);
}

void write_metadata_f32(std::ostream& out, const std::string& key, float value) {
    write_string(out, key);
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(motifcl::gguf::MetadataType::Float32));
    write_le<float>(out, value);
}

void write_metadata_string_array(std::ostream& out, const std::string& key, const std::vector<std::string>& values) {
    write_string(out, key);
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(motifcl::gguf::MetadataType::Array));
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(motifcl::gguf::MetadataType::String));
    write_le<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) write_string(out, value);
}

void write_metadata_u32_array(std::ostream& out, const std::string& key, const std::vector<std::uint32_t>& values) {
    write_string(out, key);
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(motifcl::gguf::MetadataType::Array));
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(motifcl::gguf::MetadataType::UInt32));
    write_le<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) write_le<std::uint32_t>(out, value);
}

void write_metadata_bool_array(std::ostream& out, const std::string& key, const std::vector<bool>& values) {
    write_string(out, key);
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(motifcl::gguf::MetadataType::Array));
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(motifcl::gguf::MetadataType::Bool));
    write_le<std::uint64_t>(out, static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) write_le<std::uint8_t>(out, value ? 1 : 0);
}

void pad_to_alignment(std::ostream& out, std::uint64_t alignment) {
    const auto pos = static_cast<std::uint64_t>(out.tellp());
    const auto rem = pos % alignment;
    if (rem == 0) return;
    std::vector<char> zeros(static_cast<std::size_t>(alignment - rem), 0);
    out.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
}

struct TinyTensor {
    TinyTensor(std::string tensor_name,
               std::vector<std::uint64_t> tensor_dims,
               motifcl::gguf::TensorType tensor_type,
               std::vector<float> tensor_values)
        : name(std::move(tensor_name)),
          dims(std::move(tensor_dims)),
          type(tensor_type),
          values(std::move(tensor_values)) {}

    TinyTensor(std::string tensor_name,
               std::vector<std::uint64_t> tensor_dims,
               motifcl::gguf::TensorType tensor_type,
               std::vector<float> tensor_values,
               std::vector<std::uint8_t> tensor_raw_payload)
        : name(std::move(tensor_name)),
          dims(std::move(tensor_dims)),
          type(tensor_type),
          values(std::move(tensor_values)),
          raw_payload(std::move(tensor_raw_payload)) {}

    std::string name;
    std::vector<std::uint64_t> dims;
    motifcl::gguf::TensorType type = motifcl::gguf::TensorType::F32;
    std::vector<float> values;
    std::vector<std::uint8_t> raw_payload;
    std::uint64_t offset = 0;
};

std::vector<float> sequence(std::size_t n, float first = 1.0f) {
    std::vector<float> values(n);
    for (std::size_t i = 0; i < n; ++i) values[i] = first + static_cast<float>(i) * 0.01f;
    return values;
}

std::size_t tiny_tensor_nbytes(const TinyTensor& tensor) {
    if (!tensor.raw_payload.empty()) return tensor.raw_payload.size();
    std::uint64_t n = 1;
    for (auto dim : tensor.dims) n *= dim;
    return static_cast<std::size_t>(n * (tensor.type == motifcl::gguf::TensorType::F16 ? 2 : 4));
}

void write_tensor_info(std::ostream& out, const TinyTensor& tensor) {
    write_string(out, tensor.name);
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(tensor.dims.size()));
    for (auto dim : tensor.dims) write_le<std::uint64_t>(out, dim);
    write_le<std::uint32_t>(out, static_cast<std::uint32_t>(tensor.type));
    write_le<std::uint64_t>(out, tensor.offset);
}

void write_tensor_payload(std::ostream& out, const TinyTensor& tensor) {
    if (!tensor.raw_payload.empty()) {
        out.write(reinterpret_cast<const char*>(tensor.raw_payload.data()),
                  static_cast<std::streamsize>(tensor.raw_payload.size()));
        return;
    }
    if (tensor.type == motifcl::gguf::TensorType::F16) {
        for (std::size_t i = 0; i < tensor.values.size(); ++i) write_le<std::uint16_t>(out, 0x3c00u);
    } else {
        for (float value : tensor.values) write_le<float>(out, value);
    }
}

std::vector<std::uint8_t> quant_q4_0_block(std::uint8_t packed = 0x98u) {
    std::vector<std::uint8_t> raw(18, 0);
    raw[0] = 0x00; // f16 1.0
    raw[1] = 0x3c;
    std::fill(raw.begin() + 2, raw.end(), packed);
    return raw;
}

std::vector<std::uint8_t> quant_q8_0_block() {
    std::vector<std::uint8_t> raw(34, 0);
    raw[0] = 0x00; // f16 1.0
    raw[1] = 0x3c;
    for (int i = 0; i < 32; ++i) raw[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(i - 16));
    return raw;
}

std::vector<std::uint8_t> repeat_quant_block(const std::vector<std::uint8_t>& block, std::size_t count) {
    std::vector<std::uint8_t> raw;
    raw.reserve(block.size() * count);
    for (std::size_t i = 0; i < count; ++i) raw.insert(raw.end(), block.begin(), block.end());
    return raw;
}

std::vector<std::uint8_t> k_scales_one() {
    // Encodes scale=1, min=0 for all eight 32-element K-quant sub-blocks.
    return {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
}

std::vector<std::uint8_t> quant_q4_k_block() {
    std::vector<std::uint8_t> raw(144, 0);
    raw[0] = 0x00; // d = f16 1.0
    raw[1] = 0x3c;
    raw[2] = 0x00; // dmin = f16 0.0
    raw[3] = 0x00;
    const auto scales = k_scales_one();
    std::copy(scales.begin(), scales.end(), raw.begin() + 4);
    std::fill(raw.begin() + 16, raw.end(), 0x21u);
    return raw;
}

std::vector<std::uint8_t> quant_q5_k_block() {
    std::vector<std::uint8_t> raw(176, 0);
    raw[0] = 0x00; // d = f16 1.0
    raw[1] = 0x3c;
    raw[2] = 0x00; // dmin = f16 0.0
    raw[3] = 0x00;
    const auto scales = k_scales_one();
    std::copy(scales.begin(), scales.end(), raw.begin() + 4);
    raw[16] = 0x01; // high bit for the first low-nibble value only.
    std::fill(raw.begin() + 48, raw.end(), 0x21u);
    return raw;
}

std::vector<std::uint8_t> quant_q6_k_zero_block() {
    std::vector<std::uint8_t> raw(210, 0);
    std::fill(raw.begin() + 128, raw.begin() + 192, 0xAAu); // high bits encode code 32.
    std::fill(raw.begin() + 192, raw.begin() + 208, 0x01u); // int8 scales = 1.
    raw[208] = 0x00; // d = f16 1.0
    raw[209] = 0x3c;
    return raw;
}

} // namespace

int main() {
    const auto path = std::filesystem::current_path() / "tiny_test.gguf";
    {
        constexpr std::uint64_t vocab = 8;
        constexpr std::uint64_t hidden = 4;
        constexpr std::uint64_t heads = 2;
        constexpr std::uint64_t kv_heads = 1;
        constexpr std::uint64_t kv_dim = 2;
        constexpr std::uint64_t mlp = 8;
        std::vector<TinyTensor> tensors{
            {"token_embd.weight", {hidden, vocab}, motifcl::gguf::TensorType::F32, sequence(hidden * vocab, 1.0f)},
            {"output_norm.weight", {hidden}, motifcl::gguf::TensorType::F16, std::vector<float>(hidden, 1.0f)},
            {"output.weight", {hidden, vocab}, motifcl::gguf::TensorType::F32, sequence(hidden * vocab, 2.0f)},
            {"blk.0.attn_norm.weight", {hidden}, motifcl::gguf::TensorType::F32, std::vector<float>(hidden, 1.0f)},
            {"blk.0.ffn_norm.weight", {hidden}, motifcl::gguf::TensorType::F32, std::vector<float>(hidden, 1.0f)},
            {"blk.0.attn_q.weight", {hidden, hidden}, motifcl::gguf::TensorType::F32, sequence(hidden * hidden, 3.0f)},
            {"blk.0.attn_k.weight", {hidden, kv_dim}, motifcl::gguf::TensorType::F32, sequence(hidden * kv_dim, 4.0f)},
            {"blk.0.attn_v.weight", {hidden, kv_dim}, motifcl::gguf::TensorType::F32, sequence(hidden * kv_dim, 5.0f)},
            {"blk.0.attn_output.weight", {hidden, hidden}, motifcl::gguf::TensorType::F32, sequence(hidden * hidden, 6.0f)},
            {"blk.0.ffn_gate.weight", {hidden, mlp}, motifcl::gguf::TensorType::F32, sequence(hidden * mlp, 7.0f)},
            {"blk.0.ffn_up.weight", {hidden, mlp}, motifcl::gguf::TensorType::F32, sequence(hidden * mlp, 8.0f)},
            {"blk.0.ffn_down.weight", {hidden, mlp}, motifcl::gguf::TensorType::F32, sequence(hidden * mlp, 9.0f)},
            {"quant_q4_0.weight", {32}, motifcl::gguf::TensorType::Q4_0, {}, quant_q4_0_block()},
            {"quant_q8_0.weight", {32}, motifcl::gguf::TensorType::Q8_0, {}, quant_q8_0_block()},
            {"quant_q4_k.weight", {256}, motifcl::gguf::TensorType::Q4_K, {}, quant_q4_k_block()},
            {"quant_q5_k.weight", {256}, motifcl::gguf::TensorType::Q5_K, {}, quant_q5_k_block()},
            {"quant_q6_k.weight", {256}, motifcl::gguf::TensorType::Q6_K, {}, quant_q6_k_zero_block()},
            {"quant_q4_k_mat.weight", {256, 1}, motifcl::gguf::TensorType::Q4_K, {}, quant_q4_k_block()},
            {"quant_q5_k_mat.weight", {256, 1}, motifcl::gguf::TensorType::Q5_K, {}, quant_q5_k_block()},
            {"quant_q6_k_mat.weight", {256, 1}, motifcl::gguf::TensorType::Q6_K, {}, quant_q6_k_zero_block()},
        };
        std::uint64_t offset = 0;
        for (auto& tensor : tensors) {
            tensor.offset = offset;
            offset += tiny_tensor_nbytes(tensor);
        }

        std::ofstream out(path, std::ios::binary);
        write_le<std::uint32_t>(out, 0x46554747u);
        write_le<std::uint32_t>(out, 3u);
        write_le<std::uint64_t>(out, static_cast<std::uint64_t>(tensors.size()));
        write_le<std::uint64_t>(out, 14u); // metadata_kv_count

        write_metadata_string(out, "general.architecture", "llama");
        write_metadata_u32(out, "general.alignment", 32);
        write_metadata_u32(out, "llama.context_length", 2048);
        write_metadata_u32(out, "llama.embedding_length", static_cast<std::uint32_t>(hidden));
        write_metadata_u32(out, "llama.block_count", 1);
        write_metadata_u32(out, "llama.feed_forward_length", static_cast<std::uint32_t>(mlp));
        write_metadata_u32(out, "llama.attention.head_count", static_cast<std::uint32_t>(heads));
        write_metadata_u32(out, "llama.attention.head_count_kv", static_cast<std::uint32_t>(kv_heads));
        write_metadata_f32(out, "llama.attention.layer_norm_rms_epsilon", 0.00001f);
        write_metadata_f32(out, "llama.rope.freq_base", 10000.0f);
        write_metadata_u32(out, "tokenizer.ggml.bos_token_id", 0);
        write_metadata_u32(out, "tokenizer.ggml.eos_token_id", 1);
        write_metadata_string(out, "tokenizer.ggml.model", "llama");
        write_metadata_string_array(out, "tokenizer.ggml.tokens", {"<s>", "</s>", "A", "B", "C", "D", "E", "F"});

        for (const auto& tensor : tensors) write_tensor_info(out, tensor);

        pad_to_alignment(out, 32);
        for (const auto& tensor : tensors) write_tensor_payload(out, tensor);
    }

    auto file = motifcl::gguf::File::open(path.string());
    require(file.version() == 3, "GGUF version parse failed");
    require(file.tensor_count() == 20, "GGUF tensor count parse failed");
    require(file.metadata_count() == 14, "GGUF metadata count parse failed");
    require(file.alignment() == 32, "GGUF alignment parse failed");
    require(file.metadata_string_or("general.architecture") == "llama", "GGUF string metadata parse failed");
    require(file.metadata_u64_or("llama.context_length") == 2048, "GGUF uint metadata parse failed");

    const auto& tokens = file.metadata("tokenizer.ggml.tokens");
    require(tokens.is_array(), "GGUF array metadata type failed");
    require(tokens.array_type == motifcl::gguf::MetadataType::String, "GGUF array element type failed");
    require(tokens.as_array().size() == 8 && tokens.as_array()[2].as_string() == "A",
            "GGUF string array parse failed");

    require(file.contains_tensor("blk.0.attn_q.weight"), "GGUF tensor lookup failed");
    const auto& info = file.tensor_info("blk.0.attn_q.weight");
    require(info.type == motifcl::gguf::TensorType::F32, "GGUF tensor type parse failed");
    require(info.dimensions == std::vector<std::uint64_t>({4, 4}), "GGUF tensor dimensions parse failed");
    require(motifcl::gguf::tensor_element_count(info) == 16, "GGUF tensor element count failed");
    require(motifcl::gguf::tensor_nbytes(info) == 64, "GGUF tensor byte size failed");
    require(motifcl::gguf::tensor_type_block_nbytes(motifcl::gguf::TensorType::Q8_1) == 36,
            "GGUF Q8_1 block byte size failed");
    require(motifcl::gguf::tensor_type_block_size(motifcl::gguf::TensorType::Q4_K) == 256,
            "GGUF Q4_K block size failed");
    require(motifcl::gguf::tensor_type_block_nbytes(motifcl::gguf::TensorType::Q4_K) == 144,
            "GGUF Q4_K block byte size failed");
    require(motifcl::gguf::tensor_type_can_dequantize_to_f32(motifcl::gguf::TensorType::Q5_K),
            "GGUF Q5_K dequant capability failed");
    require(motifcl::gguf::tensor_type_can_dequantize_to_f32(motifcl::gguf::TensorType::Q6_K),
            "GGUF Q6_K dequant capability failed");
    require(motifcl::gguf::tensor_type_can_repack_to_motifcl_quant(motifcl::gguf::TensorType::Q4_0) &&
                motifcl::gguf::tensor_type_can_repack_to_motifcl_quant(motifcl::gguf::TensorType::Q8_0) &&
                motifcl::gguf::tensor_type_can_repack_to_motifcl_quant(motifcl::gguf::TensorType::Q4_K) &&
                motifcl::gguf::tensor_type_can_repack_to_motifcl_quant(motifcl::gguf::TensorType::Q5_K) &&
                motifcl::gguf::tensor_type_can_repack_to_motifcl_quant(motifcl::gguf::TensorType::Q6_K),
            "GGUF native MotifCL quant repack capability failed");

    const auto raw = file.read_tensor_data("blk.0.attn_q.weight");
    require(raw.size() == 64, "GGUF tensor raw read size failed");
    float first = 0.0f;
    std::memcpy(&first, raw.data(), sizeof(first));
    require(first == 3.0f, "GGUF tensor raw read content failed");
    const auto q4_0 = file.read_tensor_f32("quant_q4_0.weight");
    require(q4_0.size() == 32 && q4_0[0] == 0.0f && q4_0[16] == 1.0f,
            "GGUF Q4_0 dequant failed");
    const auto q8_0 = file.read_tensor_f32("quant_q8_0.weight");
    require(q8_0.size() == 32 && q8_0[0] == -16.0f && q8_0[31] == 15.0f,
            "GGUF Q8_0 dequant failed");
    const auto q4_k = file.read_tensor_f32("quant_q4_k.weight");
    require(q4_k.size() == 256 && q4_k[0] == 1.0f && q4_k[32] == 2.0f && q4_k[255] == 2.0f,
            "GGUF Q4_K dequant failed");
    const auto q5_k = file.read_tensor_f32("quant_q5_k.weight");
    require(q5_k.size() == 256 && q5_k[0] == 17.0f && q5_k[1] == 1.0f && q5_k[32] == 2.0f,
            "GGUF Q5_K dequant failed");
    const auto q6_k = file.read_tensor_f32("quant_q6_k.weight");
    require(q6_k.size() == 256 && q6_k[0] == 0.0f && q6_k[255] == 0.0f,
            "GGUF Q6_K dequant failed");

    auto cfg = motifcl::nn::load_hf_transformer_config_gguf(path.string());
    require(cfg.architecture == motifcl::nn::HFArchitecture::Llama, "GGUF architecture mapping failed");
    require(cfg.transformer.vocab_size == 8 && cfg.transformer.n_embd == 4, "GGUF config size mapping failed");
    require(cfg.transformer.n_layer == 1 && cfg.transformer.n_head == 2 && cfg.transformer.n_kv_head == 1,
            "GGUF config attention mapping failed");
    require(cfg.transformer.mlp_hidden == 8 && cfg.bos_token_id == 0 && cfg.eos_token_id == 1,
            "GGUF config tokenizer metadata mapping failed");

    auto mapped = motifcl::nn::map_gguf_transformer_weight_name(cfg.architecture, "blk.0.attn_q.weight");
    require(mapped.kind == "q_proj" && mapped.layer == 0, "GGUF weight mapper failed");
    auto tokenizer = motifcl::nn::load_hf_tokenizer_gguf(path.string(), cfg);
    require(tokenizer.encode("AB", false, false) == std::vector<std::int32_t>({2, 3}),
            "GGUF tokenizer metadata load failed");

    const auto gemma4_path = std::filesystem::current_path() / "tiny_gemma4_meta.gguf";
    {
        std::ofstream out(gemma4_path, std::ios::binary);
        write_le<std::uint32_t>(out, 0x46554747u);
        write_le<std::uint32_t>(out, 3u);
        write_le<std::uint64_t>(out, 0u);
        write_le<std::uint64_t>(out, 21u);
        write_metadata_string(out, "general.architecture", "gemma4");
        write_metadata_u32(out, "general.alignment", 32);
        write_metadata_u32(out, "gemma4.context_length", 32);
        write_metadata_u32(out, "gemma4.embedding_length", 8);
        write_metadata_u32(out, "gemma4.block_count", 2);
        write_metadata_u32_array(out, "gemma4.feed_forward_length", {8, 16});
        write_metadata_u32(out, "gemma4.attention.head_count", 2);
        write_metadata_u32(out, "gemma4.attention.head_count_kv", 1);
        write_metadata_f32(out, "gemma4.attention.layer_norm_rms_epsilon", 0.000001f);
        write_metadata_f32(out, "gemma4.rope.freq_base", 1000000.0f);
        write_metadata_f32(out, "gemma4.rope.freq_base_swa", 10000.0f);
        write_metadata_u32(out, "gemma4.attention.key_length", 4);
        write_metadata_u32(out, "gemma4.attention.key_length_swa", 2);
        write_metadata_u32(out, "gemma4.rope.dimension_count", 4);
        write_metadata_u32(out, "gemma4.rope.dimension_count_swa", 2);
        write_metadata_u32(out, "gemma4.attention.sliding_window", 3);
        write_metadata_bool_array(out, "gemma4.attention.sliding_window_pattern", {true, false});
        write_metadata_u32(out, "tokenizer.ggml.bos_token_id", 0);
        write_metadata_u32(out, "tokenizer.ggml.eos_token_id", 1);
        write_metadata_string(out, "tokenizer.ggml.model", "gemma4");
        write_metadata_string_array(out, "tokenizer.ggml.tokens", {"<s>", "</s>", "A", "B", "C", "D", "E", "F"});
        pad_to_alignment(out, 32);
    }
    auto gemma4_cfg = motifcl::nn::load_hf_transformer_config_gguf(gemma4_path.string());
    require(gemma4_cfg.architecture == motifcl::nn::HFArchitecture::Gemma4,
            "Gemma4 GGUF architecture mapping failed");
    require(gemma4_cfg.transformer.mlp_hidden == 8 &&
                gemma4_cfg.transformer.layer_mlp_hiddens == std::vector<int>({8, 16}),
            "Gemma4 GGUF per-layer MLP metadata mapping failed");
    require(gemma4_cfg.transformer.head_dim == 2 &&
                gemma4_cfg.transformer.layer_head_dims == std::vector<int>({2, 4}),
            "Gemma4 GGUF per-layer head-dim metadata mapping failed");
    require(gemma4_cfg.transformer.layer_rope_thetas.size() == 2 &&
                std::fabs(gemma4_cfg.transformer.layer_rope_thetas[0] - 10000.0f) < 0.5f &&
                std::fabs(gemma4_cfg.transformer.layer_rope_thetas[1] - 1000000.0f) < 0.5f,
            "Gemma4 GGUF per-layer rope theta metadata mapping failed");
    require(gemma4_cfg.layer_types.size() == 2 &&
                gemma4_cfg.layer_types[0] == "sliding_attention" &&
                gemma4_cfg.layer_types[1] == "full_attention" &&
                gemma4_cfg.local_attention_layers == std::vector<int>({0}) &&
                gemma4_cfg.global_attention_layers == std::vector<int>({1}),
            "Gemma4 GGUF sliding/full attention pattern mapping failed");
    auto backend = motifcl::Backend::create_opencl();
    auto gemma4_model = motifcl::nn::make_hf_transformer_model(backend, gemma4_cfg);
    require(gemma4_model.blocks[0]->attention().head_dim() == 2 &&
                gemma4_model.blocks[1]->attention().head_dim() == 4 &&
                gemma4_model.blocks[0]->mlp().gate_proj().out_features() == 8 &&
                gemma4_model.blocks[1]->mlp().gate_proj().out_features() == 16,
            "Gemma4 GGUF per-layer model construction failed");
    const auto native_q4 = file.read_tensor_quantized(backend, "quant_q4_0.weight");
    require(native_q4.dtype() == motifcl::DType::Q4_0 && native_q4.has_quant_scales() &&
                native_q4.quant_block_size() == 32,
            "GGUF Q4_0 native quant repack metadata failed");
    const auto native_q4_f32 = motifcl::dequantize_q4(native_q4).to_vector<float>();
    require(native_q4_f32.size() == 32 && native_q4_f32[0] == 0.0f && native_q4_f32[16] == 1.0f,
            "GGUF Q4_0 native quant repack values failed");
    const auto native_q8 = file.read_tensor_quantized(backend, "quant_q8_0.weight");
    require(native_q8.dtype() == motifcl::DType::Q8_0 && native_q8.has_quant_scales() &&
                native_q8.quant_block_size() == 32,
            "GGUF Q8_0 native quant repack metadata failed");
    const auto native_q8_f32 = motifcl::dequantize_q8(native_q8).to_vector<float>();
    require(native_q8_f32.size() == 32 && native_q8_f32[0] == -16.0f && native_q8_f32[31] == 15.0f,
            "GGUF Q8_0 native quant repack values failed");
    const auto native_q4k = file.read_tensor_quantized(backend, "quant_q4_k_mat.weight");
    require(native_q4k.dtype() == motifcl::DType::Q4_K && native_q4k.shape() == motifcl::Shape({256, 1}) &&
                native_q4k.nbytes() == 144,
            "GGUF Q4_K native packed tensor metadata failed");
    const auto native_q5k = file.read_tensor_quantized(backend, "quant_q5_k_mat.weight");
    require(native_q5k.dtype() == motifcl::DType::Q5_K && native_q5k.shape() == motifcl::Shape({256, 1}) &&
                native_q5k.nbytes() == 176,
            "GGUF Q5_K native packed tensor metadata failed");
    const auto native_q6k = file.read_tensor_quantized(backend, "quant_q6_k_mat.weight");
    require(native_q6k.dtype() == motifcl::DType::Q6_K && native_q6k.shape() == motifcl::Shape({256, 1}) &&
                native_q6k.nbytes() == 210,
            "GGUF Q6_K native packed tensor metadata failed");
    std::vector<float> ones(256, 1.0f);
    auto x = motifcl::Tensor::from_cpu(backend, {1, 256}, motifcl::DType::F32, ones.data());
    auto xq = motifcl::quantize_q8_symmetric_rows(x);
    const auto q4k_out = motifcl::matmul(xq, native_q4k).to_vector<float>();
    const auto q5k_out = motifcl::matmul(xq, native_q5k).to_vector<float>();
    const auto q6k_out = motifcl::matmul(xq, native_q6k).to_vector<float>();
    const auto q4k_decode_out = motifcl::matmul(x, native_q4k).to_vector<float>();
    const auto q5k_decode_out = motifcl::matmul(x, native_q5k).to_vector<float>();
    const auto q6k_decode_out = motifcl::matmul(x, native_q6k).to_vector<float>();
    require(q4k_out.size() == 1 && std::fabs(q4k_out[0] - 384.0f) < 0.05f,
            "GGUF Q4_K native packed matmul failed");
    require(q5k_out.size() == 1 && std::fabs(q5k_out[0] - 400.0f) < 0.05f,
            "GGUF Q5_K native packed matmul failed");
    require(q6k_out.size() == 1 && std::fabs(q6k_out[0]) < 0.05f,
            "GGUF Q6_K native packed matmul failed");
    require(q4k_decode_out.size() == 1 && std::fabs(q4k_decode_out[0] - 384.0f) < 0.05f,
            "GGUF Q4_K native packed F32 decode matmul failed");
    require(q5k_decode_out.size() == 1 && std::fabs(q5k_decode_out[0] - 400.0f) < 0.05f,
            "GGUF Q5_K native packed F32 decode matmul failed");
    require(q6k_decode_out.size() == 1 && std::fabs(q6k_decode_out[0]) < 0.05f,
            "GGUF Q6_K native packed F32 decode matmul failed");

    auto model = motifcl::nn::make_hf_transformer_model(backend, cfg);
    auto report = motifcl::nn::load_hf_transformer_gguf_weights(backend, model, path.string(), cfg, true, false);
    require(report.loaded_tensors == 12 && report.missing.empty(), "GGUF F16/F32 weight load report failed");
    require(model.token_embedding.weight.data.shape() == motifcl::Shape({8, 4}), "GGUF token embedding shape failed");
    require(model.lm_head.data.shape() == motifcl::Shape({4, 8}), "GGUF lm_head shape failed");
    require(model.blocks[0]->attention().qkv_proj().weight.data.shape() == motifcl::Shape({4, 8}),
            "GGUF packed qkv shape failed");
    require(model.blocks[0]->mlp().gate_up_proj.weight.data.shape() == motifcl::Shape({4, 16}),
            "GGUF packed MLP shape failed");
    const auto norm_values = model.final_norm.weight.data.to_vector<float>();
    require(!norm_values.empty() && norm_values[0] == 1.0f, "GGUF F16 norm load failed");

    const auto packed_path = std::filesystem::current_path() / "tiny_packed_model.gguf";
    {
        constexpr std::uint64_t vocab = 64;
        constexpr std::uint64_t hidden = 32;
        constexpr std::uint64_t heads = 4;
        constexpr std::uint64_t kv_heads = 4;
        constexpr std::uint64_t head_dim = hidden / heads;
        constexpr std::uint64_t kv_dim = 32;
        constexpr std::uint64_t mlp = 32;
        constexpr std::uint64_t ple = 8;
        const auto q4 = quant_q4_0_block(0x98u);
        const auto q5 = quant_q5_k_block();
        const auto q6 = quant_q6_k_zero_block();
        auto q4_payload = [&](std::uint64_t elements) {
            return repeat_quant_block(q4, static_cast<std::size_t>(elements / 32));
        };
        auto q5_payload = [&](std::uint64_t elements) {
            return repeat_quant_block(q5, static_cast<std::size_t>(elements / 256));
        };
        auto q6_payload = [&](std::uint64_t elements) {
            return repeat_quant_block(q6, static_cast<std::size_t>(elements / 256));
        };
        std::vector<TinyTensor> tensors{
            {"token_embd.weight", {hidden, vocab}, motifcl::gguf::TensorType::F32, sequence(hidden * vocab, 1.0f)},
            {"output_norm.weight", {hidden}, motifcl::gguf::TensorType::F32, std::vector<float>(hidden, 1.0f)},
            {"output.weight", {vocab, hidden}, motifcl::gguf::TensorType::Q4_0, {}, q4_payload(hidden * vocab)},
            {"per_layer_token_embd.weight", {ple, vocab}, motifcl::gguf::TensorType::Q5_K, {}, q5_payload(ple * vocab)},
            {"per_layer_model_proj.weight", {hidden, ple}, motifcl::gguf::TensorType::F32, sequence(hidden * ple, 0.01f)},
            {"per_layer_proj_norm.weight", {ple}, motifcl::gguf::TensorType::F32, std::vector<float>(ple, 1.0f)},
            {"blk.0.attn_norm.weight", {hidden}, motifcl::gguf::TensorType::F32, std::vector<float>(hidden, 1.0f)},
            {"blk.0.ffn_norm.weight", {hidden}, motifcl::gguf::TensorType::F32, std::vector<float>(hidden, 1.0f)},
            {"blk.0.attn_q_norm.weight", {head_dim}, motifcl::gguf::TensorType::F32, std::vector<float>(head_dim, 1.0f)},
            {"blk.0.attn_k_norm.weight", {head_dim}, motifcl::gguf::TensorType::F32, std::vector<float>(head_dim, 1.0f)},
            {"blk.0.post_attention_norm.weight", {hidden}, motifcl::gguf::TensorType::F32, std::vector<float>(hidden, 1.0f)},
            {"blk.0.post_ffw_norm.weight", {hidden}, motifcl::gguf::TensorType::F32, std::vector<float>(hidden, 1.0f)},
            {"blk.0.layer_output_scale.weight", {1}, motifcl::gguf::TensorType::F32, std::vector<float>{1.0f}},
            {"blk.0.inp_gate.weight", {hidden, ple}, motifcl::gguf::TensorType::F32, sequence(hidden * ple, 0.02f)},
            {"blk.0.proj.weight", {ple, hidden}, motifcl::gguf::TensorType::F32, sequence(ple * hidden, 0.03f)},
            {"blk.0.post_norm.weight", {hidden}, motifcl::gguf::TensorType::F32, std::vector<float>(hidden, 1.0f)},
            {"blk.0.attn_q.weight", {hidden, hidden}, motifcl::gguf::TensorType::Q4_0, {}, q4_payload(hidden * hidden)},
            {"blk.0.attn_k.weight", {hidden, kv_dim}, motifcl::gguf::TensorType::Q4_0, {}, q4_payload(hidden * kv_dim)},
            {"blk.0.attn_v.weight", {hidden, kv_dim}, motifcl::gguf::TensorType::Q6_K, {}, q6_payload(hidden * kv_dim)},
            {"blk.0.attn_output.weight", {hidden, hidden}, motifcl::gguf::TensorType::Q4_0, {}, q4_payload(hidden * hidden)},
            {"blk.0.ffn_gate.weight", {hidden, mlp}, motifcl::gguf::TensorType::Q4_0, {}, q4_payload(hidden * mlp)},
            {"blk.0.ffn_up.weight", {hidden, mlp}, motifcl::gguf::TensorType::Q4_0, {}, q4_payload(hidden * mlp)},
            {"blk.0.ffn_down.weight", {mlp, hidden}, motifcl::gguf::TensorType::Q4_0, {}, q4_payload(mlp * hidden)},
        };
        std::uint64_t offset = 0;
        for (auto& tensor : tensors) {
            tensor.offset = offset;
            offset += tiny_tensor_nbytes(tensor);
        }
        std::ofstream out(packed_path, std::ios::binary);
        write_le<std::uint32_t>(out, 0x46554747u);
        write_le<std::uint32_t>(out, 3u);
        write_le<std::uint64_t>(out, static_cast<std::uint64_t>(tensors.size()));
        write_le<std::uint64_t>(out, 15u);
        write_metadata_string(out, "general.architecture", "llama");
        write_metadata_u32(out, "general.alignment", 32);
        write_metadata_u32(out, "llama.context_length", 64);
        write_metadata_u32(out, "llama.embedding_length", static_cast<std::uint32_t>(hidden));
        write_metadata_u32(out, "llama.block_count", 1);
        write_metadata_u32(out, "llama.feed_forward_length", static_cast<std::uint32_t>(mlp));
        write_metadata_u32(out, "llama.embedding_length_per_layer_input", static_cast<std::uint32_t>(ple));
        write_metadata_u32(out, "llama.attention.head_count", static_cast<std::uint32_t>(heads));
        write_metadata_u32(out, "llama.attention.head_count_kv", static_cast<std::uint32_t>(kv_heads));
        write_metadata_f32(out, "llama.attention.layer_norm_rms_epsilon", 0.00001f);
        write_metadata_f32(out, "llama.rope.freq_base", 10000.0f);
        write_metadata_u32(out, "tokenizer.ggml.bos_token_id", 0);
        write_metadata_u32(out, "tokenizer.ggml.eos_token_id", 1);
        write_metadata_string(out, "tokenizer.ggml.model", "llama");
        std::vector<std::string> tokens;
        for (std::uint64_t i = 0; i < vocab; ++i) tokens.push_back("T" + std::to_string(i));
        write_metadata_string_array(out, "tokenizer.ggml.tokens", tokens);
        for (const auto& tensor : tensors) write_tensor_info(out, tensor);
        pad_to_alignment(out, 32);
        for (const auto& tensor : tensors) write_tensor_payload(out, tensor);
    }

    auto packed_cfg = motifcl::nn::load_hf_transformer_config_gguf(packed_path.string());
    require(packed_cfg.transformer.use_qk_norm &&
                packed_cfg.transformer.use_post_attention_norm &&
                packed_cfg.transformer.use_post_ffw_norm &&
                packed_cfg.transformer.use_layer_output_scale,
            "GGUF optional modern norm/scale flags were not detected");
    require(packed_cfg.transformer.use_per_layer_inputs &&
                packed_cfg.transformer.per_layer_input_dim == 8 &&
                packed_cfg.transformer.per_layer_input_vocab_size == 64,
            "GGUF PLE metadata/tensor flags were not detected");
    auto packed_model = motifcl::nn::make_hf_transformer_model(backend, packed_cfg);
    auto packed_report = motifcl::nn::load_hf_transformer_gguf_weights(backend, packed_model, packed_path.string(),
                                                                       packed_cfg, true, false);
    require(packed_report.loaded_tensors == 23 && packed_report.missing.empty(),
            "GGUF packed model load report failed");
    require(packed_model.blocks[0]->attention().qk_norm_enabled(),
            "GGUF packed q/k RMSNorm runtime flag was not enabled");
    require(packed_model.per_layer_token_embedding &&
                packed_model.per_layer_token_embedding->quantized_inference_enabled() &&
                packed_model.blocks[0]->per_layer_input_enabled(),
            "GGUF PLE runtime modules were not enabled");
    bool saw_repacked_transpose = false;
    for (const auto& item : packed_report.applied) {
        if (item.find("packed_quant_repacked_transpose") != std::string::npos) saw_repacked_transpose = true;
    }
    require(saw_repacked_transpose, "GGUF packed transpose repack path was not exercised");
    require(packed_model.quantized_inference_enabled() &&
                packed_model.quantized_weight_dtype() == motifcl::DType::Q4_0,
            "GGUF packed lm_head was not installed");
    require(packed_model.blocks[0]->attention().split_projections_enabled() &&
                packed_model.blocks[0]->attention().q_proj().quantized_weight_dtype() == motifcl::DType::Q4_0 &&
                packed_model.blocks[0]->attention().o_proj().quantized_weight_dtype() == motifcl::DType::Q4_0,
            "GGUF packed attention projections were not installed");
    require(packed_model.blocks[0]->attention().v_proj().quantized_inference_enabled() &&
                packed_model.blocks[0]->attention().v_proj().quantized_weight_dtype() == motifcl::DType::Q6_K,
            "GGUF Q6_K native packed v_proj was not installed");
    require(packed_model.blocks[0]->mlp().split_projections_enabled() &&
                packed_model.blocks[0]->mlp().gate_proj().quantized_weight_dtype() == motifcl::DType::Q4_0 &&
                packed_model.blocks[0]->mlp().down_proj.quantized_weight_dtype() == motifcl::DType::Q4_0,
            "GGUF packed MLP projections were not installed");
    std::vector<std::int32_t> packed_ids{1, 2};
    auto packed_tokens = motifcl::Tensor::from_cpu(backend, {1, 2}, motifcl::DType::I32, packed_ids.data());
    auto packed_logits = packed_model.forward(packed_tokens);
    require(packed_logits.shape() == motifcl::Shape({1, 2, 64}), "GGUF packed model forward shape failed");
    for (float v : packed_logits.to_vector<float>()) {
        require(std::isfinite(v), "GGUF packed model forward produced non-finite logits");
    }
    std::filesystem::remove(packed_path);

    std::filesystem::remove(gemma4_path);
    std::filesystem::remove(path);
    return 0;
}
