#include <motifcl/nn/hf_compat.hpp>

#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/gguf.hpp>
#include <motifcl/ops/indexing.hpp>
#include <motifcl/ops/quant.hpp>
#include <motifcl/ops/reduce.hpp>
#include <motifcl/tensor/storage.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

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

bool json_has_key(const std::string& text, const std::string& key) {
    return text.find("\"" + key + "\"") != std::string::npos;
}

std::int64_t json_integer_or(const std::string& text, const std::string& key, std::int64_t fallback) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(-?\\d+)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return fallback;
    return std::stoll(m[1].str());
}

bool json_bool_or(const std::string& text, const std::string& key, bool fallback) {
    const std::regex re("\"" + key + "\"\\s*:\\s*(true|false|1|0)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return fallback;
    const auto value = lower_ascii(m[1].str());
    return value == "true" || value == "1";
}

std::string extract_json_object_for_key(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return {};
    const auto colon = text.find(':', key_pos);
    if (colon == std::string::npos) return {};
    const auto open = text.find('{', colon + 1);
    if (open == std::string::npos) return {};
    bool in_string = false;
    bool escape = false;
    int depth = 0;
    for (std::size_t i = open; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == '{') ++depth;
        else if (c == '}') {
            --depth;
            if (depth == 0) return text.substr(open + 1, i - open - 1);
        }
    }
    return {};
}

std::string extract_json_array_for_key(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return {};
    const auto colon = text.find(':', key_pos);
    if (colon == std::string::npos) return {};
    const auto open = text.find('[', colon + 1);
    if (open == std::string::npos) return {};
    bool in_string = false;
    bool escape = false;
    int depth = 0;
    for (std::size_t i = open; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escape) escape = false;
            else if (c == '\\') escape = true;
            else if (c == '"') in_string = false;
            continue;
        }
        if (c == '"') in_string = true;
        else if (c == '[') ++depth;
        else if (c == ']') {
            --depth;
            if (depth == 0) return text.substr(open + 1, i - open - 1);
        }
    }
    return {};
}

std::vector<std::string> json_string_array_or_empty(const std::string& text, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        const auto array = extract_json_array_for_key(text, key);
        if (array.empty()) continue;
        std::vector<std::string> out;
        std::smatch m;
        std::string::const_iterator search_start(array.cbegin());
        const std::regex string_re("\"((?:\\\\.|[^\"\\\\])*)\"");
        while (std::regex_search(search_start, array.cend(), m, string_re)) {
            out.push_back(json_unescape(m[1].str()));
            search_start = m.suffix().first;
        }
        if (!out.empty()) return out;
    }
    return {};
}

std::vector<int> json_int_array_or_empty(const std::string& text, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        const auto array = extract_json_array_for_key(text, key);
        if (array.empty()) continue;
        std::vector<int> out;
        std::smatch m;
        std::string::const_iterator search_start(array.cbegin());
        const std::regex int_re("(-?\\d+)");
        while (std::regex_search(search_start, array.cend(), m, int_re)) {
            out.push_back(std::stoi(m[1].str()));
            search_start = m.suffix().first;
        }
        if (!out.empty()) return out;
    }
    return {};
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
           architecture == HFArchitecture::Gemma2 ||
           architecture == HFArchitecture::Gemma3 ||
           architecture == HFArchitecture::Gemma4 ||
           architecture == HFArchitecture::Llama ||
           architecture == HFArchitecture::Mistral ||
           architecture == HFArchitecture::Qwen2;
}

HFWeightName map_qwen35_weight_name(const std::string& hf_name) {
    if (hf_name == "model.embed_tokens.weight") return {hf_name, "token_embedding.weight", "token_embedding", -1};
    if (hf_name == "lm_head.weight") return {hf_name, "lm_head.weight", "lm_head", -1};
    if (hf_name == "model.norm.weight") return {hf_name, "final_norm.weight", "final_norm", -1};

    std::smatch m;
    const std::regex expert_re(R"(^model\.layers\.(\d+)\.(?:mlp|block_sparse_moe)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$)");
    if (std::regex_match(hf_name, m, expert_re)) {
        const int layer = std::stoi(m[1].str());
        const int expert = std::stoi(m[2].str());
        const std::string proj = m[3].str();
        const std::string canonical_proj = proj == "w1" ? "gate_proj" : (proj == "w3" ? "up_proj" : (proj == "w2" ? "down_proj" : proj));
        return {hf_name,
                "blocks." + std::to_string(layer) + ".moe.experts." + std::to_string(expert) + "." + canonical_proj + ".weight",
                "moe_expert_" + canonical_proj,
                layer};
    }

    const std::regex layer_re(R"(^model\.layers\.(\d+)\.(.+)$)");
    if (!std::regex_match(hf_name, m, layer_re)) return {hf_name, hf_name, "unknown", -1};
    const int layer = std::stoi(m[1].str());
    const std::string suffix = m[2].str();
    const std::string base = "blocks." + std::to_string(layer) + ".";
    if (suffix == "input_layernorm.weight") return {hf_name, base + "norm1.weight", "input_layernorm", layer};
    if (suffix == "post_attention_layernorm.weight") return {hf_name, base + "norm2.weight", "post_attention_layernorm", layer};
    if (suffix == "self_attn.q_proj.weight") return {hf_name, base + "attention.q_proj.weight", "q_proj", layer};
    if (suffix == "self_attn.k_proj.weight") return {hf_name, base + "attention.k_proj.weight", "k_proj", layer};
    if (suffix == "self_attn.v_proj.weight") return {hf_name, base + "attention.v_proj.weight", "v_proj", layer};
    if (suffix == "self_attn.o_proj.weight") return {hf_name, base + "attention.o_proj.weight", "o_proj", layer};
    if (suffix == "self_attn.gate_proj.weight") return {hf_name, base + "gated_attention.gate_proj.weight", "gated_attention_gate_proj", layer};
    if (suffix == "linear_attn.q_proj.weight" || suffix == "gated_delta_net.q_proj.weight") return {hf_name, base + "delta.q_proj.weight", "delta_q_proj", layer};
    if (suffix == "linear_attn.k_proj.weight" || suffix == "gated_delta_net.k_proj.weight") return {hf_name, base + "delta.k_proj.weight", "delta_k_proj", layer};
    if (suffix == "linear_attn.v_proj.weight" || suffix == "gated_delta_net.v_proj.weight") return {hf_name, base + "delta.v_proj.weight", "delta_v_proj", layer};
    if (suffix == "linear_attn.gate_proj.weight" || suffix == "gated_delta_net.gate_proj.weight") return {hf_name, base + "delta.gate_proj.weight", "delta_gate_proj", layer};
    if (suffix == "linear_attn.o_proj.weight" || suffix == "gated_delta_net.o_proj.weight") return {hf_name, base + "delta.o_proj.weight", "delta_o_proj", layer};
    if (suffix == "mlp.gate.weight" || suffix == "mlp.router.weight" || suffix == "mlp.gate_proj.weight" ||
        suffix == "block_sparse_moe.gate.weight" || suffix == "block_sparse_moe.router.weight") {
        return {hf_name, base + "moe.router_weight", "moe_router", layer};
    }
    if (suffix == "mlp.up_proj.weight") return {hf_name, base + "mlp.up_proj.weight", "up_proj", layer};
    if (suffix == "mlp.down_proj.weight") return {hf_name, base + "mlp.down_proj.weight", "down_proj", layer};
    return {hf_name, hf_name, "unknown", layer};
}

std::string extension_lower(const std::filesystem::path& path) {
    return lower_ascii(path.extension().string());
}

bool is_gguf_path(const std::filesystem::path& path) {
    return extension_lower(path) == ".gguf";
}

std::vector<std::string> find_safetensors_files(const std::filesystem::path& dir) {
    std::vector<std::string> out;
    if (!std::filesystem::is_directory(dir)) return out;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && extension_lower(entry.path()) == ".safetensors") {
            out.push_back(entry.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
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

std::uint64_t gguf_metadata_scalar_to_u64(const gguf::MetadataScalar& scalar,
                                          const std::string& key) {
    switch (scalar.type) {
        case gguf::MetadataType::UInt8:
        case gguf::MetadataType::UInt16:
        case gguf::MetadataType::UInt32:
        case gguf::MetadataType::UInt64:
            return scalar.as_u64();
        case gguf::MetadataType::Int8:
        case gguf::MetadataType::Int16:
        case gguf::MetadataType::Int32:
        case gguf::MetadataType::Int64: {
            const auto value = scalar.as_i64();
            MCL_CHECK(value >= 0, "GGUF metadata integer is negative: " + key);
            return static_cast<std::uint64_t>(value);
        }
        default:
            MCL_CHECK(false, "GGUF metadata is not integer: " + key);
    }
    return 0;
}

std::uint64_t metadata_integer_or(const gguf::File& file,
                                  const std::vector<std::string>& keys,
                                  std::uint64_t fallback) {
    for (const auto& key : keys) {
        if (!file.has_metadata(key)) continue;
        return gguf_metadata_scalar_to_u64(file.metadata(key).as_scalar(), key);
    }
    return fallback;
}

std::vector<std::uint64_t> metadata_integer_array_or_empty(const gguf::File& file,
                                                           const std::string& key) {
    if (!file.has_metadata(key)) return {};
    const auto& value = file.metadata(key);
    if (!value.is_array()) return {gguf_metadata_scalar_to_u64(value.as_scalar(), key)};
    std::vector<std::uint64_t> out;
    out.reserve(value.as_array().size());
    for (const auto& scalar : value.as_array()) {
        out.push_back(gguf_metadata_scalar_to_u64(scalar, key));
    }
    return out;
}

std::vector<bool> metadata_bool_array_or_empty(const gguf::File& file, const std::string& key) {
    if (!file.has_metadata(key)) return {};
    const auto& value = file.metadata(key);
    MCL_CHECK(value.is_array(), "GGUF metadata is not a bool/integer array: " + key);
    std::vector<bool> out;
    out.reserve(value.as_array().size());
    for (const auto& scalar : value.as_array()) {
        if (scalar.type == gguf::MetadataType::Bool) {
            out.push_back(scalar.as_bool());
        } else {
            out.push_back(gguf_metadata_scalar_to_u64(scalar, key) != 0);
        }
    }
    return out;
}

double metadata_float_or(const gguf::File& file,
                         const std::vector<std::string>& keys,
                         double fallback) {
    for (const auto& key : keys) {
        if (!file.has_metadata(key)) continue;
        const auto& scalar = file.metadata(key).as_scalar();
        if (scalar.type == gguf::MetadataType::Float32 || scalar.type == gguf::MetadataType::Float64) {
            return scalar.as_f64();
        }
        if (scalar.type == gguf::MetadataType::UInt8 || scalar.type == gguf::MetadataType::UInt16 ||
            scalar.type == gguf::MetadataType::UInt32 || scalar.type == gguf::MetadataType::UInt64 ||
            scalar.type == gguf::MetadataType::Int8 || scalar.type == gguf::MetadataType::Int16 ||
            scalar.type == gguf::MetadataType::Int32 || scalar.type == gguf::MetadataType::Int64) {
            return static_cast<double>(metadata_integer_or(file, {key}, 0));
        }
        MCL_CHECK(false, "GGUF metadata is not numeric: " + key);
    }
    return fallback;
}

std::vector<std::string> metadata_string_array_or_empty(const gguf::File& file, const std::string& key) {
    if (!file.has_metadata(key)) return {};
    const auto& value = file.metadata(key);
    MCL_CHECK(value.is_array() && value.array_type == gguf::MetadataType::String,
              "GGUF metadata is not a string array: " + key);
    std::vector<std::string> out;
    out.reserve(value.as_array().size());
    for (const auto& scalar : value.as_array()) out.push_back(scalar.as_string());
    return out;
}

std::vector<float> gguf_tensor_f32(const gguf::File& file, const std::string& name) {
    return file.read_tensor_f32(name);
}

bool gguf_type_is_native_quant(gguf::TensorType type) {
    return type == gguf::TensorType::Q8_0 ||
           type == gguf::TensorType::Q4_0 ||
           type == gguf::TensorType::Q4_K ||
           type == gguf::TensorType::Q5_K ||
           type == gguf::TensorType::Q6_K;
}

bool gguf_tensor_is_native_quant(const gguf::File& file, const std::string& name) {
    return file.contains_tensor(name) && gguf_type_is_native_quant(file.tensor_info(name).type);
}

bool env_flag_enabled(const char* name) {
    const char* env = std::getenv(name);
    return env && *env && std::string(env) != "0" && std::string(env) != "false" && std::string(env) != "FALSE";
}

bool repack_gguf_k_quant_to_q4_0() {
    return env_flag_enabled("MOTIFCL_REPACK_GGUF_K_QUANT_TO_Q4_0");
}

bool repack_gguf_k_quant_to_q4_0_col() {
    return env_flag_enabled("MOTIFCL_REPACK_GGUF_K_QUANT_TO_Q4_0_COL");
}

bool repack_gguf_k_quant_to_q4_0_tile8() {
    return env_flag_enabled("MOTIFCL_REPACK_GGUF_K_QUANT_TO_Q4_0_TILE8") ||
           (repack_gguf_k_quant_to_q4_0_col() && !env_flag_enabled("MOTIFCL_DISABLE_Q4_0_TILE8_REPACK"));
}

bool gguf_type_is_k_quant(gguf::TensorType type) {
    return type == gguf::TensorType::Q4_K || type == gguf::TensorType::Q5_K || type == gguf::TensorType::Q6_K;
}

bool q4_0_col_cache_enabled() {
    return !env_flag_enabled("MOTIFCL_DISABLE_Q4_0_COL_CACHE");
}

std::filesystem::path q4_0_col_cache_dir() {
    if (const char* env = std::getenv("MOTIFCL_Q4_0_COL_CACHE_DIR"); env && *env) {
        return std::filesystem::path(env);
    }
    return std::filesystem::temp_directory_path() / "motifcl_q4_0_col_cache";
}

std::filesystem::path q4_0_col_cache_path(const gguf::File& file,
                                          const gguf::TensorInfo& info,
                                          const std::string& name,
                                          int64_t rows,
                                          int64_t cols,
                                          bool direct) {
    std::ostringstream key;
    key << file.path() << '|' << file.total_file_size() << '|' << file.tensor_data_offset()
        << '|' << name << '|' << info.raw_type << '|' << info.offset
        << '|' << rows << 'x' << cols << '|' << (direct ? 'D' : 'T');
    for (auto dim : info.dimensions) key << '|' << dim;
    const auto h0 = std::hash<std::string>{}(key.str());
    const auto h1 = std::hash<std::string>{}(name + key.str());
    std::ostringstream filename;
    filename << std::hex << h0 << "_" << h1 << ".mclq4col";
    return q4_0_col_cache_dir() / filename.str();
}

std::filesystem::path q4_0_tile8_cache_path(const gguf::File& file,
                                            const gguf::TensorInfo& info,
                                            const std::string& name,
                                            int64_t rows,
                                            int64_t cols,
                                            bool direct) {
    std::ostringstream key;
    key << file.path() << '|' << file.total_file_size() << '|' << file.tensor_data_offset()
        << '|' << name << '|' << info.raw_type << '|' << info.offset
        << '|' << rows << 'x' << cols << "|T8|" << (direct ? 'D' : 'T');
    for (auto dim : info.dimensions) key << '|' << dim;
    const auto h0 = std::hash<std::string>{}(key.str());
    const auto h1 = std::hash<std::string>{}(name + key.str());
    std::ostringstream filename;
    filename << std::hex << h0 << "_" << h1 << ".mclq4t8";
    return q4_0_col_cache_dir() / filename.str();
}

Tensor try_read_q4_0_col_cache(Backend& backend,
                               const std::filesystem::path& path,
                               int64_t rows,
                               int64_t cols) {
    if (!q4_0_col_cache_enabled()) return Tensor{};
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) return Tensor{};
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) return Tensor{};
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (!in.good() || std::string(magic, sizeof(magic)) != std::string("MCLQ4C1\n", 8)) return Tensor{};
    auto read_u64 = [&]() {
        std::uint64_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };
    const auto version = read_u64();
    const auto cached_rows = read_u64();
    const auto cached_cols = read_u64();
    const auto block = read_u64();
    const auto packed_bytes = read_u64();
    const auto scale_count = read_u64();
    if (!in.good() || version != 1 || cached_rows != static_cast<std::uint64_t>(rows) ||
        cached_cols != static_cast<std::uint64_t>(cols) || block == 0 ||
        packed_bytes != dtype_storage_nbytes(DType::Q4_0_COL, static_cast<std::size_t>(rows * cols)) ||
        scale_count == 0 || scale_count > (std::numeric_limits<std::uint64_t>::max() / sizeof(float))) {
        return Tensor{};
    }
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(packed_bytes));
    std::vector<float> scales(static_cast<std::size_t>(scale_count));
    if (!packed.empty()) in.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    if (!scales.empty()) in.read(reinterpret_cast<char*>(scales.data()),
                                 static_cast<std::streamsize>(scales.size() * sizeof(float)));
    if (!in.good()) return Tensor{};
    auto tensor = Tensor::from_cpu(backend, {rows, cols}, DType::Q4_0_COL, packed.data());
    auto scale_tensor = Tensor::from_cpu(backend, {static_cast<int64_t>(scales.size())}, DType::F32, scales.data());
    tensor._set_quant_scales(scale_tensor, 3, static_cast<int64_t>(block));
    return tensor;
}

Tensor try_read_q4_0_tile8_cache(Backend& backend,
                                 const std::filesystem::path& path,
                                 int64_t rows,
                                 int64_t cols) {
    if (!q4_0_col_cache_enabled()) return Tensor{};
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) return Tensor{};
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) return Tensor{};
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (!in.good() || std::string(magic, sizeof(magic)) != std::string("MCLQ4T8\n", 8)) return Tensor{};
    auto read_u64 = [&]() {
        std::uint64_t v = 0;
        in.read(reinterpret_cast<char*>(&v), sizeof(v));
        return v;
    };
    const auto version = read_u64();
    const auto cached_rows = read_u64();
    const auto cached_cols = read_u64();
    const auto block = read_u64();
    const auto packed_bytes = read_u64();
    const auto scale_count = read_u64();
    if (!in.good() || version != 1 || cached_rows != static_cast<std::uint64_t>(rows) ||
        cached_cols != static_cast<std::uint64_t>(cols) || block == 0 ||
        packed_bytes != dtype_storage_nbytes(DType::Q4_0_COL, static_cast<std::size_t>(rows * cols)) ||
        scale_count == 0 || scale_count > (std::numeric_limits<std::uint64_t>::max() / sizeof(float))) {
        return Tensor{};
    }
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(packed_bytes));
    std::vector<float> scales(static_cast<std::size_t>(scale_count));
    if (!packed.empty()) in.read(reinterpret_cast<char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    if (!scales.empty()) in.read(reinterpret_cast<char*>(scales.data()),
                                 static_cast<std::streamsize>(scales.size() * sizeof(float)));
    if (!in.good()) return Tensor{};
    auto tensor = Tensor::from_cpu(backend, {rows, cols}, DType::Q4_0_COL, packed.data());
    auto scale_tensor = Tensor::from_cpu(backend, {static_cast<int64_t>(scales.size())}, DType::F32, scales.data());
    tensor._set_quant_scales(scale_tensor, 4, static_cast<int64_t>(block));
    return tensor;
}

void write_q4_0_col_cache(const std::filesystem::path& path,
                          int64_t rows,
                          int64_t cols,
                          int64_t block,
                          const std::vector<std::uint8_t>& packed,
                          const std::vector<float>& scales) {
    if (!q4_0_col_cache_enabled()) return;
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return;
    const auto tmp = path.string() + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out.good()) return;
    const char magic[8] = {'M', 'C', 'L', 'Q', '4', 'C', '1', '\n'};
    auto write_u64 = [&](std::uint64_t v) {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    out.write(magic, sizeof(magic));
    write_u64(1);
    write_u64(static_cast<std::uint64_t>(rows));
    write_u64(static_cast<std::uint64_t>(cols));
    write_u64(static_cast<std::uint64_t>(block));
    write_u64(static_cast<std::uint64_t>(packed.size()));
    write_u64(static_cast<std::uint64_t>(scales.size()));
    if (!packed.empty()) out.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    if (!scales.empty()) out.write(reinterpret_cast<const char*>(scales.data()),
                                   static_cast<std::streamsize>(scales.size() * sizeof(float)));
    out.close();
    if (!out.good()) {
        std::filesystem::remove(tmp, ec);
        return;
    }
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tmp, path, ec);
    if (ec) std::filesystem::remove(tmp, ec);
}

void write_q4_0_tile8_cache(const std::filesystem::path& path,
                            int64_t rows,
                            int64_t cols,
                            int64_t block,
                            const std::vector<std::uint8_t>& packed,
                            const std::vector<float>& scales) {
    if (!q4_0_col_cache_enabled()) return;
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return;
    const auto tmp = path.string() + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out.good()) return;
    const char magic[8] = {'M', 'C', 'L', 'Q', '4', 'T', '8', '\n'};
    auto write_u64 = [&](std::uint64_t v) {
        out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    out.write(magic, sizeof(magic));
    write_u64(1);
    write_u64(static_cast<std::uint64_t>(rows));
    write_u64(static_cast<std::uint64_t>(cols));
    write_u64(static_cast<std::uint64_t>(block));
    write_u64(static_cast<std::uint64_t>(packed.size()));
    write_u64(static_cast<std::uint64_t>(scales.size()));
    if (!packed.empty()) out.write(reinterpret_cast<const char*>(packed.data()), static_cast<std::streamsize>(packed.size()));
    if (!scales.empty()) out.write(reinterpret_cast<const char*>(scales.data()),
                                   static_cast<std::streamsize>(scales.size() * sizeof(float)));
    out.close();
    if (!out.good()) {
        std::filesystem::remove(tmp, ec);
        return;
    }
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tmp, path, ec);
    if (ec) std::filesystem::remove(tmp, ec);
}

std::uint16_t read_le_u16_host(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

float f16_to_f32_host(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x03ffu;
    std::uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03ffu;
            bits = sign | (static_cast<std::uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::int8_t byte_to_i8_host(std::uint8_t value) {
    std::int8_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

void gguf_k_scale_min_host(int j, const std::uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = static_cast<int>(q[j] & 63u);
        m = static_cast<int>(q[j + 4] & 63u);
    } else {
        d = static_cast<int>((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4));
        m = static_cast<int>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

float gguf_k_quant_value_at(const std::vector<std::uint8_t>& data,
                            gguf::TensorType type,
                            std::uint64_t idx) {
    const std::uint64_t block = idx >> 8; // / 256
    const int loc = static_cast<int>(idx & 255u);
    if (type == gguf::TensorType::Q4_K) {
        const auto base = static_cast<std::size_t>(block * 144u);
        const int sub = loc >> 5;
        const int l = loc & 31;
        const int group = sub >> 1;
        const auto* scales = data.data() + base + 4;
        const auto* qs = data.data() + base + 16 + static_cast<std::size_t>(group * 32);
        int sc = 0;
        int mn = 0;
        gguf_k_scale_min_host(sub, scales, sc, mn);
        const int code = (sub & 1) ? static_cast<int>(qs[l] >> 4)
                                   : static_cast<int>(qs[l] & 0x0fu);
        const float d = f16_to_f32_host(read_le_u16_host(data.data() + base));
        const float dmin = f16_to_f32_host(read_le_u16_host(data.data() + base + 2));
        return d * static_cast<float>(sc) * static_cast<float>(code) -
               dmin * static_cast<float>(mn);
    }
    if (type == gguf::TensorType::Q5_K) {
        const auto base = static_cast<std::size_t>(block * 176u);
        const int sub = loc >> 5;
        const int l = loc & 31;
        const int group = sub >> 1;
        const auto* scales = data.data() + base + 4;
        const auto* qh = data.data() + base + 16;
        const auto* ql = data.data() + base + 48 + static_cast<std::size_t>(group * 32);
        int sc = 0;
        int mn = 0;
        gguf_k_scale_min_host(sub, scales, sc, mn);
        int code = (sub & 1) ? static_cast<int>(ql[l] >> 4)
                             : static_cast<int>(ql[l] & 0x0fu);
        if ((static_cast<int>(qh[l]) & (1 << sub)) != 0) code += 16;
        const float d = f16_to_f32_host(read_le_u16_host(data.data() + base));
        const float dmin = f16_to_f32_host(read_le_u16_host(data.data() + base + 2));
        return d * static_cast<float>(sc) * static_cast<float>(code) -
               dmin * static_cast<float>(mn);
    }
    if (type == gguf::TensorType::Q6_K) {
        const auto base = static_cast<std::size_t>(block * 210u);
        const int half_idx = loc >> 7;
        const int l128 = loc & 127;
        const auto* ql = data.data() + base + static_cast<std::size_t>(half_idx * 64);
        const auto* qh = data.data() + base + 128 + static_cast<std::size_t>(half_idx * 32);
        const auto* sc = data.data() + base + 192 + static_cast<std::size_t>(half_idx * 8);
        int l = 0;
        int code = 0;
        int scale_idx = 0;
        if (l128 < 32) {
            l = l128;
            code = static_cast<int>(ql[l] & 0x0fu) | ((static_cast<int>(qh[l] >> 0) & 0x03) << 4);
            scale_idx = (l >> 4) + 0;
        } else if (l128 < 64) {
            l = l128 - 32;
            code = static_cast<int>(ql[l + 32] & 0x0fu) | ((static_cast<int>(qh[l] >> 2) & 0x03) << 4);
            scale_idx = (l >> 4) + 2;
        } else if (l128 < 96) {
            l = l128 - 64;
            code = static_cast<int>((ql[l] >> 4) & 0x0fu) | ((static_cast<int>(qh[l] >> 4) & 0x03) << 4);
            scale_idx = (l >> 4) + 4;
        } else {
            l = l128 - 96;
            code = static_cast<int>((ql[l + 32] >> 4) & 0x0fu) | ((static_cast<int>(qh[l] >> 6) & 0x03) << 4);
            scale_idx = (l >> 4) + 6;
        }
        code -= 32;
        const float d = f16_to_f32_host(read_le_u16_host(data.data() + base + 208));
        return d * static_cast<float>(byte_to_i8_host(sc[scale_idx])) * static_cast<float>(code);
    }
    MCL_CHECK(false, "unsupported GGUF K-quant type for Q4_0_COL repack");
    return 0.0f;
}

Tensor gguf_k_quant_matrix_to_q4_0_col(Backend& backend,
                                       const gguf::File& file,
                                       const std::string& name,
                                       int64_t rows,
                                       int64_t cols,
                                       bool direct) {
    MCL_CHECK(rows > 0 && cols > 0, "Q4_0_COL repack expects non-empty matrix");
    const auto& info = file.tensor_info(name);
    MCL_CHECK(gguf_type_is_k_quant(info.type), "Q4_0_COL repack expects GGUF K-quant tensor");
    MCL_CHECK(cols <= std::numeric_limits<int64_t>::max() / rows,
              "Q4_0_COL repack tensor shape overflows: " + name);
    const auto total = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
    const auto cache_path = q4_0_col_cache_path(file, info, name, rows, cols, direct);
    if (auto cached = try_read_q4_0_col_cache(backend, cache_path, rows, cols); cached.valid()) {
        return cached;
    }
    MCL_CHECK(total <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
              "Q4_0_COL repack tensor is too large on this host: " + name);
    const auto data = file.read_tensor_data(name);
    const auto block_nbytes = gguf::tensor_type_block_nbytes(info.type);
    const auto expected_nbytes = (total / gguf::tensor_type_block_size(info.type)) * block_nbytes;
    MCL_CHECK(total % gguf::tensor_type_block_size(info.type) == 0 &&
                  data.size() == static_cast<std::size_t>(expected_nbytes),
              "Q4_0_COL repack GGUF K-quant byte size mismatch: " + name);

    constexpr int64_t q_block = 32;
    const auto k_rows = static_cast<std::size_t>(rows);
    const auto n_cols = static_cast<std::size_t>(cols);
    const auto blocks_per_col = static_cast<std::size_t>((rows + q_block - 1) / q_block);
    std::vector<std::uint8_t> packed(dtype_storage_nbytes(DType::Q4_0_COL, static_cast<std::size_t>(total)), 0);
    std::vector<float> scales(n_cols * blocks_per_col, 1.0f);
    std::vector<float> tmp(static_cast<std::size_t>(q_block));

    for (std::size_t c = 0; c < n_cols; ++c) {
        for (std::size_t kb = 0; kb < blocks_per_col; ++kb) {
            const std::size_t begin = kb * static_cast<std::size_t>(q_block);
            const std::size_t end = std::min(k_rows, begin + static_cast<std::size_t>(q_block));
            float max_abs = 0.0f;
            for (std::size_t k = begin; k < end; ++k) {
                const std::uint64_t raw_idx = direct
                    ? static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(n_cols) + static_cast<std::uint64_t>(c)
                    : static_cast<std::uint64_t>(c) * static_cast<std::uint64_t>(k_rows) + static_cast<std::uint64_t>(k);
                const float v = gguf_k_quant_value_at(data, info.type, raw_idx);
                tmp[k - begin] = v;
                max_abs = std::max(max_abs, std::fabs(v));
            }
            const float scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? (max_abs / 7.0f) : 1.0f;
            scales[c * blocks_per_col + kb] = scale;
            for (std::size_t k = begin; k < end; ++k) {
                int q = static_cast<int>(std::lrint(tmp[k - begin] / scale));
                q = std::max(-7, std::min(7, q));
                const std::uint8_t code = static_cast<std::uint8_t>((q + 8) & 0x0f);
                const std::size_t pos = c * k_rows + k;
                auto& byte = packed[pos >> 1];
                if ((pos & 1u) == 0) byte = static_cast<std::uint8_t>((byte & 0xf0u) | code);
                else byte = static_cast<std::uint8_t>((byte & 0x0fu) | (code << 4));
            }
        }
    }

    auto tensor = Tensor::from_cpu(backend, {rows, cols}, DType::Q4_0_COL, packed.data());
    auto scale_tensor = Tensor::from_cpu(backend, {static_cast<int64_t>(scales.size())}, DType::F32, scales.data());
    tensor._set_quant_scales(scale_tensor, 3, q_block);
    write_q4_0_col_cache(cache_path, rows, cols, q_block, packed, scales);
    return tensor;
}

Tensor gguf_k_quant_matrix_to_q4_0_tile8(Backend& backend,
                                         const gguf::File& file,
                                         const std::string& name,
                                         int64_t rows,
                                         int64_t cols,
                                         bool direct) {
    MCL_CHECK(rows > 0 && cols > 0, "Q4_0 tile8 repack expects non-empty matrix");
    const auto& info = file.tensor_info(name);
    MCL_CHECK(gguf_type_is_k_quant(info.type), "Q4_0 tile8 repack expects GGUF K-quant tensor");
    MCL_CHECK(cols <= std::numeric_limits<int64_t>::max() / rows,
              "Q4_0 tile8 repack tensor shape overflows: " + name);
    constexpr int64_t q_block = 32;
    constexpr int64_t tile_cols = 8;
    MCL_CHECK((rows % q_block) == 0,
              "Q4_0 tile8 repack requires input rows divisible by 32: " + name);
    const auto total = static_cast<std::uint64_t>(rows) * static_cast<std::uint64_t>(cols);
    const auto cache_path = q4_0_tile8_cache_path(file, info, name, rows, cols, direct);
    if (auto cached = try_read_q4_0_tile8_cache(backend, cache_path, rows, cols); cached.valid()) {
        return cached;
    }
    MCL_CHECK(total <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
              "Q4_0 tile8 repack tensor is too large on this host: " + name);
    const auto data = file.read_tensor_data(name);
    const auto block_nbytes = gguf::tensor_type_block_nbytes(info.type);
    const auto expected_nbytes = (total / gguf::tensor_type_block_size(info.type)) * block_nbytes;
    MCL_CHECK(total % gguf::tensor_type_block_size(info.type) == 0 &&
                  data.size() == static_cast<std::size_t>(expected_nbytes),
              "Q4_0 tile8 repack GGUF K-quant byte size mismatch: " + name);

    const auto k_rows = static_cast<std::size_t>(rows);
    const auto n_cols = static_cast<std::size_t>(cols);
    const auto blocks_per_col = static_cast<std::size_t>(rows / q_block);
    std::vector<std::uint8_t> packed(dtype_storage_nbytes(DType::Q4_0_COL, static_cast<std::size_t>(total)), 0);
    std::vector<float> scales(n_cols * blocks_per_col, 1.0f);
    std::vector<float> tmp(static_cast<std::size_t>(tile_cols * q_block), 0.0f);

    for (std::size_t c0 = 0; c0 < n_cols; c0 += static_cast<std::size_t>(tile_cols)) {
        const std::size_t tile = c0 / static_cast<std::size_t>(tile_cols);
        const std::size_t cols_in_tile = std::min(static_cast<std::size_t>(tile_cols), n_cols - c0);
        for (std::size_t kb = 0; kb < blocks_per_col; ++kb) {
            const std::size_t begin = kb * static_cast<std::size_t>(q_block);
            for (std::size_t tc = 0; tc < cols_in_tile; ++tc) {
                const std::size_t c = c0 + tc;
                float max_abs = 0.0f;
                for (std::size_t kk = 0; kk < static_cast<std::size_t>(q_block); ++kk) {
                    const std::size_t k = begin + kk;
                    const std::uint64_t raw_idx = direct
                        ? static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(n_cols) + static_cast<std::uint64_t>(c)
                        : static_cast<std::uint64_t>(c) * static_cast<std::uint64_t>(k_rows) + static_cast<std::uint64_t>(k);
                    const float v = gguf_k_quant_value_at(data, info.type, raw_idx);
                    tmp[tc * static_cast<std::size_t>(q_block) + kk] = v;
                    max_abs = std::max(max_abs, std::fabs(v));
                }
                const float scale = (max_abs > 0.0f && std::isfinite(max_abs)) ? (max_abs / 7.0f) : 1.0f;
                scales[tile * blocks_per_col * static_cast<std::size_t>(tile_cols) + kb * cols_in_tile + tc] = scale;
            }
            for (std::size_t kk = 0; kk < static_cast<std::size_t>(q_block); ++kk) {
                for (std::size_t tc = 0; tc < cols_in_tile; ++tc) {
                    const float scale =
                        scales[tile * blocks_per_col * static_cast<std::size_t>(tile_cols) + kb * cols_in_tile + tc];
                    int q = static_cast<int>(std::lrint(tmp[tc * static_cast<std::size_t>(q_block) + kk] / scale));
                    q = std::max(-7, std::min(7, q));
                    const std::uint8_t code = static_cast<std::uint8_t>((q + 8) & 0x0f);
                    const std::size_t pos =
                        tile * blocks_per_col * static_cast<std::size_t>(q_block * tile_cols) +
                        (kb * static_cast<std::size_t>(q_block) + kk) * cols_in_tile + tc;
                    auto& byte = packed[pos >> 1];
                    if ((pos & 1u) == 0) byte = static_cast<std::uint8_t>((byte & 0xf0u) | code);
                    else byte = static_cast<std::uint8_t>((byte & 0x0fu) | (code << 4));
                }
            }
        }
    }

    auto tensor = Tensor::from_cpu(backend, {rows, cols}, DType::Q4_0_COL, packed.data());
    auto scale_tensor = Tensor::from_cpu(backend, {static_cast<int64_t>(scales.size())}, DType::F32, scales.data());
    tensor._set_quant_scales(scale_tensor, 4, q_block);
    write_q4_0_tile8_cache(cache_path, rows, cols, q_block, packed, scales);
    return tensor;
}

std::uint64_t shape_numel_u64(const std::vector<int64_t>& shape) {
    std::uint64_t n = 1;
    for (auto dim : shape) {
        MCL_CHECK(dim >= 0, "negative shape dimension");
        n *= static_cast<std::uint64_t>(dim);
    }
    return n;
}

std::vector<float> transpose_2d(const std::vector<float>& src, int64_t rows, int64_t cols) {
    MCL_CHECK(rows >= 0 && cols >= 0, "transpose_2d invalid shape");
    MCL_CHECK(src.size() == static_cast<std::size_t>(rows * cols), "transpose_2d source size mismatch");
    std::vector<float> dst(static_cast<std::size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            dst[static_cast<std::size_t>(c * rows + r)] = src[static_cast<std::size_t>(r * cols + c)];
        }
    }
    return dst;
}

Tensor tensor_from_f32(Backend& backend, const std::vector<int64_t>& shape, const std::vector<float>& values) {
    MCL_CHECK(shape_numel_u64(shape) == values.size(), "tensor_from_f32 shape/value mismatch");
    return Tensor::from_cpu(backend, Shape(shape), DType::F32, values.data());
}

void assign_parameter(Parameter& parameter, Tensor tensor, bool trainable) {
    if (parameter.data.valid()) {
        MCL_CHECK(parameter.data.shape() == tensor.shape(), "loaded parameter shape mismatch");
    }
    parameter.data = std::move(tensor);
    parameter.trainable = trainable;
    parameter.data.set_requires_grad(trainable);
}

bool shape_is(const SafeTensorInfo& info, std::initializer_list<int64_t> dims) {
    return info.shape == std::vector<int64_t>(dims);
}

struct SafeTensorsArchiveLite {
    explicit SafeTensorsArchiveLite(const std::vector<std::string>& paths) {
        files.reserve(paths.size());
        for (const auto& path : paths) {
            files.push_back(SafeTensorsFile::open(path));
            const int file_index = static_cast<int>(files.size() - 1);
            for (const auto& name : files.back().tensor_names()) {
                MCL_CHECK(index.emplace(name, file_index).second, "duplicate safetensors tensor name: " + name);
            }
        }
    }

    bool contains(const std::string& name) const { return index.find(name) != index.end(); }

    const SafeTensorInfo& info(const std::string& name) const {
        auto it = index.find(name);
        MCL_CHECK(it != index.end(), "missing safetensors tensor: " + name);
        return files[static_cast<std::size_t>(it->second)].tensor_info(name);
    }

    std::vector<float> f32(const std::string& name) const {
        auto it = index.find(name);
        MCL_CHECK(it != index.end(), "missing safetensors tensor: " + name);
        return files[static_cast<std::size_t>(it->second)].load_f32_vector(name);
    }

    std::vector<std::string> names() const {
        std::vector<std::string> out;
        out.reserve(index.size());
        for (const auto& kv : index) out.push_back(kv.first);
        std::sort(out.begin(), out.end());
        return out;
    }

private:
    std::vector<SafeTensorsFile> files;
    std::unordered_map<std::string, int> index;
};

std::string first_existing(const SafeTensorsArchiveLite& archive,
                           const std::vector<std::string>& names) {
    for (const auto& name : names) {
        if (archive.contains(name)) return name;
    }
    return {};
}

bool gguf_shape_is_direct_matrix(const gguf::TensorInfo& info, int64_t rows, int64_t cols) {
    return info.dimensions.size() == 2 &&
           static_cast<int64_t>(info.dimensions[0]) == rows &&
           static_cast<int64_t>(info.dimensions[1]) == cols;
}

bool gguf_shape_is_transposed_matrix(const gguf::TensorInfo& info, int64_t rows, int64_t cols) {
    return info.dimensions.size() == 2 &&
           static_cast<int64_t>(info.dimensions[0]) == cols &&
           static_cast<int64_t>(info.dimensions[1]) == rows;
}

Tensor gguf_quantized_matrix_for_shape(Backend& backend,
                                       const gguf::File& file,
                                       const std::string& name,
                                       int64_t rows,
                                       int64_t cols) {
    const auto& info = file.tensor_info(name);
    MCL_CHECK(gguf_type_is_native_quant(info.type),
              "GGUF tensor is not a MotifCL-native quant tensor: " + name);
    const bool direct = gguf_shape_is_direct_matrix(info, rows, cols);
    const bool transposed = gguf_shape_is_transposed_matrix(info, rows, cols);
    if (repack_gguf_k_quant_to_q4_0_tile8() && gguf_type_is_k_quant(info.type)) {
        MCL_CHECK(direct || transposed, "packed GGUF tensor shape mismatch for " + name);
        return gguf_k_quant_matrix_to_q4_0_tile8(backend, file, name, rows, cols, direct);
    }
    if (repack_gguf_k_quant_to_q4_0_col() && gguf_type_is_k_quant(info.type)) {
        MCL_CHECK(direct || transposed, "packed GGUF tensor shape mismatch for " + name);
        return gguf_k_quant_matrix_to_q4_0_col(backend, file, name, rows, cols, direct);
    }
    if (repack_gguf_k_quant_to_q4_0() && gguf_type_is_k_quant(info.type) && cols <= 32768) {
        MCL_CHECK(direct || transposed, "packed GGUF tensor shape mismatch for " + name);
        const auto values = direct ? gguf_tensor_f32(file, name) : transpose_2d(gguf_tensor_f32(file, name), cols, rows);
        Tensor q;
        {
            auto dense = tensor_from_f32(backend, {rows, cols}, values);
            q = quantize_q4_symmetric_cols(dense);
        }
        clear_memory_pool();
        return q;
    }
    if (direct) return file.read_tensor_quantized(backend, name);
    MCL_CHECK(transposed, "packed GGUF tensor shape mismatch for " + name);

    const auto values = transpose_2d(gguf_tensor_f32(file, name), cols, rows);
    auto dense = tensor_from_f32(backend, {rows, cols}, values);
    if (info.type == gguf::TensorType::Q4_0) return quantize_q4_symmetric_cols(dense);
    return quantize_q8_symmetric_cols(dense);
}

bool apply_gguf_quantized_linear(Backend& backend,
                                 const gguf::File& file,
                                 const std::string& name,
                                 Linear& linear,
                                 HFWeightLoadReport& report) {
    if (!gguf_tensor_is_native_quant(file, name)) return false;
    const auto direct = gguf_shape_is_direct_matrix(file.tensor_info(name),
                                                   linear.in_features(), linear.out_features());
    auto q = gguf_quantized_matrix_for_shape(backend, file, name,
                                            linear.in_features(), linear.out_features());
    linear.set_quantized_weight(q);
    report.applied.push_back(name + (direct ? "->packed_quant" : "->packed_quant_repacked_transpose"));
    ++report.loaded_tensors;
    return true;
}

bool apply_gguf_quantized_lm_head(Backend& backend,
                                  const gguf::File& file,
                                  const std::string& name,
                                  ModernGPTModel& model,
                                  HFWeightLoadReport& report) {
    if (!gguf_tensor_is_native_quant(file, name)) return false;
    const auto direct = gguf_shape_is_direct_matrix(file.tensor_info(name),
                                                   model.config.n_embd, model.config.vocab_size);
    auto q = gguf_quantized_matrix_for_shape(backend, file, name,
                                            model.config.n_embd, model.config.vocab_size);
    model.set_quantized_lm_head(q);
    report.applied.push_back(name + (direct ? "->lm_head.packed_quant" : "->lm_head.packed_quant_repacked_transpose"));
    ++report.loaded_tensors;
    return true;
}

bool apply_gguf_quantized_embedding_transposed(Backend& backend,
                                               const gguf::File& file,
                                               const std::string& name,
                                               Embedding& embedding,
                                               int64_t vocab_size,
                                               int64_t embed_dim,
                                               HFWeightLoadReport& report) {
    if (!gguf_tensor_is_native_quant(file, name)) return false;
    const auto& info = file.tensor_info(name);
    if (info.type != gguf::TensorType::Q4_K && info.type != gguf::TensorType::Q5_K) return false;
    MCL_CHECK(gguf_shape_is_direct_matrix(info, embed_dim, vocab_size),
              "GGUF quantized embedding expects transposed [embed_dim,vocab] layout: " + name);
    auto q = file.read_tensor_quantized(backend, name);
    embedding.set_quantized_weight_transposed(q);
    report.applied.push_back(name + "->embedding.packed_quant_transposed");
    ++report.loaded_tensors;
    return true;
}

std::vector<float> gguf_matrix_for_shape(const gguf::File& file,
                                         const std::string& name,
                                         int64_t rows,
                                         int64_t cols) {
    const auto& info = file.tensor_info(name);
    MCL_CHECK(info.dimensions.size() == 2, "expected rank-2 GGUF tensor: " + name);
    const auto values = gguf_tensor_f32(file, name);
    const auto d0 = static_cast<int64_t>(info.dimensions[0]);
    const auto d1 = static_cast<int64_t>(info.dimensions[1]);
    if (d0 == rows && d1 == cols) return values;
    if (d0 == cols && d1 == rows) return transpose_2d(values, cols, rows);
    MCL_CHECK(false, "GGUF tensor shape mismatch for " + name);
    return {};
}

std::vector<float> gguf_vector_for_size(const gguf::File& file, const std::string& name, int64_t size) {
    const auto& info = file.tensor_info(name);
    MCL_CHECK(info.dimensions.size() == 1 && static_cast<int64_t>(info.dimensions[0]) == size,
              "GGUF vector shape mismatch for " + name);
    return gguf_tensor_f32(file, name);
}

void apply_gguf_matrix(Backend& backend,
                       const gguf::File& file,
                       const std::string& name,
                       Parameter& parameter,
                       int64_t rows,
                       int64_t cols,
                       bool trainable,
                       HFWeightLoadReport& report) {
    if (!file.contains_tensor(name)) return;
    assign_parameter(parameter, tensor_from_f32(backend, {rows, cols}, gguf_matrix_for_shape(file, name, rows, cols)), trainable);
    report.applied.push_back(name);
    ++report.loaded_tensors;
}

bool apply_gguf_linear_weight(Backend& backend,
                              const gguf::File& file,
                              const std::string& name,
                              Linear& linear,
                              bool trainable,
                              HFWeightLoadReport& report) {
    if (!file.contains_tensor(name)) return false;
    if (apply_gguf_quantized_linear(backend, file, name, linear, report)) return true;
    apply_gguf_matrix(backend, file, name, linear.weight,
                      linear.in_features(), linear.out_features(), trainable, report);
    return true;
}

void apply_gguf_vector(Backend& backend,
                       const gguf::File& file,
                       const std::string& name,
                       Parameter& parameter,
                       int64_t size,
                       bool trainable,
                       HFWeightLoadReport& report) {
    if (!file.contains_tensor(name)) return;
    assign_parameter(parameter, tensor_from_f32(backend, {size}, gguf_vector_for_size(file, name, size)), trainable);
    report.applied.push_back(name);
    ++report.loaded_tensors;
}

void pack_direct_projection(std::vector<float>& dst,
                            int64_t dst_cols,
                            int64_t col_offset,
                            const std::vector<float>& src,
                            int64_t rows,
                            int64_t cols) {
    MCL_CHECK(dst.size() >= static_cast<std::size_t>(rows * dst_cols), "packed projection destination too small");
    MCL_CHECK(src.size() == static_cast<std::size_t>(rows * cols), "GGUF projection size mismatch");
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            dst[static_cast<std::size_t>(r * dst_cols + col_offset + c)] =
                src[static_cast<std::size_t>(r * cols + c)];
        }
    }
}

bool is_supported_gguf_weight_name(const std::string& name) {
    return name == "token_embd.weight" || name == "output_norm.weight" || name == "output.weight" ||
           name == "per_layer_token_embd.weight" ||
           name == "per_layer_model_proj.weight" ||
           name == "per_layer_proj_norm.weight" ||
           std::regex_match(name, std::regex(R"(^blk\.\d+\.(attn_norm|ffn_norm)\.weight$)")) ||
           std::regex_match(name, std::regex(R"(^blk\.\d+\.attn_(q|k)_norm\.weight$)")) ||
           std::regex_match(name, std::regex(R"(^blk\.\d+\.(post_attention_norm|post_ffw_norm|layer_output_scale)\.weight$)")) ||
           std::regex_match(name, std::regex(R"(^blk\.\d+\.(inp_gate|proj|post_norm)\.weight$)")) ||
           std::regex_match(name, std::regex(R"(^blk\.\d+\.attn_(q|k|v|output)\.(weight|bias)$)")) ||
           std::regex_match(name, std::regex(R"(^blk\.\d+\.ffn_(gate|up|down)\.weight$)"));
}

} // namespace

std::string hf_architecture_name(HFArchitecture architecture) {
    switch (architecture) {
    case HFArchitecture::Auto: return "auto";
    case HFArchitecture::GenericDecoder: return "generic_decoder";
    case HFArchitecture::Gemma: return "gemma";
    case HFArchitecture::Gemma2: return "gemma2";
    case HFArchitecture::Gemma3: return "gemma3";
    case HFArchitecture::Gemma4: return "gemma4";
    case HFArchitecture::Llama: return "llama";
    case HFArchitecture::Mistral: return "mistral";
    case HFArchitecture::Qwen2: return "qwen2";
    case HFArchitecture::Qwen3: return "qwen3";
    case HFArchitecture::Qwen35: return "qwen3.5";
    case HFArchitecture::Phi3: return "phi3";
    case HFArchitecture::Phi4: return "phi4";
    case HFArchitecture::Mixtral: return "mixtral";
    case HFArchitecture::DeepSeek: return "deepseek";
    case HFArchitecture::Falcon: return "falcon";
    case HFArchitecture::GPTNeoX: return "gpt_neox";
    case HFArchitecture::Mamba: return "mamba";
    }
    return "generic_decoder";
}

HFArchitecture parse_hf_architecture(const std::string& value) {
    const auto v = lower_ascii(value);
    if (v.empty() || v == "auto") return HFArchitecture::Auto;
    if (v.find("qwen3.5") != std::string::npos || v.find("qwen35") != std::string::npos ||
        v.find("qwen3_5") != std::string::npos || v.find("qwen_3_5") != std::string::npos) {
        return HFArchitecture::Qwen35;
    }
    if (v.find("qwen3") != std::string::npos || v.find("qwen-3") != std::string::npos) return HFArchitecture::Qwen3;
    if (v.find("gemma4") != std::string::npos || v.find("gemma-4") != std::string::npos) return HFArchitecture::Gemma4;
    if (v.find("gemma3") != std::string::npos || v.find("gemma-3") != std::string::npos) return HFArchitecture::Gemma3;
    if (v.find("gemma2") != std::string::npos || v.find("gemma-2") != std::string::npos) return HFArchitecture::Gemma2;
    if (v.find("phi4") != std::string::npos || v.find("phi-4") != std::string::npos) return HFArchitecture::Phi4;
    if (v.find("phi3") != std::string::npos || v.find("phi-3") != std::string::npos) return HFArchitecture::Phi3;
    if (v.find("mixtral") != std::string::npos) return HFArchitecture::Mixtral;
    if (v.find("deepseek") != std::string::npos) return HFArchitecture::DeepSeek;
    if (v.find("falcon") != std::string::npos) return HFArchitecture::Falcon;
    if (v.find("gpt_neox") != std::string::npos || v.find("gpt-neox") != std::string::npos) return HFArchitecture::GPTNeoX;
    if (v.find("mamba") != std::string::npos) return HFArchitecture::Mamba;
    if (v.find("gemma") != std::string::npos) return HFArchitecture::Gemma;
    if (v.find("mistral") != std::string::npos) return HFArchitecture::Mistral;
    if (v.find("qwen2") != std::string::npos || v.find("qwen") != std::string::npos) return HFArchitecture::Qwen2;
    if (v.find("llama") != std::string::npos || v.find("llm") != std::string::npos) return HFArchitecture::Llama;
    if (v == "generic" || v == "generic_decoder" || v == "decoder") return HFArchitecture::GenericDecoder;
    return HFArchitecture::GenericDecoder;
}

std::vector<HFArchitectureInfo> hf_architecture_registry() {
    return {
        {HFArchitecture::GenericDecoder, "generic_decoder", {"generic", "decoder"}, true, true, true, true, {}, "generic common decoder-only LLaMA-like path"},
        {HFArchitecture::Llama, "llama", {"llama2", "llama3", "llama3.1", "llama3.2", "llama3.3"}, true, true, true, true, {}, "common LLaMA-style decoder-only layout"},
        {HFArchitecture::Mistral, "mistral", {"mistral-nemo", "ministral"}, true, true, true, true, {}, "common Mistral decoder-only layout when tensors match LLaMA-style names"},
        {HFArchitecture::Qwen2, "qwen2", {"qwen", "qwen2.5"}, true, true, true, true, {}, "Qwen2/Qwen2.5 common decoder-only layout"},
        {HFArchitecture::Gemma, "gemma", {"gemma1"}, true, true, true, true, {}, "original Gemma common decoder-only layout"},
        {HFArchitecture::Gemma2, "gemma2", {"gemma-2"}, true, false, false, true, {"Gemma2 soft-capping/sliding-attention details are not fully validated"}, "detected but not treated as numerically supported yet"},
        {HFArchitecture::Gemma3, "gemma3", {"gemma-3"}, true, false, false, true, {"Gemma3 text/VLM variants need dedicated validation"}, "detected for probe/UX only"},
        {HFArchitecture::Gemma4, "gemma4", {"gemma-4"}, true, true, true, true, {}, "Gemma4 dense text-core path; multimodal/MoE variants are blocked by model spec"},
        {HFArchitecture::Qwen3, "qwen3", {"qwen-3"}, true, false, false, true, {"Qwen3 variants need dedicated mapper and model-family validation"}, "detected for probe/UX only"},
        {HFArchitecture::Qwen35, "qwen3.5", {"qwen35", "qwen-3.5"}, true, true, false, true, {}, "HF safetensors text-core path uses HybridGPTModel for DeltaNet/Gated-Attention/MoE layers"},
        {HFArchitecture::Phi3, "phi3", {"phi-3"}, true, false, false, true, {"Phi rotary/config and tensor layout adapter is not implemented"}, "detected for probe/UX only"},
        {HFArchitecture::Phi4, "phi4", {"phi-4"}, true, false, false, true, {"Phi4 tensor layout and any multimodal variants need a dedicated adapter"}, "detected for probe/UX only"},
        {HFArchitecture::Mixtral, "mixtral", {"mixtral-moe"}, true, true, false, true, {}, "HF safetensors MoE text-core path uses HybridGPTModel router/expert execution"},
        {HFArchitecture::DeepSeek, "deepseek", {"deepseek-v2", "deepseek-v3", "deepseek-r1"}, true, false, false, true, {"DeepSeek MoE/MLA-style model details are not implemented"}, "requires dedicated attention/MoE adapters"},
        {HFArchitecture::Falcon, "falcon", {}, true, false, false, true, {"Falcon parallel-attention/tensor layout adapter is not implemented"}, "detected for probe/UX only"},
        {HFArchitecture::GPTNeoX, "gpt_neox", {"gpt-neox", "pythia"}, true, false, false, true, {"GPT-NeoX tensor layout adapter is not implemented"}, "detected for probe/UX only"},
        {HFArchitecture::Mamba, "mamba", {"mamba2"}, false, false, false, false, {"State-space/Mamba blocks are not transformer decoder blocks"}, "outside current transformer runner"},
    };
}

const HFArchitectureInfo& hf_architecture_info(HFArchitecture architecture) {
    static const auto registry = hf_architecture_registry();
    for (const auto& info : registry) {
        if (info.architecture == architecture) return info;
    }
    for (const auto& info : registry) {
        if (info.architecture == HFArchitecture::GenericDecoder) return info;
    }
    MCL_CHECK(false, "HF architecture registry is empty");
    return registry.front();
}

bool hf_architecture_supported_now(HFArchitecture architecture, HFModelFormat format) {
    const auto& info = hf_architecture_info(architecture == HFArchitecture::Auto ? HFArchitecture::GenericDecoder : architecture);
    if (!info.decoder_only || !info.blockers.empty()) return false;
    if (format == HFModelFormat::GGUF) return info.supports_gguf;
    if (format == HFModelFormat::HuggingFaceDirectory || format == HFModelFormat::HuggingFaceConfig) {
        return info.supports_hf_safetensors;
    }
    return info.supports_hf_safetensors || info.supports_gguf;
}

std::vector<std::string> hf_architecture_blockers(HFArchitecture architecture) {
    return hf_architecture_info(architecture == HFArchitecture::Auto ? HFArchitecture::GenericDecoder : architecture).blockers;
}

std::string hf_model_format_name(HFModelFormat format) {
    switch (format) {
    case HFModelFormat::Unknown: return "unknown";
    case HFModelFormat::HuggingFaceDirectory: return "hf_directory";
    case HFModelFormat::HuggingFaceConfig: return "hf_config";
    case HFModelFormat::GGUF: return "gguf";
    }
    return "unknown";
}

std::string hf_chat_template_name(HFChatTemplateKind kind) {
    switch (kind) {
    case HFChatTemplateKind::Auto: return "auto";
    case HFChatTemplateKind::None: return "none";
    case HFChatTemplateKind::Generic: return "generic";
    case HFChatTemplateKind::ChatML: return "chatml";
    case HFChatTemplateKind::Llama2: return "llama2";
    case HFChatTemplateKind::Llama3: return "llama3";
    case HFChatTemplateKind::Mistral: return "mistral";
    case HFChatTemplateKind::Gemma: return "gemma";
    }
    return "generic";
}

HFChatTemplateKind parse_hf_chat_template_kind(const std::string& value) {
    const auto v = lower_ascii(value);
    if (v.empty() || v == "auto") return HFChatTemplateKind::Auto;
    if (v == "none" || v == "raw") return HFChatTemplateKind::None;
    if (v == "chatml" || v.find("qwen") != std::string::npos) return HFChatTemplateKind::ChatML;
    if (v == "llama2" || v.find("llama-2") != std::string::npos) return HFChatTemplateKind::Llama2;
    if (v == "llama" || v == "llama3" || v.find("llama-3") != std::string::npos) return HFChatTemplateKind::Llama3;
    if (v.find("mistral") != std::string::npos || v == "inst") return HFChatTemplateKind::Mistral;
    if (v.find("gemma") != std::string::npos) return HFChatTemplateKind::Gemma;
    if (v == "generic" || v == "plain") return HFChatTemplateKind::Generic;
    return HFChatTemplateKind::Generic;
}

std::string modern_layer_kind_name(ModernLayerKind kind) {
    switch (kind) {
    case ModernLayerKind::FullAttention: return "full_attention";
    case ModernLayerKind::SlidingAttention: return "sliding_attention";
    case ModernLayerKind::GatedDeltaNet: return "gated_delta_net";
    case ModernLayerKind::GatedAttention: return "gated_attention";
    case ModernLayerKind::MoEFFN: return "moe_ffn";
    case ModernLayerKind::VisionProjector: return "vision_projector";
    case ModernLayerKind::AudioProjector: return "audio_projector";
    case ModernLayerKind::SwiGLUFFN: return "swiglu_ffn";
    }
    return "full_attention";
}

ModernLayerKind parse_modern_layer_kind(const std::string& value) {
    const auto v = lower_ascii(value);
    if (v.find("delta") != std::string::npos || v.find("linear_attention") != std::string::npos) {
        return ModernLayerKind::GatedDeltaNet;
    }
    if (v.find("gated_attention") != std::string::npos || v.find("gated-attention") != std::string::npos) {
        return ModernLayerKind::GatedAttention;
    }
    if (v.find("sliding") != std::string::npos || v.find("local") != std::string::npos ||
        v.find("window") != std::string::npos) {
        return ModernLayerKind::SlidingAttention;
    }
    if (v.find("moe") != std::string::npos || v.find("expert") != std::string::npos) {
        return ModernLayerKind::MoEFFN;
    }
    if (v.find("vision") != std::string::npos || v.find("image") != std::string::npos) {
        return ModernLayerKind::VisionProjector;
    }
    if (v.find("audio") != std::string::npos || v.find("speech") != std::string::npos) {
        return ModernLayerKind::AudioProjector;
    }
    if (v.find("swiglu") != std::string::npos || v.find("ffn") != std::string::npos ||
        v.find("mlp") != std::string::npos) {
        return ModernLayerKind::SwiGLUFFN;
    }
    return ModernLayerKind::FullAttention;
}

namespace {

bool contains_int(const std::vector<int>& values, int needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

LayerSpec make_layer_spec(int graph_index,
                          int transformer_layer,
                          ModernLayerKind kind,
                          int sliding_window,
                          const HFTransformerConfig& cfg) {
    LayerSpec layer;
    layer.graph_index = graph_index;
    layer.transformer_layer = transformer_layer;
    layer.kind = kind;
    layer.sliding_window = kind == ModernLayerKind::SlidingAttention ? sliding_window : 0;
    layer.num_experts = kind == ModernLayerKind::MoEFFN ? cfg.num_experts : 0;
    layer.experts_per_token = kind == ModernLayerKind::MoEFFN ? cfg.experts_per_token : 0;
    layer.uses_kv_cache = kind == ModernLayerKind::FullAttention ||
                          kind == ModernLayerKind::SlidingAttention ||
                          kind == ModernLayerKind::GatedAttention;
    layer.uses_state_cache = kind == ModernLayerKind::GatedDeltaNet;
    layer.consumes_per_layer_input = cfg.per_layer_inputs &&
                                     transformer_layer >= 0 &&
                                     (kind == ModernLayerKind::FullAttention ||
                                      kind == ModernLayerKind::SlidingAttention ||
                                      kind == ModernLayerKind::GatedAttention ||
                                      kind == ModernLayerKind::GatedDeltaNet);
    layer.name = transformer_layer >= 0
        ? "layers." + std::to_string(transformer_layer) + "." + modern_layer_kind_name(kind)
        : modern_layer_kind_name(kind);
    return layer;
}

ModernLayerKind attention_kind_for_layer(const HFTransformerConfig& cfg, int layer) {
    if (layer >= 0 && layer < static_cast<int>(cfg.layer_types.size())) {
        const auto parsed = parse_modern_layer_kind(cfg.layer_types[static_cast<std::size_t>(layer)]);
        if (parsed == ModernLayerKind::SlidingAttention ||
            parsed == ModernLayerKind::GatedDeltaNet ||
            parsed == ModernLayerKind::GatedAttention ||
            parsed == ModernLayerKind::FullAttention) {
            return parsed;
        }
    }
    if (contains_int(cfg.global_attention_layers, layer)) return ModernLayerKind::FullAttention;
    if (contains_int(cfg.local_attention_layers, layer)) return ModernLayerKind::SlidingAttention;
    if (cfg.sliding_window > 0 && cfg.architecture == HFArchitecture::Gemma4) {
        const auto global_every = cfg.transformer.n_layer >= 4 ? 4 : 2;
        return ((layer + 1) % global_every == 0) ? ModernLayerKind::FullAttention : ModernLayerKind::SlidingAttention;
    }
    if (cfg.sliding_window > 0 && cfg.architecture != HFArchitecture::GenericDecoder) {
        return ModernLayerKind::SlidingAttention;
    }
    if (cfg.has_gated_delta_net) return ModernLayerKind::GatedDeltaNet;
    if (cfg.has_gated_attention) return ModernLayerKind::GatedAttention;
    return ModernLayerKind::FullAttention;
}

ModernLayerKind ffn_kind_for_layer(const HFTransformerConfig& cfg, int layer) {
    if (layer >= 0 && layer < static_cast<int>(cfg.layer_types.size())) {
        const auto parsed = parse_modern_layer_kind(cfg.layer_types[static_cast<std::size_t>(layer)]);
        if (parsed == ModernLayerKind::MoEFFN || parsed == ModernLayerKind::SwiGLUFFN) return parsed;
    }
    return cfg.has_moe ? ModernLayerKind::MoEFFN : ModernLayerKind::SwiGLUFFN;
}

void add_unique(std::vector<int>& values, int value) {
    if (!contains_int(values, value)) values.push_back(value);
}

} // namespace

ModernModelSpec modern_model_spec_from_config(const HFTransformerConfig& cfg) {
    ModernModelSpec spec;
    spec.architecture = cfg.architecture == HFArchitecture::Auto ? HFArchitecture::GenericDecoder : cfg.architecture;
    spec.architecture_name = hf_architecture_name(spec.architecture);
    spec.transformer = cfg.transformer;
    spec.text_core = !cfg.has_vision_projector && !cfg.has_audio_projector;
    spec.per_layer_inputs = cfg.per_layer_inputs;
    spec.has_moe = cfg.has_moe;
    spec.has_vision_projector = cfg.has_vision_projector;
    spec.has_audio_projector = cfg.has_audio_projector;
    int graph_index = 0;

    if (cfg.has_vision_projector) {
        spec.layers.push_back(make_layer_spec(graph_index++, -1, ModernLayerKind::VisionProjector, 0, cfg));
    }
    if (cfg.has_audio_projector) {
        spec.layers.push_back(make_layer_spec(graph_index++, -1, ModernLayerKind::AudioProjector, 0, cfg));
    }

    for (int layer = 0; layer < cfg.transformer.n_layer; ++layer) {
        const auto attn_kind = attention_kind_for_layer(cfg, layer);
        spec.layers.push_back(make_layer_spec(graph_index++, layer, attn_kind, cfg.sliding_window, cfg));
        const auto ffn_kind = ffn_kind_for_layer(cfg, layer);
        spec.layers.push_back(make_layer_spec(graph_index++, layer, ffn_kind, 0, cfg));
    }

    for (const auto& layer : spec.layers) {
        if (layer.uses_state_cache) spec.has_recurrent_state = true;
    }
    const auto arch_blockers = hf_architecture_blockers(spec.architecture);
    spec.blockers.insert(spec.blockers.end(), arch_blockers.begin(), arch_blockers.end());
    if (spec.has_vision_projector) spec.blockers.push_back("vision projector execution is represented in the model graph but not executable by ModernGPTModel");
    if (spec.has_audio_projector) spec.blockers.push_back("audio projector execution is represented in the model graph but not executable by ModernGPTModel");
    if (spec.per_layer_inputs && !spec.transformer.use_per_layer_inputs) {
        spec.blockers.push_back("per-layer input embeddings/projections were detected but hidden_size_per_layer_input is unavailable");
    }
    if (spec.has_moe) spec.blockers.push_back("MoE FFN routing/expert execution is represented in the model graph but not executable by ModernGPTModel");
    if (spec.has_recurrent_state) spec.blockers.push_back("Gated DeltaNet recurrent state cache is represented in the model graph but not executable by ModernGPTModel");
    if (spec.architecture == HFArchitecture::Qwen35) {
        spec.blockers.push_back("Qwen3.5 hybrid scheduler is outside the dense ModernGPTModel runner");
    }
    return spec;
}

ModernModelSpec load_modern_model_spec_json(const std::string& path, HFArchitecture architecture) {
    return modern_model_spec_from_config(load_hf_transformer_config_json(path, architecture));
}

bool modern_model_spec_runnable_by_modern_gpt(const ModernModelSpec& spec) {
    return spec.blockers.empty();
}

bool modern_model_spec_runnable_by_hybrid(const ModernModelSpec& spec) {
    if (!spec.text_core || spec.per_layer_inputs || spec.has_vision_projector || spec.has_audio_projector) return false;
    if (spec.architecture == HFArchitecture::Mamba) return false;
    for (const auto& layer : spec.layers) {
        if (layer.transformer_layer < 0) return false;
        switch (layer.kind) {
        case ModernLayerKind::FullAttention:
        case ModernLayerKind::SlidingAttention:
        case ModernLayerKind::GatedAttention:
        case ModernLayerKind::GatedDeltaNet:
        case ModernLayerKind::SwiGLUFFN:
        case ModernLayerKind::MoEFFN:
            break;
        case ModernLayerKind::VisionProjector:
        case ModernLayerKind::AudioProjector:
            return false;
        }
    }
    return true;
}

std::vector<std::string> modern_model_spec_blockers(const ModernModelSpec& spec) {
    return spec.blockers;
}

LongContextRuntimeSpec long_context_runtime_spec_from_model_spec(const ModernModelSpec& spec, int page_size) {
    LongContextRuntimeSpec runtime;
    runtime.max_context = spec.transformer.block_size;
    runtime.page_size = page_size > 0 ? page_size : 256;
    for (const auto& layer : spec.layers) {
        if (layer.transformer_layer < 0) continue;
        if (layer.uses_kv_cache) add_unique(runtime.kv_cache_layers, layer.transformer_layer);
        if (layer.kind == ModernLayerKind::SlidingAttention) {
            add_unique(runtime.sliding_window_layers, layer.transformer_layer);
            runtime.sliding_window = std::max(runtime.sliding_window, layer.sliding_window);
        }
        if (layer.uses_state_cache) add_unique(runtime.state_cache_layers, layer.transformer_layer);
    }
    runtime.needs_paged_kv = !runtime.kv_cache_layers.empty() && runtime.max_context > runtime.page_size;
    runtime.needs_sliding_window_cache = !runtime.sliding_window_layers.empty();
    runtime.needs_state_cache = !runtime.state_cache_layers.empty();
    return runtime;
}

std::string format_modern_model_spec(const ModernModelSpec& spec) {
    std::ostringstream out;
    out << "ModernModelSpec\n";
    out << "  architecture: " << spec.architecture_name << "\n";
    out << "  layers: " << spec.layers.size() << "\n";
    out << "  text_core: " << (spec.text_core ? "yes" : "no") << "\n";
    out << "  runnable_by_modern_gpt: " << (modern_model_spec_runnable_by_modern_gpt(spec) ? "yes" : "no") << "\n";
    for (const auto& layer : spec.layers) {
        out << "  [" << layer.graph_index << "] " << layer.name
            << " transformer_layer=" << layer.transformer_layer
            << " kind=" << modern_layer_kind_name(layer.kind);
        if (layer.sliding_window > 0) out << " window=" << layer.sliding_window;
        if (layer.num_experts > 0) out << " experts=" << layer.num_experts
                                       << " top_k=" << layer.experts_per_token;
        if (layer.uses_state_cache) out << " state_cache=yes";
        if (layer.uses_kv_cache) out << " kv_cache=yes";
        if (layer.consumes_per_layer_input) out << " per_layer_input=yes";
        out << "\n";
    }
    for (const auto& blocker : spec.blockers) out << "  blocker: " << blocker << "\n";
    return out.str();
}

HFChatTemplateKind infer_hf_chat_template_kind(HFArchitecture architecture,
                                               const std::string& model_dir_or_tokenizer_config) {
    namespace fs = std::filesystem;
    if (!model_dir_or_tokenizer_config.empty()) {
        fs::path path(model_dir_or_tokenizer_config);
        if (fs::is_directory(path)) path /= "tokenizer_config.json";
        if (fs::exists(path)) {
            const auto text = read_text_file(path.string());
            const auto lower = lower_ascii(text);
            if (lower.find("<|im_start|>") != std::string::npos) return HFChatTemplateKind::ChatML;
            if (lower.find("<|start_header_id|>") != std::string::npos) return HFChatTemplateKind::Llama3;
            if (lower.find("<<sys>>") != std::string::npos) return HFChatTemplateKind::Llama2;
            if (lower.find("[inst]") != std::string::npos) return HFChatTemplateKind::Mistral;
            if (lower.find("<start_of_turn>") != std::string::npos) return HFChatTemplateKind::Gemma;
            if (json_has_key(text, "chat_template")) return HFChatTemplateKind::Generic;
        }
    }
    switch (architecture) {
    case HFArchitecture::Qwen2:
    case HFArchitecture::Qwen3:
    case HFArchitecture::Qwen35:
        return HFChatTemplateKind::ChatML;
    case HFArchitecture::Llama:
        return HFChatTemplateKind::Llama3;
    case HFArchitecture::Mistral:
        return HFChatTemplateKind::Mistral;
    case HFArchitecture::Gemma:
    case HFArchitecture::Gemma2:
    case HFArchitecture::Gemma3:
    case HFArchitecture::Gemma4:
        return HFChatTemplateKind::Gemma;
    default:
        return HFChatTemplateKind::Generic;
    }
}

std::string apply_hf_chat_template(const std::vector<HFChatMessage>& messages,
                                   HFArchitecture architecture,
                                   HFChatTemplateKind kind,
                                   bool add_generation_prompt) {
    if (kind == HFChatTemplateKind::Auto) kind = infer_hf_chat_template_kind(architecture);
    if (kind == HFChatTemplateKind::None) {
        std::string out;
        for (const auto& message : messages) {
            if (!out.empty()) out.push_back('\n');
            out += message.content;
        }
        return out;
    }

    auto role_or = [](const std::string& role, const std::string& fallback) {
        return role.empty() ? fallback : lower_ascii(role);
    };

    if (kind == HFChatTemplateKind::ChatML) {
        std::string out;
        for (const auto& message : messages) {
            out += "<|im_start|>" + role_or(message.role, "user") + "\n" + message.content + "<|im_end|>\n";
        }
        if (add_generation_prompt) out += "<|im_start|>assistant\n";
        return out;
    }
    if (kind == HFChatTemplateKind::Llama3) {
        std::string out = "<|begin_of_text|>";
        for (const auto& message : messages) {
            out += "<|start_header_id|>" + role_or(message.role, "user") +
                   "<|end_header_id|>\n\n" + message.content + "<|eot_id|>";
        }
        if (add_generation_prompt) out += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        return out;
    }
    if (kind == HFChatTemplateKind::Llama2) {
        std::string system;
        std::string out;
        bool open_inst = false;
        bool first_user = true;
        for (const auto& message : messages) {
            const auto role = role_or(message.role, "user");
            if (role == "system") {
                system = message.content;
            } else if (role == "user") {
                if (!out.empty() && !open_inst) out += "</s>";
                out += first_user ? "<s>[INST] " : "[INST] ";
                if (first_user && !system.empty()) {
                    out += "<<SYS>>\n" + system + "\n<</SYS>>\n\n";
                }
                out += message.content + " [/INST]";
                open_inst = true;
                first_user = false;
            } else {
                out += " " + message.content;
                open_inst = false;
            }
        }
        if (!add_generation_prompt && !out.empty() && !open_inst) out += "</s>";
        return out;
    }
    if (kind == HFChatTemplateKind::Gemma) {
        std::string out;
        for (const auto& message : messages) {
            auto role = role_or(message.role, "user");
            if (role == "assistant") role = "model";
            if (role == "system") role = "user";
            out += "<start_of_turn>" + role + "\n" + message.content + "<end_of_turn>\n";
        }
        if (add_generation_prompt) out += "<start_of_turn>model\n";
        return out;
    }
    if (kind == HFChatTemplateKind::Mistral) {
        std::string system;
        std::string out;
        bool open_inst = false;
        bool first_user = true;
        for (const auto& message : messages) {
            const auto role = role_or(message.role, "user");
            if (role == "system") {
                system = message.content;
            } else if (role == "user") {
                if (!out.empty() && !open_inst) out += "</s>";
                out += first_user ? "<s>[INST] " : "[INST] ";
                if (first_user && !system.empty()) out += system + "\n\n";
                out += message.content + " [/INST]";
                open_inst = true;
                first_user = false;
            } else {
                out += " " + message.content;
                open_inst = false;
            }
        }
        if (!add_generation_prompt && !out.empty() && !open_inst) out += "</s>";
        return out;
    }

    std::string out;
    for (const auto& message : messages) {
        const auto role = role_or(message.role, "user");
        if (role == "system") out += "System: " + message.content + "\n";
        else if (role == "assistant") out += "Assistant: " + message.content + "\n";
        else out += "User: " + message.content + "\n";
    }
    if (add_generation_prompt) out += "Assistant: ";
    return out;
}

HFModelProbe probe_hf_transformer_model(const std::string& path, HFArchitecture architecture) {
    namespace fs = std::filesystem;
    HFModelProbe probe;
    probe.source_path = path;
    const fs::path input(path);
    const auto forced_arch = architecture;

    if (is_gguf_path(input)) {
        probe.format = HFModelFormat::GGUF;
        const auto file = gguf::File::open(input.string());
        probe.config = load_hf_transformer_config_gguf(input.string(), forced_arch);
        probe.has_config = true;
        probe.gguf_tensors = static_cast<std::size_t>(file.tensor_count());
        for (const auto& tensor : file.tensors()) {
            if (gguf::tensor_type_is_quantized(tensor.type)) ++probe.gguf_quantized_tensors;
        }
        probe.has_tokenizer = file.has_metadata("tokenizer.ggml.tokens");
        if (hf_architecture_supported_now(probe.config.architecture, probe.format)) {
            const auto expected = expected_gguf_transformer_weight_names(probe.config, !probe.config.tie_word_embeddings);
            probe.has_weights = std::all_of(expected.begin(), expected.end(),
                                            [&](const std::string& name) { return file.contains_tensor(name); });
        } else {
            probe.has_weights = file.contains_tensor("token_embd.weight");
        }
        if (!probe.has_weights) probe.warnings.push_back("some expected GGUF tensors are missing");
    } else {
        fs::path model_root = input;
        fs::path config_path = input;
        if (fs::is_directory(input)) {
            probe.format = HFModelFormat::HuggingFaceDirectory;
            model_root = input;
            config_path = model_root / "config.json";
        } else {
            probe.format = HFModelFormat::HuggingFaceConfig;
            model_root = input.parent_path();
        }
        if (!fs::exists(config_path)) {
            probe.blockers.push_back("config.json not found");
            return probe;
        }
        probe.config = load_hf_transformer_config_json(config_path.string(), forced_arch);
        probe.has_config = true;
        const auto tokenizer = resolve_tokenizer_path(model_root.string());
        probe.has_tokenizer = !tokenizer.empty() && fs::exists(tokenizer);
        auto weights = find_safetensors_files(model_root);
        probe.weight_files = weights.size();
        probe.has_weights = !weights.empty();
        if (!probe.has_weights) probe.warnings.push_back("no .safetensors weights found");
    }

    ModernModelSpec spec;
    bool dense_graph_runnable = false;
    bool hybrid_graph_runnable = false;
    if (probe.has_config) {
        spec = modern_model_spec_from_config(probe.config);
        dense_graph_runnable = modern_model_spec_runnable_by_modern_gpt(spec);
        hybrid_graph_runnable = !dense_graph_runnable && modern_model_spec_runnable_by_hybrid(spec) &&
                                probe.format != HFModelFormat::GGUF;
        if (!dense_graph_runnable && !hybrid_graph_runnable) {
            const auto arch_blockers = hf_architecture_blockers(probe.config.architecture);
            probe.blockers.insert(probe.blockers.end(), arch_blockers.begin(), arch_blockers.end());
            const auto spec_blockers = modern_model_spec_blockers(spec);
            probe.blockers.insert(probe.blockers.end(), spec_blockers.begin(), spec_blockers.end());
        }
    }
    if (!hybrid_graph_runnable && !hf_architecture_supported_now(probe.config.architecture, probe.format)) {
        probe.blockers.push_back("architecture is detected but not supported by the current ModernGPTModel runner");
    }
    if (!probe.has_config) probe.blockers.push_back("model config is unavailable");
    if (!probe.has_weights) probe.blockers.push_back("model weights are unavailable or incomplete");
    if (!probe.has_tokenizer) probe.warnings.push_back("tokenizer metadata not found; byte fallback may be used");

    probe.can_instantiate = probe.has_config &&
                            ((hf_architecture_supported_now(probe.config.architecture, probe.format) && dense_graph_runnable) ||
                             hybrid_graph_runnable);
    probe.can_load_weights = probe.can_instantiate && probe.has_weights;
    probe.can_generate_text = probe.can_load_weights;
    return probe;
}

std::string format_hf_model_probe(const HFModelProbe& probe) {
    std::ostringstream out;
    out << "Model probe\n";
    out << "  source: " << probe.source_path << "\n";
    out << "  format: " << hf_model_format_name(probe.format) << "\n";
    if (probe.has_config) {
        out << "  architecture: " << probe.config.architecture_name << "\n";
        out << "  vocab_size: " << probe.config.transformer.vocab_size << "\n";
        out << "  context_length: " << probe.config.transformer.block_size << "\n";
        out << "  hidden_size: " << probe.config.transformer.n_embd << "\n";
        out << "  layers: " << probe.config.transformer.n_layer << "\n";
        out << "  heads: " << probe.config.transformer.n_head
            << " kv_heads=" << probe.config.transformer.n_kv_head << "\n";
        out << "  mlp_hidden: " << probe.config.transformer.mlp_hidden << "\n";
    }
    out << "  tokenizer: " << (probe.has_tokenizer ? "yes" : "no") << "\n";
    out << "  weights: " << (probe.has_weights ? "yes" : "no");
    if (probe.weight_files > 0) out << " files=" << probe.weight_files;
    if (probe.gguf_tensors > 0) {
        out << " tensors=" << probe.gguf_tensors
            << " quantized=" << probe.gguf_quantized_tensors;
    }
    out << "\n";
    out << "  can_instantiate: " << (probe.can_instantiate ? "yes" : "no") << "\n";
    out << "  can_load_weights: " << (probe.can_load_weights ? "yes" : "no") << "\n";
    out << "  can_generate_text: " << (probe.can_generate_text ? "yes" : "no") << "\n";
    for (const auto& warning : probe.warnings) out << "  warning: " << warning << "\n";
    for (const auto& blocker : probe.blockers) out << "  blocker: " << blocker << "\n";
    return out.str();
}

HFTransformerConfig load_hf_transformer_config_json(const std::string& path, HFArchitecture architecture) {
    const auto text = read_text_file(path);
    const auto text_config = extract_json_object_for_key(text, "text_config");
    HFTransformerConfig out;
    out.architecture = architecture == HFArchitecture::Auto ? parse_hf_architecture(first_architecture_name(text)) : architecture;
    if (out.architecture == HFArchitecture::Auto) out.architecture = HFArchitecture::GenericDecoder;
    out.architecture_name = hf_architecture_name(out.architecture);
    const bool text_core_wrapper = (out.architecture == HFArchitecture::Gemma4 ||
                                    out.architecture == HFArchitecture::Qwen35) &&
                                   !text_config.empty();
    const auto& model_text = text_core_wrapper ? text_config : text;

    const auto gemma_like = load_gemma_config_json(path);
    out.transformer = to_transformer_config(gemma_like);
    const bool gemma_scaled_embeddings = out.architecture == HFArchitecture::Gemma ||
                                         out.architecture == HFArchitecture::Gemma2 ||
                                         out.architecture == HFArchitecture::Gemma3 ||
                                         out.architecture == HFArchitecture::Gemma4;
    out.transformer.embedding_scale = gemma_scaled_embeddings
        ? std::sqrt(static_cast<float>(out.transformer.n_embd))
        : 1.0f;
    out.rms_norm_eps = gemma_like.rms_norm_eps;
    out.transformer.rms_norm_eps = out.rms_norm_eps;
    out.tie_word_embeddings = gemma_like.tie_word_embeddings;
    out.attention_bias = gemma_like.attention_bias;
    out.attention_k_eq_v = gemma_like.attention_k_eq_v;
    out.bos_token_id = gemma_like.bos_token_id;
    out.eos_token_id = gemma_like.eos_token_id;
    out.pad_token_id = gemma_like.pad_token_id;
    out.sliding_window = gemma_like.sliding_window;
    out.transformer.sliding_window = gemma_like.sliding_window;
    out.layer_types = json_string_array_or_empty(model_text, {"layer_types", "layer_type", "attention_types", "block_types"});
    out.local_attention_layers = json_int_array_or_empty(model_text, {"local_attention_layers", "sliding_window_layers"});
    out.global_attention_layers = json_int_array_or_empty(model_text, {"global_attention_layers", "full_attention_layers"});
    out.per_layer_inputs = json_bool_or(model_text, "per_layer_inputs", false) ||
                           json_bool_or(model_text, "per_layer_input_embeddings", false) ||
                           json_has_key(model_text, "per_layer_embedding") ||
                           json_has_key(model_text, "embed_tokens_per_layer") ||
                           json_has_key(model_text, "hidden_size_per_layer_input") ||
                           json_has_key(model_text, "vocab_size_per_layer_input");
    out.transformer.per_layer_input_dim =
        static_cast<int>(json_integer_or(model_text, "hidden_size_per_layer_input", 0));
    out.transformer.per_layer_input_vocab_size =
        static_cast<int>(json_integer_or(model_text, "vocab_size_per_layer_input", out.transformer.vocab_size));
    out.transformer.use_per_layer_inputs = out.per_layer_inputs && out.transformer.per_layer_input_dim > 0;
    out.num_experts = static_cast<int>(json_integer_or(model_text, "num_experts", 0));
    if (out.num_experts <= 0) out.num_experts = static_cast<int>(json_integer_or(model_text, "num_local_experts", 0));
    if (out.num_experts <= 0) out.num_experts = static_cast<int>(json_integer_or(model_text, "n_routed_experts", 0));
    if (out.num_experts <= 0) out.num_experts = static_cast<int>(json_integer_or(model_text, "n_experts", 0));
    out.experts_per_token = static_cast<int>(json_integer_or(model_text, "num_experts_per_tok", 0));
    if (out.experts_per_token <= 0) out.experts_per_token = static_cast<int>(json_integer_or(model_text, "moe_top_k", 0));
    if (out.experts_per_token <= 0) out.experts_per_token = static_cast<int>(json_integer_or(model_text, "num_selected_experts", 0));
    if (out.experts_per_token <= 0) out.experts_per_token = static_cast<int>(json_integer_or(model_text, "top_k", 0));
    const auto lower = lower_ascii(model_text);
    const auto raw_lower = lower_ascii(text);
    out.has_moe = out.num_experts > 0 ||
                  lower.find("\"moe") != std::string::npos ||
                  lower.find("sparse_moe") != std::string::npos ||
                  lower.find("block_sparse_moe") != std::string::npos ||
                  lower.find("num_local_experts") != std::string::npos;
    if (out.has_moe && (out.num_experts <= 0 || out.experts_per_token <= 0)) out.has_moe = false;
    out.has_vision_projector = !text_core_wrapper &&
                               (json_has_key(text, "vision_config") ||
                                json_has_key(text, "vision_projector") ||
                                json_has_key(text, "mm_projector") ||
                                raw_lower.find("vision") != std::string::npos);
    out.has_audio_projector = !text_core_wrapper &&
                              (json_has_key(text, "audio_config") ||
                               json_has_key(text, "audio_projector") ||
                               raw_lower.find("audio") != std::string::npos);
    if (!out.layer_types.empty()) {
        const int local_head_dim = static_cast<int>(json_integer_or(model_text, "head_dim", out.transformer.head_dim));
        const int global_head_dim = static_cast<int>(json_integer_or(model_text, "global_head_dim", local_head_dim));
        out.transformer.layer_head_dims.clear();
        out.transformer.layer_head_dims.reserve(static_cast<std::size_t>(out.transformer.n_layer));
        for (int i = 0; i < out.transformer.n_layer; ++i) {
            const auto kind = attention_kind_for_layer(out, i);
            out.transformer.layer_head_dims.push_back(kind == ModernLayerKind::FullAttention ? global_head_dim : local_head_dim);
        }
        if (!out.transformer.layer_head_dims.empty()) out.transformer.head_dim = out.transformer.layer_head_dims.front();
    }
    out.has_gated_delta_net = lower.find("gated_deltanet") != std::string::npos ||
                              lower.find("gated_delta_net") != std::string::npos ||
                              lower.find("delta_net") != std::string::npos ||
                              lower.find("linear_attention") != std::string::npos;
    out.has_gated_attention = lower.find("gated_attention") != std::string::npos ||
                              lower.find("gated-attention") != std::string::npos;
    return out;
}

HFTransformerConfig load_hf_transformer_config_gguf(const std::string& path, HFArchitecture architecture) {
    const auto file = gguf::File::open(path);
    const auto gguf_arch_name = file.metadata_string_or("general.architecture", "generic_decoder");
    HFTransformerConfig out;
    out.architecture = architecture == HFArchitecture::Auto ? parse_hf_architecture(gguf_arch_name) : architecture;
    if (out.architecture == HFArchitecture::Auto) out.architecture = HFArchitecture::GenericDecoder;
    out.architecture_name = hf_architecture_name(out.architecture);

    const auto prefix = lower_ascii(gguf_arch_name.empty() ? out.architecture_name : gguf_arch_name);
    const auto tokens = metadata_string_array_or_empty(file, "tokenizer.ggml.tokens");
    out.transformer.vocab_size = static_cast<int>(metadata_integer_or(file,
        {"tokenizer.ggml.vocab_size", prefix + ".vocab_size"}, tokens.size()));
    if (!tokens.empty()) out.transformer.vocab_size = static_cast<int>(tokens.size());
    out.transformer.block_size = static_cast<int>(metadata_integer_or(file,
        {prefix + ".context_length", prefix + ".max_position_embeddings"}, 0));
    out.transformer.n_embd = static_cast<int>(metadata_integer_or(file,
        {prefix + ".embedding_length", prefix + ".hidden_size"}, 0));
    out.transformer.n_layer = static_cast<int>(metadata_integer_or(file,
        {prefix + ".block_count", prefix + ".num_hidden_layers"}, 0));
    out.transformer.n_head = static_cast<int>(metadata_integer_or(file,
        {prefix + ".attention.head_count", prefix + ".attention_head_count"}, 0));
    out.transformer.n_kv_head = static_cast<int>(metadata_integer_or(file,
        {prefix + ".attention.head_count_kv", prefix + ".attention_head_count_kv"}, out.transformer.n_head));
    const auto mlp_lengths = metadata_integer_array_or_empty(file, prefix + ".feed_forward_length");
    if (!mlp_lengths.empty()) {
        out.transformer.mlp_hidden = static_cast<int>(mlp_lengths.front());
        if (static_cast<int>(mlp_lengths.size()) == out.transformer.n_layer) {
            out.transformer.layer_mlp_hiddens.clear();
            out.transformer.layer_mlp_hiddens.reserve(mlp_lengths.size());
            for (const auto value : mlp_lengths) {
                out.transformer.layer_mlp_hiddens.push_back(static_cast<int>(value));
            }
        }
    } else {
        out.transformer.mlp_hidden = static_cast<int>(metadata_integer_or(file,
            {prefix + ".intermediate_size"}, 0));
    }
    out.rms_norm_eps = static_cast<float>(metadata_float_or(file,
        {prefix + ".attention.layer_norm_rms_epsilon", prefix + ".attention.layer_norm_epsilon"}, 1e-6));
    out.transformer.rms_norm_eps = out.rms_norm_eps;
    out.transformer.rope_theta = static_cast<float>(metadata_float_or(file,
        {prefix + ".rope.freq_base", "rope.freq_base"}, 10000.0));
    const int default_head_dim = out.transformer.n_head > 0
        ? out.transformer.n_embd / out.transformer.n_head
        : 0;
    const int full_head_dim = static_cast<int>(metadata_integer_or(file,
        {prefix + ".attention.key_length", prefix + ".rope.dimension_count"}, default_head_dim));
    const int sliding_head_dim = static_cast<int>(metadata_integer_or(file,
        {prefix + ".attention.key_length_swa", prefix + ".rope.dimension_count_swa"}, full_head_dim));
    out.transformer.head_dim = full_head_dim;
    out.transformer.rotary_dim = static_cast<int>(metadata_integer_or(file,
        {prefix + ".rope.dimension_count", prefix + ".attention.key_length"}, full_head_dim));
    out.transformer.dropout = 0.0f;
    out.transformer.use_rope = true;
    out.transformer.use_swiglu = true;
    out.transformer.causal = true;
    out.transformer.learned_position_embeddings = false;
    out.transformer.skip_weight_init = true;
    out.transformer.embedding_scale =
        (out.architecture == HFArchitecture::Gemma ||
         out.architecture == HFArchitecture::Gemma2 ||
         out.architecture == HFArchitecture::Gemma3 ||
         out.architecture == HFArchitecture::Gemma4 ||
         prefix.find("gemma") != std::string::npos)
            ? std::sqrt(static_cast<float>(out.transformer.n_embd))
            : 1.0f;
    out.attention_bias = file.contains_tensor("blk.0.attn_q.bias") ||
                         file.contains_tensor("blk.0.attn_k.bias") ||
                         file.contains_tensor("blk.0.attn_v.bias") ||
                         file.contains_tensor("blk.0.attn_output.bias");
    out.transformer.use_qkv_bias = out.attention_bias;
    out.tie_word_embeddings = !file.contains_tensor("output.weight") && file.contains_tensor("token_embd.weight");
    out.transformer.use_qk_norm = file.contains_tensor("blk.0.attn_q_norm.weight") ||
                                  file.contains_tensor("blk.0.attn_k_norm.weight");
    out.transformer.use_post_attention_norm = file.contains_tensor("blk.0.post_attention_norm.weight");
    out.transformer.use_post_ffw_norm = file.contains_tensor("blk.0.post_ffw_norm.weight");
    out.transformer.use_layer_output_scale = file.contains_tensor("blk.0.layer_output_scale.weight");
    out.transformer.per_layer_input_dim = static_cast<int>(metadata_integer_or(file,
        {prefix + ".embedding_length_per_layer_input", prefix + ".hidden_size_per_layer_input"}, 0));
    out.transformer.per_layer_input_vocab_size = static_cast<int>(metadata_integer_or(file,
        {prefix + ".vocab_size_per_layer_input"}, out.transformer.vocab_size));
    if (file.contains_tensor("per_layer_token_embd.weight")) {
        const auto& ple_info = file.tensor_info("per_layer_token_embd.weight");
        if (ple_info.dimensions.size() == 2 && out.transformer.n_layer > 0) {
            const auto d0 = static_cast<int64_t>(ple_info.dimensions[0]);
            const auto d1 = static_cast<int64_t>(ple_info.dimensions[1]);
            if (out.transformer.per_layer_input_dim <= 0 && d0 % out.transformer.n_layer == 0) {
                out.transformer.per_layer_input_dim = static_cast<int>(d0 / out.transformer.n_layer);
            }
            if (d1 > 0) out.transformer.per_layer_input_vocab_size = static_cast<int>(d1);
        }
    }
    out.transformer.use_per_layer_inputs = file.contains_tensor("per_layer_token_embd.weight") ||
                                           file.contains_tensor("per_layer_model_proj.weight") ||
                                           file.contains_tensor("blk.0.inp_gate.weight");
    out.per_layer_inputs = out.transformer.use_per_layer_inputs;
    out.bos_token_id = static_cast<int>(metadata_integer_or(file, {"tokenizer.ggml.bos_token_id"}, 1));
    out.eos_token_id = static_cast<int>(metadata_integer_or(file, {"tokenizer.ggml.eos_token_id"}, 2));
    out.pad_token_id = static_cast<int>(metadata_integer_or(file, {"tokenizer.ggml.padding_token_id"}, 0));
    out.sliding_window = static_cast<int>(metadata_integer_or(file,
        {prefix + ".attention.sliding_window", prefix + ".sliding_window"}, 0));
    out.transformer.sliding_window = out.sliding_window;
    if (out.architecture == HFArchitecture::Gemma4 || prefix == "gemma4") {
        const auto sliding_pattern = metadata_bool_array_or_empty(file, prefix + ".attention.sliding_window_pattern");
        if (static_cast<int>(sliding_pattern.size()) == out.transformer.n_layer) {
            const float full_rope_theta = static_cast<float>(metadata_float_or(file,
                {prefix + ".rope.freq_base", "rope.freq_base"}, out.transformer.rope_theta));
            const float sliding_rope_theta = static_cast<float>(metadata_float_or(file,
                {prefix + ".rope.freq_base_swa"}, full_rope_theta));
            out.layer_types.clear();
            out.local_attention_layers.clear();
            out.global_attention_layers.clear();
            out.transformer.layer_head_dims.clear();
            out.transformer.layer_rope_thetas.clear();
            out.layer_types.reserve(sliding_pattern.size());
            out.transformer.layer_head_dims.reserve(sliding_pattern.size());
            out.transformer.layer_rope_thetas.reserve(sliding_pattern.size());
            for (int i = 0; i < static_cast<int>(sliding_pattern.size()); ++i) {
                if (sliding_pattern[static_cast<std::size_t>(i)]) {
                    out.layer_types.push_back("sliding_attention");
                    out.local_attention_layers.push_back(i);
                    out.transformer.layer_head_dims.push_back(sliding_head_dim);
                    out.transformer.layer_rope_thetas.push_back(sliding_rope_theta);
                } else {
                    out.layer_types.push_back("full_attention");
                    out.global_attention_layers.push_back(i);
                    out.transformer.layer_head_dims.push_back(full_head_dim);
                    out.transformer.layer_rope_thetas.push_back(full_rope_theta);
                }
            }
            if (!out.transformer.layer_head_dims.empty()) {
                out.transformer.head_dim = out.transformer.layer_head_dims.front();
            }
        }
    }

    MCL_CHECK(out.transformer.vocab_size > 0, "GGUF config missing tokenizer vocab_size/tokens");
    MCL_CHECK(out.transformer.block_size > 0, "GGUF config missing context_length");
    MCL_CHECK(out.transformer.n_embd > 0, "GGUF config missing embedding_length");
    MCL_CHECK(out.transformer.n_layer > 0, "GGUF config missing block_count");
    MCL_CHECK(out.transformer.n_head > 0, "GGUF config missing attention.head_count");
    MCL_CHECK(out.transformer.n_kv_head > 0, "GGUF config missing attention.head_count_kv");
    MCL_CHECK(out.transformer.mlp_hidden > 0, "GGUF config missing feed_forward_length");
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
    out.head_dim = cfg.transformer.head_dim;
    out.rms_norm_eps = cfg.rms_norm_eps;
    out.rope_theta = cfg.transformer.rope_theta;
    out.attention_dropout = cfg.transformer.dropout;
    out.attention_bias = cfg.attention_bias || cfg.transformer.use_qkv_bias;
    out.attention_k_eq_v = cfg.attention_k_eq_v;
    out.tie_word_embeddings = cfg.tie_word_embeddings;
    out.bos_token_id = cfg.bos_token_id;
    out.eos_token_id = cfg.eos_token_id;
    out.pad_token_id = cfg.pad_token_id;
    out.sliding_window = cfg.sliding_window;
    return out;
}

ModernGPTModel make_hf_transformer_model(Backend& backend, const HFTransformerConfig& cfg) {
    const auto spec = modern_model_spec_from_config(cfg);
    if (!modern_model_spec_runnable_by_modern_gpt(spec)) {
        MCL_CHECK(false, "HF model graph is not executable by ModernGPTModel; first blocker: " + spec.blockers.front());
    }
    ModernGPTModel model(backend, cfg.transformer);
    model.final_norm.eps = cfg.rms_norm_eps;
    if (model.per_layer_projection_norm) model.per_layer_projection_norm->eps = cfg.rms_norm_eps;
    for (auto& block : model.blocks) {
        block->norm1().eps = cfg.rms_norm_eps;
        block->norm2().eps = cfg.rms_norm_eps;
        block->attention().q_norm().eps = cfg.rms_norm_eps;
        block->attention().k_norm().eps = cfg.rms_norm_eps;
        block->post_attention_norm().eps = cfg.rms_norm_eps;
        block->post_ffw_norm().eps = cfg.rms_norm_eps;
        if (auto* post_ple = block->post_per_layer_norm()) post_ple->eps = cfg.rms_norm_eps;
    }
    for (const auto& layer : spec.layers) {
        if (layer.kind == ModernLayerKind::SlidingAttention && layer.transformer_layer >= 0) {
            model.set_layer_attention_window(layer.transformer_layer, layer.sliding_window);
        } else if ((layer.kind == ModernLayerKind::FullAttention ||
                    layer.kind == ModernLayerKind::GatedAttention) &&
                   layer.transformer_layer >= 0) {
            model.set_layer_attention_window(layer.transformer_layer, 0);
        }
    }
    return model;
}

std::vector<HybridLayerConfig> hybrid_layer_configs_from_model_spec(const ModernModelSpec& spec) {
    MCL_CHECK(modern_model_spec_runnable_by_hybrid(spec),
              "HF model graph is not executable by HybridGPTModel");
    std::vector<HybridLayerConfig> layers(static_cast<std::size_t>(spec.transformer.n_layer));
    for (const auto& layer : spec.layers) {
        MCL_CHECK(layer.transformer_layer >= 0 &&
                      layer.transformer_layer < static_cast<int>(layers.size()),
                  "Hybrid layer spec index out of range");
        auto& dst = layers[static_cast<std::size_t>(layer.transformer_layer)];
        switch (layer.kind) {
        case ModernLayerKind::FullAttention:
            dst.attention = HybridAttentionKind::FullAttention;
            dst.sliding_window = 0;
            break;
        case ModernLayerKind::SlidingAttention:
            dst.attention = HybridAttentionKind::SlidingAttention;
            dst.sliding_window = layer.sliding_window > 0 ? layer.sliding_window : spec.transformer.sliding_window;
            break;
        case ModernLayerKind::GatedAttention:
            dst.attention = HybridAttentionKind::GatedAttention;
            dst.sliding_window = layer.sliding_window;
            break;
        case ModernLayerKind::GatedDeltaNet:
            dst.attention = HybridAttentionKind::GatedDeltaNet;
            break;
        case ModernLayerKind::SwiGLUFFN:
            dst.ffn = HybridFFNKind::SwiGLUFFN;
            break;
        case ModernLayerKind::MoEFFN:
            dst.ffn = HybridFFNKind::MoEFFN;
            dst.num_experts = layer.num_experts;
            dst.experts_per_token = layer.experts_per_token;
            break;
        case ModernLayerKind::VisionProjector:
        case ModernLayerKind::AudioProjector:
            MCL_CHECK(false, "projector layers are not executable by HybridGPTModel");
        }
    }
    return layers;
}

HybridGPTModel make_hf_hybrid_transformer_model(Backend& backend, const HFTransformerConfig& cfg) {
    auto spec = modern_model_spec_from_config(cfg);
    MCL_CHECK(modern_model_spec_runnable_by_hybrid(spec),
              "HF model graph is not executable by HybridGPTModel");
    auto layer_configs = hybrid_layer_configs_from_model_spec(spec);
    HybridGPTModel model(backend, cfg.transformer, layer_configs);
    model.final_norm.eps = cfg.rms_norm_eps;
    for (auto& block : model.blocks) {
        block->norm1().eps = cfg.rms_norm_eps;
        block->norm2().eps = cfg.rms_norm_eps;
        if (auto* attn = block->attention()) {
            attn->q_norm().eps = cfg.rms_norm_eps;
            attn->k_norm().eps = cfg.rms_norm_eps;
        }
        if (auto* gated = block->gated_attention()) {
            gated->attention.q_norm().eps = cfg.rms_norm_eps;
            gated->attention.k_norm().eps = cfg.rms_norm_eps;
        }
    }
    return model;
}

HFWeightName map_hf_transformer_weight_name(HFArchitecture architecture, const std::string& hf_name) {
    if (architecture == HFArchitecture::Qwen35 ||
        architecture == HFArchitecture::Mixtral ||
        architecture == HFArchitecture::DeepSeek) return map_qwen35_weight_name(hf_name);
    MCL_CHECK(architecture_uses_llama_like_weights(architecture),
              "HF architecture weight mapping is not implemented: " + hf_architecture_name(architecture));
    return map_gemma_hf_weight_name(hf_name);
}

HFWeightName map_gguf_transformer_weight_name(HFArchitecture architecture, const std::string& gguf_name) {
    MCL_CHECK(architecture_uses_llama_like_weights(architecture),
              "GGUF architecture weight mapping is not implemented: " + hf_architecture_name(architecture));
    if (gguf_name == "token_embd.weight") return {gguf_name, "token_embedding.weight", "token_embedding", -1};
    if (gguf_name == "output_norm.weight") return {gguf_name, "final_norm.weight", "final_norm", -1};
    if (gguf_name == "output.weight") return {gguf_name, "lm_head.weight", "lm_head", -1};

    std::smatch m;
    const std::regex layer_re(R"(^blk\.(\d+)\.(.+)$)");
    if (!std::regex_match(gguf_name, m, layer_re)) return {gguf_name, gguf_name, "unknown", -1};
    const int layer = std::stoi(m[1].str());
    const std::string suffix = m[2].str();
    const std::string base = "blocks." + std::to_string(layer) + ".";
    if (suffix == "attn_norm.weight") return {gguf_name, base + "norm1.weight", "input_layernorm", layer};
    if (suffix == "ffn_norm.weight") return {gguf_name, base + "norm2.weight", "post_attention_layernorm", layer};
    if (suffix == "attn_q.weight") return {gguf_name, base + "attention.q_proj.weight", "q_proj", layer};
    if (suffix == "attn_k.weight") return {gguf_name, base + "attention.k_proj.weight", "k_proj", layer};
    if (suffix == "attn_v.weight") return {gguf_name, base + "attention.v_proj.weight", "v_proj", layer};
    if (suffix == "attn_output.weight") return {gguf_name, base + "attention.o_proj.weight", "o_proj", layer};
    if (suffix == "attn_q.bias") return {gguf_name, base + "attention.q_proj.bias", "q_proj_bias", layer};
    if (suffix == "attn_k.bias") return {gguf_name, base + "attention.k_proj.bias", "k_proj_bias", layer};
    if (suffix == "attn_v.bias") return {gguf_name, base + "attention.v_proj.bias", "v_proj_bias", layer};
    if (suffix == "attn_output.bias") return {gguf_name, base + "attention.o_proj.bias", "o_proj_bias", layer};
    if (suffix == "ffn_gate.weight") return {gguf_name, base + "mlp.gate_proj.weight", "gate_proj", layer};
    if (suffix == "ffn_up.weight") return {gguf_name, base + "mlp.up_proj.weight", "up_proj", layer};
    if (suffix == "ffn_down.weight") return {gguf_name, base + "mlp.down_proj.weight", "down_proj", layer};
    return {gguf_name, gguf_name, "unknown", layer};
}

std::vector<std::string> expected_hf_transformer_weight_names(const HFTransformerConfig& cfg, bool include_lm_head) {
    MCL_CHECK(architecture_uses_llama_like_weights(cfg.architecture),
              "HF architecture expected weights are not implemented: " + cfg.architecture_name);
    return expected_gemma_hf_weight_names(to_gemma_compatible_config(cfg), include_lm_head);
}

std::vector<std::string> expected_gguf_transformer_weight_names(const HFTransformerConfig& cfg, bool include_lm_head) {
    MCL_CHECK(architecture_uses_llama_like_weights(cfg.architecture),
              "GGUF architecture expected weights are not implemented: " + cfg.architecture_name);
    std::vector<std::string> names{"token_embd.weight", "output_norm.weight"};
    if (cfg.transformer.use_per_layer_inputs) {
        names.push_back("per_layer_token_embd.weight");
        names.push_back("per_layer_model_proj.weight");
        names.push_back("per_layer_proj_norm.weight");
    }
    if (include_lm_head) names.push_back("output.weight");
    for (int i = 0; i < cfg.transformer.n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        names.push_back(p + "attn_norm.weight");
        names.push_back(p + "ffn_norm.weight");
        if (cfg.transformer.use_qk_norm) {
            names.push_back(p + "attn_q_norm.weight");
            names.push_back(p + "attn_k_norm.weight");
        }
        if (cfg.transformer.use_post_attention_norm) names.push_back(p + "post_attention_norm.weight");
        if (cfg.transformer.use_post_ffw_norm) names.push_back(p + "post_ffw_norm.weight");
        if (cfg.transformer.use_layer_output_scale) names.push_back(p + "layer_output_scale.weight");
        if (cfg.transformer.use_per_layer_inputs) {
            names.push_back(p + "inp_gate.weight");
            names.push_back(p + "proj.weight");
            names.push_back(p + "post_norm.weight");
        }
        names.push_back(p + "attn_q.weight");
        names.push_back(p + "attn_k.weight");
        names.push_back(p + "attn_v.weight");
        names.push_back(p + "attn_output.weight");
        if (cfg.attention_bias || cfg.transformer.use_qkv_bias) {
            names.push_back(p + "attn_q.bias");
            names.push_back(p + "attn_k.bias");
            names.push_back(p + "attn_v.bias");
            names.push_back(p + "attn_output.bias");
        }
        names.push_back(p + "ffn_gate.weight");
        names.push_back(p + "ffn_up.weight");
        names.push_back(p + "ffn_down.weight");
    }
    return names;
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

HFWeightLoadReport load_hf_hybrid_transformer_weights(Backend& backend,
                                                      HybridGPTModel& model,
                                                      const std::vector<std::string>& safetensors_paths,
                                                      const HFTransformerConfig& cfg,
                                                      bool strict,
                                                      bool trainable) {
    MCL_CHECK(model.config.vocab_size == cfg.transformer.vocab_size &&
                  model.config.n_embd == cfg.transformer.n_embd &&
                  model.config.n_layer == cfg.transformer.n_layer &&
                  model.config.n_head == cfg.transformer.n_head &&
                  model.config.n_kv_head == cfg.transformer.n_kv_head,
              "HF config does not match HybridGPTModel config");
    SafeTensorsArchiveLite archive(safetensors_paths);
    HFWeightLoadReport report;

    auto apply_direct = [&](const std::vector<std::string>& aliases,
                            Parameter& parameter,
                            bool required) -> bool {
        const auto name = first_existing(archive, aliases);
        if (name.empty()) {
            if (required && !aliases.empty()) report.missing.push_back(aliases.front());
            return false;
        }
        assign_parameter(parameter, tensor_from_f32(backend, archive.info(name).shape, archive.f32(name)), trainable);
        report.applied.push_back(name);
        ++report.loaded_tensors;
        return true;
    };

    auto apply_linear = [&](const std::vector<std::string>& aliases,
                            Linear& linear,
                            bool required) -> bool {
        const auto name = first_existing(archive, aliases);
        if (name.empty()) {
            if (required && !aliases.empty()) report.missing.push_back(aliases.front());
            return false;
        }
        const auto& info = archive.info(name);
        MCL_CHECK(info.shape.size() == 2, "expected rank-2 HF linear weight: " + name);
        const int64_t in = linear.in_features();
        const int64_t out = linear.out_features();
        Tensor tensor;
        if (shape_is(info, {out, in})) {
            tensor = tensor_from_f32(backend, {in, out}, transpose_2d(archive.f32(name), out, in));
        } else if (shape_is(info, {in, out})) {
            tensor = tensor_from_f32(backend, {in, out}, archive.f32(name));
        } else {
            MCL_CHECK(false, "HF linear weight shape mismatch: " + name);
        }
        assign_parameter(linear.weight, std::move(tensor), trainable);
        report.applied.push_back(name);
        ++report.loaded_tensors;
        return true;
    };

    auto apply_bias = [&](const std::vector<std::string>& aliases,
                          Linear& linear) -> bool {
        if (!linear.has_bias()) return false;
        return apply_direct(aliases, linear.bias, false);
    };

    auto attention_ptr = [](HybridTransformerBlock& block) -> ModernSelfAttention* {
        if (block.attention()) return block.attention();
        if (block.gated_attention()) return &block.gated_attention()->attention;
        return nullptr;
    };

    auto apply_attention = [&](int layer, ModernSelfAttention& attention, bool required) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        apply_linear({p + "self_attn.q_proj.weight", p + "attention.q_proj.weight", p + "attn.q_proj.weight"},
                     attention.q_proj(), required);
        apply_linear({p + "self_attn.k_proj.weight", p + "attention.k_proj.weight", p + "attn.k_proj.weight"},
                     attention.k_proj(), required);
        apply_linear({p + "self_attn.v_proj.weight", p + "attention.v_proj.weight", p + "attn.v_proj.weight"},
                     attention.v_proj(), required);
        apply_linear({p + "self_attn.o_proj.weight", p + "attention.o_proj.weight", p + "attn.o_proj.weight",
                      p + "self_attn.out_proj.weight", p + "attention.out_proj.weight"},
                     attention.o_proj(), required);
        apply_bias({p + "self_attn.q_proj.bias", p + "attention.q_proj.bias", p + "attn.q_proj.bias"},
                   attention.q_proj());
        apply_bias({p + "self_attn.k_proj.bias", p + "attention.k_proj.bias", p + "attn.k_proj.bias"},
                   attention.k_proj());
        apply_bias({p + "self_attn.v_proj.bias", p + "attention.v_proj.bias", p + "attn.v_proj.bias"},
                   attention.v_proj());
        apply_bias({p + "self_attn.o_proj.bias", p + "attention.o_proj.bias", p + "attn.o_proj.bias",
                    p + "self_attn.out_proj.bias", p + "attention.out_proj.bias"},
                   attention.o_proj());
    };

    auto apply_delta = [&](int layer, GatedDeltaNetLayer& delta, bool required) {
        const std::string p = "model.layers." + std::to_string(layer) + ".";
        apply_linear({p + "linear_attn.q_proj.weight", p + "gated_delta_net.q_proj.weight",
                      p + "delta.q_proj.weight"},
                     delta.q_proj, required);
        apply_linear({p + "linear_attn.k_proj.weight", p + "gated_delta_net.k_proj.weight",
                      p + "delta.k_proj.weight"},
                     delta.k_proj, required);
        apply_linear({p + "linear_attn.v_proj.weight", p + "gated_delta_net.v_proj.weight",
                      p + "delta.v_proj.weight"},
                     delta.v_proj, required);
        apply_linear({p + "linear_attn.gate_proj.weight", p + "gated_delta_net.gate_proj.weight",
                      p + "delta.gate_proj.weight"},
                     delta.gate_proj, required);
        apply_linear({p + "linear_attn.o_proj.weight", p + "gated_delta_net.o_proj.weight",
                      p + "delta.o_proj.weight"},
                     delta.o_proj, required);
    };

    auto linear_matrix_values = [&](const std::string& name, int64_t in, int64_t out) -> std::vector<float> {
        const auto& info = archive.info(name);
        MCL_CHECK(info.shape.size() == 2, "expected rank-2 HF weight: " + name);
        if (shape_is(info, {out, in})) return transpose_2d(archive.f32(name), out, in);
        if (shape_is(info, {in, out})) return archive.f32(name);
        MCL_CHECK(false, "HF weight shape mismatch: " + name);
        return {};
    };

    auto expert_aliases = [](const std::string& p, int expert, const std::string& proj) {
        const std::string e = std::to_string(expert);
        std::string mixtral_proj = proj == "gate_proj" ? "w1" : (proj == "up_proj" ? "w3" : "w2");
        return std::vector<std::string>{
            p + "mlp.experts." + e + "." + proj + ".weight",
            p + "block_sparse_moe.experts." + e + "." + mixtral_proj + ".weight",
            p + "mlp.experts." + e + "." + mixtral_proj + ".weight",
        };
    };

    apply_direct({"model.embed_tokens.weight", "model.tok_embeddings.weight", "transformer.wte.weight"},
                 model.token_embedding.weight, true);
    apply_direct({"model.norm.weight", "model.final_layernorm.weight", "transformer.ln_f.weight"},
                 model.final_norm.weight, true);
    if (archive.contains("lm_head.weight")) {
        const auto& info = archive.info("lm_head.weight");
        Tensor tensor;
        if (shape_is(info, {cfg.transformer.vocab_size, cfg.transformer.n_embd})) {
            tensor = tensor_from_f32(backend, {cfg.transformer.n_embd, cfg.transformer.vocab_size},
                                    transpose_2d(archive.f32("lm_head.weight"),
                                                 cfg.transformer.vocab_size,
                                                 cfg.transformer.n_embd));
        } else if (shape_is(info, {cfg.transformer.n_embd, cfg.transformer.vocab_size})) {
            tensor = tensor_from_f32(backend, {cfg.transformer.n_embd, cfg.transformer.vocab_size},
                                    archive.f32("lm_head.weight"));
        } else {
            MCL_CHECK(false, "HF lm_head.weight shape mismatch");
        }
        assign_parameter(model.lm_head, std::move(tensor), trainable);
        report.applied.push_back("lm_head.weight");
        ++report.loaded_tensors;
    } else if (cfg.tie_word_embeddings && archive.contains("model.embed_tokens.weight")) {
        const auto& info = archive.info("model.embed_tokens.weight");
        MCL_CHECK(shape_is(info, {cfg.transformer.vocab_size, cfg.transformer.n_embd}),
                  "tied embedding weight shape mismatch");
        assign_parameter(model.lm_head,
                         tensor_from_f32(backend,
                                         {cfg.transformer.n_embd, cfg.transformer.vocab_size},
                                         transpose_2d(archive.f32("model.embed_tokens.weight"),
                                                      cfg.transformer.vocab_size,
                                                      cfg.transformer.n_embd)),
                         trainable);
        report.applied.push_back("model.embed_tokens.weight->lm_head.weight");
    } else {
        report.missing.push_back("lm_head.weight");
    }

    for (int i = 0; i < cfg.transformer.n_layer; ++i) {
        auto& block = *model.blocks[static_cast<std::size_t>(i)];
        const std::string p = "model.layers." + std::to_string(i) + ".";
        apply_direct({p + "input_layernorm.weight", p + "input_layer_norm.weight", p + "ln_1.weight"},
                     block.norm1().weight, true);
        apply_direct({p + "post_attention_layernorm.weight", p + "pre_feedforward_layernorm.weight",
                      p + "post_attention_layer_norm.weight", p + "ln_2.weight"},
                     block.norm2().weight, true);

        if (auto* delta = block.delta()) {
            apply_delta(i, *delta, true);
        } else if (auto* attn = attention_ptr(block)) {
            apply_attention(i, *attn, true);
            if (block.gated_attention()) {
                apply_linear({p + "self_attn.gate_proj.weight", p + "attention.gate_proj.weight",
                              p + "gated_attention.gate_proj.weight"},
                             block.gated_attention()->gate_proj, true);
            }
        }

        if (auto* mlp = block.mlp()) {
            apply_linear({p + "mlp.gate_proj.weight", p + "feed_forward.w1.weight", p + "mlp.w1.weight"},
                         mlp->gate_proj(), true);
            apply_linear({p + "mlp.up_proj.weight", p + "feed_forward.w3.weight", p + "mlp.w3.weight"},
                         mlp->up_proj(), true);
            apply_linear({p + "mlp.down_proj.weight", p + "feed_forward.w2.weight", p + "mlp.w2.weight"},
                         mlp->down_proj, true);
        } else if (auto* moe = block.moe()) {
            const auto router_name = first_existing(archive, {p + "mlp.gate.weight", p + "mlp.router.weight",
                                                              p + "mlp.gate_proj.weight",
                                                              p + "block_sparse_moe.gate.weight",
                                                              p + "block_sparse_moe.router.weight"});
            if (router_name.empty()) {
                report.missing.push_back(p + "mlp.gate.weight");
            } else {
                assign_parameter(moe->router_weight,
                                 tensor_from_f32(backend, {moe->in_features, moe->num_experts},
                                                 linear_matrix_values(router_name, moe->in_features, moe->num_experts)),
                                 trainable);
                report.applied.push_back(router_name);
                ++report.loaded_tensors;
            }

            auto load_expert_projection = [&](const std::string& proj,
                                              Parameter& parameter,
                                              int64_t in,
                                              int64_t out) {
                std::vector<float> values(static_cast<std::size_t>(moe->num_experts * in * out));
                bool all_present = true;
                int loaded = 0;
                for (int e = 0; e < moe->num_experts; ++e) {
                    const auto name = first_existing(archive, expert_aliases(p, e, proj));
                    if (name.empty()) {
                        all_present = false;
                        report.missing.push_back(expert_aliases(p, e, proj).front());
                        continue;
                    }
                    const auto matrix = linear_matrix_values(name, in, out);
                    std::copy(matrix.begin(), matrix.end(),
                              values.begin() + static_cast<std::ptrdiff_t>(e * in * out));
                    report.applied.push_back(name);
                    ++loaded;
                }
                if (all_present) {
                    assign_parameter(parameter,
                                     tensor_from_f32(backend, {moe->num_experts, in, out}, values),
                                     trainable);
                    report.loaded_tensors += loaded;
                }
            };
            load_expert_projection("gate_proj", moe->expert_gate_weight, moe->in_features, moe->hidden_features);
            load_expert_projection("up_proj", moe->expert_up_weight, moe->in_features, moe->hidden_features);
            load_expert_projection("down_proj", moe->expert_down_weight, moe->hidden_features, moe->in_features);
        }
    }

    if (strict && !report.missing.empty()) {
        MCL_CHECK(false, "missing required HF hybrid weights; first missing: " + report.missing.front());
    }
    return report;
}

HFWeightLoadReport load_hf_transformer_gguf_weights(Backend& backend,
                                                    ModernGPTModel& model,
                                                    const std::string& gguf_path,
                                                    const HFTransformerConfig& cfg,
                                                    bool strict,
                                                    bool trainable) {
    MCL_CHECK(architecture_uses_llama_like_weights(cfg.architecture),
              "GGUF architecture weight loading is not implemented: " + cfg.architecture_name);
    MCL_CHECK(model.config.vocab_size == cfg.transformer.vocab_size &&
                  model.config.n_embd == cfg.transformer.n_embd &&
                  model.config.n_layer == cfg.transformer.n_layer &&
                  model.config.n_head == cfg.transformer.n_head &&
                  model.config.n_kv_head == cfg.transformer.n_kv_head,
              "GGUF config does not match ModernGPTModel config");
    const auto file = gguf::File::open(gguf_path);
    HFWeightLoadReport report;

    const auto expected = expected_gguf_transformer_weight_names(cfg, !cfg.tie_word_embeddings);
    const std::unordered_set<std::string> expected_set(expected.begin(), expected.end());
    for (const auto& tensor : file.tensors()) {
        if (expected_set.find(tensor.name) == expected_set.end() && is_supported_gguf_weight_name(tensor.name)) {
            report.unexpected.push_back(tensor.name);
        }
    }
    for (const auto& name : expected) {
        if (!file.contains_tensor(name)) report.missing.push_back(name);
    }
    if (strict && !report.missing.empty()) {
        MCL_CHECK(false, "missing required GGUF weights; first missing: " + report.missing.front());
    }

    const int64_t hidden = cfg.transformer.n_embd;

    if (!apply_gguf_quantized_embedding_transposed(backend, file, "token_embd.weight", model.token_embedding,
                                                  cfg.transformer.vocab_size, hidden, report)) {
        apply_gguf_matrix(backend, file, "token_embd.weight", model.token_embedding.weight,
                          cfg.transformer.vocab_size, hidden, trainable, report);
    }
    apply_gguf_vector(backend, file, "output_norm.weight", model.final_norm.weight, hidden, trainable, report);
    if (cfg.transformer.use_per_layer_inputs) {
        MCL_CHECK(model.per_layer_token_embedding && model.per_layer_model_projection && model.per_layer_projection_norm,
                  "ModernGPTModel PLE modules are not initialized for GGUF load");
        const int64_t ple_dim = cfg.transformer.per_layer_input_dim;
        const int64_t ple_total = static_cast<int64_t>(cfg.transformer.n_layer) * ple_dim;
        const int64_t ple_vocab = cfg.transformer.per_layer_input_vocab_size > 0
            ? cfg.transformer.per_layer_input_vocab_size
            : cfg.transformer.vocab_size;
        if (!apply_gguf_quantized_embedding_transposed(backend, file, "per_layer_token_embd.weight",
                                                      *model.per_layer_token_embedding,
                                                      ple_vocab, ple_total, report)) {
            apply_gguf_matrix(backend, file, "per_layer_token_embd.weight",
                              model.per_layer_token_embedding->weight, ple_vocab, ple_total, trainable, report);
        }
        apply_gguf_linear_weight(backend, file, "per_layer_model_proj.weight",
                                 *model.per_layer_model_projection, trainable, report);
        apply_gguf_vector(backend, file, "per_layer_proj_norm.weight",
                          model.per_layer_projection_norm->weight, ple_dim, trainable, report);
    }
    if (file.contains_tensor("output.weight")) {
        if (!apply_gguf_quantized_lm_head(backend, file, "output.weight", model, report)) {
            apply_gguf_matrix(backend, file, "output.weight", model.lm_head, hidden,
                              cfg.transformer.vocab_size, trainable, report);
        }
    } else if (cfg.tie_word_embeddings && file.contains_tensor("token_embd.weight")) {
        if (!apply_gguf_quantized_lm_head(backend, file, "token_embd.weight", model, report)) {
            const auto embedding = gguf_matrix_for_shape(file, "token_embd.weight", cfg.transformer.vocab_size, hidden);
            assign_parameter(model.lm_head,
                             tensor_from_f32(backend, {hidden, cfg.transformer.vocab_size},
                                             transpose_2d(embedding, cfg.transformer.vocab_size, hidden)),
                             trainable);
            report.applied.push_back("token_embd.weight->output.weight");
        }
    }

    for (int i = 0; i < cfg.transformer.n_layer; ++i) {
        auto& block = *model.blocks[static_cast<std::size_t>(i)];
        const std::string p = "blk." + std::to_string(i) + ".";
        const int64_t q_dim = block.attention().q_proj().out_features();
        const int64_t kv_dim = block.attention().k_proj().out_features();
        const int64_t qkv_dim = q_dim + 2 * kv_dim;
        const int64_t mlp_hidden = block.mlp().gate_proj().out_features();
        apply_gguf_vector(backend, file, p + "attn_norm.weight", block.norm1().weight, hidden, trainable, report);
        apply_gguf_vector(backend, file, p + "ffn_norm.weight", block.norm2().weight, hidden, trainable, report);
        if (cfg.transformer.use_qk_norm) {
            const int64_t head_dim = block.attention().head_dim();
            apply_gguf_vector(backend, file, p + "attn_q_norm.weight", block.attention().q_norm().weight,
                              head_dim, trainable, report);
            apply_gguf_vector(backend, file, p + "attn_k_norm.weight", block.attention().k_norm().weight,
                              head_dim, trainable, report);
        }
        if (cfg.transformer.use_post_attention_norm) {
            apply_gguf_vector(backend, file, p + "post_attention_norm.weight",
                              block.post_attention_norm().weight, hidden, trainable, report);
        }
        if (cfg.transformer.use_post_ffw_norm) {
            apply_gguf_vector(backend, file, p + "post_ffw_norm.weight",
                              block.post_ffw_norm().weight, hidden, trainable, report);
        }
        if (cfg.transformer.use_layer_output_scale) {
            apply_gguf_vector(backend, file, p + "layer_output_scale.weight",
                              block.layer_output_scale(), 1, trainable, report);
        }
        if (cfg.transformer.use_per_layer_inputs) {
            MCL_CHECK(block.per_layer_input_gate() && block.per_layer_projection() && block.post_per_layer_norm(),
                      "ModernTransformerBlock PLE modules are not initialized for GGUF load");
            apply_gguf_linear_weight(backend, file, p + "inp_gate.weight",
                                     *block.per_layer_input_gate(), trainable, report);
            apply_gguf_linear_weight(backend, file, p + "proj.weight",
                                     *block.per_layer_projection(), trainable, report);
            apply_gguf_vector(backend, file, p + "post_norm.weight",
                              block.post_per_layer_norm()->weight, hidden, trainable, report);
        }
        apply_gguf_linear_weight(backend, file, p + "attn_output.weight", block.attention().o_proj(),
                                 trainable, report);
        if (block.attention().o_proj().has_bias()) {
            apply_gguf_vector(backend, file, p + "attn_output.bias", block.attention().o_proj().bias,
                              hidden, trainable, report);
        }
        apply_gguf_linear_weight(backend, file, p + "ffn_down.weight", block.mlp().down_proj,
                                 trainable, report);

        const auto q_name = p + "attn_q.weight";
        const auto k_name = p + "attn_k.weight";
        const auto v_name = p + "attn_v.weight";
        const bool qkv_present = file.contains_tensor(q_name) && file.contains_tensor(k_name) && file.contains_tensor(v_name);
        const bool qkv_quant = gguf_tensor_is_native_quant(file, q_name) ||
                               gguf_tensor_is_native_quant(file, k_name) ||
                               gguf_tensor_is_native_quant(file, v_name);
        if (qkv_present && qkv_quant) {
            block.attention().enable_split_projections(true);
            apply_gguf_linear_weight(backend, file, q_name, block.attention().q_proj(), trainable, report);
            apply_gguf_linear_weight(backend, file, k_name, block.attention().k_proj(), trainable, report);
            apply_gguf_linear_weight(backend, file, v_name, block.attention().v_proj(), trainable, report);
        } else if (qkv_present) {
            std::vector<float> packed(static_cast<std::size_t>(hidden * qkv_dim));
            pack_direct_projection(packed, qkv_dim, 0, gguf_matrix_for_shape(file, q_name, hidden, q_dim), hidden, q_dim);
            pack_direct_projection(packed, qkv_dim, q_dim, gguf_matrix_for_shape(file, k_name, hidden, kv_dim), hidden, kv_dim);
            pack_direct_projection(packed, qkv_dim, q_dim + kv_dim,
                                   gguf_matrix_for_shape(file, v_name, hidden, kv_dim), hidden, kv_dim);
            assign_parameter(block.attention().qkv_proj().weight,
                             tensor_from_f32(backend, {hidden, qkv_dim}, packed),
                             trainable);
            report.applied.push_back(q_name + "+" + k_name + "+" + v_name + "->qkv_proj.weight");
            report.loaded_tensors += 3;
        }

        const auto qb_name = p + "attn_q.bias";
        const auto kb_name = p + "attn_k.bias";
        const auto vb_name = p + "attn_v.bias";
        if (block.attention().qkv_proj().has_bias() &&
            file.contains_tensor(qb_name) && file.contains_tensor(kb_name) && file.contains_tensor(vb_name)) {
            const auto qb = gguf_vector_for_size(file, qb_name, q_dim);
            const auto kb = gguf_vector_for_size(file, kb_name, kv_dim);
            const auto vb = gguf_vector_for_size(file, vb_name, kv_dim);
            if (block.attention().split_projections_enabled()) {
                assign_parameter(block.attention().q_proj().bias, tensor_from_f32(backend, {q_dim}, qb), trainable);
                assign_parameter(block.attention().k_proj().bias, tensor_from_f32(backend, {kv_dim}, kb), trainable);
                assign_parameter(block.attention().v_proj().bias, tensor_from_f32(backend, {kv_dim}, vb), trainable);
                report.applied.push_back(qb_name + "+" + kb_name + "+" + vb_name + "->split_qkv.bias");
                report.loaded_tensors += 3;
            } else {
                std::vector<float> packed(static_cast<std::size_t>(qkv_dim));
                std::copy(qb.begin(), qb.end(), packed.begin());
                std::copy(kb.begin(), kb.end(), packed.begin() + q_dim);
                std::copy(vb.begin(), vb.end(), packed.begin() + q_dim + kv_dim);
                assign_parameter(block.attention().qkv_proj().bias,
                                 tensor_from_f32(backend, {qkv_dim}, packed),
                                 trainable);
                report.applied.push_back(qb_name + "+" + kb_name + "+" + vb_name + "->qkv_proj.bias");
                report.loaded_tensors += 3;
            }
        }

        const auto gate_name = p + "ffn_gate.weight";
        const auto up_name = p + "ffn_up.weight";
        const bool gate_up_present = file.contains_tensor(gate_name) && file.contains_tensor(up_name);
        const bool gate_up_quant = gguf_tensor_is_native_quant(file, gate_name) || gguf_tensor_is_native_quant(file, up_name);
        if (gate_up_present && gate_up_quant) {
            block.mlp().enable_split_projections(true);
            apply_gguf_linear_weight(backend, file, gate_name, block.mlp().gate_proj(), trainable, report);
            apply_gguf_linear_weight(backend, file, up_name, block.mlp().up_proj(), trainable, report);
        } else if (gate_up_present) {
            std::vector<float> packed(static_cast<std::size_t>(hidden * mlp_hidden * 2));
            pack_direct_projection(packed, mlp_hidden * 2, 0,
                                   gguf_matrix_for_shape(file, gate_name, hidden, mlp_hidden),
                                   hidden, mlp_hidden);
            pack_direct_projection(packed, mlp_hidden * 2, mlp_hidden,
                                   gguf_matrix_for_shape(file, up_name, hidden, mlp_hidden),
                                   hidden, mlp_hidden);
            assign_parameter(block.mlp().gate_up_proj.weight,
                             tensor_from_f32(backend, {hidden, mlp_hidden * 2}, packed),
                             trainable);
            report.applied.push_back(gate_name + "+" + up_name + "->gate_up_proj.weight");
            report.loaded_tensors += 2;
        }
    }

    return report;
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

HFTokenizer load_hf_tokenizer_gguf(const std::string& path, const HFTransformerConfig& cfg) {
    const auto file = gguf::File::open(path);
    auto tokens = metadata_string_array_or_empty(file, "tokenizer.ggml.tokens");
    if (!tokens.empty()) {
        return HFTokenizer::from_tokens(tokens, cfg.bos_token_id, cfg.eos_token_id,
                                        file.metadata_string_or("tokenizer.ggml.model", "GGUF"));
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

std::string generate_hf_hybrid_text(Backend& backend,
                                    HybridGPTModel& model,
                                    const HFTokenizer& tokenizer,
                                    const std::string& prompt,
                                    const GenerateOptions& options) {
    MCL_CHECK(options.max_new_tokens >= 0, "GenerateOptions max_new_tokens must be non-negative");
    MCL_CHECK(options.kv_page_size > 0, "GenerateOptions kv_page_size must be positive");
    autograd::NoGradGuard no_grad;
    auto tokens = tokenizer.encode(prompt, options.add_bos, false);
    if (tokens.empty() && options.add_bos) tokens.push_back(options.bos_token_id);
    MCL_CHECK(!tokens.empty(), "generate_hf_hybrid_text requires a non-empty prompt or add_bos=true");
    MCL_CHECK(static_cast<int>(tokens.size()) <= model.config.block_size, "prompt exceeds model block_size");
    if (options.max_new_tokens == 0) return tokenizer.decode(tokens, true);

    auto cache = model.create_runtime_cache(backend, 1, options.use_paged_kv_cache, options.kv_page_size);
    auto last_logits = [&](const Tensor& logits, int64_t seq_len) {
        if (seq_len == 1) return logits.view({1, model.config.vocab_size});
        std::int32_t pos = static_cast<std::int32_t>(seq_len - 1);
        auto positions = Tensor::from_cpu(backend, {1}, DType::I32, &pos);
        return gather_last_token_logits(logits, positions, 1, seq_len, model.config.vocab_size);
    };

    Tensor logits;
    if (options.prefill_prompt) {
        auto input = Tensor::from_cpu(backend,
                                      {1, static_cast<int64_t>(tokens.size())},
                                      DType::I32,
                                      tokens.data());
        logits = last_logits(model.forward_with_cache(input, cache), static_cast<int64_t>(tokens.size()));
    } else {
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            std::int32_t id = tokens[i];
            auto input = Tensor::from_cpu(backend, {1, 1}, DType::I32, &id);
            logits = last_logits(model.forward_with_cache(input, cache), 1);
        }
    }

    std::mt19937 rng(options.seed);
    for (int step = 0; step < options.max_new_tokens; ++step) {
        MCL_CHECK(static_cast<int>(tokens.size()) < model.config.block_size,
                  "generate_hf_hybrid_text reached model block_size");
        auto next_tensor = rowwise_sample_top_p(logits,
                                                options.temperature,
                                                options.top_k,
                                                options.top_p,
                                                static_cast<std::uint32_t>(rng()));
        const auto next = next_tensor.to_vector<std::int32_t>()[0];
        tokens.push_back(next);
        if (options.eos_token_id >= 0 && next == options.eos_token_id) break;
        if (step + 1 >= options.max_new_tokens) break;
        auto input = next_tensor.view({1, 1});
        logits = last_logits(model.forward_with_cache(input, cache), 1);
    }
    return tokenizer.decode(tokens, true);
}

std::vector<std::string> generate_hf_batch_text(Backend& backend,
                                                ModernGPTModel& model,
                                                const HFTokenizer& tokenizer,
                                                const std::vector<std::string>& prompts,
                                                const GenerateOptions& options) {
    return generate_batch_text(backend, model, tokenizer, prompts, options);
}

} // namespace motifcl::nn
