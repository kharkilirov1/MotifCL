#include <motifcl/nn/gemma.hpp>

#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/ops/indexing.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/ops/quant.hpp>
#include <motifcl/ops/reduce.hpp>
#include <motifcl/runtime/backend.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <regex>
#include <sstream>
#include <unordered_set>

namespace motifcl {
namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open text file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string regex_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() * 2);
    for (char c : value) {
        if (std::string(R"(\.^$|()[]{}*+?)").find(c) != std::string::npos) out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

void append_utf8(std::string& out, unsigned int cp) {
    if (cp <= 0x7fu) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7ffu) {
        out.push_back(static_cast<char>(0xc0u | (cp >> 6)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    } else if (cp <= 0xffffu) {
        out.push_back(static_cast<char>(0xe0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    } else {
        out.push_back(static_cast<char>(0xf0u | (cp >> 18)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 12) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    }
}

std::string json_unescape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '\\' || i + 1 >= value.size()) {
            out.push_back(value[i]);
            continue;
        }
        const char n = value[++i];
        switch (n) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (i + 4 <= value.size()) {
                    unsigned int cp = 0;
                    bool ok = true;
                    for (int j = 0; j < 4; ++j) {
                        const char h = value[i + 1 + static_cast<std::size_t>(j)];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= static_cast<unsigned int>(h - '0');
                        else if (h >= 'a' && h <= 'f') cp |= static_cast<unsigned int>(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') cp |= static_cast<unsigned int>(h - 'A' + 10);
                        else {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        append_utf8(out, cp);
                        i += 4;
                    } else {
                        out.push_back('u');
                    }
                } else {
                    out.push_back('u');
                }
                break;
            }
            default:
                out.push_back(n);
                break;
        }
    }
    return out;
}

std::string trim(std::string s) {
    auto first = std::find_if_not(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c) != 0; });
    auto last = std::find_if_not(s.rbegin(), s.rend(), [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (first >= last) return {};
    return std::string(first, last);
}

std::string lower_ascii(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::size_t find_matching_json(const std::string& text, std::size_t open_pos, char open, char close) {
    bool in_string = false;
    bool escape = false;
    int depth = 0;
    for (std::size_t i = open_pos; i < text.size(); ++i) {
        const char c = text[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
        } else if (c == open) {
            ++depth;
        } else if (c == close) {
            --depth;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

std::string extract_json_object_for_key(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return {};
    const auto colon = text.find(':', key_pos);
    if (colon == std::string::npos) return {};
    auto brace = text.find_first_not_of(" \t\r\n", colon + 1);
    if (brace == std::string::npos || text[brace] != '{') return {};
    const auto end = find_matching_json(text, brace, '{', '}');
    if (end == std::string::npos) return {};
    return text.substr(brace + 1, end - brace - 1);
}

std::string extract_json_array_for_key(const std::string& text, const std::string& key) {
    const auto key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return {};
    const auto colon = text.find(':', key_pos);
    if (colon == std::string::npos) return {};
    auto bracket = text.find_first_not_of(" \t\r\n", colon + 1);
    if (bracket == std::string::npos || text[bracket] != '[') return {};
    const auto end = find_matching_json(text, bracket, '[', ']');
    if (end == std::string::npos) return {};
    return text.substr(bracket + 1, end - bracket - 1);
}

std::string json_string_or_empty(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    std::size_t search = 0;
    while (search < text.size()) {
        const auto key_pos = text.find(needle, search);
        if (key_pos == std::string::npos) return {};
        const auto colon = text.find(':', key_pos + needle.size());
        if (colon == std::string::npos) return {};
        auto pos = text.find_first_not_of(" \t\r\n", colon + 1);
        if (pos == std::string::npos) return {};
        if (text[pos] != '"') {
            search = pos + 1;
            continue;
        }
        ++pos;
        std::string raw;
        bool escape = false;
        for (; pos < text.size(); ++pos) {
            const char c = text[pos];
            if (escape) {
                raw.push_back('\\');
                raw.push_back(c);
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                return json_unescape(raw);
            } else {
                raw.push_back(c);
            }
        }
        return {};
    }
    return {};
}

std::vector<std::string> json_strings_in(const std::string& text) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < text.size();) {
        if (text[i] != '"') {
            ++i;
            continue;
        }
        ++i;
        std::string raw;
        bool escape = false;
        for (; i < text.size(); ++i) {
            const char c = text[i];
            if (escape) {
                raw.push_back('\\');
                raw.push_back(c);
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                ++i;
                break;
            } else {
                raw.push_back(c);
            }
        }
        out.push_back(json_unescape(raw));
    }
    return out;
}

std::vector<std::pair<std::string, std::int32_t>> json_object_int_pairs(const std::string& text) {
    std::vector<std::pair<std::string, std::int32_t>> out;
    for (std::size_t i = 0; i < text.size();) {
        if (text[i] != '"') {
            ++i;
            continue;
        }
        ++i;
        std::string raw_key;
        bool escape = false;
        for (; i < text.size(); ++i) {
            const char c = text[i];
            if (escape) {
                raw_key.push_back('\\');
                raw_key.push_back(c);
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                ++i;
                break;
            } else {
                raw_key.push_back(c);
            }
        }
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        if (i >= text.size() || text[i] != ':') continue;
        ++i;
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
        bool negative = false;
        if (i < text.size() && text[i] == '-') { negative = true; ++i; }
        if (i >= text.size() || !std::isdigit(static_cast<unsigned char>(text[i]))) continue;
        std::int64_t value = 0;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            value = value * 10 + static_cast<std::int64_t>(text[i] - '0');
            ++i;
        }
        if (negative) value = -value;
        if (value >= 0 && value <= std::numeric_limits<std::int32_t>::max()) {
            out.emplace_back(json_unescape(raw_key), static_cast<std::int32_t>(value));
        }
    }
    return out;
}

std::string bpe_pair_key(const std::string& left, const std::string& right) {
    std::string key = left;
    key.push_back('\0');
    key += right;
    return key;
}

std::vector<std::string> utf8_pieces(const std::string& text) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < text.size();) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        std::size_t len = 1;
        if ((c & 0xe0u) == 0xc0u) len = 2;
        else if ((c & 0xf0u) == 0xe0u) len = 3;
        else if ((c & 0xf8u) == 0xf0u) len = 4;
        if (i + len > text.size()) len = 1;
        out.push_back(text.substr(i, len));
        i += len;
    }
    return out;
}


bool protobuf_read_varint(const std::string& data, std::size_t& pos, std::uint64_t& value) {
    value = 0;
    int shift = 0;
    while (pos < data.size() && shift <= 63) {
        const auto byte = static_cast<unsigned char>(data[pos++]);
        value |= static_cast<std::uint64_t>(byte & 0x7fu) << shift;
        if ((byte & 0x80u) == 0) return true;
        shift += 7;
    }
    return false;
}

bool protobuf_skip_field(const std::string& data, std::size_t& pos, int wire_type) {
    std::uint64_t ignored = 0;
    switch (wire_type) {
    case 0:
        return protobuf_read_varint(data, pos, ignored);
    case 1:
        if (pos + 8 > data.size()) return false;
        pos += 8;
        return true;
    case 2:
        if (!protobuf_read_varint(data, pos, ignored)) return false;
        if (pos + static_cast<std::size_t>(ignored) > data.size()) return false;
        pos += static_cast<std::size_t>(ignored);
        return true;
    case 5:
        if (pos + 4 > data.size()) return false;
        pos += 4;
        return true;
    default:
        return false;
    }
}

struct SentencePieceModelLite {
    std::vector<std::string> pieces;
    std::string model_type = "SentencePiece";
    std::string normalizer_name = "nmt_nfkc";
    bool add_dummy_prefix = true;
    bool remove_extra_whitespaces = true;
    bool escape_whitespaces = true;
};

std::string parse_sentencepiece_piece_message(const std::string& msg) {
    std::size_t pos = 0;
    std::string piece;
    while (pos < msg.size()) {
        std::uint64_t key = 0;
        if (!protobuf_read_varint(msg, pos, key)) break;
        const int field = static_cast<int>(key >> 3);
        const int wire = static_cast<int>(key & 0x7u);
        if (field == 1 && wire == 2) {
            std::uint64_t len = 0;
            if (!protobuf_read_varint(msg, pos, len)) break;
            if (pos + static_cast<std::size_t>(len) > msg.size()) break;
            piece = msg.substr(pos, static_cast<std::size_t>(len));
            pos += static_cast<std::size_t>(len);
        } else if (!protobuf_skip_field(msg, pos, wire)) {
            break;
        }
    }
    return piece;
}

std::string parse_sentencepiece_trainer_spec(const std::string& msg) {
    std::size_t pos = 0;
    while (pos < msg.size()) {
        std::uint64_t key = 0;
        if (!protobuf_read_varint(msg, pos, key)) break;
        const int field = static_cast<int>(key >> 3);
        const int wire = static_cast<int>(key & 0x7u);
        if (field == 3 && wire == 0) {
            std::uint64_t value = 0;
            if (!protobuf_read_varint(msg, pos, value)) break;
            if (value == 1) return "Unigram";
            if (value == 2) return "BPE";
            if (value == 3) return "Word";
            if (value == 4) return "Char";
            return "SentencePiece";
        }
        if (!protobuf_skip_field(msg, pos, wire)) break;
    }
    return "SentencePiece";
}

void parse_sentencepiece_normalizer_spec(const std::string& msg, SentencePieceModelLite& out) {
    std::size_t pos = 0;
    while (pos < msg.size()) {
        std::uint64_t key = 0;
        if (!protobuf_read_varint(msg, pos, key)) break;
        const int field = static_cast<int>(key >> 3);
        const int wire = static_cast<int>(key & 0x7u);
        if (field == 1 && wire == 2) {
            std::uint64_t len = 0;
            if (!protobuf_read_varint(msg, pos, len)) break;
            if (pos + static_cast<std::size_t>(len) > msg.size()) break;
            out.normalizer_name = msg.substr(pos, static_cast<std::size_t>(len));
            pos += static_cast<std::size_t>(len);
        } else if ((field == 3 || field == 4 || field == 5) && wire == 0) {
            std::uint64_t value = 0;
            if (!protobuf_read_varint(msg, pos, value)) break;
            const bool enabled = value != 0;
            if (field == 3) out.add_dummy_prefix = enabled;
            else if (field == 4) out.remove_extra_whitespaces = enabled;
            else if (field == 5) out.escape_whitespaces = enabled;
        } else if (!protobuf_skip_field(msg, pos, wire)) {
            break;
        }
    }
}

SentencePieceModelLite parse_sentencepiece_model_proto(const std::string& data) {
    SentencePieceModelLite out;
    std::size_t pos = 0;
    while (pos < data.size()) {
        std::uint64_t key = 0;
        if (!protobuf_read_varint(data, pos, key)) break;
        const int field = static_cast<int>(key >> 3);
        const int wire = static_cast<int>(key & 0x7u);
        if ((field == 1 || field == 2 || field == 3) && wire == 2) {
            std::uint64_t len = 0;
            if (!protobuf_read_varint(data, pos, len)) break;
            if (pos + static_cast<std::size_t>(len) > data.size()) break;
            const std::string msg = data.substr(pos, static_cast<std::size_t>(len));
            pos += static_cast<std::size_t>(len);
            if (field == 1) {
                auto piece = parse_sentencepiece_piece_message(msg);
                if (!piece.empty()) out.pieces.push_back(std::move(piece));
            } else {
                if (field == 2) out.model_type = parse_sentencepiece_trainer_spec(msg);
                else parse_sentencepiece_normalizer_spec(msg, out);
            }
        } else if (!protobuf_skip_field(data, pos, wire)) {
            break;
        }
    }
    return out;
}

std::uint64_t read_le_u64(std::istream& in) {
    unsigned char bytes[8] = {};
    in.read(reinterpret_cast<char*>(bytes), sizeof(bytes));
    MCL_CHECK(in.good(), "failed to read safetensors header length");
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
    return value;
}

std::uint16_t read_le_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

std::vector<int64_t> parse_i64_list(const std::string& text) {
    std::vector<int64_t> out;
    std::regex number_re(R"([-+]?\d+)");
    for (std::sregex_iterator it(text.begin(), text.end(), number_re), end; it != end; ++it) {
        out.push_back(std::stoll((*it)[0].str()));
    }
    return out;
}

std::vector<std::uint64_t> parse_u64_list(const std::string& text) {
    std::vector<std::uint64_t> out;
    std::regex number_re(R"(\d+)");
    for (std::sregex_iterator it(text.begin(), text.end(), number_re), end; it != end; ++it) {
        out.push_back(static_cast<std::uint64_t>(std::stoull((*it)[0].str())));
    }
    return out;
}

std::size_t dtype_file_size(const std::string& dtype) {
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
    if (dtype == "U8" || dtype == "I8" || dtype == "BOOL") return 1;
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
    MCL_CHECK(false, "unsupported safetensors dtype: " + dtype);
    return 0;
}

std::uint64_t shape_numel_u64(const std::vector<int64_t>& shape) {
    std::uint64_t n = 1;
    for (auto dim : shape) {
        MCL_CHECK(dim >= 0, "negative safetensors shape dimension");
        n *= static_cast<std::uint64_t>(dim);
    }
    return n;
}

float f16_to_f32(std::uint16_t h) {
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

std::vector<float> raw_to_f32_vector(const std::string& dtype,
                                     const std::vector<int64_t>& shape,
                                     const std::vector<std::uint8_t>& raw) {
    const auto numel = static_cast<std::size_t>(shape_numel_u64(shape));
    std::vector<float> out(numel);
    if (dtype == "F32") {
        MCL_CHECK(raw.size() == numel * sizeof(float), "F32 safetensors byte size mismatch");
        std::memcpy(out.data(), raw.data(), raw.size());
        return out;
    }
    if (dtype == "F16") {
        MCL_CHECK(raw.size() == numel * 2, "F16 safetensors byte size mismatch");
        for (std::size_t i = 0; i < numel; ++i) out[i] = f16_to_f32(read_le_u16(raw.data() + i * 2));
        return out;
    }
    if (dtype == "BF16") {
        MCL_CHECK(raw.size() == numel * 2, "BF16 safetensors byte size mismatch");
        for (std::size_t i = 0; i < numel; ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(read_le_u16(raw.data() + i * 2)) << 16;
            std::memcpy(&out[i], &bits, sizeof(float));
        }
        return out;
    }
    MCL_CHECK(false, "cannot convert safetensors dtype to f32: " + dtype);
    return out;
}

std::vector<std::uint8_t> read_tensor_raw(const std::string& path,
                                          std::uint64_t data_start,
                                          const SafeTensorInfo& info) {
    MCL_CHECK(info.data_end >= info.data_begin, "invalid safetensors tensor offsets");
    const auto nbytes = static_cast<std::size_t>(info.data_end - info.data_begin);
    MCL_CHECK(nbytes == static_cast<std::size_t>(shape_numel_u64(info.shape) * dtype_file_size(info.dtype)),
              "safetensors byte size does not match dtype/shape for " + info.name);
    std::vector<std::uint8_t> raw(nbytes);
    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open safetensors file: " + path);
    in.seekg(static_cast<std::streamoff>(data_start + info.data_begin), std::ios::beg);
    MCL_CHECK(in.good(), "failed to seek safetensors tensor: " + info.name);
    if (!raw.empty()) in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    MCL_CHECK(in.good(), "failed to read safetensors tensor: " + info.name);
    return raw;
}

bool json_number(const std::string& text, const std::string& key, double& out) {
    std::regex re("\"" + regex_escape(key) + R"("\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?))");
    std::smatch m;
    if (!std::regex_search(text, m, re)) return false;
    out = std::stod(m[1].str());
    return true;
}

int json_int_or(const std::string& text, const std::vector<std::string>& keys, int fallback) {
    for (const auto& key : keys) {
        double value = 0.0;
        if (json_number(text, key, value)) return static_cast<int>(value);
    }
    return fallback;
}

float json_float_or(const std::string& text, const std::vector<std::string>& keys, float fallback) {
    for (const auto& key : keys) {
        double value = 0.0;
        if (json_number(text, key, value)) return static_cast<float>(value);
    }
    return fallback;
}

bool json_bool_or(const std::string& text, const std::vector<std::string>& keys, bool fallback) {
    for (const auto& key : keys) {
        std::regex re("\"" + regex_escape(key) + R"("\s*:\s*(true|false))");
        std::smatch m;
        if (std::regex_search(text, m, re)) return m[1].str() == "true";
    }
    return fallback;
}

nn::GemmaConfig normalize_gemma_config(nn::GemmaConfig cfg) {
    if (cfg.num_key_value_heads <= 0) cfg.num_key_value_heads = cfg.num_attention_heads;
    if (cfg.head_dim <= 0 && cfg.hidden_size > 0 && cfg.num_attention_heads > 0) {
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    }
    if (cfg.intermediate_size <= 0 && cfg.hidden_size > 0) cfg.intermediate_size = cfg.hidden_size * 4;
    MCL_CHECK(cfg.vocab_size > 0, "GemmaConfig vocab_size must be positive");
    MCL_CHECK(cfg.max_position_embeddings > 0, "GemmaConfig max_position_embeddings must be positive");
    MCL_CHECK(cfg.hidden_size > 0, "GemmaConfig hidden_size must be positive");
    MCL_CHECK(cfg.intermediate_size > 0, "GemmaConfig intermediate_size must be positive");
    MCL_CHECK(cfg.num_hidden_layers > 0, "GemmaConfig num_hidden_layers must be positive");
    MCL_CHECK(cfg.num_attention_heads > 0, "GemmaConfig num_attention_heads must be positive");
    MCL_CHECK(cfg.num_key_value_heads > 0, "GemmaConfig num_key_value_heads must be positive");
    MCL_CHECK(cfg.head_dim > 0, "GemmaConfig head_dim must be positive");
    MCL_CHECK(cfg.num_attention_heads % cfg.num_key_value_heads == 0,
              "GemmaConfig num_attention_heads must be divisible by num_key_value_heads");
    return cfg;
}

std::vector<float> transpose_2d(const std::vector<float>& src, int64_t rows, int64_t cols) {
    MCL_CHECK(rows >= 0 && cols >= 0, "transpose_2d invalid shape");
    MCL_CHECK(src.size() == static_cast<std::size_t>(rows * cols), "transpose_2d source size mismatch");
    std::vector<float> dst(static_cast<std::size_t>(rows * cols));
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) dst[static_cast<std::size_t>(c * rows + r)] = src[static_cast<std::size_t>(r * cols + c)];
    }
    return dst;
}

void pack_transposed_projection(std::vector<float>& dst,
                                int64_t dst_cols,
                                int64_t col_offset,
                                const std::vector<float>& hf_weight,
                                int64_t out_dim,
                                int64_t in_dim) {
    MCL_CHECK(dst.size() >= static_cast<std::size_t>(in_dim * dst_cols), "packed projection destination too small");
    MCL_CHECK(hf_weight.size() == static_cast<std::size_t>(out_dim * in_dim), "HF projection size mismatch");
    for (int64_t out = 0; out < out_dim; ++out) {
        for (int64_t in = 0; in < in_dim; ++in) {
            dst[static_cast<std::size_t>(in * dst_cols + col_offset + out)] =
                hf_weight[static_cast<std::size_t>(out * in_dim + in)];
        }
    }
}

Tensor tensor_from_f32(Backend& backend, const std::vector<int64_t>& shape, const std::vector<float>& values) {
    MCL_CHECK(shape_numel_u64(shape) == values.size(), "tensor_from_f32 shape/value mismatch");
    return Tensor::from_cpu(backend, Shape(shape), DType::F32, values.data());
}

void assign_parameter(nn::Parameter& parameter, Tensor tensor, bool trainable) {
    MCL_CHECK(parameter.data.shape() == tensor.shape(), "loaded parameter shape mismatch");
    parameter.data = std::move(tensor);
    parameter.trainable = trainable;
    parameter.data.set_requires_grad(trainable);
}

class SafeTensorsArchive {
public:
    explicit SafeTensorsArchive(const std::vector<std::string>& paths) {
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

bool shape_is(const SafeTensorInfo& info, std::initializer_list<int64_t> dims) {
    return info.shape == std::vector<int64_t>(dims);
}

int choose_next_token(const std::vector<float>& logits, const nn::GenerateOptions& options, std::mt19937& rng) {
    MCL_CHECK(!logits.empty(), "generate received empty logits");
    if (options.temperature <= 0.0f) {
        return static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
    }

    std::vector<int> indices(logits.size());
    for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = static_cast<int>(i);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)]; });
    if (options.top_k > 0 && options.top_k < static_cast<int>(indices.size())) {
        indices.resize(static_cast<std::size_t>(options.top_k));
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int idx : indices) max_logit = std::max(max_logit, logits[static_cast<std::size_t>(idx)]);
    const float inv_temp = 1.0f / std::max(options.temperature, 1e-6f);
    std::vector<double> weights(indices.size());
    double total = 0.0;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        weights[i] = std::exp(static_cast<double>((logits[static_cast<std::size_t>(indices[i])] - max_logit) * inv_temp));
        total += weights[i];
    }
    if (options.top_p > 0.0f && options.top_p < 0.999999f && total > 0.0) {
        const double threshold = static_cast<double>(std::min(std::max(options.top_p, 1e-6f), 1.0f)) * total;
        double cumulative = 0.0;
        std::size_t keep = 0;
        for (; keep < weights.size(); ++keep) {
            cumulative += weights[keep];
            if (cumulative >= threshold) {
                ++keep;
                break;
            }
        }
        keep = std::max<std::size_t>(1, std::min(keep, indices.size()));
        indices.resize(keep);
        weights.resize(keep);
    }
    std::discrete_distribution<std::size_t> dist(weights.begin(), weights.end());
    return indices[dist(rng)];
}

[[maybe_unused]] Tensor last_token_logits_tensor(const Tensor& logits, int vocab_size) {
    MCL_CHECK(vocab_size > 0, "generate requires positive vocab_size");
    MCL_CHECK(logits.dtype() == DType::F32, "generate logits must be f32");
    MCL_CHECK(logits.numel() >= vocab_size && logits.numel() % vocab_size == 0,
              "generate logits size is not divisible by vocab_size");
    const int64_t rows = logits.numel() / vocab_size;
    if (rows == 1) return logits.view({1, vocab_size});
    return slice_rows(logits.view({rows, vocab_size}), rows - 1, rows);
}

std::int32_t choose_next_token(const Tensor& logits,
                               const nn::GenerateOptions& options,
                               std::mt19937& rng,
                               std::vector<float>& cpu_logits) {
    if (options.gpu_greedy_sampling) {
        return rowwise_sample_top_p(logits, options.temperature, options.top_k, options.top_p, static_cast<std::uint32_t>(rng()))
            .to_vector<std::int32_t>()[0];
    }
    cpu_logits = logits.to_vector<float>();
    return static_cast<std::int32_t>(choose_next_token(cpu_logits, options, rng));
}

Tensor last_token_logits_batch_tensor(const Tensor& logits,
                                      const std::vector<std::int32_t>& positions,
                                      int64_t batch_size,
                                      int64_t seq_len,
                                      int vocab_size) {
    MCL_CHECK(static_cast<int64_t>(positions.size()) == batch_size, "batch logits positions size mismatch");
    auto pos = Tensor::from_cpu(logits.backend(), {batch_size}, DType::I32, positions.data());
    return gather_last_token_logits(logits, pos, batch_size, seq_len, vocab_size);
}

std::vector<std::int32_t> choose_next_tokens(const Tensor& logits,
                                            const nn::GenerateOptions& options,
                                            std::mt19937& rng,
                                            int64_t batch_size,
                                            int vocab_size) {
    MCL_CHECK(logits.dtype() == DType::F32 && logits.ndim() == 2, "batch sampler expects [B,V] f32 logits");
    MCL_CHECK(logits.shape()[0] == batch_size && logits.shape()[1] == vocab_size, "batch sampler logits shape mismatch");
    if (options.gpu_greedy_sampling) {
        return rowwise_sample_top_p(logits, options.temperature, options.top_k, options.top_p, static_cast<std::uint32_t>(rng()))
            .to_vector<std::int32_t>();
    }
    const auto cpu = logits.to_vector<float>();
    std::vector<std::int32_t> out(static_cast<std::size_t>(batch_size));
    for (int64_t b = 0; b < batch_size; ++b) {
        const auto begin = cpu.begin() + b * vocab_size;
        const auto end = begin + vocab_size;
        std::vector<float> row(begin, end);
        out[static_cast<std::size_t>(b)] = static_cast<std::int32_t>(choose_next_token(row, options, rng));
    }
    return out;
}

void replace_all(std::string& s, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

bool utf8_decode_one(const std::string& s, std::size_t pos, std::uint32_t& cp, std::size_t& len) {
    if (pos >= s.size()) return false;
    const auto c0 = static_cast<unsigned char>(s[pos]);
    if (c0 < 0x80) {
        cp = c0;
        len = 1;
        return true;
    }
    auto cont = [&](std::size_t off) -> int {
        if (pos + off >= s.size()) return -1;
        const auto c = static_cast<unsigned char>(s[pos + off]);
        return (c & 0xc0) == 0x80 ? static_cast<int>(c & 0x3f) : -1;
    };
    if ((c0 & 0xe0) == 0xc0) {
        const int c1 = cont(1);
        if (c1 < 0) return false;
        cp = ((c0 & 0x1fu) << 6) | static_cast<std::uint32_t>(c1);
        len = 2;
        return true;
    }
    if ((c0 & 0xf0) == 0xe0) {
        const int c1 = cont(1);
        const int c2 = cont(2);
        if (c1 < 0 || c2 < 0) return false;
        cp = ((c0 & 0x0fu) << 12) | (static_cast<std::uint32_t>(c1) << 6) | static_cast<std::uint32_t>(c2);
        len = 3;
        return true;
    }
    if ((c0 & 0xf8) == 0xf0) {
        const int c1 = cont(1);
        const int c2 = cont(2);
        const int c3 = cont(3);
        if (c1 < 0 || c2 < 0 || c3 < 0) return false;
        cp = ((c0 & 0x07u) << 18) | (static_cast<std::uint32_t>(c1) << 12) |
             (static_cast<std::uint32_t>(c2) << 6) | static_cast<std::uint32_t>(c3);
        len = 4;
        return true;
    }
    return false;
}

bool sentencepiece_is_space(std::uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v' ||
           cp == 0x00a0 || cp == 0x3000;
}

std::string sentencepiece_nfkc_lite(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t pos = 0; pos < text.size();) {
        std::uint32_t cp = 0;
        std::size_t len = 0;
        if (!utf8_decode_one(text, pos, cp, len)) {
            out.push_back(text[pos++]);
            continue;
        }
        pos += len;
        if (cp == 0x00a0 || cp == 0x3000) {
            out.push_back(' ');
        } else if (cp >= 0xff01 && cp <= 0xff5e) {
            out.push_back(static_cast<char>(cp - 0xfee0));
        } else if (cp == 0x2018 || cp == 0x2019) {
            out.push_back('\'');
        } else if (cp == 0x201c || cp == 0x201d) {
            out.push_back('"');
        } else {
            append_utf8(out, cp);
        }
    }
    return out;
}

std::string sentencepiece_normalize_text(const std::string& text,
                                         bool add_dummy_prefix,
                                         bool remove_extra_whitespaces,
                                         bool escape_whitespaces) {
    auto s = sentencepiece_nfkc_lite(text);
    if (remove_extra_whitespaces) {
        std::string collapsed;
        collapsed.reserve(s.size());
        bool pending_space = false;
        bool emitted = false;
        for (std::size_t pos = 0; pos < s.size();) {
            std::uint32_t cp = 0;
            std::size_t len = 0;
            if (!utf8_decode_one(s, pos, cp, len)) {
                cp = static_cast<unsigned char>(s[pos]);
                len = 1;
            }
            pos += len;
            if (sentencepiece_is_space(cp)) {
                pending_space = emitted;
                continue;
            }
            if (pending_space) collapsed.push_back(' ');
            append_utf8(collapsed, cp);
            pending_space = false;
            emitted = true;
        }
        s = std::move(collapsed);
    }
    if (add_dummy_prefix && (s.empty() || s.front() != ' ')) s.insert(s.begin(), ' ');
    if (escape_whitespaces) replace_all(s, " ", "\xE2\x96\x81");
    return s;
}

} // namespace

SafeTensorsFile SafeTensorsFile::open(const std::string& path) {
    SafeTensorsFile file;
    file.path_ = path;
    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open safetensors file: " + path);
    const auto header_len = read_le_u64(in);
    MCL_CHECK(header_len > 0 && header_len < (1ull << 32), "invalid safetensors header length");
    file.header_.resize(static_cast<std::size_t>(header_len));
    in.read(file.header_.data(), static_cast<std::streamsize>(file.header_.size()));
    MCL_CHECK(in.good(), "failed to read safetensors header");
    file.data_start_ = 8 + header_len;

    std::regex entry_re(R"json("((?:\\.|[^"\\])*)"\s*:\s*\{([^{}]*)\})json");
    for (std::sregex_iterator it(file.header_.begin(), file.header_.end(), entry_re), end; it != end; ++it) {
        const std::string name = json_unescape((*it)[1].str());
        if (name == "__metadata__") continue;
        const std::string object = (*it)[2].str();
        std::smatch dtype_m;
        std::smatch shape_m;
        std::smatch offsets_m;
        if (!std::regex_search(object, dtype_m, std::regex(R"json("dtype"\s*:\s*"([^"]+)")json")) ||
            !std::regex_search(object, shape_m, std::regex(R"json("shape"\s*:\s*\[([^\]]*)\])json")) ||
            !std::regex_search(object, offsets_m, std::regex(R"json("data_offsets"\s*:\s*\[([^\]]*)\])json"))) {
            continue;
        }
        auto offsets = parse_u64_list(offsets_m[1].str());
        MCL_CHECK(offsets.size() == 2, "invalid safetensors data_offsets for " + name);
        SafeTensorInfo info;
        info.name = name;
        info.dtype = dtype_m[1].str();
        info.shape = parse_i64_list(shape_m[1].str());
        info.data_begin = offsets[0];
        info.data_end = offsets[1];
        MCL_CHECK(info.data_end >= info.data_begin, "invalid safetensors offsets for " + name);
        (void)dtype_file_size(info.dtype);
        MCL_CHECK(file.infos_.emplace(name, std::move(info)).second, "duplicate tensor in safetensors header: " + name);
    }
    MCL_CHECK(!file.infos_.empty(), "safetensors header contains no tensors: " + path);
    return file;
}

std::vector<std::string> SafeTensorsFile::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(infos_.size());
    for (const auto& kv : infos_) names.push_back(kv.first);
    std::sort(names.begin(), names.end());
    return names;
}

bool SafeTensorsFile::contains(const std::string& name) const {
    return infos_.find(name) != infos_.end();
}

const SafeTensorInfo& SafeTensorsFile::tensor_info(const std::string& name) const {
    auto it = infos_.find(name);
    MCL_CHECK(it != infos_.end(), "safetensors tensor not found: " + name);
    return it->second;
}

Tensor SafeTensorsFile::load_tensor(Backend& backend, const std::string& name, bool force_f32) const {
    const auto& info = tensor_info(name);
    auto raw = read_tensor_raw(path_, data_start_, info);
    if (info.dtype == "F32" || info.dtype == "F16" || info.dtype == "BF16") {
        if (force_f32 || info.dtype == "BF16") {
            auto values = raw_to_f32_vector(info.dtype, info.shape, raw);
            return tensor_from_f32(backend, info.shape, values);
        }
        MCL_CHECK(info.dtype == "F16", "only F16 can be loaded as non-f32 tensor here");
        return Tensor::from_cpu(backend, Shape(info.shape), DType::F16, raw.data());
    }
    if (info.dtype == "I32") return Tensor::from_cpu(backend, Shape(info.shape), DType::I32, raw.data());
    if (info.dtype == "U8") return Tensor::from_cpu(backend, Shape(info.shape), DType::U8, raw.data());
    MCL_CHECK(false, "unsupported safetensors tensor load dtype: " + info.dtype);
    return {};
}

std::vector<float> SafeTensorsFile::load_f32_vector(const std::string& name) const {
    const auto& info = tensor_info(name);
    auto raw = read_tensor_raw(path_, data_start_, info);
    return raw_to_f32_vector(info.dtype, info.shape, raw);
}

std::unordered_map<std::string, Tensor> load_safetensors(Backend& backend,
                                                         const std::vector<std::string>& paths,
                                                         bool force_f32) {
    std::unordered_map<std::string, Tensor> result;
    for (const auto& path : paths) {
        auto file = SafeTensorsFile::open(path);
        for (const auto& name : file.tensor_names()) {
            MCL_CHECK(result.emplace(name, file.load_tensor(backend, name, force_f32)).second,
                      "duplicate safetensors tensor name: " + name);
        }
    }
    return result;
}

namespace nn {

GemmaConfig load_gemma_config_json(const std::string& path) {
    const auto raw_text = read_text_file(path);
    const auto text_config = extract_json_object_for_key(raw_text, "text_config");
    const bool use_text_config = !text_config.empty() &&
                                 (json_string_or_empty(raw_text, "model_type") == "gemma4" ||
                                  json_string_or_empty(text_config, "model_type").find("gemma4") != std::string::npos);
    const auto& text = use_text_config ? text_config : raw_text;
    GemmaConfig cfg;
    cfg.vocab_size = json_int_or(text, {"vocab_size"}, cfg.vocab_size);
    cfg.max_position_embeddings = json_int_or(text, {"max_position_embeddings", "block_size", "seq_length", "max_seq_len", "max_sequence_length", "context_length"}, cfg.max_position_embeddings);
    cfg.hidden_size = json_int_or(text, {"hidden_size", "n_embd", "d_model"}, cfg.hidden_size);
    cfg.intermediate_size = json_int_or(text, {"intermediate_size", "mlp_hidden", "ffn_hidden_size"}, cfg.intermediate_size);
    cfg.num_hidden_layers = json_int_or(text, {"num_hidden_layers", "n_layer", "num_layers", "n_layers"}, cfg.num_hidden_layers);
    cfg.num_attention_heads = json_int_or(text, {"num_attention_heads", "n_head", "num_heads", "n_heads"}, cfg.num_attention_heads);
    cfg.num_key_value_heads = json_int_or(text, {"num_key_value_heads", "n_kv_head", "num_kv_heads", "n_kv_heads", "kv_heads"}, cfg.num_key_value_heads);
    cfg.head_dim = json_int_or(text, {"head_dim"}, cfg.head_dim);
    cfg.rms_norm_eps = json_float_or(text, {"rms_norm_eps", "layer_norm_eps"}, cfg.rms_norm_eps);
    cfg.rope_theta = json_float_or(text, {"rope_theta", "rope_base"}, cfg.rope_theta);
    cfg.attention_dropout = json_float_or(text, {"attention_dropout", "dropout"}, cfg.attention_dropout);
    cfg.attention_bias = json_bool_or(text, {"attention_bias", "use_qkv_bias"}, cfg.attention_bias);
    cfg.attention_k_eq_v = json_bool_or(text, {"attention_k_eq_v", "k_eq_v", "key_equals_value"}, cfg.attention_k_eq_v);
    cfg.tie_word_embeddings = json_bool_or(text, {"tie_word_embeddings"}, cfg.tie_word_embeddings);
    cfg.bos_token_id = json_int_or(text, {"bos_token_id"}, json_int_or(raw_text, {"bos_token_id"}, cfg.bos_token_id));
    cfg.eos_token_id = json_int_or(text, {"eos_token_id"}, json_int_or(raw_text, {"eos_token_id"}, cfg.eos_token_id));
    cfg.pad_token_id = json_int_or(text, {"pad_token_id"}, json_int_or(raw_text, {"pad_token_id"}, cfg.pad_token_id));
    cfg.sliding_window = json_int_or(text, {"sliding_window"}, cfg.sliding_window);
    return normalize_gemma_config(cfg);
}

TransformerConfig to_transformer_config(const GemmaConfig& raw_cfg) {
    const auto cfg = normalize_gemma_config(raw_cfg);
    TransformerConfig out;
    out.vocab_size = cfg.vocab_size;
    out.block_size = cfg.max_position_embeddings;
    out.n_embd = cfg.hidden_size;
    out.n_head = cfg.num_attention_heads;
    out.n_kv_head = cfg.num_key_value_heads;
    out.head_dim = cfg.head_dim;
    out.n_layer = cfg.num_hidden_layers;
    out.mlp_hidden = cfg.intermediate_size;
    out.dropout = cfg.attention_dropout;
    out.rms_norm_eps = cfg.rms_norm_eps;
    out.embedding_scale = std::sqrt(static_cast<float>(cfg.hidden_size));
    out.use_rope = true;
    out.use_swiglu = true;
    out.use_qkv_bias = cfg.attention_bias;
    out.causal = true;
    out.learned_position_embeddings = false;
    out.rope_theta = cfg.rope_theta;
    out.rotary_dim = cfg.head_dim;
    out.sliding_window = cfg.sliding_window;
    return out;
}

ModernGPTModel make_gemma_model(Backend& backend, const GemmaConfig& cfg) {
    ModernGPTModel model(backend, to_transformer_config(cfg));
    model.final_norm.eps = cfg.rms_norm_eps;
    for (auto& block : model.blocks) {
        block->norm1().eps = cfg.rms_norm_eps;
        block->norm2().eps = cfg.rms_norm_eps;
        block->attention().q_norm().eps = cfg.rms_norm_eps;
        block->attention().k_norm().eps = cfg.rms_norm_eps;
        block->post_attention_norm().eps = cfg.rms_norm_eps;
        block->post_ffw_norm().eps = cfg.rms_norm_eps;
    }
    return model;
}

GemmaWeightName map_gemma_hf_weight_name(const std::string& hf_name) {
    std::string name = hf_name;
    const std::string text_prefix = "model.language_model.";
    if (name.rfind(text_prefix, 0) == 0) {
        const auto rest = name.substr(text_prefix.size());
        name = rest == "lm_head.weight" ? rest : "model." + rest;
    }
    if (name == "model.lm_head.weight") name = "lm_head.weight";
    if (name == "model.embed_tokens.weight") return {hf_name, "token_embedding.weight", "token_embedding", -1};
    if (name == "lm_head.weight") return {hf_name, "lm_head.weight", "lm_head", -1};
    if (name == "model.norm.weight") return {hf_name, "final_norm.weight", "final_norm", -1};

    std::smatch m;
    std::regex layer_re(R"(^model\.layers\.(\d+)\.(.+)$)");
    if (!std::regex_match(name, m, layer_re)) return {hf_name, hf_name, "unknown", -1};
    const int layer = std::stoi(m[1].str());
    const std::string suffix = m[2].str();
    const std::string base = "blocks." + std::to_string(layer) + ".";
    if (suffix == "input_layernorm.weight") return {hf_name, base + "norm1.weight", "input_layernorm", layer};
    if (suffix == "post_attention_layernorm.weight") return {hf_name, base + "norm2.weight", "post_attention_layernorm", layer};
    if (suffix == "pre_feedforward_layernorm.weight") return {hf_name, base + "norm2.weight", "pre_feedforward_layernorm", layer};
    if (suffix == "self_attn.q_proj.weight") return {hf_name, base + "attention.q_proj.weight", "q_proj", layer};
    if (suffix == "self_attn.k_proj.weight") return {hf_name, base + "attention.k_proj.weight", "k_proj", layer};
    if (suffix == "self_attn.v_proj.weight") return {hf_name, base + "attention.v_proj.weight", "v_proj", layer};
    if (suffix == "self_attn.o_proj.weight") return {hf_name, base + "attention.o_proj.weight", "o_proj", layer};
    if (suffix == "self_attn.q_proj.bias") return {hf_name, base + "attention.q_proj.bias", "q_proj_bias", layer};
    if (suffix == "self_attn.k_proj.bias") return {hf_name, base + "attention.k_proj.bias", "k_proj_bias", layer};
    if (suffix == "self_attn.v_proj.bias") return {hf_name, base + "attention.v_proj.bias", "v_proj_bias", layer};
    if (suffix == "self_attn.o_proj.bias") return {hf_name, base + "attention.o_proj.bias", "o_proj_bias", layer};
    if (suffix == "mlp.gate_proj.weight") return {hf_name, base + "mlp.gate_proj.weight", "gate_proj", layer};
    if (suffix == "mlp.up_proj.weight") return {hf_name, base + "mlp.up_proj.weight", "up_proj", layer};
    if (suffix == "mlp.down_proj.weight") return {hf_name, base + "mlp.down_proj.weight", "down_proj", layer};
    return {hf_name, hf_name, "unknown", layer};
}

std::vector<std::string> expected_gemma_hf_weight_names(const GemmaConfig& raw_cfg, bool include_lm_head) {
    const auto cfg = normalize_gemma_config(raw_cfg);
    std::vector<std::string> names{"model.embed_tokens.weight", "model.norm.weight"};
    if (include_lm_head) names.push_back("lm_head.weight");
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string p = "model.layers." + std::to_string(i) + ".";
        names.push_back(p + "input_layernorm.weight");
        names.push_back(p + "post_attention_layernorm.weight");
        names.push_back(p + "self_attn.q_proj.weight");
        names.push_back(p + "self_attn.k_proj.weight");
        names.push_back(p + "self_attn.v_proj.weight");
        names.push_back(p + "self_attn.o_proj.weight");
        if (cfg.attention_bias) {
            names.push_back(p + "self_attn.q_proj.bias");
            names.push_back(p + "self_attn.k_proj.bias");
            names.push_back(p + "self_attn.v_proj.bias");
            names.push_back(p + "self_attn.o_proj.bias");
        }
        names.push_back(p + "mlp.gate_proj.weight");
        names.push_back(p + "mlp.up_proj.weight");
        names.push_back(p + "mlp.down_proj.weight");
    }
    return names;
}

GemmaWeightLoadReport load_gemma_hf_weights(Backend& backend,
                                            ModernGPTModel& model,
                                            const std::vector<std::string>& safetensors_paths,
                                            const GemmaConfig& raw_cfg,
                                            bool strict,
                                            bool trainable) {
    const auto cfg = normalize_gemma_config(raw_cfg);
    MCL_CHECK(model.config.vocab_size == cfg.vocab_size && model.config.n_embd == cfg.hidden_size &&
              model.config.n_layer == cfg.num_hidden_layers && model.config.n_head == cfg.num_attention_heads &&
              model.config.n_kv_head == cfg.num_key_value_heads,
              "Gemma config does not match ModernGPTModel config");
    SafeTensorsArchive archive(safetensors_paths);
    GemmaWeightLoadReport report;

    auto aliases_for = [](const std::string& canonical) {
        std::vector<std::string> aliases{canonical};
        if (canonical == "lm_head.weight") {
            aliases.push_back("model.lm_head.weight");
            aliases.push_back("model.language_model.lm_head.weight");
        }
        const std::string model_prefix = "model.";
        if (canonical.rfind(model_prefix, 0) == 0) {
            aliases.push_back("model.language_model." + canonical.substr(model_prefix.size()));
        }
        return aliases;
    };

    auto first_existing = [&](const std::vector<std::string>& aliases) -> std::string {
        for (const auto& name : aliases) {
            if (archive.contains(name)) return name;
        }
        return {};
    };

    auto any_existing = [&](const std::vector<std::string>& aliases) -> bool {
        return !first_existing(aliases).empty();
    };

    auto layer_aliases = [&](int layer, const std::string& suffix) {
        return aliases_for("model.layers." + std::to_string(layer) + "." + suffix);
    };

    auto k_alias_for_v = [&](const std::string& v_name) {
        auto k_name = v_name;
        const auto pos = k_name.find("self_attn.v_proj.");
        if (pos != std::string::npos) k_name.replace(pos, std::string("self_attn.v_proj.").size(), "self_attn.k_proj.");
        return aliases_for(k_name);
    };

    const auto expected = expected_gemma_hf_weight_names(cfg, !cfg.tie_word_embeddings);
    std::unordered_set<std::string> expected_set;
    for (const auto& name : expected) {
        for (const auto& alias : aliases_for(name)) expected_set.insert(alias);
    }
    if (cfg.tie_word_embeddings) {
        for (const auto& alias : aliases_for("lm_head.weight")) expected_set.insert(alias);
    }
    for (const auto& name : archive.names()) {
        if (expected_set.find(name) == expected_set.end()) report.unexpected.push_back(name);
    }
    for (const auto& name : expected) {
        const auto aliases = aliases_for(name);
        if (any_existing(aliases)) continue;
        if (cfg.attention_k_eq_v && name.find(".self_attn.v_proj.") != std::string::npos && any_existing(k_alias_for_v(name))) {
            continue;
        }
        report.missing.push_back(name);
    }
    if (strict && !report.missing.empty()) {
        MCL_CHECK(false, "missing required Gemma HF weights; first missing: " + report.missing.front());
    }

    auto apply_direct = [&](const std::vector<std::string>& aliases, Parameter& parameter) -> bool {
        const auto name = first_existing(aliases);
        if (name.empty()) return false;
        const auto values = archive.f32(name);
        assign_parameter(parameter, tensor_from_f32(backend, archive.info(name).shape, values), trainable);
        report.applied.push_back(name);
        ++report.loaded_tensors;
        return true;
    };
    auto apply_transposed = [&](const std::vector<std::string>& aliases, Parameter& parameter) -> bool {
        const auto name = first_existing(aliases);
        if (name.empty()) return false;
        const auto& info = archive.info(name);
        MCL_CHECK(info.shape.size() == 2, "expected rank-2 HF weight: " + name);
        const auto values = transpose_2d(archive.f32(name), info.shape[0], info.shape[1]);
        assign_parameter(parameter, tensor_from_f32(backend, {info.shape[1], info.shape[0]}, values), trainable);
        report.applied.push_back(name);
        ++report.loaded_tensors;
        return true;
    };

    const auto embed_aliases = aliases_for("model.embed_tokens.weight");
    apply_direct(embed_aliases, model.token_embedding.weight);
    apply_direct(aliases_for("model.norm.weight"), model.final_norm.weight);
    if (!apply_transposed(aliases_for("lm_head.weight"), model.lm_head) &&
        cfg.tie_word_embeddings) {
        const auto embed_name = first_existing(embed_aliases);
        if (!embed_name.empty()) {
            const auto& info = archive.info(embed_name);
            const auto values = transpose_2d(archive.f32(embed_name), info.shape[0], info.shape[1]);
            assign_parameter(model.lm_head, tensor_from_f32(backend, {info.shape[1], info.shape[0]}, values), trainable);
            report.applied.push_back(embed_name + "->lm_head.weight");
        }
    }

    const int64_t hidden = cfg.hidden_size;
    const int64_t mlp_hidden = cfg.intermediate_size;

    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
        auto& block = *model.blocks[static_cast<std::size_t>(i)];
        apply_direct(layer_aliases(i, "input_layernorm.weight"), block.norm1().weight);
        if (!apply_direct(layer_aliases(i, "post_attention_layernorm.weight"), block.norm2().weight)) {
            apply_direct(layer_aliases(i, "pre_feedforward_layernorm.weight"), block.norm2().weight);
        }
        apply_transposed(layer_aliases(i, "self_attn.o_proj.weight"), block.attention().o_proj().weight);
        if (block.attention().o_proj().has_bias()) apply_direct(layer_aliases(i, "self_attn.o_proj.bias"), block.attention().o_proj().bias);
        apply_transposed(layer_aliases(i, "mlp.down_proj.weight"), block.mlp().down_proj.weight);

        const int64_t q_dim = block.attention().q_proj().out_features();
        const int64_t kv_dim = block.attention().k_proj().out_features();
        const int64_t qkv_dim = block.attention().qkv_proj().out_features();
        MCL_CHECK(qkv_dim == q_dim + 2 * kv_dim, "ModernSelfAttention qkv shape mismatch for Gemma layer " + std::to_string(i));

        const auto q_name = first_existing(layer_aliases(i, "self_attn.q_proj.weight"));
        const auto k_name = first_existing(layer_aliases(i, "self_attn.k_proj.weight"));
        const auto v_name = first_existing(layer_aliases(i, "self_attn.v_proj.weight"));
        const bool v_from_k = v_name.empty() && cfg.attention_k_eq_v && !k_name.empty();
        if (!q_name.empty() && !k_name.empty() && (!v_name.empty() || v_from_k)) {
            const auto& q_info = archive.info(q_name);
            const auto& k_info = archive.info(k_name);
            const auto& v_info = archive.info(v_from_k ? k_name : v_name);
            MCL_CHECK(shape_is(q_info, {q_dim, hidden}), "q_proj shape mismatch: " + q_name);
            MCL_CHECK(shape_is(k_info, {kv_dim, hidden}), "k_proj shape mismatch: " + k_name);
            MCL_CHECK(shape_is(v_info, {kv_dim, hidden}), "v_proj shape mismatch: " + (v_from_k ? k_name : v_name));
            std::vector<float> packed(static_cast<std::size_t>(hidden * qkv_dim));
            const auto q_values = archive.f32(q_name);
            const auto k_values = archive.f32(k_name);
            const auto v_values = v_from_k ? k_values : archive.f32(v_name);
            pack_transposed_projection(packed, qkv_dim, 0, q_values, q_dim, hidden);
            pack_transposed_projection(packed, qkv_dim, q_dim, k_values, kv_dim, hidden);
            pack_transposed_projection(packed, qkv_dim, q_dim + kv_dim, v_values, kv_dim, hidden);
            assign_parameter(block.attention().qkv_proj().weight, tensor_from_f32(backend, {hidden, qkv_dim}, packed), trainable);
            report.applied.push_back(q_name + "+" + k_name + "+" + (v_from_k ? (k_name + "(as_v)") : v_name) + "->qkv_proj.weight");
            report.loaded_tensors += v_from_k ? 2 : 3;
        }

        const auto qb_name = first_existing(layer_aliases(i, "self_attn.q_proj.bias"));
        const auto kb_name = first_existing(layer_aliases(i, "self_attn.k_proj.bias"));
        const auto vb_name = first_existing(layer_aliases(i, "self_attn.v_proj.bias"));
        const bool vb_from_k = vb_name.empty() && cfg.attention_k_eq_v && !kb_name.empty();
        if (block.attention().qkv_proj().has_bias() &&
            !qb_name.empty() && !kb_name.empty() && (!vb_name.empty() || vb_from_k)) {
            MCL_CHECK(shape_is(archive.info(qb_name), {q_dim}), "q_proj bias shape mismatch: " + qb_name);
            MCL_CHECK(shape_is(archive.info(kb_name), {kv_dim}), "k_proj bias shape mismatch: " + kb_name);
            MCL_CHECK(shape_is(archive.info(vb_from_k ? kb_name : vb_name), {kv_dim}),
                      "v_proj bias shape mismatch: " + (vb_from_k ? kb_name : vb_name));
            std::vector<float> packed(static_cast<std::size_t>(qkv_dim));
            auto qv = archive.f32(qb_name);
            auto kv = archive.f32(kb_name);
            auto vv = vb_from_k ? kv : archive.f32(vb_name);
            std::copy(qv.begin(), qv.end(), packed.begin());
            std::copy(kv.begin(), kv.end(), packed.begin() + q_dim);
            std::copy(vv.begin(), vv.end(), packed.begin() + q_dim + kv_dim);
            assign_parameter(block.attention().qkv_proj().bias, tensor_from_f32(backend, {qkv_dim}, packed), trainable);
            report.applied.push_back(qb_name + "+" + kb_name + "+" + (vb_from_k ? (kb_name + "(as_v)") : vb_name) + "->qkv_proj.bias");
            report.loaded_tensors += vb_from_k ? 2 : 3;
        }

        const auto gate_name = first_existing(layer_aliases(i, "mlp.gate_proj.weight"));
        const auto up_name = first_existing(layer_aliases(i, "mlp.up_proj.weight"));
        if (!gate_name.empty() && !up_name.empty()) {
            MCL_CHECK(shape_is(archive.info(gate_name), {mlp_hidden, hidden}), "gate_proj shape mismatch: " + gate_name);
            MCL_CHECK(shape_is(archive.info(up_name), {mlp_hidden, hidden}), "up_proj shape mismatch: " + up_name);
            std::vector<float> packed(static_cast<std::size_t>(hidden * mlp_hidden * 2));
            pack_transposed_projection(packed, mlp_hidden * 2, 0, archive.f32(gate_name), mlp_hidden, hidden);
            pack_transposed_projection(packed, mlp_hidden * 2, mlp_hidden, archive.f32(up_name), mlp_hidden, hidden);
            assign_parameter(block.mlp().gate_up_proj.weight, tensor_from_f32(backend, {hidden, mlp_hidden * 2}, packed), trainable);
            report.applied.push_back(gate_name + "+" + up_name + "->gate_up_proj.weight");
            report.loaded_tensors += 2;
        }
    }

    return report;
}

GemmaTokenizer GemmaTokenizer::byte_fallback(int vocab_size, int bos_token_id, int eos_token_id) {
    GemmaTokenizer tok;
    tok.vocab_size_ = std::max(vocab_size, 256);
    tok.bos_token_id_ = bos_token_id;
    tok.eos_token_id_ = eos_token_id;
    tok.byte_fallback_ = true;
    return tok;
}

GemmaTokenizer GemmaTokenizer::from_tokens(const std::vector<std::string>& tokens,
                                           int bos_token_id,
                                           int eos_token_id,
                                           const std::string& tokenizer_model_type) {
    if (tokens.empty()) return byte_fallback(256, bos_token_id, eos_token_id);
    GemmaTokenizer tok;
    tok.byte_fallback_ = false;
    tok.bos_token_id_ = bos_token_id;
    tok.eos_token_id_ = eos_token_id;
    tok.tokenizer_model_type_ = tokenizer_model_type.empty() ? "GGUF" : tokenizer_model_type;
    tok.vocab_size_ = static_cast<int>(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const auto id = static_cast<std::int32_t>(i);
        tok.token_to_id_[tokens[i]] = id;
        tok.id_to_token_[id] = tokens[i];
        if (tokens[i].find("\xE2\x96\x81") != std::string::npos) tok.sentencepiece_style_ = true;
    }
    const auto model_type = lower_ascii(tok.tokenizer_model_type_);
    if (model_type.find("unigram") != std::string::npos ||
        model_type.find("sentencepiece") != std::string::npos) {
        tok.sentencepiece_style_ = true;
    }
    if (tok.sentencepiece_style_) {
        tok.sp_add_dummy_prefix_ = true;
        tok.sp_remove_extra_whitespaces_ = true;
        tok.sp_escape_whitespaces_ = true;
        tok.sp_normalizer_name_ = "nmt_nfkc";
    }
    return tok;
}

GemmaTokenizer GemmaTokenizer::load_vocab(const std::string& path, int bos_token_id, int eos_token_id) {
    GemmaTokenizer tok;
    tok.byte_fallback_ = false;
    tok.bos_token_id_ = bos_token_id;
    tok.eos_token_id_ = eos_token_id;
    const auto text = read_text_file(path);
    const auto path_ext = lower_ascii(std::filesystem::path(path).extension().string());
    const auto filename = lower_ascii(std::filesystem::path(path).filename().string());
    const bool is_plain_vocab_json = filename == "vocab.json";
    const auto model_object_for_type = is_plain_vocab_json ? std::string{} : extract_json_object_for_key(text, "model");
    tok.tokenizer_model_type_ = is_plain_vocab_json ? "BPE" :
        json_string_or_empty(model_object_for_type.empty() ? text : model_object_for_type, "type");
    if (tok.tokenizer_model_type_.empty()) tok.tokenizer_model_type_ = "vocab";

    auto add_token = [&](const std::string& token, std::int32_t id) {
        tok.token_to_id_[token] = id;
        tok.id_to_token_[id] = token;
        tok.vocab_size_ = std::max(tok.vocab_size_, static_cast<int>(id) + 1);
        if (token.find("\xE2\x96\x81") != std::string::npos) tok.sentencepiece_style_ = true;
    };

    if (path_ext == ".model") {
        const auto sp = parse_sentencepiece_model_proto(text);
        if (!sp.pieces.empty()) {
            tok.tokenizer_model_type_ = sp.model_type.empty() ? "SentencePiece" : sp.model_type;
            tok.sentencepiece_style_ = true;
            tok.sp_add_dummy_prefix_ = sp.add_dummy_prefix;
            tok.sp_remove_extra_whitespaces_ = sp.remove_extra_whitespaces;
            tok.sp_escape_whitespaces_ = sp.escape_whitespaces;
            tok.sp_normalizer_name_ = sp.normalizer_name;
            for (std::size_t i = 0; i < sp.pieces.size(); ++i) {
                add_token(sp.pieces[i], static_cast<std::int32_t>(i));
            }
            return tok;
        }
    }

    if (is_plain_vocab_json) {
        for (const auto& kv : json_object_int_pairs(text)) {
            add_token(kv.first, kv.second);
        }
    }

    const std::string vocab_object = is_plain_vocab_json ? std::string{} : extract_json_object_for_key(text, "vocab");
    if (!vocab_object.empty()) {
        for (const auto& kv : json_object_int_pairs(vocab_object)) {
            add_token(kv.first, kv.second);
        }
    }

    if (tok.token_to_id_.empty()) {
        const std::string vocab_array = extract_json_array_for_key(text, "vocab");
        if (!vocab_array.empty()) {
            std::int32_t id = 0;
            for (std::size_t i = 0; i < vocab_array.size();) {
                if (vocab_array[i] != '[') {
                    ++i;
                    continue;
                }
                ++i;
                while (i < vocab_array.size() && std::isspace(static_cast<unsigned char>(vocab_array[i]))) ++i;
                if (i >= vocab_array.size() || vocab_array[i] != '"') continue;
                ++i;
                std::string raw;
                bool escape = false;
                for (; i < vocab_array.size(); ++i) {
                    const char c = vocab_array[i];
                    if (escape) {
                        raw.push_back('\\');
                        raw.push_back(c);
                        escape = false;
                    } else if (c == '\\') {
                        escape = true;
                    } else if (c == '"') {
                        ++i;
                        break;
                    } else {
                        raw.push_back(c);
                    }
                }
                add_token(json_unescape(raw), id++);
            }
        }
    }

    if (tok.token_to_id_.empty()) {
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ls(line);
            std::string first;
            std::string second;
            ls >> first >> second;
            if (first.empty() || second.empty()) continue;
            std::int32_t id = 0;
            std::string token;
            if (std::all_of(first.begin(), first.end(), [](unsigned char c) { return std::isdigit(c) != 0; })) {
                id = static_cast<std::int32_t>(std::stol(first));
                token = second;
            } else {
                token = first;
                id = static_cast<std::int32_t>(std::stol(second));
            }
            add_token(token, id);
        }
    }

    if (!is_plain_vocab_json) {
        const std::string added_array = extract_json_array_for_key(text, "added_tokens");
        for (std::size_t i = 0; i < added_array.size();) {
            if (added_array[i] != '{') {
                ++i;
                continue;
            }
            const auto end = find_matching_json(added_array, i, '{', '}');
            if (end == std::string::npos) break;
            const auto object = added_array.substr(i + 1, end - i - 1);
            const auto content = json_string_or_empty(object, "content");
            const auto id = json_int_or(object, {"id"}, -1);
            if (!content.empty() && id >= 0 && id <= std::numeric_limits<std::int32_t>::max()) {
                add_token(content, static_cast<std::int32_t>(id));
            }
            i = end + 1;
        }
    }

    auto add_merge = [&](const std::string& left, const std::string& right) {
        if (left.empty() || right.empty()) return;
        const auto key = bpe_pair_key(left, right);
        if (tok.bpe_merge_rank_.find(key) == tok.bpe_merge_rank_.end()) {
            tok.bpe_merge_rank_[key] = static_cast<int>(tok.bpe_merge_rank_.size());
        }
    };
    const std::string merges_array = extract_json_array_for_key(text, "merges");
    if (!merges_array.empty()) {
        std::vector<std::string> pairless;
        for (const auto& item : json_strings_in(merges_array)) {
            std::istringstream ls(item);
            std::string left;
            std::string right;
            if (ls >> left >> right) {
                add_merge(left, right);
            } else {
                pairless.push_back(item);
            }
        }
        if (tok.bpe_merge_rank_.empty() && pairless.size() >= 2) {
            for (std::size_t i = 0; i + 1 < pairless.size(); i += 2) add_merge(pairless[i], pairless[i + 1]);
        }
    }
    if (tok.bpe_merge_rank_.empty()) {
        const auto merges_path = std::filesystem::path(path).parent_path() / "merges.txt";
        if (std::filesystem::exists(merges_path)) {
            std::istringstream in(read_text_file(merges_path.string()));
            std::string line;
            while (std::getline(in, line)) {
                line = trim(line);
                if (line.empty() || line[0] == '#') continue;
                std::istringstream ls(line);
                std::string left;
                std::string right;
                if (ls >> left >> right) add_merge(left, right);
            }
        }
    }

    const auto model_type = lower_ascii(tok.tokenizer_model_type_);
    if (model_type.find("unigram") != std::string::npos ||
        model_type.find("sentencepiece") != std::string::npos) {
        tok.sentencepiece_style_ = true;
    }
    if (!tok.bpe_merge_rank_.empty() && model_type.find("bpe") == std::string::npos) {
        tok.tokenizer_model_type_ = "BPE";
    }
    if (tok.token_to_id_.empty()) return byte_fallback(256, bos_token_id, eos_token_id);
    return tok;
}

std::vector<std::int32_t> GemmaTokenizer::encode(const std::string& text, bool add_bos, bool add_eos) const {
    std::vector<std::int32_t> ids;
    if (add_bos && bos_token_id_ >= 0) ids.push_back(bos_token_id_);
    if (byte_fallback_) {
        for (unsigned char c : text) ids.push_back(static_cast<std::int32_t>(c));
    } else {
        auto push_longest_match = [&](const std::string& segment) {
            for (std::size_t pos = 0; pos < segment.size();) {
                std::int32_t best_id = -1;
                std::size_t best_len = 0;
                for (const auto& kv : token_to_id_) {
                    const auto& token = kv.first;
                    if (token.empty() || pos + token.size() > segment.size()) continue;
                    if (token.size() <= best_len) continue;
                    if (segment.compare(pos, token.size(), token) == 0) {
                        best_id = kv.second;
                        best_len = token.size();
                    }
                }
                if (best_id >= 0) {
                    ids.push_back(best_id);
                    pos += best_len;
                } else {
                    if (sentencepiece_style_ && sp_escape_whitespaces_ &&
                        segment.compare(pos, 3, "\xE2\x96\x81") == 0) {
                        pos += 3;
                        continue;
                    }
                    ids.push_back(static_cast<std::int32_t>(static_cast<unsigned char>(segment[pos])));
                    ++pos;
                }
            }
        };

        std::string normalized = text;
        if (sentencepiece_style_) {
            normalized = sentencepiece_normalize_text(text,
                                                      sp_add_dummy_prefix_,
                                                      sp_remove_extra_whitespaces_,
                                                      sp_escape_whitespaces_);
        }

        if (!bpe_merge_rank_.empty()) {
            auto pieces = utf8_pieces(normalized);
            while (pieces.size() > 1) {
                int best_rank = std::numeric_limits<int>::max();
                std::size_t best_pos = pieces.size();
                for (std::size_t i = 0; i + 1 < pieces.size(); ++i) {
                    const auto it = bpe_merge_rank_.find(bpe_pair_key(pieces[i], pieces[i + 1]));
                    if (it != bpe_merge_rank_.end() && it->second < best_rank) {
                        best_rank = it->second;
                        best_pos = i;
                    }
                }
                if (best_pos >= pieces.size()) break;
                pieces[best_pos] += pieces[best_pos + 1];
                pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best_pos + 1));
            }
            for (const auto& piece : pieces) {
                const auto it = token_to_id_.find(piece);
                if (it != token_to_id_.end()) ids.push_back(it->second);
                else push_longest_match(piece);
            }
        } else {
            push_longest_match(normalized);
        }
    }
    if (add_eos && eos_token_id_ >= 0) ids.push_back(eos_token_id_);
    return ids;
}

std::string GemmaTokenizer::decode(const std::vector<std::int32_t>& ids, bool skip_special) const {
    std::string out;
    for (auto id : ids) {
        if (skip_special && (id == bos_token_id_ || id == eos_token_id_)) continue;
        if (byte_fallback_) {
            if (id >= 0 && id <= 255) out.push_back(static_cast<char>(id));
        } else {
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) {
                out += it->second;
            } else if (id >= 0 && id <= 255) {
                out.push_back(static_cast<char>(id));
            }
        }
    }
    if (sentencepiece_style_ && sp_escape_whitespaces_) replace_all(out, "\xE2\x96\x81", " ");
    if (sentencepiece_style_ && sp_add_dummy_prefix_ && !out.empty() && out.front() == ' ') out.erase(out.begin());
    return out;
}

QuantizedLinear::QuantizedLinear(const Tensor& f32_weight, DType qdtype)
    : QuantizedLinear(f32_weight, Tensor{}, qdtype) {}

QuantizedLinear::QuantizedLinear(const Tensor& f32_weight, const Tensor& bias, DType qdtype) {
    MCL_CHECK(f32_weight.valid() && f32_weight.dtype() == DType::F32 && f32_weight.ndim() == 2,
              "QuantizedLinear expects rank-2 f32 weight");
    MCL_CHECK(qdtype == DType::Q8_0 || qdtype == DType::Q4_0, "QuantizedLinear qdtype must be Q8_0 or Q4_0");
    in_features_ = static_cast<int>(f32_weight.shape()[0]);
    out_features_ = static_cast<int>(f32_weight.shape()[1]);
    weight_dtype_ = qdtype;
    weight_ = qdtype == DType::Q4_0 ? quantize_q4_symmetric_cols(f32_weight) : quantize_q8_symmetric_cols(f32_weight);
    if (bias.valid()) {
        MCL_CHECK(bias.dtype() == DType::F32 && bias.ndim() == 1 && bias.shape()[0] == out_features_,
                  "QuantizedLinear bias shape mismatch");
        bias_ = bias;
        use_bias_ = true;
    }
}

QuantizedLinear QuantizedLinear::from_linear(const Linear& linear, DType qdtype) {
    if (linear.has_bias()) return QuantizedLinear(linear.weight.data, linear.bias.data, qdtype);
    return QuantizedLinear(linear.weight.data, qdtype);
}

Tensor QuantizedLinear::forward(const Tensor& x) {
    MCL_CHECK(weight_.valid(), "QuantizedLinear is not initialized");
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "QuantizedLinear expects rank-2 f32 input");
    MCL_CHECK(x.shape()[1] == in_features_, "QuantizedLinear input feature mismatch");
    auto xq = quantize_q8_symmetric_rows(x);
    auto y = matmul(xq, weight_);
    return use_bias_ ? add_bias_rows(y, bias_) : y;
}

std::vector<std::int32_t> generate(Backend& backend,
                                   ModernGPTModel& model,
                                   const std::vector<std::int32_t>& prompt_tokens,
                                   const GenerateOptions& options) {
    MCL_CHECK(options.max_new_tokens >= 0, "GenerateOptions max_new_tokens must be non-negative");
    autograd::NoGradGuard no_grad;
    std::vector<std::int32_t> tokens;
    if (options.add_bos) tokens.push_back(options.bos_token_id);
    tokens.insert(tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
    MCL_CHECK(!tokens.empty(), "generate requires a non-empty prompt or add_bos=true");
    MCL_CHECK(static_cast<int>(tokens.size()) <= model.config.block_size, "prompt exceeds model block_size");
    MCL_CHECK(options.kv_page_size > 0, "GenerateOptions kv_page_size must be positive");
    if (options.max_new_tokens == 0) return tokens;

    Tensor logits;
    if (options.use_paged_kv_cache) {
        auto caches = model.create_paged_kv_cache(backend, 1, options.kv_page_size);
        if (options.prefill_prompt) {
            auto input = Tensor::from_cpu(backend,
                                          {1, static_cast<int64_t>(tokens.size())},
                                          DType::I32,
                                          tokens.data());
            logits = model.forward_with_cache_last_logits(input, caches);
        } else {
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                std::int32_t id = tokens[i];
                auto input = Tensor::from_cpu(backend, {1, 1}, DType::I32, &id);
                logits = model.decode_step(input, caches);
            }
        }

        std::mt19937 rng(options.seed);
        std::vector<float> cpu_logits;
        for (int step = 0; step < options.max_new_tokens; ++step) {
            MCL_CHECK(static_cast<int>(tokens.size()) < model.config.block_size, "generate reached model block_size");
            Tensor next_tensor;
            std::int32_t next = 0;
            if (options.gpu_greedy_sampling) {
                next_tensor = rowwise_sample_top_p(logits, options.temperature, options.top_k, options.top_p,
                                                   static_cast<std::uint32_t>(rng()));
                next = next_tensor.to_vector<std::int32_t>()[0];
            } else {
                next = choose_next_token(logits, options, rng, cpu_logits);
            }
            tokens.push_back(next);
            if (options.eos_token_id >= 0 && next == options.eos_token_id) break;
            if (step + 1 >= options.max_new_tokens) break;
            auto input = options.gpu_greedy_sampling
                ? next_tensor.view({1, 1})
                : Tensor::from_cpu(backend, {1, 1}, DType::I32, &next);
            logits = model.decode_step(input, caches);
        }
        return tokens;
    } else {
        auto caches = model.create_kv_cache(backend, 1);
        if (options.prefill_prompt) {
            auto input = Tensor::from_cpu(backend,
                                          {1, static_cast<int64_t>(tokens.size())},
                                          DType::I32,
                                          tokens.data());
            logits = model.forward_with_cache_last_logits(input, caches);
        } else {
            for (std::size_t i = 0; i < tokens.size(); ++i) {
                std::int32_t id = tokens[i];
                auto input = Tensor::from_cpu(backend, {1, 1}, DType::I32, &id);
                logits = model.decode_step(input, caches);
            }
        }

        std::mt19937 rng(options.seed);
        std::vector<float> cpu_logits;
        for (int step = 0; step < options.max_new_tokens; ++step) {
            MCL_CHECK(static_cast<int>(tokens.size()) < model.config.block_size, "generate reached model block_size");
            Tensor next_tensor;
            std::int32_t next = 0;
            if (options.gpu_greedy_sampling) {
                next_tensor = rowwise_sample_top_p(logits, options.temperature, options.top_k, options.top_p,
                                                   static_cast<std::uint32_t>(rng()));
                next = next_tensor.to_vector<std::int32_t>()[0];
            } else {
                next = choose_next_token(logits, options, rng, cpu_logits);
            }
            tokens.push_back(next);
            if (options.eos_token_id >= 0 && next == options.eos_token_id) break;
            if (step + 1 >= options.max_new_tokens) break;
            auto input = options.gpu_greedy_sampling
                ? next_tensor.view({1, 1})
                : Tensor::from_cpu(backend, {1, 1}, DType::I32, &next);
            logits = model.decode_step(input, caches);
        }
        return tokens;
    }
}

std::string generate_text(Backend& backend,
                          ModernGPTModel& model,
                          const GemmaTokenizer& tokenizer,
                          const std::string& prompt,
                          const GenerateOptions& options) {
    auto prompt_ids = tokenizer.encode(prompt, options.add_bos, false);
    auto local_options = options;
    local_options.add_bos = false;
    if (local_options.eos_token_id < 0) local_options.eos_token_id = tokenizer.eos_token_id();
    return tokenizer.decode(generate(backend, model, prompt_ids, local_options), true);
}

std::vector<std::vector<std::int32_t>> generate_batch(
    Backend& backend,
    ModernGPTModel& model,
    const std::vector<std::vector<std::int32_t>>& prompt_tokens,
    const GenerateOptions& options) {
    MCL_CHECK(options.max_new_tokens >= 0, "GenerateOptions max_new_tokens must be non-negative");
    if (prompt_tokens.empty()) return {};
    if (!options.prefill_prompt || options.use_paged_kv_cache) {
        std::vector<std::vector<std::int32_t>> debug_out;
        debug_out.reserve(prompt_tokens.size());
        for (const auto& prompt : prompt_tokens) debug_out.push_back(generate(backend, model, prompt, options));
        return debug_out;
    }

    autograd::NoGradGuard no_grad;
    const int64_t batch_size = static_cast<int64_t>(prompt_tokens.size());
    std::vector<std::vector<std::int32_t>> out(prompt_tokens.size());
    int64_t max_prompt = 0;
    for (std::size_t b = 0; b < prompt_tokens.size(); ++b) {
        if (options.add_bos) out[b].push_back(options.bos_token_id);
        out[b].insert(out[b].end(), prompt_tokens[b].begin(), prompt_tokens[b].end());
        MCL_CHECK(!out[b].empty(), "generate_batch requires non-empty prompts or add_bos=true");
        MCL_CHECK(static_cast<int>(out[b].size()) <= model.config.block_size, "prompt exceeds model block_size");
        max_prompt = std::max<int64_t>(max_prompt, static_cast<int64_t>(out[b].size()));
    }
    if (options.max_new_tokens == 0) return out;

    const std::int32_t pad = options.pad_token_id >= 0
        ? static_cast<std::int32_t>(options.pad_token_id)
        : static_cast<std::int32_t>(std::max(options.eos_token_id, 0));
    std::vector<std::int32_t> input_ids(static_cast<std::size_t>(batch_size * max_prompt), pad);
    std::vector<std::int32_t> mask(static_cast<std::size_t>(batch_size * model.config.block_size), 1);
    std::vector<std::int32_t> last_positions(static_cast<std::size_t>(batch_size), 0);
    for (int64_t b = 0; b < batch_size; ++b) {
        const auto& row = out[static_cast<std::size_t>(b)];
        last_positions[static_cast<std::size_t>(b)] = static_cast<std::int32_t>(row.size() - 1);
        for (std::size_t t = 0; t < row.size(); ++t) {
            input_ids[static_cast<std::size_t>(b * max_prompt + static_cast<int64_t>(t))] = row[t];
            mask[static_cast<std::size_t>(b * model.config.block_size + static_cast<int64_t>(t))] = 0;
        }
    }

    auto caches = model.create_kv_cache(backend, batch_size);
    auto input = Tensor::from_cpu(backend, {batch_size, max_prompt}, DType::I32, input_ids.data());
    auto mask_tensor = Tensor::from_cpu(backend, {batch_size, model.config.block_size}, DType::I32, mask.data());
    auto logits = last_token_logits_batch_tensor(
        model.forward_with_cache_masked(input, mask_tensor, caches),
        last_positions,
        batch_size,
        max_prompt,
        model.config.vocab_size);

    std::mt19937 rng(options.seed);
    std::vector<char> active(static_cast<std::size_t>(batch_size), 1);
    std::vector<std::int32_t> row_lengths(static_cast<std::size_t>(batch_size), 0);
    for (int64_t b = 0; b < batch_size; ++b) {
        row_lengths[static_cast<std::size_t>(b)] = static_cast<std::int32_t>(out[static_cast<std::size_t>(b)].size());
    }
    for (int step = 0; step < options.max_new_tokens; ++step) {
        const auto next_tokens = choose_next_tokens(logits, options, rng, batch_size, model.config.vocab_size);

        std::vector<std::int32_t> step_ids(static_cast<std::size_t>(batch_size), pad);
        std::vector<std::int32_t> step_positions(static_cast<std::size_t>(batch_size), -1);
        bool any_active_after = false;
        int64_t cache_length_after = caches.empty() ? max_prompt : caches[0].length;
        for (int64_t b = 0; b < batch_size; ++b) {
            const auto idx = static_cast<std::size_t>(b);
            if (!active[idx]) continue;
            const std::int32_t pos = row_lengths[idx];
            MCL_CHECK(pos < model.config.block_size, "generate_batch reached model block_size");
            const std::int32_t next = next_tokens[idx];
            step_ids[idx] = next;
            step_positions[idx] = pos;
            out[idx].push_back(next);
            mask[static_cast<std::size_t>(b * model.config.block_size + pos)] = 0;
            row_lengths[idx] = pos + 1;
            cache_length_after = std::max<int64_t>(cache_length_after, row_lengths[idx]);
            if (options.eos_token_id >= 0 && next == options.eos_token_id) active[idx] = 0;
            if (active[idx]) any_active_after = true;
        }
        if (!any_active_after) break;

        input = Tensor::from_cpu(backend, {batch_size, 1}, DType::I32, step_ids.data());
        auto positions_tensor = Tensor::from_cpu(backend, {batch_size, 1}, DType::I32, step_positions.data());
        mask_tensor = Tensor::from_cpu(backend, {batch_size, model.config.block_size}, DType::I32, mask.data());
        logits = model.forward_with_cache_positions_masked(input, positions_tensor, mask_tensor, caches, cache_length_after, false)
            .view({batch_size, model.config.vocab_size});
    }
    return out;
}

std::vector<std::string> generate_batch_text(Backend& backend,
                                             ModernGPTModel& model,
                                             const GemmaTokenizer& tokenizer,
                                             const std::vector<std::string>& prompts,
                                             const GenerateOptions& options) {
    std::vector<std::string> out;
    out.reserve(prompts.size());
    std::vector<std::vector<std::int32_t>> encoded;
    encoded.reserve(prompts.size());
    for (const auto& prompt : prompts) {
        encoded.push_back(tokenizer.encode(prompt, options.add_bos, false));
    }
    auto local_options = options;
    local_options.add_bos = false;
    if (local_options.eos_token_id < 0) local_options.eos_token_id = tokenizer.eos_token_id();
    auto generated = generate_batch(backend, model, encoded, local_options);
    for (const auto& ids : generated) out.push_back(tokenizer.decode(ids, true));
    return out;
}

} // namespace nn
} // namespace motifcl
