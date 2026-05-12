#include <motifcl/gguf.hpp>

#include <motifcl/core/error.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <type_traits>

namespace motifcl::gguf {
namespace {

constexpr std::uint32_t kMagic = 0x46554747u; // "GGUF" little-endian

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        MCL_CHECK(false, std::string("GGUF overflow while computing ") + what);
    }
    return a * b;
}

std::uint16_t read_le_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
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

std::int8_t byte_to_i8(std::uint8_t value) {
    std::int8_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        d = (q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    MCL_CHECK(alignment > 0, "GGUF alignment must be positive");
    const auto rem = value % alignment;
    return rem == 0 ? value : value + (alignment - rem);
}

void read_exact(std::istream& in, void* dst, std::size_t n, const std::string& what) {
    if (n == 0) return;
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
    MCL_CHECK(in.good(), "failed to read GGUF " + what);
}

template <typename T>
T read_le(std::istream& in, const std::string& what) {
    unsigned char bytes[sizeof(T)] = {};
    read_exact(in, bytes, sizeof(T), what);
    if constexpr (std::is_same<T, float>::value) {
        std::uint32_t bits = 0;
        for (std::size_t i = 0; i < sizeof(bits); ++i) bits |= static_cast<std::uint32_t>(bytes[i]) << (8 * i);
        float out = 0.0f;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    } else if constexpr (std::is_same<T, double>::value) {
        std::uint64_t bits = 0;
        for (std::size_t i = 0; i < sizeof(bits); ++i) bits |= static_cast<std::uint64_t>(bytes[i]) << (8 * i);
        double out = 0.0;
        std::memcpy(&out, &bits, sizeof(out));
        return out;
    } else {
        using U = typename std::make_unsigned<T>::type;
        U value = 0;
        for (std::size_t i = 0; i < sizeof(T); ++i) value |= static_cast<U>(bytes[i]) << (8 * i);
        return static_cast<T>(value);
    }
}

std::string read_string(std::istream& in, const std::string& what) {
    const auto len = read_le<std::uint64_t>(in, what + " length");
    MCL_CHECK(len <= (1ull << 32), "GGUF string is too large: " + what);
    std::string out(static_cast<std::size_t>(len), '\0');
    if (len > 0) read_exact(in, out.data(), out.size(), what);
    return out;
}

MetadataType read_metadata_type(std::istream& in, const std::string& what) {
    const auto raw = read_le<std::uint32_t>(in, what);
    MCL_CHECK(raw <= static_cast<std::uint32_t>(MetadataType::Float64), "unknown GGUF metadata type");
    return static_cast<MetadataType>(raw);
}

MetadataScalar read_scalar_payload(std::istream& in, MetadataType type, const std::string& key) {
    MetadataScalar out;
    out.type = type;
    switch (type) {
    case MetadataType::UInt8:
        out.u64 = read_le<std::uint8_t>(in, key);
        break;
    case MetadataType::Int8:
        out.i64 = read_le<std::int8_t>(in, key);
        break;
    case MetadataType::UInt16:
        out.u64 = read_le<std::uint16_t>(in, key);
        break;
    case MetadataType::Int16:
        out.i64 = read_le<std::int16_t>(in, key);
        break;
    case MetadataType::UInt32:
        out.u64 = read_le<std::uint32_t>(in, key);
        break;
    case MetadataType::Int32:
        out.i64 = read_le<std::int32_t>(in, key);
        break;
    case MetadataType::Float32:
        out.f64 = read_le<float>(in, key);
        break;
    case MetadataType::Bool:
        out.boolean = read_le<std::uint8_t>(in, key) != 0;
        break;
    case MetadataType::String:
        out.string = read_string(in, key);
        break;
    case MetadataType::UInt64:
        out.u64 = read_le<std::uint64_t>(in, key);
        break;
    case MetadataType::Int64:
        out.i64 = read_le<std::int64_t>(in, key);
        break;
    case MetadataType::Float64:
        out.f64 = read_le<double>(in, key);
        break;
    case MetadataType::Array:
        MCL_CHECK(false, "nested GGUF metadata arrays are not supported");
        break;
    }
    return out;
}

MetadataValue read_metadata_value(std::istream& in, const std::string& key) {
    MetadataValue out;
    out.type = read_metadata_type(in, key + " type");
    if (out.type != MetadataType::Array) {
        out.scalar = read_scalar_payload(in, out.type, key);
        return out;
    }

    out.array_type = read_metadata_type(in, key + " array type");
    MCL_CHECK(out.array_type != MetadataType::Array, "nested GGUF metadata arrays are not supported: " + key);
    const auto n = read_le<std::uint64_t>(in, key + " array length");
    MCL_CHECK(n <= (1ull << 32), "GGUF metadata array is too large: " + key);
    out.array.reserve(static_cast<std::size_t>(n));
    for (std::uint64_t i = 0; i < n; ++i) {
        out.array.push_back(read_scalar_payload(in, out.array_type, key));
    }
    return out;
}

std::uint64_t stream_pos_u64(std::istream& in) {
    const auto pos = in.tellg();
    MCL_CHECK(pos >= 0, "failed to query GGUF stream position");
    return static_cast<std::uint64_t>(pos);
}

std::uint64_t file_size(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    MCL_CHECK(in.good(), "failed to open GGUF file: " + path);
    const auto size = in.tellg();
    MCL_CHECK(size >= 0, "failed to query GGUF file size: " + path);
    return static_cast<std::uint64_t>(size);
}

} // namespace

const char* metadata_type_name(MetadataType type) {
    switch (type) {
    case MetadataType::UInt8: return "uint8";
    case MetadataType::Int8: return "int8";
    case MetadataType::UInt16: return "uint16";
    case MetadataType::Int16: return "int16";
    case MetadataType::UInt32: return "uint32";
    case MetadataType::Int32: return "int32";
    case MetadataType::Float32: return "float32";
    case MetadataType::Bool: return "bool";
    case MetadataType::String: return "string";
    case MetadataType::Array: return "array";
    case MetadataType::UInt64: return "uint64";
    case MetadataType::Int64: return "int64";
    case MetadataType::Float64: return "float64";
    }
    return "unknown";
}

std::uint64_t MetadataScalar::as_u64() const {
    MCL_CHECK(type == MetadataType::UInt8 || type == MetadataType::UInt16 ||
                  type == MetadataType::UInt32 || type == MetadataType::UInt64,
              "GGUF metadata scalar is not unsigned integer");
    return u64;
}

std::int64_t MetadataScalar::as_i64() const {
    if (type == MetadataType::Int8 || type == MetadataType::Int16 ||
        type == MetadataType::Int32 || type == MetadataType::Int64) {
        return i64;
    }
    if (type == MetadataType::UInt8 || type == MetadataType::UInt16 ||
        type == MetadataType::UInt32 || type == MetadataType::UInt64) {
        MCL_CHECK(u64 <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
                  "GGUF unsigned metadata does not fit int64");
        return static_cast<std::int64_t>(u64);
    }
    MCL_CHECK(false, "GGUF metadata scalar is not integer");
    return 0;
}

double MetadataScalar::as_f64() const {
    MCL_CHECK(type == MetadataType::Float32 || type == MetadataType::Float64,
              "GGUF metadata scalar is not floating point");
    return f64;
}

bool MetadataScalar::as_bool() const {
    MCL_CHECK(type == MetadataType::Bool, "GGUF metadata scalar is not bool");
    return boolean;
}

const std::string& MetadataScalar::as_string() const {
    MCL_CHECK(type == MetadataType::String, "GGUF metadata scalar is not string");
    return string;
}

const MetadataScalar& MetadataValue::as_scalar() const {
    MCL_CHECK(!is_array(), "GGUF metadata value is an array");
    return scalar;
}

const std::vector<MetadataScalar>& MetadataValue::as_array() const {
    MCL_CHECK(is_array(), "GGUF metadata value is not an array");
    return array;
}

TensorType tensor_type_from_u32(std::uint32_t value) {
    switch (value) {
    case 0: return TensorType::F32;
    case 1: return TensorType::F16;
    case 2: return TensorType::Q4_0;
    case 3: return TensorType::Q4_1;
    case 6: return TensorType::Q5_0;
    case 7: return TensorType::Q5_1;
    case 8: return TensorType::Q8_0;
    case 9: return TensorType::Q8_1;
    case 10: return TensorType::Q2_K;
    case 11: return TensorType::Q3_K;
    case 12: return TensorType::Q4_K;
    case 13: return TensorType::Q5_K;
    case 14: return TensorType::Q6_K;
    case 15: return TensorType::Q8_K;
    case 16: return TensorType::IQ2_XXS;
    case 17: return TensorType::IQ2_XS;
    case 18: return TensorType::IQ3_XXS;
    case 19: return TensorType::IQ1_S;
    case 20: return TensorType::IQ4_NL;
    case 21: return TensorType::IQ3_S;
    case 22: return TensorType::IQ2_S;
    case 23: return TensorType::IQ4_XS;
    case 24: return TensorType::I8;
    case 25: return TensorType::I16;
    case 26: return TensorType::I32;
    case 27: return TensorType::I64;
    case 28: return TensorType::F64;
    case 29: return TensorType::IQ1_M;
    case 30: return TensorType::BF16;
    default: return TensorType::Unknown;
    }
}

const char* tensor_type_name(TensorType type) {
    switch (type) {
    case TensorType::F32: return "F32";
    case TensorType::F16: return "F16";
    case TensorType::Q4_0: return "Q4_0";
    case TensorType::Q4_1: return "Q4_1";
    case TensorType::Q5_0: return "Q5_0";
    case TensorType::Q5_1: return "Q5_1";
    case TensorType::Q8_0: return "Q8_0";
    case TensorType::Q8_1: return "Q8_1";
    case TensorType::Q2_K: return "Q2_K";
    case TensorType::Q3_K: return "Q3_K";
    case TensorType::Q4_K: return "Q4_K";
    case TensorType::Q5_K: return "Q5_K";
    case TensorType::Q6_K: return "Q6_K";
    case TensorType::Q8_K: return "Q8_K";
    case TensorType::IQ2_XXS: return "IQ2_XXS";
    case TensorType::IQ2_XS: return "IQ2_XS";
    case TensorType::IQ3_XXS: return "IQ3_XXS";
    case TensorType::IQ1_S: return "IQ1_S";
    case TensorType::IQ4_NL: return "IQ4_NL";
    case TensorType::IQ3_S: return "IQ3_S";
    case TensorType::IQ2_S: return "IQ2_S";
    case TensorType::IQ4_XS: return "IQ4_XS";
    case TensorType::I8: return "I8";
    case TensorType::I16: return "I16";
    case TensorType::I32: return "I32";
    case TensorType::I64: return "I64";
    case TensorType::F64: return "F64";
    case TensorType::IQ1_M: return "IQ1_M";
    case TensorType::BF16: return "BF16";
    case TensorType::Unknown: return "UNKNOWN";
    }
    return "UNKNOWN";
}

std::uint32_t tensor_type_raw(TensorType type) {
    return static_cast<std::uint32_t>(type);
}

std::uint64_t tensor_type_block_size(TensorType type) {
    switch (type) {
    case TensorType::F32:
    case TensorType::F16:
    case TensorType::I8:
    case TensorType::I16:
    case TensorType::I32:
    case TensorType::I64:
    case TensorType::F64:
    case TensorType::BF16:
        return 1;
    case TensorType::Q4_0:
    case TensorType::Q4_1:
    case TensorType::Q5_0:
    case TensorType::Q5_1:
    case TensorType::Q8_0:
    case TensorType::Q8_1:
    case TensorType::IQ4_NL:
        return 32;
    case TensorType::Q2_K:
    case TensorType::Q3_K:
    case TensorType::Q4_K:
    case TensorType::Q5_K:
    case TensorType::Q6_K:
    case TensorType::Q8_K:
        return 256;
    case TensorType::IQ2_XXS:
    case TensorType::IQ2_XS:
    case TensorType::IQ3_XXS:
    case TensorType::IQ1_S:
    case TensorType::IQ3_S:
    case TensorType::IQ2_S:
    case TensorType::IQ4_XS:
    case TensorType::IQ1_M:
        return 0;
    case TensorType::Unknown:
        return 0;
    }
    return 0;
}

std::uint64_t tensor_type_block_nbytes(TensorType type) {
    switch (type) {
    case TensorType::F32: return 4;
    case TensorType::F16: return 2;
    case TensorType::Q4_0: return 18;
    case TensorType::Q4_1: return 20;
    case TensorType::Q5_0: return 22;
    case TensorType::Q5_1: return 24;
    case TensorType::Q8_0: return 34;
    case TensorType::Q8_1: return 36;
    case TensorType::Q2_K: return 84;
    case TensorType::Q3_K: return 110;
    case TensorType::Q4_K: return 144;
    case TensorType::Q5_K: return 176;
    case TensorType::Q6_K: return 210;
    case TensorType::Q8_K: return 292;
    case TensorType::IQ2_XXS: return 0;
    case TensorType::IQ2_XS: return 0;
    case TensorType::IQ3_XXS: return 0;
    case TensorType::IQ1_S: return 0;
    case TensorType::IQ4_NL: return 18;
    case TensorType::IQ3_S: return 0;
    case TensorType::IQ2_S: return 0;
    case TensorType::IQ4_XS: return 0;
    case TensorType::I8: return 1;
    case TensorType::I16: return 2;
    case TensorType::I32: return 4;
    case TensorType::I64: return 8;
    case TensorType::F64: return 8;
    case TensorType::IQ1_M: return 0;
    case TensorType::BF16: return 2;
    case TensorType::Unknown: return 0;
    }
    return 0;
}

bool tensor_type_is_quantized(TensorType type) {
    return tensor_type_block_size(type) > 1;
}

bool tensor_type_can_dequantize_to_f32(TensorType type) {
    switch (type) {
    case TensorType::F32:
    case TensorType::F16:
    case TensorType::BF16:
    case TensorType::Q4_0:
    case TensorType::Q8_0:
    case TensorType::Q4_K:
    case TensorType::Q5_K:
    case TensorType::Q6_K:
        return true;
    default:
        return false;
    }
}

bool tensor_type_can_repack_to_motifcl_quant(TensorType type) {
    return type == TensorType::Q8_0 || type == TensorType::Q4_0 ||
           type == TensorType::Q4_K || type == TensorType::Q5_K ||
           type == TensorType::Q6_K;
}

std::uint64_t tensor_element_count(const TensorInfo& info) {
    std::uint64_t n = 1;
    for (const auto dim : info.dimensions) n = checked_mul(n, dim, "tensor element count");
    return n;
}

std::uint64_t tensor_nbytes(const TensorInfo& info) {
    const auto block = tensor_type_block_size(info.type);
    const auto block_bytes = tensor_type_block_nbytes(info.type);
    MCL_CHECK(block > 0 && block_bytes > 0, "cannot compute GGUF tensor byte size for type " +
                                           std::string(tensor_type_name(info.type)));
    const auto n = tensor_element_count(info);
    MCL_CHECK(n % block == 0, "GGUF tensor element count is not divisible by type block size: " + info.name);
    return checked_mul(n / block, block_bytes, "tensor byte size");
}

std::vector<float> dequantize_tensor_data_to_f32(const TensorInfo& info,
                                                 const std::vector<std::uint8_t>& data) {
    MCL_CHECK(tensor_type_can_dequantize_to_f32(info.type),
              "GGUF tensor type cannot be dequantized to F32: " + info.name +
                  " type=" + tensor_type_name(info.type));
    const auto numel_u64 = tensor_element_count(info);
    MCL_CHECK(numel_u64 <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
              "GGUF tensor is too large to dequantize on this host: " + info.name);
    const auto numel = static_cast<std::size_t>(numel_u64);
    MCL_CHECK(data.size() == tensor_nbytes(info), "GGUF tensor byte size mismatch: " + info.name);

    std::vector<float> out(numel);
    if (info.type == TensorType::F32) {
        for (std::size_t i = 0; i < numel; ++i) {
            std::uint32_t bits = 0;
            for (int b = 0; b < 4; ++b) bits |= static_cast<std::uint32_t>(data[i * 4 + b]) << (8 * b);
            std::memcpy(&out[i], &bits, sizeof(float));
        }
        return out;
    }
    if (info.type == TensorType::F16) {
        for (std::size_t i = 0; i < numel; ++i) out[i] = f16_to_f32(read_le_u16(data.data() + i * 2));
        return out;
    }
    if (info.type == TensorType::BF16) {
        for (std::size_t i = 0; i < numel; ++i) {
            const std::uint32_t bits = static_cast<std::uint32_t>(read_le_u16(data.data() + i * 2)) << 16;
            std::memcpy(&out[i], &bits, sizeof(float));
        }
        return out;
    }

    const auto block = tensor_type_block_size(info.type);
    const auto block_bytes = tensor_type_block_nbytes(info.type);
    MCL_CHECK(block > 0 && block_bytes > 0 && numel_u64 % block == 0,
              "invalid GGUF quantized tensor shape/type: " + info.name);
    const auto nb = numel_u64 / block;

    if (info.type == TensorType::Q4_0) {
        for (std::uint64_t i = 0; i < nb; ++i) {
            const auto off = static_cast<std::size_t>(i * block_bytes);
            const float d = f16_to_f32(read_le_u16(data.data() + off));
            const auto* q = data.data() + off + 2;
            const auto out_base = static_cast<std::size_t>(i * block);
            for (std::size_t j = 0; j < 16; ++j) {
                out[out_base + j] = static_cast<float>((q[j] & 0x0f) - 8) * d;
                out[out_base + j + 16] = static_cast<float>((q[j] >> 4) - 8) * d;
            }
        }
        return out;
    }

    if (info.type == TensorType::Q8_0) {
        for (std::uint64_t i = 0; i < nb; ++i) {
            const auto off = static_cast<std::size_t>(i * block_bytes);
            const float d = f16_to_f32(read_le_u16(data.data() + off));
            const auto* q = data.data() + off + 2;
            const auto out_base = static_cast<std::size_t>(i * block);
            for (std::size_t j = 0; j < 32; ++j) {
                out[out_base + j] = static_cast<float>(byte_to_i8(q[j])) * d;
            }
        }
        return out;
    }

    if (info.type == TensorType::Q4_K) {
        for (std::uint64_t i = 0; i < nb; ++i) {
            const auto off = static_cast<std::size_t>(i * block_bytes);
            const float d = f16_to_f32(read_le_u16(data.data() + off));
            const float min = f16_to_f32(read_le_u16(data.data() + off + 2));
            const auto* scales = data.data() + off + 4;
            const auto* q = data.data() + off + 16;
            auto out_pos = static_cast<std::size_t>(i * block);
            int is = 0;
            for (int group = 0; group < 4; ++group) {
                std::uint8_t sc = 0;
                std::uint8_t m = 0;
                get_scale_min_k4(is + 0, scales, sc, m);
                const float d1 = d * static_cast<float>(sc);
                const float m1 = min * static_cast<float>(m);
                get_scale_min_k4(is + 1, scales, sc, m);
                const float d2 = d * static_cast<float>(sc);
                const float m2 = min * static_cast<float>(m);
                for (int l = 0; l < 32; ++l) out[out_pos++] = d1 * static_cast<float>(q[l] & 0x0f) - m1;
                for (int l = 0; l < 32; ++l) out[out_pos++] = d2 * static_cast<float>(q[l] >> 4) - m2;
                q += 32;
                is += 2;
            }
        }
        return out;
    }

    if (info.type == TensorType::Q5_K) {
        for (std::uint64_t i = 0; i < nb; ++i) {
            const auto off = static_cast<std::size_t>(i * block_bytes);
            const float d = f16_to_f32(read_le_u16(data.data() + off));
            const float min = f16_to_f32(read_le_u16(data.data() + off + 2));
            const auto* scales = data.data() + off + 4;
            const auto* qh = data.data() + off + 16;
            const auto* ql = data.data() + off + 48;
            auto out_pos = static_cast<std::size_t>(i * block);
            int is = 0;
            std::uint8_t u1 = 1;
            std::uint8_t u2 = 2;
            for (int group = 0; group < 4; ++group) {
                std::uint8_t sc = 0;
                std::uint8_t m = 0;
                get_scale_min_k4(is + 0, scales, sc, m);
                const float d1 = d * static_cast<float>(sc);
                const float m1 = min * static_cast<float>(m);
                get_scale_min_k4(is + 1, scales, sc, m);
                const float d2 = d * static_cast<float>(sc);
                const float m2 = min * static_cast<float>(m);
                for (int l = 0; l < 32; ++l) {
                    const int value = (ql[l] & 0x0f) + ((qh[l] & u1) ? 16 : 0);
                    out[out_pos++] = d1 * static_cast<float>(value) - m1;
                }
                for (int l = 0; l < 32; ++l) {
                    const int value = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
                    out[out_pos++] = d2 * static_cast<float>(value) - m2;
                }
                ql += 32;
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        return out;
    }

    if (info.type == TensorType::Q6_K) {
        for (std::uint64_t i = 0; i < nb; ++i) {
            const auto off = static_cast<std::size_t>(i * block_bytes);
            const auto* ql_base = data.data() + off;
            const auto* qh_base = data.data() + off + 128;
            const auto* sc_base = data.data() + off + 192;
            const float d = f16_to_f32(read_le_u16(data.data() + off + 208));
            const auto out_base = static_cast<std::size_t>(i * block);

            for (int n = 0; n < 256; n += 128) {
                const auto* ql = ql_base + (n / 128) * 64;
                const auto* qh = qh_base + (n / 128) * 32;
                const auto* sc = sc_base + (n / 128) * 8;
                const auto y = out_base + static_cast<std::size_t>(n);
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = static_cast<int>((ql[l + 0] & 0x0fu) |
                                                    (((qh[l] >> 0) & 0x03u) << 4)) - 32;
                    const int q2 = static_cast<int>((ql[l + 32] & 0x0fu) |
                                                    (((qh[l] >> 2) & 0x03u) << 4)) - 32;
                    const int q3 = static_cast<int>(((ql[l + 0] >> 4) & 0x0fu) |
                                                    (((qh[l] >> 4) & 0x03u) << 4)) - 32;
                    const int q4 = static_cast<int>(((ql[l + 32] >> 4) & 0x0fu) |
                                                    (((qh[l] >> 6) & 0x03u) << 4)) - 32;
                    out[y + static_cast<std::size_t>(l + 0)] =
                        d * static_cast<float>(byte_to_i8(sc[is + 0])) * static_cast<float>(q1);
                    out[y + static_cast<std::size_t>(l + 32)] =
                        d * static_cast<float>(byte_to_i8(sc[is + 2])) * static_cast<float>(q2);
                    out[y + static_cast<std::size_t>(l + 64)] =
                        d * static_cast<float>(byte_to_i8(sc[is + 4])) * static_cast<float>(q3);
                    out[y + static_cast<std::size_t>(l + 96)] =
                        d * static_cast<float>(byte_to_i8(sc[is + 6])) * static_cast<float>(q4);
                }
            }
        }
        return out;
    }

    MCL_CHECK(false, "unhandled GGUF dequantization type: " + std::string(tensor_type_name(info.type)));
    return {};
}

File File::open(const std::string& path) {
    File file;
    file.path_ = path;
    file.file_size_ = file_size(path);

    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open GGUF file: " + path);

    const auto magic = read_le<std::uint32_t>(in, "magic");
    MCL_CHECK(magic == kMagic, "invalid GGUF magic: " + path);
    file.version_ = read_le<std::uint32_t>(in, "version");
    MCL_CHECK(file.version_ >= 2 && file.version_ <= 3, "unsupported GGUF version");

    const auto tensor_count = read_le<std::uint64_t>(in, "tensor_count");
    const auto metadata_count = read_le<std::uint64_t>(in, "metadata_kv_count");
    MCL_CHECK(tensor_count <= (1ull << 32), "GGUF tensor count is too large");
    MCL_CHECK(metadata_count <= (1ull << 32), "GGUF metadata count is too large");

    file.metadata_.reserve(static_cast<std::size_t>(metadata_count));
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        auto key = read_string(in, "metadata key");
        MCL_CHECK(!key.empty(), "GGUF metadata key cannot be empty");
        auto value = read_metadata_value(in, key);
        MCL_CHECK(file.metadata_index_.emplace(key, file.metadata_.size()).second,
                  "duplicate GGUF metadata key: " + key);
        file.metadata_.emplace_back(std::move(key), std::move(value));
    }

    file.alignment_ = file.metadata_u64_or("general.alignment", 32);
    MCL_CHECK(file.alignment_ > 0 && file.alignment_ <= (1u << 20), "invalid GGUF tensor alignment");

    file.tensors_.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo info;
        info.name = read_string(in, "tensor name");
        MCL_CHECK(!info.name.empty(), "GGUF tensor name cannot be empty");
        const auto ndim = read_le<std::uint32_t>(in, info.name + " ndim");
        MCL_CHECK(ndim > 0 && ndim <= 16, "invalid GGUF tensor rank: " + info.name);
        info.dimensions.reserve(ndim);
        for (std::uint32_t d = 0; d < ndim; ++d) {
            const auto dim = read_le<std::uint64_t>(in, info.name + " dim");
            MCL_CHECK(dim > 0, "GGUF tensor dimension must be positive: " + info.name);
            info.dimensions.push_back(dim);
        }
        info.raw_type = read_le<std::uint32_t>(in, info.name + " type");
        info.type = tensor_type_from_u32(info.raw_type);
        info.offset = read_le<std::uint64_t>(in, info.name + " offset");
        if (tensor_type_block_nbytes(info.type) > 0) (void)tensor_nbytes(info);
        MCL_CHECK(file.tensor_index_.emplace(info.name, file.tensors_.size()).second,
                  "duplicate GGUF tensor name: " + info.name);
        file.tensors_.push_back(std::move(info));
    }

    file.tensor_data_offset_ = align_up(stream_pos_u64(in), file.alignment_);
    MCL_CHECK(file.tensor_data_offset_ <= file.file_size_, "GGUF tensor data offset is past EOF");

    for (const auto& info : file.tensors_) {
        MCL_CHECK(info.offset <= file.file_size_ - file.tensor_data_offset_,
                  "GGUF tensor offset is past data section: " + info.name);
        if (tensor_type_block_nbytes(info.type) > 0) {
            const auto bytes = tensor_nbytes(info);
            MCL_CHECK(bytes <= file.file_size_ - file.tensor_data_offset_ - info.offset,
                      "GGUF tensor byte range is past EOF: " + info.name);
        }
    }

    return file;
}

std::vector<std::string> File::metadata_keys() const {
    std::vector<std::string> keys;
    keys.reserve(metadata_.size());
    for (const auto& kv : metadata_) keys.push_back(kv.first);
    return keys;
}

bool File::has_metadata(const std::string& key) const {
    return metadata_index_.find(key) != metadata_index_.end();
}

const MetadataValue& File::metadata(const std::string& key) const {
    auto it = metadata_index_.find(key);
    MCL_CHECK(it != metadata_index_.end(), "GGUF metadata key not found: " + key);
    return metadata_[it->second].second;
}

std::string File::metadata_string_or(const std::string& key, const std::string& fallback) const {
    if (!has_metadata(key)) return fallback;
    return metadata(key).as_scalar().as_string();
}

std::uint64_t File::metadata_u64_or(const std::string& key, std::uint64_t fallback) const {
    if (!has_metadata(key)) return fallback;
    return metadata(key).as_scalar().as_u64();
}

std::vector<std::string> File::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& info : tensors_) names.push_back(info.name);
    return names;
}

bool File::contains_tensor(const std::string& name) const {
    return tensor_index_.find(name) != tensor_index_.end();
}

const TensorInfo& File::tensor_info(const std::string& name) const {
    auto it = tensor_index_.find(name);
    MCL_CHECK(it != tensor_index_.end(), "GGUF tensor not found: " + name);
    return tensors_[it->second];
}

std::vector<std::uint8_t> File::read_tensor_data(const std::string& name) const {
    const auto& info = tensor_info(name);
    auto nbytes = std::uint64_t{0};
    if (tensor_type_block_nbytes(info.type) > 0) {
        nbytes = tensor_nbytes(info);
    } else {
        const auto data_span = file_size_ - tensor_data_offset_;
        auto end = data_span;
        for (const auto& other : tensors_) {
            if (other.offset > info.offset) end = std::min(end, other.offset);
        }
        MCL_CHECK(end >= info.offset, "invalid GGUF tensor offsets around: " + name);
        nbytes = end - info.offset;
    }
    MCL_CHECK(nbytes <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
              "GGUF tensor payload is too large to read on this host: " + name);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(nbytes));
    std::ifstream in(path_, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open GGUF file: " + path_);
    in.seekg(static_cast<std::streamoff>(tensor_data_offset_ + info.offset), std::ios::beg);
    MCL_CHECK(in.good(), "failed to seek GGUF tensor: " + name);
    if (!data.empty()) read_exact(in, data.data(), data.size(), "tensor " + name);
    return data;
}

std::vector<float> File::read_tensor_f32(const std::string& name) const {
    const auto& info = tensor_info(name);
    return dequantize_tensor_data_to_f32(info, read_tensor_data(name));
}

Tensor File::read_tensor_quantized(Backend& backend, const std::string& name) const {
    const auto& info = tensor_info(name);
    MCL_CHECK(tensor_type_can_repack_to_motifcl_quant(info.type),
              "GGUF tensor type cannot be represented as MotifCL native quant tensor: " + name +
                  " type=" + tensor_type_name(info.type));
    const auto numel_u64 = tensor_element_count(info);
    MCL_CHECK(numel_u64 <= static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()),
              "GGUF tensor is too large to repack on this host: " + name);
    const auto numel = static_cast<std::size_t>(numel_u64);
    const auto block = tensor_type_block_size(info.type);
    const auto block_bytes = tensor_type_block_nbytes(info.type);
    MCL_CHECK(numel_u64 % block == 0, "GGUF native quant tensor has incomplete quant block: " + name);
    const auto nb = static_cast<std::size_t>(numel_u64 / block);
    const auto data = read_tensor_data(name);
    MCL_CHECK(data.size() == nb * block_bytes, "GGUF tensor byte size mismatch while repacking: " + name);

    std::vector<int64_t> shape;
    shape.reserve(info.dimensions.size());
    for (const auto dim : info.dimensions) {
        MCL_CHECK(dim <= static_cast<std::uint64_t>(std::numeric_limits<int64_t>::max()),
                  "GGUF tensor dimension is too large: " + name);
        shape.push_back(static_cast<int64_t>(dim));
    }

    if (info.type == TensorType::Q4_K || info.type == TensorType::Q5_K || info.type == TensorType::Q6_K) {
        MCL_CHECK(block == 256, "GGUF K-quant native tensor requires 256-element blocks: " + name);
        const auto dtype = info.type == TensorType::Q4_K
            ? DType::Q4_K
            : (info.type == TensorType::Q5_K ? DType::Q5_K : DType::Q6_K);
        return Tensor::from_cpu(backend, Shape(shape), dtype, data.data());
    }

    MCL_CHECK(block == 32, "GGUF native Q4_0/Q8_0 tensor requires 32-element blocks: " + name);
    std::vector<float> scales(nb);
    if (info.type == TensorType::Q8_0) {
        std::vector<std::int8_t> q(numel);
        for (std::size_t i = 0; i < nb; ++i) {
            const auto off = i * static_cast<std::size_t>(block_bytes);
            scales[i] = f16_to_f32(read_le_u16(data.data() + off));
            std::memcpy(q.data() + i * 32, data.data() + off + 2, 32);
        }
        auto tensor = Tensor::from_cpu(backend, Shape(shape), DType::Q8_0, q.data());
        auto scale_tensor = Tensor::from_cpu(backend, {static_cast<int64_t>(scales.size())}, DType::F32, scales.data());
        tensor._set_quant_scales(scale_tensor, 2, 32);
        return tensor;
    }

    std::vector<std::uint8_t> q(dtype_storage_nbytes(DType::Q4_0, numel));
    for (std::size_t i = 0; i < nb; ++i) {
        const auto off = i * static_cast<std::size_t>(block_bytes);
        scales[i] = f16_to_f32(read_le_u16(data.data() + off));
        const auto* gguf_q = data.data() + off + 2;
        auto* out_q = q.data() + i * 16;
        for (int elem = 0; elem < 32; ++elem) {
            const std::uint8_t code = elem < 16
                ? static_cast<std::uint8_t>(gguf_q[elem] & 0x0fu)
                : static_cast<std::uint8_t>((gguf_q[elem - 16] >> 4) & 0x0fu);
            const int packed = elem >> 1;
            if ((elem & 1) == 0) out_q[packed] = static_cast<std::uint8_t>((out_q[packed] & 0xf0u) | code);
            else out_q[packed] = static_cast<std::uint8_t>((out_q[packed] & 0x0fu) | (code << 4));
        }
    }
    auto tensor = Tensor::from_cpu(backend, Shape(shape), DType::Q4_0, q.data());
    auto scale_tensor = Tensor::from_cpu(backend, {static_cast<int64_t>(scales.size())}, DType::F32, scales.data());
    tensor._set_quant_scales(scale_tensor, 2, 32);
    return tensor;
}

} // namespace motifcl::gguf
