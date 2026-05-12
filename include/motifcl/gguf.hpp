#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <motifcl/tensor/tensor.hpp>

namespace motifcl::gguf {

enum class MetadataType : std::uint32_t {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12
};

const char* metadata_type_name(MetadataType type);

struct MetadataScalar {
    MetadataType type = MetadataType::UInt8;
    std::uint64_t u64 = 0;
    std::int64_t i64 = 0;
    double f64 = 0.0;
    bool boolean = false;
    std::string string;

    std::uint64_t as_u64() const;
    std::int64_t as_i64() const;
    double as_f64() const;
    bool as_bool() const;
    const std::string& as_string() const;
};

struct MetadataValue {
    MetadataType type = MetadataType::UInt8;
    MetadataType array_type = MetadataType::UInt8;
    MetadataScalar scalar;
    std::vector<MetadataScalar> array;

    bool is_array() const { return type == MetadataType::Array; }
    const MetadataScalar& as_scalar() const;
    const std::vector<MetadataScalar>& as_array() const;
};

enum class TensorType : std::uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Unknown = 0xffffffffu
};

TensorType tensor_type_from_u32(std::uint32_t value);
const char* tensor_type_name(TensorType type);
std::uint32_t tensor_type_raw(TensorType type);
std::uint64_t tensor_type_block_size(TensorType type);
std::uint64_t tensor_type_block_nbytes(TensorType type);
bool tensor_type_is_quantized(TensorType type);
bool tensor_type_can_dequantize_to_f32(TensorType type);
bool tensor_type_can_repack_to_motifcl_quant(TensorType type);

struct TensorInfo {
    std::string name;
    std::vector<std::uint64_t> dimensions;
    TensorType type = TensorType::Unknown;
    std::uint32_t raw_type = 0;
    std::uint64_t offset = 0;
};

std::uint64_t tensor_element_count(const TensorInfo& info);
std::uint64_t tensor_nbytes(const TensorInfo& info);
std::vector<float> dequantize_tensor_data_to_f32(const TensorInfo& info,
                                                 const std::vector<std::uint8_t>& data);

class File {
public:
    File() = default;

    static File open(const std::string& path);

    const std::string& path() const { return path_; }
    std::uint32_t version() const { return version_; }
    std::uint64_t tensor_count() const { return tensors_.size(); }
    std::uint64_t metadata_count() const { return metadata_.size(); }
    std::uint64_t alignment() const { return alignment_; }
    std::uint64_t tensor_data_offset() const { return tensor_data_offset_; }
    std::uint64_t total_file_size() const { return file_size_; }

    std::vector<std::string> metadata_keys() const;
    bool has_metadata(const std::string& key) const;
    const MetadataValue& metadata(const std::string& key) const;

    std::string metadata_string_or(const std::string& key, const std::string& fallback = "") const;
    std::uint64_t metadata_u64_or(const std::string& key, std::uint64_t fallback = 0) const;

    std::vector<std::string> tensor_names() const;
    bool contains_tensor(const std::string& name) const;
    const TensorInfo& tensor_info(const std::string& name) const;
    const std::vector<TensorInfo>& tensors() const { return tensors_; }

    std::vector<std::uint8_t> read_tensor_data(const std::string& name) const;
    std::vector<float> read_tensor_f32(const std::string& name) const;
    Tensor read_tensor_quantized(Backend& backend, const std::string& name) const;

private:
    std::string path_;
    std::uint32_t version_ = 0;
    std::uint64_t alignment_ = 32;
    std::uint64_t tensor_data_offset_ = 0;
    std::uint64_t file_size_ = 0;
    std::vector<std::pair<std::string, MetadataValue>> metadata_;
    std::unordered_map<std::string, std::size_t> metadata_index_;
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, std::size_t> tensor_index_;
};

} // namespace motifcl::gguf
