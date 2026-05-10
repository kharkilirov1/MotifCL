#include <motifcl/serialization.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/ops/quant.hpp>
#include <motifcl/runtime/backend.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace motifcl {

namespace {

constexpr char kTensorMagicV1[8] = {'M', 'C', 'L', 'T', 'E', 'N', '1', '\0'};
constexpr char kParamMagicV1[8] = {'M', 'C', 'L', 'P', 'A', 'R', '1', '\0'};
constexpr char kTensorMagicV2[8] = {'M', 'C', 'L', 'T', 'E', 'N', '2', '\0'};
constexpr char kParamMagicV2[8] = {'M', 'C', 'L', 'P', 'A', 'R', '2', '\0'};

template <typename T>
void write_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    MCL_CHECK(out.good(), "failed to write serialization stream");
}

template <typename T>
T read_value(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    MCL_CHECK(in.good(), "failed to read serialization stream");
    return value;
}

void write_tensor_payload(std::ostream& out, const Tensor& tensor) {
    const std::int32_t dtype = static_cast<std::int32_t>(tensor.dtype());
    const std::uint64_t ndim = static_cast<std::uint64_t>(tensor.shape().dims.size());
    const std::uint64_t nbytes = static_cast<std::uint64_t>(tensor.nbytes());
    write_value(out, dtype);
    write_value(out, ndim);
    for (auto dim : tensor.shape().dims) write_value(out, static_cast<std::int64_t>(dim));
    write_value(out, nbytes);
    write_value(out, tensor.quant_scale());
    write_value(out, static_cast<std::int32_t>(tensor.has_quant_scales() ? 1 : 0));
    write_value(out, static_cast<std::int32_t>(tensor.quant_scale_axis()));
    write_value(out, static_cast<std::int64_t>(tensor.quant_block_size()));
    if (tensor.has_quant_scales()) {
        write_tensor_payload(out, tensor.quant_scales());
    }
    std::vector<std::uint8_t> bytes(tensor.nbytes());
    if (!bytes.empty()) tensor.to_cpu(bytes.data(), bytes.size());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    MCL_CHECK(out.good(), "failed to write tensor payload");
}

Tensor read_tensor_payload_v1(Backend& backend, std::istream& in) {
    const auto dtype = static_cast<DType>(read_value<std::int32_t>(in));
    const auto ndim = read_value<std::uint64_t>(in);
    std::vector<int64_t> dims(ndim);
    for (auto& dim : dims) dim = read_value<std::int64_t>(in);
    const auto nbytes = read_value<std::uint64_t>(in);
    Shape shape(std::move(dims));
    MCL_CHECK(dtype_storage_nbytes(dtype, static_cast<std::size_t>(shape.numel())) == nbytes,
              "serialized tensor byte size does not match dtype/shape");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(nbytes));
    if (!bytes.empty()) in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    MCL_CHECK(in.good(), "failed to read tensor payload");
    return Tensor::from_cpu(backend, shape, dtype, bytes.data());
}

Tensor read_tensor_payload_v2(Backend& backend, std::istream& in) {
    const auto dtype = static_cast<DType>(read_value<std::int32_t>(in));
    const auto ndim = read_value<std::uint64_t>(in);
    std::vector<int64_t> dims(ndim);
    for (auto& dim : dims) dim = read_value<std::int64_t>(in);
    const auto nbytes = read_value<std::uint64_t>(in);
    const auto quant_scale = read_value<float>(in);
    const auto has_quant_scales = read_value<std::int32_t>(in) != 0;
    const auto quant_scale_axis = read_value<std::int32_t>(in);
    const auto quant_block_size = read_value<std::int64_t>(in);
    Tensor quant_scales;
    if (has_quant_scales) quant_scales = read_tensor_payload_v2(backend, in);

    Shape shape(std::move(dims));
    MCL_CHECK(dtype_storage_nbytes(dtype, static_cast<std::size_t>(shape.numel())) == nbytes,
              "serialized tensor byte size does not match dtype/shape");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(nbytes));
    if (!bytes.empty()) in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    MCL_CHECK(in.good(), "failed to read tensor payload");
    auto tensor = Tensor::from_cpu(backend, shape, dtype, bytes.data());
    if (has_quant_scales) {
        tensor._set_quant_scales(quant_scales, quant_scale_axis, quant_block_size);
    } else if (dtype == DType::Q8_0 || dtype == DType::Q4_0) {
        tensor._set_quant_scale(quant_scale);
    }
    return tensor;
}

std::array<char, 8> read_magic(std::istream& in) {
    char actual[8] = {};
    in.read(actual, sizeof(actual));
    MCL_CHECK(in.good(), "failed to read serialization magic");
    std::array<char, 8> out{};
    std::memcpy(out.data(), actual, out.size());
    return out;
}

bool magic_equals(const std::array<char, 8>& actual, const char expected[8]) {
    return std::memcmp(actual.data(), expected, actual.size()) == 0;
}

} // namespace

void save_tensor(const Tensor& tensor, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    MCL_CHECK(out.good(), "failed to open tensor file for write: " + path);
    out.write(kTensorMagicV2, sizeof(kTensorMagicV2));
    write_tensor_payload(out, tensor);
}

Tensor load_tensor(Backend& backend, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open tensor file for read: " + path);
    const auto magic = read_magic(in);
    if (magic_equals(magic, kTensorMagicV1)) return read_tensor_payload_v1(backend, in);
    if (magic_equals(magic, kTensorMagicV2)) return read_tensor_payload_v2(backend, in);
    MCL_CHECK(false, "serialization magic mismatch");
    return {};
}

void save_parameters(const std::vector<nn::Parameter*>& params, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    MCL_CHECK(out.good(), "failed to open parameter checkpoint for write: " + path);
    out.write(kParamMagicV2, sizeof(kParamMagicV2));
    write_value(out, static_cast<std::uint64_t>(params.size()));
    for (auto* param : params) {
        MCL_CHECK(param != nullptr, "cannot save null parameter");
        write_tensor_payload(out, param->data);
    }
}

void load_parameters(const std::vector<nn::Parameter*>& params, Backend& backend, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open parameter checkpoint for read: " + path);
    const auto magic = read_magic(in);
    const bool is_v1 = magic_equals(magic, kParamMagicV1);
    MCL_CHECK(is_v1 || magic_equals(magic, kParamMagicV2), "serialization magic mismatch");
    const auto count = read_value<std::uint64_t>(in);
    MCL_CHECK(count == params.size(), "checkpoint parameter count mismatch");
    for (auto* param : params) {
        MCL_CHECK(param != nullptr, "cannot load into null parameter");
        auto loaded = is_v1 ? read_tensor_payload_v1(backend, in) : read_tensor_payload_v2(backend, in);
        MCL_CHECK(loaded.shape() == param->data.shape(), "checkpoint parameter shape mismatch");
        MCL_CHECK(loaded.dtype() == param->data.dtype(), "checkpoint parameter dtype mismatch");
        param->data = std::move(loaded);
        param->data.set_requires_grad(param->trainable);
    }
}

namespace {

std::string checkpoint_dtype_name(DType dtype) {
    switch (dtype) {
    case DType::Q4_0: return "Q4_0";
    case DType::Q8_0: return "Q8_0";
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::I32: return "I32";
    case DType::U8: return "U8";
    }
    return "unknown";
}

Tensor quantized_linear_weight_for_checkpoint(const nn::Linear& linear, DType qdtype) {
    MCL_CHECK(qdtype == DType::Q4_0 || qdtype == DType::Q8_0,
              "quantized transformer checkpoint expects Q4_0 or Q8_0");
    if (linear.quantized_inference_enabled() && linear.quantized_weight_dtype() == qdtype) {
        return linear.quantized_weight();
    }
    return qdtype == DType::Q4_0
        ? quantize_q4_symmetric_cols(linear.weight.data)
        : quantize_q8_symmetric_cols(linear.weight.data);
}

Tensor quantized_lm_head_for_checkpoint(const nn::ModernGPTModel& model, DType qdtype) {
    MCL_CHECK(qdtype == DType::Q4_0 || qdtype == DType::Q8_0,
              "quantized transformer checkpoint expects Q4_0 or Q8_0");
    if (model.quantized_inference_enabled() && model.quantized_weight_dtype() == qdtype) {
        return model.quantized_lm_head();
    }
    return qdtype == DType::Q4_0
        ? quantize_q4_symmetric_cols(model.lm_head.data)
        : quantize_q8_symmetric_cols(model.lm_head.data);
}

std::filesystem::path checkpoint_path(const std::filesystem::path& dir, const std::string& name) {
    return dir / name;
}

void assign_checkpoint_parameter(nn::Parameter& parameter, Tensor tensor) {
    MCL_CHECK(parameter.data.shape() == tensor.shape(), "quantized checkpoint parameter shape mismatch");
    MCL_CHECK(parameter.data.dtype() == tensor.dtype(), "quantized checkpoint parameter dtype mismatch");
    parameter.data = std::move(tensor);
    parameter.data.set_requires_grad(parameter.trainable);
}

void save_linear_quant_checkpoint(const nn::Linear& linear,
                                  const std::filesystem::path& dir,
                                  const std::string& prefix,
                                  DType qdtype) {
    save_tensor(quantized_linear_weight_for_checkpoint(linear, qdtype),
                checkpoint_path(dir, prefix + ".qweight.mclt").string());
    if (linear.has_bias()) {
        save_tensor(linear.bias.data, checkpoint_path(dir, prefix + ".bias.mclt").string());
    }
}

void load_linear_quant_checkpoint(nn::Linear& linear,
                                  Backend& backend,
                                  const std::filesystem::path& dir,
                                  const std::string& prefix) {
    linear.set_quantized_weight(load_tensor(backend, checkpoint_path(dir, prefix + ".qweight.mclt").string()));
    if (linear.has_bias()) {
        assign_checkpoint_parameter(linear.bias, load_tensor(backend, checkpoint_path(dir, prefix + ".bias.mclt").string()));
    }
}

std::string read_manifest_text(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open quantized transformer manifest: " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

void save_quantized_transformer_checkpoint(const nn::ModernGPTModel& model,
                                           const std::string& dir,
                                           DType qdtype) {
    nn::QuantizationPolicy policy;
    policy.default_dtype = qdtype;
    policy.lm_head_dtype = qdtype;
    save_quantized_transformer_checkpoint(model, dir, policy);
}

void save_quantized_transformer_checkpoint(const nn::ModernGPTModel& model,
                                           const std::string& dir,
                                           const nn::QuantizationPolicy& policy) {
    MCL_CHECK(policy.default_dtype == DType::Q4_0 || policy.default_dtype == DType::Q8_0,
              "save_quantized_transformer_checkpoint policy default expects Q4_0 or Q8_0");
    MCL_CHECK(policy.lm_head_dtype == DType::Q4_0 || policy.lm_head_dtype == DType::Q8_0,
              "save_quantized_transformer_checkpoint policy lm_head expects Q4_0 or Q8_0");
    const std::filesystem::path root(dir);
    std::filesystem::create_directories(root);

    save_tensor(model.token_embedding.weight.data, checkpoint_path(root, "token_embedding.weight.mclt").string());
    if (model.config.learned_position_embeddings) {
        save_tensor(model.position_embedding.data, checkpoint_path(root, "position_embedding.weight.mclt").string());
    }
    save_tensor(model.final_norm.weight.data, checkpoint_path(root, "final_norm.weight.mclt").string());
    save_tensor(quantized_lm_head_for_checkpoint(model, policy.lm_head_dtype), checkpoint_path(root, "lm_head.qweight.mclt").string());

    for (std::size_t i = 0; i < model.blocks.size(); ++i) {
        const auto& block = *model.blocks[i];
        const DType layer_dtype = std::find(policy.q8_layers.begin(), policy.q8_layers.end(), static_cast<int>(i)) != policy.q8_layers.end()
            ? DType::Q8_0
            : policy.default_dtype;
        const std::string p = "blocks." + std::to_string(i) + ".";
        save_tensor(block.norm1().weight.data, checkpoint_path(root, p + "norm1.weight.mclt").string());
        save_tensor(block.norm2().weight.data, checkpoint_path(root, p + "norm2.weight.mclt").string());
        save_linear_quant_checkpoint(block.attention().qkv_proj(), root, p + "attn.qkv_proj", layer_dtype);
        save_linear_quant_checkpoint(block.attention().o_proj(), root, p + "attn.o_proj", layer_dtype);
        save_linear_quant_checkpoint(block.mlp().gate_up_proj, root, p + "mlp.gate_up_proj", layer_dtype);
        save_linear_quant_checkpoint(block.mlp().down_proj, root, p + "mlp.down_proj", layer_dtype);
    }

    std::ofstream manifest(checkpoint_path(root, "manifest.json"), std::ios::binary);
    MCL_CHECK(manifest.good(), "failed to write quantized transformer manifest");
    manifest << "{\n";
    manifest << "  \"format\": \"motifcl.quantized_transformer.v1\",\n";
    manifest << "  \"quantization\": \"" << checkpoint_dtype_name(policy.default_dtype) << "\",\n";
    manifest << "  \"lm_head_quantization\": \"" << checkpoint_dtype_name(policy.lm_head_dtype) << "\",\n";
    manifest << "  \"q8_layers\": [";
    for (std::size_t i = 0; i < policy.q8_layers.size(); ++i) {
        if (i) manifest << ", ";
        manifest << policy.q8_layers[i];
    }
    manifest << "],\n";
    manifest << "  \"vocab_size\": " << model.config.vocab_size << ",\n";
    manifest << "  \"block_size\": " << model.config.block_size << ",\n";
    manifest << "  \"n_embd\": " << model.config.n_embd << ",\n";
    manifest << "  \"n_head\": " << model.config.n_head << ",\n";
    manifest << "  \"n_kv_head\": " << model.config.n_kv_head << ",\n";
    manifest << "  \"n_layer\": " << model.config.n_layer << ",\n";
    manifest << "  \"mlp_hidden\": " << model.config.mlp_hidden << ",\n";
    manifest << "  \"learned_position_embeddings\": "
             << (model.config.learned_position_embeddings ? "true" : "false") << "\n";
    manifest << "}\n";
    MCL_CHECK(manifest.good(), "failed to finalize quantized transformer manifest");
}

void load_quantized_transformer_checkpoint(nn::ModernGPTModel& model,
                                           Backend& backend,
                                           const std::string& dir) {
    const std::filesystem::path root(dir);
    const auto manifest = read_manifest_text(checkpoint_path(root, "manifest.json"));
    MCL_CHECK(manifest.find("motifcl.quantized_transformer.v1") != std::string::npos,
              "unsupported quantized transformer checkpoint manifest");

    assign_checkpoint_parameter(model.token_embedding.weight,
                                load_tensor(backend, checkpoint_path(root, "token_embedding.weight.mclt").string()));
    if (model.config.learned_position_embeddings) {
        assign_checkpoint_parameter(model.position_embedding,
                                    load_tensor(backend, checkpoint_path(root, "position_embedding.weight.mclt").string()));
    }
    assign_checkpoint_parameter(model.final_norm.weight,
                                load_tensor(backend, checkpoint_path(root, "final_norm.weight.mclt").string()));
    model.set_quantized_lm_head(load_tensor(backend, checkpoint_path(root, "lm_head.qweight.mclt").string()));

    for (std::size_t i = 0; i < model.blocks.size(); ++i) {
        auto& block = *model.blocks[i];
        const std::string p = "blocks." + std::to_string(i) + ".";
        assign_checkpoint_parameter(block.norm1().weight,
                                    load_tensor(backend, checkpoint_path(root, p + "norm1.weight.mclt").string()));
        assign_checkpoint_parameter(block.norm2().weight,
                                    load_tensor(backend, checkpoint_path(root, p + "norm2.weight.mclt").string()));
        load_linear_quant_checkpoint(block.attention().qkv_proj(), backend, root, p + "attn.qkv_proj");
        load_linear_quant_checkpoint(block.attention().o_proj(), backend, root, p + "attn.o_proj");
        load_linear_quant_checkpoint(block.mlp().gate_up_proj, backend, root, p + "mlp.gate_up_proj");
        load_linear_quant_checkpoint(block.mlp().down_proj, backend, root, p + "mlp.down_proj");
    }
}

} // namespace motifcl
