#include <motifcl/serialization.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace motifcl {

namespace {

constexpr char kTensorMagic[8] = {'M', 'C', 'L', 'T', 'E', 'N', '1', '\0'};
constexpr char kParamMagic[8] = {'M', 'C', 'L', 'P', 'A', 'R', '1', '\0'};

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
    std::vector<std::uint8_t> bytes(tensor.nbytes());
    if (!bytes.empty()) tensor.to_cpu(bytes.data(), bytes.size());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    MCL_CHECK(out.good(), "failed to write tensor payload");
}

Tensor read_tensor_payload(Backend& backend, std::istream& in) {
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

void check_magic(std::istream& in, const char expected[8]) {
    char actual[8] = {};
    in.read(actual, sizeof(actual));
    MCL_CHECK(in.good() && std::memcmp(actual, expected, sizeof(actual)) == 0, "serialization magic mismatch");
}

} // namespace

void save_tensor(const Tensor& tensor, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    MCL_CHECK(out.good(), "failed to open tensor file for write: " + path);
    out.write(kTensorMagic, sizeof(kTensorMagic));
    write_tensor_payload(out, tensor);
}

Tensor load_tensor(Backend& backend, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open tensor file for read: " + path);
    check_magic(in, kTensorMagic);
    return read_tensor_payload(backend, in);
}

void save_parameters(const std::vector<nn::Parameter*>& params, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    MCL_CHECK(out.good(), "failed to open parameter checkpoint for write: " + path);
    out.write(kParamMagic, sizeof(kParamMagic));
    write_value(out, static_cast<std::uint64_t>(params.size()));
    for (auto* param : params) {
        MCL_CHECK(param != nullptr, "cannot save null parameter");
        write_tensor_payload(out, param->data);
    }
}

void load_parameters(const std::vector<nn::Parameter*>& params, Backend& backend, const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    MCL_CHECK(in.good(), "failed to open parameter checkpoint for read: " + path);
    check_magic(in, kParamMagic);
    const auto count = read_value<std::uint64_t>(in);
    MCL_CHECK(count == params.size(), "checkpoint parameter count mismatch");
    for (auto* param : params) {
        MCL_CHECK(param != nullptr, "cannot load into null parameter");
        auto loaded = read_tensor_payload(backend, in);
        MCL_CHECK(loaded.shape() == param->data.shape(), "checkpoint parameter shape mismatch");
        MCL_CHECK(loaded.dtype() == param->data.dtype(), "checkpoint parameter dtype mismatch");
        param->data = std::move(loaded);
        param->data.set_requires_grad(param->trainable);
    }
}

} // namespace motifcl
