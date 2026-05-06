#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

#include <motifcl/core/dtype.hpp>
#include <motifcl/core/shape.hpp>
#include <motifcl/tensor/storage.hpp>

namespace motifcl {

class Backend;
struct BackendLifetime;
class Buffer;
namespace autograd { class Node; }

class Tensor;

void manual_seed(std::uint32_t seed);

struct TensorImpl {
    int id = 0;
    Backend* backend = nullptr;
    std::shared_ptr<BackendLifetime> backend_lifetime;
    std::shared_ptr<Storage> storage;
    Shape shape;
    std::vector<int64_t> strides;
    DType dtype = DType::F32;
    std::size_t offset = 0;
    float quant_scale = 1.0f;
    std::shared_ptr<Tensor> quant_scales;
    int quant_scale_axis = -1;
    int64_t quant_block_size = 0;
    bool requires_grad = false;
    std::shared_ptr<Tensor> grad;
    std::shared_ptr<autograd::Node> grad_fn;
};

class Tensor {
public:
    Tensor() = default;
    Tensor(Backend& backend, Shape shape, DType dtype, std::shared_ptr<Storage> storage, std::size_t offset = 0);

    bool valid() const { return impl_ != nullptr; }
    Backend& backend() const;
    Backend* backend_ptr() const;
    Storage& storage() const;
    Buffer& buffer() const;
    const Shape& shape() const;
    const std::vector<int64_t>& strides() const;
    DType dtype() const;
    std::size_t offset() const;
    int id() const;
    float quant_scale() const;
    bool has_quant_scales() const;
    Tensor quant_scales() const;
    int quant_scale_axis() const;
    int64_t quant_block_size() const;
    int64_t ndim() const { return shape().ndim(); }
    int64_t numel() const { return shape().numel(); }
    std::size_t nbytes() const { return dtype_storage_nbytes(dtype(), static_cast<std::size_t>(numel())); }

    bool requires_grad() const;
    void set_requires_grad(bool value = true);
    std::shared_ptr<Tensor> grad() const;
    void zero_grad();
    void backward();
    void backward(const Tensor& grad_output);

    Tensor view(const Shape& new_shape) const;
    Tensor contiguous() const;
    void to_cpu(void* out, std::size_t bytes = 0) const;

    template <typename T>
    std::vector<T> to_vector() const {
        std::vector<T> result(static_cast<std::size_t>(numel()));
        to_cpu(result.data(), result.size() * sizeof(T));
        return result;
    }

    float item() const;

    static Tensor empty(Backend& backend, const Shape& shape, DType dtype = DType::F32);
    static Tensor from_cpu(Backend& backend, const Shape& shape, DType dtype, const void* data);
    static Tensor zeros(Backend& backend, const Shape& shape, DType dtype = DType::F32);
    static Tensor ones(Backend& backend, const Shape& shape, DType dtype = DType::F32);
    static Tensor full(Backend& backend, const Shape& shape, float value, DType dtype = DType::F32);
    static Tensor randn(Backend& backend, const Shape& shape, float std = 0.02f, DType dtype = DType::F32);
    static Tensor uniform(Backend& backend, const Shape& shape, float low = -1.0f, float high = 1.0f, DType dtype = DType::F32);

    void _set_grad_fn(std::shared_ptr<autograd::Node> node);
    std::shared_ptr<autograd::Node> _grad_fn() const;
    void _accumulate_grad(const Tensor& grad);
    void _set_quant_scale(float scale);
    void _set_quant_scales(const Tensor& scales, int axis, int64_t block_size = 0);

private:
    std::shared_ptr<TensorImpl> impl_;
};

} // namespace motifcl
