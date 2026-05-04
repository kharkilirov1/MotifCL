#include <motifcl/tensor/tensor.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/runtime/backend.hpp>

#include <cstring>
#include <cstdint>
#include <atomic>
#include <random>

namespace motifcl {

namespace {
std::atomic<int> g_next_tensor_id{1};
}

Tensor::Tensor(Backend& backend, Shape shape, DType dtype, std::shared_ptr<Storage> storage, std::size_t offset) {
    impl_ = std::make_shared<TensorImpl>();
    impl_->id = g_next_tensor_id.fetch_add(1, std::memory_order_relaxed);
    impl_->backend = &backend;
    impl_->backend_lifetime = backend.lifetime_handle();
    impl_->storage = std::move(storage);
    impl_->shape = std::move(shape);
    impl_->strides = contiguous_strides(impl_->shape);
    impl_->dtype = dtype;
    impl_->offset = offset;
    autograd::register_tensor(impl_->id, impl_->shape, impl_->dtype, nbytes());
}

Backend& Tensor::backend() const {
    MCL_CHECK(valid(), "invalid tensor");
    MCL_CHECK(impl_->backend_lifetime && impl_->backend_lifetime->alive, "tensor operation requires a live Backend");
    return *impl_->backend;
}

Backend* Tensor::backend_ptr() const {
    return (valid() && impl_->backend_lifetime && impl_->backend_lifetime->alive) ? impl_->backend : nullptr;
}

Storage& Tensor::storage() const {
    MCL_CHECK(valid(), "invalid tensor");
    return *impl_->storage;
}

Buffer& Tensor::buffer() const {
    return storage().buffer;
}

const Shape& Tensor::shape() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->shape;
}

const std::vector<int64_t>& Tensor::strides() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->strides;
}

DType Tensor::dtype() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->dtype;
}

std::size_t Tensor::offset() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->offset;
}

int Tensor::id() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->id;
}

float Tensor::quant_scale() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->quant_scale;
}

bool Tensor::has_quant_scales() const {
    return valid() && impl_->quant_scales && impl_->quant_scales->valid();
}

Tensor Tensor::quant_scales() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->quant_scales ? *impl_->quant_scales : Tensor();
}

int Tensor::quant_scale_axis() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->quant_scale_axis;
}

int64_t Tensor::quant_block_size() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->quant_block_size;
}

bool Tensor::requires_grad() const {
    return valid() && impl_->requires_grad;
}

void Tensor::set_requires_grad(bool value) {
    MCL_CHECK(valid(), "invalid tensor");
    impl_->requires_grad = value;
}

std::shared_ptr<Tensor> Tensor::grad() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->grad;
}

void Tensor::zero_grad() {
    MCL_CHECK(valid(), "invalid tensor");
    impl_->grad.reset();
}

void Tensor::backward() {
    backward(ones_like(*this));
}

void Tensor::backward(const Tensor& grad_output) {
    MCL_CHECK(valid(), "invalid tensor");
    MCL_CHECK(grad_output.shape() == shape(), "backward gradient shape mismatch: got " + grad_output.shape().str() + " expected " + shape().str());
    _accumulate_grad(grad_output);
    auto fn = impl_->grad_fn;
    if (fn) {
        autograd::NoGradGuard guard;
        fn->backward(grad_output);
    }
}

Tensor Tensor::view(const Shape& new_shape) const {
    MCL_CHECK(valid(), "invalid tensor");
    MCL_CHECK(new_shape.numel() == numel(), "view must preserve numel");
    Tensor out(backend(), new_shape, dtype(), impl_->storage, offset());
    out.impl_->requires_grad = impl_->requires_grad;
    out.impl_->grad_fn = impl_->grad_fn;
    out.impl_->quant_scale = impl_->quant_scale;
    if (impl_->quant_scales && new_shape == shape()) {
        out.impl_->quant_scales = impl_->quant_scales;
        out.impl_->quant_scale_axis = impl_->quant_scale_axis;
        out.impl_->quant_block_size = impl_->quant_block_size;
    }
    autograd::record_op("view", {id()}, {out.id()}, true);
    return out;
}

Tensor Tensor::contiguous() const {
    return *this;
}

void Tensor::to_cpu(void* out, std::size_t bytes) const {
    MCL_CHECK(valid(), "invalid tensor");
    if (bytes == 0) bytes = nbytes();
    MCL_CHECK(bytes <= nbytes(), "to_cpu byte count exceeds tensor size");
    buffer().download(out, bytes, offset());
}

float Tensor::item() const {
    MCL_CHECK(dtype() == DType::F32, "item currently supports f32 only");
    MCL_CHECK(numel() == 1, "item requires scalar tensor");
    float value = 0.0f;
    to_cpu(&value, sizeof(float));
    return value;
}

Tensor Tensor::empty(Backend& backend, const Shape& shape, DType dtype) {
    auto bytes = dtype_storage_nbytes(dtype, static_cast<std::size_t>(shape.numel()));
    auto storage = std::make_shared<Storage>(backend, bytes);
    return Tensor(backend, shape, dtype, storage);
}

Tensor Tensor::from_cpu(Backend& backend, const Shape& shape, DType dtype, const void* data) {
    auto bytes = dtype_storage_nbytes(dtype, static_cast<std::size_t>(shape.numel()));
    auto storage = std::make_shared<Storage>(backend, bytes);
    Tensor t(backend, shape, dtype, storage);
    if (bytes > 0 && data != nullptr) t.buffer().upload(data, bytes);
    return t;
}

Tensor Tensor::zeros(Backend& backend, const Shape& shape, DType dtype) {
    const auto n = static_cast<std::size_t>(shape.numel());
    switch (dtype) {
        case DType::F32: {
            std::vector<float> values(n, 0.0f);
            return from_cpu(backend, shape, dtype, values.data());
        }
        case DType::I32: {
            std::vector<std::int32_t> values(n, 0);
            return from_cpu(backend, shape, dtype, values.data());
        }
        case DType::U8: {
            std::vector<std::uint8_t> values(n, 0);
            return from_cpu(backend, shape, dtype, values.data());
        }
        default:
            MCL_CHECK(false, std::string("zeros does not support dtype ") + dtype_name(dtype));
    }
    return {};
}

Tensor Tensor::ones(Backend& backend, const Shape& shape, DType dtype) {
    const auto n = static_cast<std::size_t>(shape.numel());
    switch (dtype) {
        case DType::F32: {
            std::vector<float> values(n, 1.0f);
            return from_cpu(backend, shape, dtype, values.data());
        }
        case DType::I32: {
            std::vector<std::int32_t> values(n, 1);
            return from_cpu(backend, shape, dtype, values.data());
        }
        case DType::U8: {
            std::vector<std::uint8_t> values(n, 1);
            return from_cpu(backend, shape, dtype, values.data());
        }
        default:
            MCL_CHECK(false, std::string("ones does not support dtype ") + dtype_name(dtype));
    }
    return {};
}

Tensor Tensor::full(Backend& backend, const Shape& shape, float value, DType dtype) {
    const auto n = static_cast<std::size_t>(shape.numel());
    switch (dtype) {
        case DType::F32: {
            std::vector<float> values(n, value);
            return from_cpu(backend, shape, dtype, values.data());
        }
        case DType::I32: {
            std::vector<std::int32_t> values(n, static_cast<std::int32_t>(value));
            return from_cpu(backend, shape, dtype, values.data());
        }
        case DType::U8: {
            std::vector<std::uint8_t> values(n, static_cast<std::uint8_t>(value));
            return from_cpu(backend, shape, dtype, values.data());
        }
        default:
            MCL_CHECK(false, std::string("full does not support dtype ") + dtype_name(dtype));
    }
    return {};
}

Tensor Tensor::randn(Backend& backend, const Shape& shape, float std, DType dtype) {
    MCL_CHECK(dtype == DType::F32, "randn currently supports f32 only");
    static thread_local std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, std);
    std::vector<float> values(static_cast<std::size_t>(shape.numel()));
    for (auto& v : values) v = dist(rng);
    return from_cpu(backend, shape, dtype, values.data());
}

Tensor Tensor::uniform(Backend& backend, const Shape& shape, float low, float high, DType dtype) {
    MCL_CHECK(dtype == DType::F32, "uniform currently supports f32 only");
    static thread_local std::mt19937 rng(7331);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> values(static_cast<std::size_t>(shape.numel()));
    for (auto& v : values) v = dist(rng);
    return from_cpu(backend, shape, dtype, values.data());
}

void Tensor::_set_grad_fn(std::shared_ptr<autograd::Node> node) {
    MCL_CHECK(valid(), "invalid tensor");
    impl_->grad_fn = std::move(node);
}

std::shared_ptr<autograd::Node> Tensor::_grad_fn() const {
    MCL_CHECK(valid(), "invalid tensor");
    return impl_->grad_fn;
}

void Tensor::_accumulate_grad(const Tensor& grad_value) {
    MCL_CHECK(valid(), "invalid tensor");
    if (!impl_->grad) {
        impl_->grad = std::make_shared<Tensor>(grad_value);
    } else {
        Tensor updated = add(*impl_->grad, grad_value);
        impl_->grad = std::make_shared<Tensor>(updated);
    }
}

void Tensor::_set_quant_scale(float scale) {
    MCL_CHECK(valid(), "invalid tensor");
    MCL_CHECK(scale > 0.0f, "quantization scale must be positive");
    impl_->quant_scale = scale;
    impl_->quant_scales.reset();
    impl_->quant_scale_axis = -1;
    impl_->quant_block_size = 0;
}

void Tensor::_set_quant_scales(const Tensor& scales, int axis, int64_t block_size) {
    MCL_CHECK(valid(), "invalid tensor");
    MCL_CHECK(scales.valid(), "quant scale tensor is invalid");
    MCL_CHECK(scales.dtype() == DType::F32, "quant scale tensor must be f32");
    MCL_CHECK(scales.backend_ptr() == backend_ptr(), "quant scale tensor must be on same backend");
    MCL_CHECK(axis == 0 || axis == 1 || axis == 2, "quant scale axis must be 0, 1, or 2");
    MCL_CHECK(axis != 2 || block_size > 0, "blockwise quantization requires positive block size");
    impl_->quant_scales = std::make_shared<Tensor>(scales);
    impl_->quant_scale_axis = axis;
    impl_->quant_block_size = block_size;
}

} // namespace motifcl
