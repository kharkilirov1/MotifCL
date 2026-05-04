#include <motifcl/nn/embedding.hpp>

#include <cmath>
#include <memory>
#include <utility>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>

namespace motifcl::nn {
namespace {
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }

Tensor embedding_weight_backward(const Tensor& indices, const Tensor& grad_out, const Shape& weight_shape) {
    MCL_CHECK(indices.dtype() == DType::I32, "embedding_weight_backward indices must be i32");
    MCL_CHECK(grad_out.dtype() == DType::F32, "embedding_weight_backward grad_out must be f32");
    MCL_CHECK(weight_shape.ndim() == 2, "embedding_weight_backward weight_shape must be rank-2");
    const int vocab_size = static_cast<int>(weight_shape[0]);
    const int embed_dim = static_cast<int>(weight_shape[1]);
    MCL_CHECK(grad_out.numel() == indices.numel() * embed_dim, "embedding_weight_backward grad_out shape mismatch");
    auto out = Tensor::zeros(indices.backend(), weight_shape, DType::F32);
    auto k = indices.backend().kernels.get("embedding_weight_backward_f32_i32");
    int n = vocab_size * embed_dim;
    k.set_arg(0, indices.buffer());
    k.set_arg(1, grad_out.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, vocab_size);
    k.set_arg(4, embed_dim);
    k.set_arg(5, static_cast<int>(indices.numel()));
    k.set_arg(6, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("embedding_weight_backward_f32_i32", {indices.id(), grad_out.id()}, {out.id()});
    return out;
}

Tensor position_embedding_backward(const Tensor& grad_out, const Shape& position_shape, int64_t batch, int64_t seq_len) {
    MCL_CHECK(grad_out.dtype() == DType::F32, "position_embedding_backward grad_out must be f32");
    MCL_CHECK(position_shape.ndim() == 2, "position_embedding_backward position_shape must be rank-2");
    const int embed_dim = static_cast<int>(position_shape[1]);
    MCL_CHECK(position_shape[0] >= seq_len, "position_embedding_backward position table shorter than sequence");
    MCL_CHECK(grad_out.numel() == batch * seq_len * embed_dim, "position_embedding_backward grad_out shape mismatch");
    auto out = Tensor::zeros(grad_out.backend(), position_shape, DType::F32);
    auto k = grad_out.backend().kernels.get("position_embedding_backward_f32_i32");
    int n = static_cast<int>(seq_len * embed_dim);
    k.set_arg(0, grad_out.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, static_cast<int>(batch));
    k.set_arg(3, static_cast<int>(seq_len));
    k.set_arg(4, embed_dim);
    k.set_arg(5, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("position_embedding_backward_f32_i32", {grad_out.id()}, {out.id()});
    return out;
}

struct EmbeddingBackwardNode : autograd::Node {
    Tensor indices;
    Tensor weight;

    EmbeddingBackwardNode(Tensor indices_value, Tensor weight_value)
        : indices(std::move(indices_value)), weight(std::move(weight_value)) {}

    void backward(const Tensor& grad_output) override {
        if (weight.requires_grad()) weight.backward(embedding_weight_backward(indices, grad_output, weight.shape()));
    }
};

struct TokenPositionEmbeddingBackwardNode : autograd::Node {
    Tensor token_ids;
    Tensor token_weight;
    Tensor position_weight;
    int64_t batch = 1;
    int64_t seq_len = 0;

    TokenPositionEmbeddingBackwardNode(Tensor token_ids_value, Tensor token_weight_value, Tensor position_weight_value,
                                       int64_t batch_value, int64_t seq_len_value)
        : token_ids(std::move(token_ids_value)),
          token_weight(std::move(token_weight_value)),
          position_weight(std::move(position_weight_value)),
          batch(batch_value),
          seq_len(seq_len_value) {}

    void backward(const Tensor& grad_output) override {
        if (token_weight.requires_grad()) {
            token_weight.backward(embedding_weight_backward(token_ids, grad_output, token_weight.shape()));
        }
        if (position_weight.requires_grad()) {
            position_weight.backward(position_embedding_backward(grad_output, position_weight.shape(), batch, seq_len));
        }
    }
};
} // namespace

Embedding::Embedding(Backend& backend, int vocab_size, int embed_dim)
    : weight(Tensor::randn(backend, {vocab_size, embed_dim}, 1.0f / std::sqrt(static_cast<float>(embed_dim)))),
      vocab_size_(vocab_size),
      embed_dim_(embed_dim) {
    weight.data.set_requires_grad(true);
}

Tensor Embedding::forward(const Tensor& indices) {
    MCL_CHECK(indices.dtype() == DType::I32, "Embedding indices must be i32");
    MCL_CHECK(indices.ndim() == 1 || indices.ndim() == 2, "Embedding indices must be rank-1 or rank-2");
    MCL_CHECK(indices.backend_ptr() == weight.data.backend_ptr(), "Embedding indices and weights must share backend");

    const int64_t token_count = indices.numel();
    Shape out_shape = indices.ndim() == 1 ? Shape{token_count, embed_dim_} : Shape{indices.shape()[0], indices.shape()[1], embed_dim_};
    auto out = Tensor::zeros(indices.backend(), out_shape, DType::F32);
    auto k = indices.backend().kernels.get("embedding_gather_f32_i32");
    int n = static_cast<int>(token_count * embed_dim_);
    k.set_arg(0, weight.data.buffer());
    k.set_arg(1, indices.buffer());
    k.set_arg(2, out.buffer());
    k.set_arg(3, vocab_size_);
    k.set_arg(4, embed_dim_);
    k.set_arg(5, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("embedding_gather_f32_i32", {weight.data.id(), indices.id()}, {out.id()});
    if (autograd::is_enabled() && weight.data.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<EmbeddingBackwardNode>(indices, weight.data));
    }
    return out;
}

Tensor token_position_embedding(const Tensor& token_ids, const Tensor& token_weight, const Tensor& position_weight) {
    MCL_CHECK(token_ids.dtype() == DType::I32, "token_position_embedding token_ids must be i32");
    MCL_CHECK(token_ids.ndim() == 1 || token_ids.ndim() == 2, "token ids must be rank-1 or rank-2");
    MCL_CHECK(token_weight.dtype() == DType::F32 && position_weight.dtype() == DType::F32, "embedding weights must be f32");
    MCL_CHECK(token_weight.ndim() == 2 && position_weight.ndim() == 2, "embedding weights must be rank-2");
    MCL_CHECK(token_ids.backend_ptr() == token_weight.backend_ptr() && token_ids.backend_ptr() == position_weight.backend_ptr(), "token_position_embedding requires one backend");
    const int64_t B = token_ids.ndim() == 2 ? token_ids.shape()[0] : 1;
    const int64_t T = token_ids.ndim() == 2 ? token_ids.shape()[1] : token_ids.shape()[0];
    const int64_t V = token_weight.shape()[0];
    const int64_t D = token_weight.shape()[1];
    MCL_CHECK(position_weight.shape()[1] == D, "position embedding dim mismatch");
    MCL_CHECK(position_weight.shape()[0] >= T, "position embedding table is shorter than sequence length");

    auto out = Tensor::zeros(token_ids.backend(), {B * T, D}, DType::F32);
    auto k = token_ids.backend().kernels.get("token_position_embedding_f32_i32");
    int n = static_cast<int>(B * T * D);
    k.set_arg(0, token_weight.buffer());
    k.set_arg(1, position_weight.buffer());
    k.set_arg(2, token_ids.buffer());
    k.set_arg(3, out.buffer());
    k.set_arg(4, static_cast<int>(V));
    k.set_arg(5, static_cast<int>(T));
    k.set_arg(6, static_cast<int>(D));
    k.set_arg(7, n);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("token_position_embedding_f32_i32", {token_weight.id(), position_weight.id(), token_ids.id()}, {out.id()});
    if (autograd::is_enabled() && (token_weight.requires_grad() || position_weight.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<TokenPositionEmbeddingBackwardNode>(token_ids, token_weight, position_weight, B, T));
    }
    return out;
}

} // namespace motifcl::nn
