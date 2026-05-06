#include <motifcl/ops/attention.hpp>

#include <motifcl/autograd/graph.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/core/error.hpp>
#include <motifcl/ops/matmul.hpp>
#include <motifcl/ops/basic_ops.hpp>
#include <motifcl/runtime/backend.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <utility>

namespace motifcl {

namespace {
std::size_t round_up(std::size_t x, std::size_t multiple) { return ((x + multiple - 1) / multiple) * multiple; }
constexpr std::size_t kFlashAttentionWorkgroup = 128;
constexpr int64_t kFlashAttentionMaxHeadDim = 128;
constexpr std::size_t kFlashAttentionLocalBytes = 24 * 1024;
constexpr int64_t kStagedAttentionMaxSeqLen = 128;

struct MultiHeadShape {
    int64_t total_tokens = 0;
    int64_t channels = 0;
    int64_t batch_size = 1;
    int64_t seq_len = 0;
    int64_t head_dim = 0;
};

MultiHeadShape validate_multihead_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                            int n_head, int64_t batch_size, int64_t seq_len,
                                            const char* op) {
    MCL_CHECK(q.dtype() == DType::F32 && k.dtype() == DType::F32 && v.dtype() == DType::F32, std::string(op) + " supports f32 only");
    MCL_CHECK(q.ndim() == 2 && k.ndim() == 2 && v.ndim() == 2, std::string(op) + " expects flattened [tokens, channels] tensors");
    MCL_CHECK(q.shape() == k.shape() && q.shape() == v.shape(), std::string(op) + " q/k/v shape mismatch");
    MCL_CHECK(q.backend_ptr() == k.backend_ptr() && q.backend_ptr() == v.backend_ptr(), std::string(op) + " requires tensors on same backend");
    MCL_CHECK(n_head > 0, std::string(op) + " n_head must be positive");

    MultiHeadShape s;
    s.total_tokens = q.shape()[0];
    s.channels = q.shape()[1];
    if (seq_len == 0) seq_len = s.total_tokens;
    MCL_CHECK(batch_size > 0 && seq_len > 0, std::string(op) + " batch_size and seq_len must be positive");
    MCL_CHECK(batch_size * seq_len == s.total_tokens, std::string(op) + " batch_size * seq_len must match token count");
    MCL_CHECK(s.channels % n_head == 0, std::string(op) + " channels must be divisible by n_head");
    s.batch_size = batch_size;
    s.seq_len = seq_len;
    s.head_dim = s.channels / n_head;
    return s;
}

bool supports_flash_attention_kernel(const Backend& backend, const MultiHeadShape& s) {
    const auto info = backend.device_info();
    return s.head_dim > 0 &&
           s.head_dim <= kFlashAttentionMaxHeadDim &&
           info.max_work_group_size >= kFlashAttentionWorkgroup &&
           info.local_mem_size >= kFlashAttentionLocalBytes;
}

bool supports_staged_attention_backward_kernel(const Backend& backend, const MultiHeadShape& s) {
    return supports_flash_attention_kernel(backend, s) && s.seq_len <= kStagedAttentionMaxSeqLen;
}

struct MultiHeadAttentionGradients {
    Tensor dq;
    Tensor dk;
    Tensor dv;
};

MultiHeadAttentionGradients multihead_attention_backward_fused(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                                                int n_head, bool causal, int64_t batch_size, int64_t seq_len);

struct MultiHeadAttentionBackwardNode : autograd::Node {
    Tensor q, k, v;
    int n_head;
    bool causal;
    int64_t batch_size;
    int64_t seq_len;

    MultiHeadAttentionBackwardNode(Tensor q_value, Tensor k_value, Tensor v_value,
                                   int n_head_value, bool causal_value,
                                   int64_t batch_size_value, int64_t seq_len_value)
        : q(std::move(q_value)),
          k(std::move(k_value)),
          v(std::move(v_value)),
          n_head(n_head_value),
          causal(causal_value),
          batch_size(batch_size_value),
          seq_len(seq_len_value) {}

    std::vector<Tensor> inputs() const override { return {q, k, v}; }

    void backward(const Tensor& grad_output) override {
        if (!(q.requires_grad() || k.requires_grad() || v.requires_grad())) return;
        auto grads = multihead_attention_backward_fused(q, k, v, grad_output, n_head, causal, batch_size, seq_len);
        if (q.requires_grad()) q.backward(grads.dq);
        if (k.requires_grad()) k.backward(grads.dk);
        if (v.requires_grad()) v.backward(grads.dv);
    }
};
}

Tensor softmax_rows(const Tensor& x) {
    MCL_CHECK(x.dtype() == DType::F32, "softmax_rows supports f32 only");
    MCL_CHECK(x.ndim() == 2, "softmax_rows expects rank-2 tensor");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto k = x.backend().kernels.get("softmax_rows_f32");
    int rows = static_cast<int>(x.shape()[0]);
    int cols = static_cast<int>(x.shape()[1]);
    k.set_arg(0, x.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, rows);
    k.set_arg(3, cols);
    k.launch1d(round_up(static_cast<std::size_t>(rows), 256), 256);
    autograd::record_op("softmax_rows_f32", {x.id()}, {out.id()});
    return out;
}

Tensor causal_mask(const Tensor& scores, float mask_value) {
    MCL_CHECK(scores.dtype() == DType::F32, "causal_mask supports f32 only");
    MCL_CHECK(scores.ndim() == 2, "causal_mask expects rank-2 tensor");
    auto out = Tensor::empty(scores.backend(), scores.shape(), DType::F32);
    auto k = scores.backend().kernels.get("causal_mask_f32");
    int rows = static_cast<int>(scores.shape()[0]);
    int cols = static_cast<int>(scores.shape()[1]);
    int n = rows * cols;
    k.set_arg(0, scores.buffer());
    k.set_arg(1, out.buffer());
    k.set_arg(2, rows);
    k.set_arg(3, cols);
    k.set_arg(4, mask_value);
    k.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("causal_mask_f32", {scores.id()}, {out.id()});
    return out;
}

Tensor attention_scores(const Tensor& q, const Tensor& k, float scale_value) {
    auto scores = matmul_transpose_b(q, k);
    return scale(scores, scale_value);
}

Tensor attention_apply(const Tensor& probs, const Tensor& v) {
    return matmul(probs, v);
}

Tensor multihead_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                           int n_head, bool causal, int64_t batch_size, int64_t seq_len) {
    auto s = validate_multihead_attention(q, k, v, n_head, batch_size, seq_len, "multihead_attention");
    auto out = Tensor::empty(q.backend(), q.shape(), DType::F32);
    const bool use_flash = supports_flash_attention_kernel(q.backend(), s);
    const std::string kernel_name = use_flash ? "multihead_attention_flash_f32" : "multihead_attention_f32";
    auto kernel = q.backend().kernels.get(kernel_name);
    const int total = static_cast<int>(q.numel());
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k.buffer());
    kernel.set_arg(2, v.buffer());
    kernel.set_arg(3, out.buffer());
    kernel.set_arg(4, static_cast<int>(s.batch_size));
    kernel.set_arg(5, static_cast<int>(s.seq_len));
    kernel.set_arg(6, static_cast<int>(s.channels));
    kernel.set_arg(7, n_head);
    kernel.set_arg(8, static_cast<int>(s.head_dim));
    kernel.set_arg(9, causal ? 1 : 0);
    kernel.set_arg(10, scale);
    if (use_flash) {
        kernel.launch2d(static_cast<std::size_t>(s.batch_size * n_head) * kFlashAttentionWorkgroup,
                        static_cast<std::size_t>(s.seq_len),
                        kFlashAttentionWorkgroup,
                        1);
    } else {
        kernel.launch1d(round_up(static_cast<std::size_t>(total), 256), 256);
    }
    autograd::record_op(kernel_name, {q.id(), k.id(), v.id()}, {out.id()});
    if (autograd::is_enabled() && (q.requires_grad() || k.requires_grad() || v.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<MultiHeadAttentionBackwardNode>(q, k, v, n_head, causal, s.batch_size, s.seq_len));
    }
    return out;
}

namespace {

MultiHeadAttentionGradients multihead_attention_backward_staged(const Tensor& q, const Tensor& k, const Tensor& v,
                                                                const Tensor& grad_out,
                                                                int n_head, bool causal,
                                                                const MultiHeadShape& s) {
    auto dq = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto dk = Tensor::empty(k.backend(), k.shape(), DType::F32);
    auto dv = Tensor::empty(v.backend(), v.shape(), DType::F32);

    const auto rows = static_cast<std::size_t>(s.batch_size * n_head * s.seq_len);
    const auto pairs = rows * static_cast<std::size_t>(s.seq_len);
    auto probs = Tensor::empty(q.backend(), {static_cast<int64_t>(rows), s.seq_len}, DType::F32);
    auto dp = Tensor::empty(q.backend(), probs.shape(), DType::F32);
    auto ds = Tensor::empty(q.backend(), probs.shape(), DType::F32);
    const int batch_i = static_cast<int>(s.batch_size);
    const int tokens_i = static_cast<int>(s.seq_len);
    const int channels_i = static_cast<int>(s.channels);
    const int head_dim_i = static_cast<int>(s.head_dim);
    const int causal_i = causal ? 1 : 0;
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));

    {
        auto kernel = q.backend().kernels.get("attention_backward_probs_f32");
        kernel.set_arg(0, q.buffer());
        kernel.set_arg(1, k.buffer());
        kernel.set_arg(2, probs.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, tokens_i);
        kernel.set_arg(5, channels_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, head_dim_i);
        kernel.set_arg(8, causal_i);
        kernel.set_arg(9, scale);
        kernel.launch2d(kFlashAttentionWorkgroup, rows, kFlashAttentionWorkgroup, 1);
    }

    {
        auto kernel = q.backend().kernels.get("attention_backward_dp_f32");
        kernel.set_arg(0, grad_out.buffer());
        kernel.set_arg(1, v.buffer());
        kernel.set_arg(2, dp.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, tokens_i);
        kernel.set_arg(5, channels_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, head_dim_i);
        kernel.set_arg(8, causal_i);
        kernel.launch1d(round_up(pairs, 256), 256);
    }

    {
        auto kernel = q.backend().kernels.get("attention_backward_ds_f32");
        kernel.set_arg(0, probs.buffer());
        kernel.set_arg(1, dp.buffer());
        kernel.set_arg(2, ds.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, tokens_i);
        kernel.set_arg(5, n_head);
        kernel.set_arg(6, causal_i);
        kernel.set_arg(7, scale);
        kernel.launch2d(kFlashAttentionWorkgroup, rows, kFlashAttentionWorkgroup, 1);
    }

    const auto total = static_cast<std::size_t>(q.numel());
    {
        auto kernel = q.backend().kernels.get("attention_backward_apply_q_f32");
        kernel.set_arg(0, ds.buffer());
        kernel.set_arg(1, k.buffer());
        kernel.set_arg(2, dq.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, tokens_i);
        kernel.set_arg(5, channels_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, head_dim_i);
        kernel.set_arg(8, causal_i);
        kernel.launch1d(round_up(total, 256), 256);
    }

    {
        auto kernel = q.backend().kernels.get("attention_backward_apply_k_f32");
        kernel.set_arg(0, ds.buffer());
        kernel.set_arg(1, q.buffer());
        kernel.set_arg(2, dk.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, tokens_i);
        kernel.set_arg(5, channels_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, head_dim_i);
        kernel.set_arg(8, causal_i);
        kernel.launch1d(round_up(total, 256), 256);
    }

    {
        auto kernel = q.backend().kernels.get("attention_backward_apply_v_f32");
        kernel.set_arg(0, probs.buffer());
        kernel.set_arg(1, grad_out.buffer());
        kernel.set_arg(2, dv.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, tokens_i);
        kernel.set_arg(5, channels_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, head_dim_i);
        kernel.set_arg(8, causal_i);
        kernel.launch1d(round_up(total, 256), 256);
    }

    autograd::record_replay_op("multihead_attention_backward_staged_f32",
                               {q.id(), k.id(), v.id(), grad_out.id()},
                               {dq.id(), dk.id(), dv.id()},
                               {probs.id(), dp.id(), ds.id()},
                               []() {});
    return {dq, dk, dv};
}

MultiHeadAttentionGradients multihead_attention_backward_fused(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                                               int n_head, bool causal, int64_t batch_size, int64_t seq_len) {
    auto s = validate_multihead_attention(q, k, v, n_head, batch_size, seq_len, "multihead_attention_backward_fused");
    MCL_CHECK(grad_out.dtype() == DType::F32 && grad_out.shape() == q.shape(), "multihead_attention_backward_fused grad_out shape/dtype mismatch");
    if (supports_staged_attention_backward_kernel(q.backend(), s)) {
        return multihead_attention_backward_staged(q, k, v, grad_out, n_head, causal, s);
    }
    auto dq = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto dk = Tensor::empty(k.backend(), k.shape(), DType::F32);
    auto dv = Tensor::empty(v.backend(), v.shape(), DType::F32);
    const bool use_flash = supports_flash_attention_kernel(q.backend(), s);
    const std::string kernel_name = use_flash ? "multihead_attention_backward_flash_f32" : "multihead_attention_backward_fused_f32";
    auto kernel = q.backend().kernels.get(kernel_name);
    const int total = static_cast<int>(q.numel());
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k.buffer());
    kernel.set_arg(2, v.buffer());
    kernel.set_arg(3, grad_out.buffer());
    kernel.set_arg(4, dq.buffer());
    kernel.set_arg(5, dk.buffer());
    kernel.set_arg(6, dv.buffer());
    kernel.set_arg(7, static_cast<int>(s.batch_size));
    kernel.set_arg(8, static_cast<int>(s.seq_len));
    kernel.set_arg(9, static_cast<int>(s.channels));
    kernel.set_arg(10, n_head);
    kernel.set_arg(11, static_cast<int>(s.head_dim));
    kernel.set_arg(12, causal ? 1 : 0);
    kernel.set_arg(13, scale);
    if (use_flash) {
        kernel.launch2d(static_cast<std::size_t>(s.batch_size * n_head) * kFlashAttentionWorkgroup,
                        static_cast<std::size_t>(s.seq_len * 3),
                        kFlashAttentionWorkgroup,
                        1);
    } else {
        kernel.launch1d(round_up(static_cast<std::size_t>(total * 3), 256), 256);
    }
    autograd::record_op(kernel_name, {q.id(), k.id(), v.id(), grad_out.id()}, {dq.id(), dk.id(), dv.id()});
    return {dq, dk, dv};
}

} // namespace

Tensor multihead_attention_backward_q(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                      int n_head, bool causal, int64_t batch_size, int64_t seq_len) {
    return multihead_attention_backward_fused(q, k, v, grad_out, n_head, causal, batch_size, seq_len).dq;
}

Tensor multihead_attention_backward_k(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                      int n_head, bool causal, int64_t batch_size, int64_t seq_len) {
    return multihead_attention_backward_fused(q, k, v, grad_out, n_head, causal, batch_size, seq_len).dk;
}

Tensor multihead_attention_backward_v(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                      int n_head, bool causal, int64_t batch_size, int64_t seq_len) {
    return multihead_attention_backward_fused(q, k, v, grad_out, n_head, causal, batch_size, seq_len).dv;
}

} // namespace motifcl
