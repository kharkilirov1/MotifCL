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

struct GroupedAttentionShape {
    int64_t batch = 1;
    int64_t query_tokens = 0;
    int64_t key_tokens = 0;
    int64_t n_head = 0;
    int64_t n_kv_head = 0;
    int64_t head_dim = 0;
    int64_t q_channels = 0;
    int64_t kv_channels = 0;
};

struct AttentionMaskShape {
    int layout = 0;
    int mode = 0; // 0 = nonzero mask means blocked; 1 = additive F32 bias
    std::string suffix;
};

bool supports_staged_grouped_attention_backward_kernel(const Backend& backend, const GroupedAttentionShape& s) {
    const auto info = backend.device_info();
    return s.head_dim > 0 &&
           s.head_dim <= kFlashAttentionMaxHeadDim &&
           s.key_tokens <= kStagedAttentionMaxSeqLen &&
           info.max_work_group_size >= kFlashAttentionWorkgroup;
}

GroupedAttentionShape validate_grouped_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                                 int n_head, int n_kv_head,
                                                 int64_t batch_size, int64_t query_len, int64_t key_len,
                                                 const char* op) {
    MCL_CHECK(q.dtype() == DType::F32 && k.dtype() == DType::F32 && v.dtype() == DType::F32, std::string(op) + " supports f32 only");
    MCL_CHECK(q.ndim() == 2 && k.ndim() == 2 && v.ndim() == 2, std::string(op) + " expects flattened rank-2 tensors");
    MCL_CHECK(k.shape() == v.shape(), std::string(op) + " k/v shape mismatch");
    MCL_CHECK(q.backend_ptr() == k.backend_ptr() && q.backend_ptr() == v.backend_ptr(), std::string(op) + " requires one backend");
    MCL_CHECK(n_head > 0 && n_kv_head > 0 && n_head % n_kv_head == 0, std::string(op) + " requires n_head % n_kv_head == 0");
    if (query_len == 0) query_len = q.shape()[0] / batch_size;
    if (key_len == 0) key_len = k.shape()[0] / batch_size;
    MCL_CHECK(batch_size > 0 && query_len > 0 && key_len > 0, std::string(op) + " invalid batch/query/key lengths");
    MCL_CHECK(batch_size * query_len == q.shape()[0], std::string(op) + " batch * query_len mismatch");
    MCL_CHECK(batch_size * key_len == k.shape()[0], std::string(op) + " batch * key_len mismatch");
    MCL_CHECK(q.shape()[1] % n_head == 0, std::string(op) + " q channels must divide n_head");
    MCL_CHECK(k.shape()[1] % n_kv_head == 0, std::string(op) + " kv channels must divide n_kv_head");
    GroupedAttentionShape s;
    s.batch = batch_size;
    s.query_tokens = query_len;
    s.key_tokens = key_len;
    s.n_head = n_head;
    s.n_kv_head = n_kv_head;
    s.q_channels = q.shape()[1];
    s.kv_channels = k.shape()[1];
    s.head_dim = s.q_channels / n_head;
    MCL_CHECK(s.kv_channels / n_kv_head == s.head_dim, std::string(op) + " q/kv head_dim mismatch");
    return s;
}

AttentionMaskShape validate_attention_mask(const Tensor& mask,
                                           int64_t batch_size,
                                           int64_t query_len,
                                           int64_t key_len,
                                           bool additive_mask,
                                           const char* op) {
    MCL_CHECK(mask.dtype() == DType::F32 || mask.dtype() == DType::I32 || mask.dtype() == DType::U8,
              std::string(op) + " mask must be f32, i32, or u8");
    MCL_CHECK(!additive_mask || mask.dtype() == DType::F32,
              std::string(op) + " additive masks require f32 dtype");
    MCL_CHECK(mask.ndim() == 2 || mask.ndim() == 3,
              std::string(op) + " mask must be [Q,K], [B,K], [B*Q,K], [B,Q,K], [B,1,K], or [1,Q,K]");
    AttentionMaskShape info;
    info.mode = additive_mask ? 1 : 0;
    if (mask.dtype() == DType::F32) info.suffix = "f32";
    else if (mask.dtype() == DType::I32) info.suffix = "i32";
    else info.suffix = "u8";

    if (mask.ndim() == 2) {
        const int64_t m0 = mask.shape()[0];
        const int64_t m1 = mask.shape()[1];
        MCL_CHECK(m1 == key_len, std::string(op) + " mask last dimension must match key_len");
        if (m0 == batch_size * query_len) info.layout = 3;      // [B*Q,K]
        else if (m0 == query_len) info.layout = 1;              // [Q,K]
        else if (m0 == batch_size) info.layout = 2;             // [B,K]
        else MCL_CHECK(false, std::string(op) + " unsupported rank-2 mask shape");
    } else {
        const int64_t m0 = mask.shape()[0];
        const int64_t m1 = mask.shape()[1];
        const int64_t m2 = mask.shape()[2];
        MCL_CHECK(m2 == key_len, std::string(op) + " mask last dimension must match key_len");
        if (m0 == batch_size && m1 == query_len) info.layout = 3;      // [B,Q,K]
        else if (m0 == batch_size && m1 == 1) info.layout = 4;         // [B,1,K]
        else if (m0 == 1 && m1 == query_len) info.layout = 5;          // [1,Q,K]
        else MCL_CHECK(false, std::string(op) + " unsupported rank-3 mask shape");
    }
    return info;
}

MultiHeadAttentionGradients multihead_attention_backward_fused(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                                                int n_head, bool causal, int64_t batch_size, int64_t seq_len);
MultiHeadAttentionGradients grouped_query_attention_backward_fused(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                                                   int n_head, int n_kv_head, bool causal,
                                                                   int64_t batch_size, int64_t query_len,
                                                                   int64_t key_len, int64_t query_offset);
MultiHeadAttentionGradients grouped_query_attention_backward_masked_fused(const Tensor& q, const Tensor& k, const Tensor& v,
                                                                          const Tensor& mask, const Tensor& grad_out,
                                                                          int n_head, int n_kv_head, bool causal,
                                                                          int64_t batch_size, int64_t query_len,
                                                                          int64_t key_len, int64_t query_offset,
                                                                          bool additive_mask);

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

struct QKVSplitBackwardNode : autograd::Node {
    Tensor packed;
    int64_t q_dim;
    int64_t kv_dim;
    int part;
    QKVSplitBackwardNode(Tensor packed_value, int64_t q_dim_value, int64_t kv_dim_value, int part_value)
        : packed(std::move(packed_value)), q_dim(q_dim_value), kv_dim(kv_dim_value), part(part_value) {}
    std::vector<Tensor> inputs() const override { return {packed}; }
    void backward(const Tensor& grad_output) override {
        if (!packed.requires_grad()) return;
        auto grad_packed = Tensor::empty(packed.backend(), packed.shape(), DType::F32);
        auto kernel = packed.backend().kernels.get("qkv_split_backward_f32");
        const int rows = static_cast<int>(packed.shape()[0]);
        const int q_dim_i = static_cast<int>(q_dim);
        const int kv_dim_i = static_cast<int>(kv_dim);
        const int n = rows * (q_dim_i + 2 * kv_dim_i);
        kernel.set_arg(0, grad_output.buffer());
        kernel.set_arg(1, grad_packed.buffer());
        kernel.set_arg(2, rows);
        kernel.set_arg(3, q_dim_i);
        kernel.set_arg(4, kv_dim_i);
        kernel.set_arg(5, part);
        kernel.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
        autograd::record_op("qkv_split_backward_f32", {grad_output.id()}, {grad_packed.id()});
        packed.backward(grad_packed);
    }
};

struct RopeBackwardNode : autograd::Node {
    Tensor x;
    int n_head;
    int64_t batch_size;
    int64_t seq_len;
    float theta;
    int64_t rotary_dim;
    int64_t token_offset;
    RopeBackwardNode(Tensor x_value, int n_head_value, int64_t batch_value, int64_t seq_value,
                     float theta_value, int64_t rotary_dim_value, int64_t offset_value)
        : x(std::move(x_value)), n_head(n_head_value), batch_size(batch_value), seq_len(seq_value),
          theta(theta_value), rotary_dim(rotary_dim_value), token_offset(offset_value) {}
    std::vector<Tensor> inputs() const override { return {x}; }
    void backward(const Tensor& grad_output) override;
};

struct GroupedQueryAttentionBackwardNode : autograd::Node {
    Tensor q, k, v;
    int n_head;
    int n_kv_head;
    bool causal;
    int64_t batch_size;
    int64_t query_len;
    int64_t key_len;
    int64_t query_offset;
    GroupedQueryAttentionBackwardNode(Tensor q_value, Tensor k_value, Tensor v_value,
                                      int n_head_value, int n_kv_head_value, bool causal_value,
                                      int64_t batch_value, int64_t query_value,
                                      int64_t key_value, int64_t offset_value)
        : q(std::move(q_value)), k(std::move(k_value)), v(std::move(v_value)),
          n_head(n_head_value), n_kv_head(n_kv_head_value), causal(causal_value),
          batch_size(batch_value), query_len(query_value), key_len(key_value), query_offset(offset_value) {}
    std::vector<Tensor> inputs() const override { return {q, k, v}; }
    void backward(const Tensor& grad_output) override {
        auto grads = grouped_query_attention_backward_fused(q, k, v, grad_output, n_head, n_kv_head,
                                                            causal, batch_size, query_len, key_len, query_offset);
        if (q.requires_grad()) q.backward(grads.dq);
        if (k.requires_grad()) k.backward(grads.dk);
        if (v.requires_grad()) v.backward(grads.dv);
    }
};

struct GroupedQueryAttentionMaskedBackwardNode : autograd::Node {
    Tensor q, k, v, mask;
    int n_head;
    int n_kv_head;
    bool causal;
    int64_t batch_size;
    int64_t query_len;
    int64_t key_len;
    int64_t query_offset;
    bool additive_mask;
    GroupedQueryAttentionMaskedBackwardNode(Tensor q_value, Tensor k_value, Tensor v_value, Tensor mask_value,
                                            int n_head_value, int n_kv_head_value, bool causal_value,
                                            int64_t batch_value, int64_t query_value,
                                            int64_t key_value, int64_t offset_value,
                                            bool additive_value)
        : q(std::move(q_value)), k(std::move(k_value)), v(std::move(v_value)), mask(std::move(mask_value)),
          n_head(n_head_value), n_kv_head(n_kv_head_value), causal(causal_value),
          batch_size(batch_value), query_len(query_value), key_len(key_value), query_offset(offset_value),
          additive_mask(additive_value) {}
    std::vector<Tensor> inputs() const override { return {q, k, v}; }
    void backward(const Tensor& grad_output) override {
        auto grads = grouped_query_attention_backward_masked_fused(q, k, v, mask, grad_output, n_head, n_kv_head,
                                                                   causal, batch_size, query_len, key_len,
                                                                   query_offset, additive_mask);
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

QKV qkv_split(const Tensor& packed, int64_t q_dim, int64_t kv_dim) {
    MCL_CHECK(packed.dtype() == DType::F32 && packed.ndim() == 2, "qkv_split expects packed f32 [rows, q+2kv]");
    MCL_CHECK(q_dim > 0 && kv_dim > 0 && packed.shape()[1] == q_dim + 2 * kv_dim, "qkv_split dimension mismatch");
    auto q = Tensor::empty(packed.backend(), {packed.shape()[0], q_dim}, DType::F32);
    auto k_out = Tensor::empty(packed.backend(), {packed.shape()[0], kv_dim}, DType::F32);
    auto v_out = Tensor::empty(packed.backend(), {packed.shape()[0], kv_dim}, DType::F32);
    auto kernel = packed.backend().kernels.get("qkv_split_f32");
    const int rows = static_cast<int>(packed.shape()[0]);
    const int q_dim_i = static_cast<int>(q_dim);
    const int kv_dim_i = static_cast<int>(kv_dim);
    const int max_dim = std::max(q_dim_i, kv_dim_i);
    kernel.set_arg(0, packed.buffer());
    kernel.set_arg(1, q.buffer());
    kernel.set_arg(2, k_out.buffer());
    kernel.set_arg(3, v_out.buffer());
    kernel.set_arg(4, rows);
    kernel.set_arg(5, q_dim_i);
    kernel.set_arg(6, kv_dim_i);
    kernel.launch1d(round_up(static_cast<std::size_t>(rows * max_dim), 256), 256);
    autograd::record_op("qkv_split_f32", {packed.id()}, {q.id(), k_out.id(), v_out.id()});
    if (autograd::is_enabled() && packed.requires_grad()) {
        q.set_requires_grad(true);
        k_out.set_requires_grad(true);
        v_out.set_requires_grad(true);
        q._set_grad_fn(std::make_shared<QKVSplitBackwardNode>(packed, q_dim, kv_dim, 0));
        k_out._set_grad_fn(std::make_shared<QKVSplitBackwardNode>(packed, q_dim, kv_dim, 1));
        v_out._set_grad_fn(std::make_shared<QKVSplitBackwardNode>(packed, q_dim, kv_dim, 2));
    }
    return {q, k_out, v_out};
}

namespace {
Tensor rope_impl(const Tensor& x, int n_head, int64_t batch_size, int64_t seq_len,
                 float theta, int64_t rotary_dim, int64_t token_offset, bool inverse) {
    MCL_CHECK(x.dtype() == DType::F32 && x.ndim() == 2, "rope expects f32 [B*T, channels]");
    MCL_CHECK(n_head > 0 && x.shape()[1] % n_head == 0, "rope channels must divide n_head");
    if (seq_len == 0) seq_len = x.shape()[0] / batch_size;
    MCL_CHECK(batch_size > 0 && seq_len > 0 && batch_size * seq_len == x.shape()[0], "rope invalid batch/seq shape");
    const int head_dim = static_cast<int>(x.shape()[1] / n_head);
    int64_t rd = rotary_dim <= 0 ? head_dim : rotary_dim;
    MCL_CHECK(rd <= head_dim && rd % 2 == 0, "rope rotary_dim must be even and <= head_dim");
    auto out = Tensor::empty(x.backend(), x.shape(), DType::F32);
    auto kernel = x.backend().kernels.get("rope_f32");
    const int total = static_cast<int>(x.numel());
    kernel.set_arg(0, x.buffer());
    kernel.set_arg(1, out.buffer());
    kernel.set_arg(2, static_cast<int>(batch_size));
    kernel.set_arg(3, static_cast<int>(seq_len));
    kernel.set_arg(4, static_cast<int>(x.shape()[1]));
    kernel.set_arg(5, n_head);
    kernel.set_arg(6, head_dim);
    kernel.set_arg(7, static_cast<int>(rd));
    kernel.set_arg(8, static_cast<int>(token_offset));
    kernel.set_arg(9, theta);
    kernel.set_arg(10, inverse ? 1 : 0);
    kernel.launch1d(round_up(static_cast<std::size_t>(total), 256), 256);
    autograd::record_op(inverse ? "rope_backward_f32" : "rope_f32", {x.id()}, {out.id()});
    return out;
}
}

Tensor rope(const Tensor& x, int n_head, int64_t batch_size, int64_t seq_len,
            float theta, int64_t rotary_dim, int64_t token_offset) {
    auto out = rope_impl(x, n_head, batch_size, seq_len, theta, rotary_dim, token_offset, false);
    if (autograd::is_enabled() && x.requires_grad()) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<RopeBackwardNode>(x, n_head, batch_size, seq_len, theta, rotary_dim, token_offset));
    }
    return out;
}

void RopeBackwardNode::backward(const Tensor& grad_output) {
    if (x.requires_grad()) x.backward(rope_impl(grad_output, n_head, batch_size, seq_len, theta, rotary_dim, token_offset, true));
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

Tensor grouped_query_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                               int n_head, int n_kv_head, bool causal,
                               int64_t batch_size, int64_t query_len,
                               int64_t key_len, int64_t query_offset) {
    auto s = validate_grouped_attention(q, k, v, n_head, n_kv_head, batch_size, query_len, key_len, "grouped_query_attention");
    if (n_head == n_kv_head && s.query_tokens == s.key_tokens && query_offset == 0) {
        return multihead_attention(q, k, v, n_head, causal, batch_size, s.query_tokens);
    }
    auto out = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto kernel = q.backend().kernels.get("grouped_query_attention_f32");
    const float scale_value = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k.buffer());
    kernel.set_arg(2, v.buffer());
    kernel.set_arg(3, out.buffer());
    kernel.set_arg(4, static_cast<int>(s.batch));
    kernel.set_arg(5, static_cast<int>(s.query_tokens));
    kernel.set_arg(6, static_cast<int>(s.key_tokens));
    kernel.set_arg(7, n_head);
    kernel.set_arg(8, n_kv_head);
    kernel.set_arg(9, static_cast<int>(s.head_dim));
    kernel.set_arg(10, causal ? 1 : 0);
    kernel.set_arg(11, static_cast<int>(query_offset));
    kernel.set_arg(12, scale_value);
    kernel.launch1d(round_up(static_cast<std::size_t>(out.numel()), 256), 256);
    autograd::record_op("grouped_query_attention_f32", {q.id(), k.id(), v.id()}, {out.id()});
    if (autograd::is_enabled() && (q.requires_grad() || k.requires_grad() || v.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<GroupedQueryAttentionBackwardNode>(q, k, v, n_head, n_kv_head, causal,
                                                                             s.batch, s.query_tokens, s.key_tokens, query_offset));
    }
    return out;
}

Tensor grouped_query_attention_masked(const Tensor& q, const Tensor& k, const Tensor& v,
                                      const Tensor& mask, int n_head, int n_kv_head,
                                      bool causal, int64_t batch_size,
                                      int64_t query_len, int64_t key_len,
                                      int64_t query_offset, bool additive_mask) {
    auto s = validate_grouped_attention(q, k, v, n_head, n_kv_head, batch_size, query_len, key_len, "grouped_query_attention_masked");
    MCL_CHECK(mask.backend_ptr() == q.backend_ptr(), "grouped_query_attention_masked mask must share backend with q/k/v");
    auto mask_info = validate_attention_mask(mask, s.batch, s.query_tokens, s.key_tokens, additive_mask, "grouped_query_attention_masked");
    auto out = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto kernel = q.backend().kernels.get("grouped_query_attention_mask_" + mask_info.suffix);
    const float scale_value = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k.buffer());
    kernel.set_arg(2, v.buffer());
    kernel.set_arg(3, mask.buffer());
    kernel.set_arg(4, out.buffer());
    kernel.set_arg(5, static_cast<int>(s.batch));
    kernel.set_arg(6, static_cast<int>(s.query_tokens));
    kernel.set_arg(7, static_cast<int>(s.key_tokens));
    kernel.set_arg(8, n_head);
    kernel.set_arg(9, n_kv_head);
    kernel.set_arg(10, static_cast<int>(s.head_dim));
    kernel.set_arg(11, causal ? 1 : 0);
    kernel.set_arg(12, static_cast<int>(query_offset));
    kernel.set_arg(13, mask_info.layout);
    kernel.set_arg(14, mask_info.mode);
    kernel.set_arg(15, scale_value);
    kernel.launch1d(round_up(static_cast<std::size_t>(out.numel()), 256), 256);
    autograd::record_op("grouped_query_attention_mask_" + mask_info.suffix, {q.id(), k.id(), v.id(), mask.id()}, {out.id()});
    if (autograd::is_enabled() && (q.requires_grad() || k.requires_grad() || v.requires_grad())) {
        out.set_requires_grad(true);
        out._set_grad_fn(std::make_shared<GroupedQueryAttentionMaskedBackwardNode>(q, k, v, mask, n_head, n_kv_head,
                                                                                   causal, s.batch, s.query_tokens,
                                                                                   s.key_tokens, query_offset,
                                                                                   additive_mask));
    }
    return out;
}

void kv_cache_append(const Tensor& new_k, const Tensor& new_v, Tensor& cache_k, Tensor& cache_v,
                     int64_t batch_size, int64_t new_tokens, int64_t max_tokens, int64_t start_pos) {
    MCL_CHECK(new_k.dtype() == DType::F32 && new_v.dtype() == DType::F32 &&
              cache_k.dtype() == DType::F32 && cache_v.dtype() == DType::F32, "kv_cache_append supports f32 only");
    MCL_CHECK(new_k.shape() == new_v.shape() && cache_k.shape() == cache_v.shape(), "kv_cache_append k/v shape mismatch");
    MCL_CHECK(new_k.ndim() == 2 && cache_k.ndim() == 2, "kv_cache_append expects flattened rank-2 tensors");
    MCL_CHECK(new_k.backend_ptr() == new_v.backend_ptr() && new_k.backend_ptr() == cache_k.backend_ptr() &&
              new_k.backend_ptr() == cache_v.backend_ptr(), "kv_cache_append requires one backend");
    const int64_t kv_channels = new_k.shape()[1];
    MCL_CHECK(cache_k.shape()[1] == kv_channels, "kv_cache_append channel mismatch");
    MCL_CHECK(batch_size * new_tokens == new_k.shape()[0], "kv_cache_append new shape mismatch");
    MCL_CHECK(batch_size * max_tokens == cache_k.shape()[0], "kv_cache_append cache shape mismatch");
    MCL_CHECK(start_pos >= 0 && start_pos + new_tokens <= max_tokens, "kv_cache_append out of cache range");
    auto kernel = new_k.backend().kernels.get("kv_cache_append_f32");
    const int n = static_cast<int>(new_k.numel());
    kernel.set_arg(0, new_k.buffer());
    kernel.set_arg(1, new_v.buffer());
    kernel.set_arg(2, cache_k.buffer());
    kernel.set_arg(3, cache_v.buffer());
    kernel.set_arg(4, static_cast<int>(batch_size));
    kernel.set_arg(5, static_cast<int>(new_tokens));
    kernel.set_arg(6, static_cast<int>(max_tokens));
    kernel.set_arg(7, static_cast<int>(kv_channels));
    kernel.set_arg(8, static_cast<int>(start_pos));
    kernel.set_arg(9, n);
    kernel.launch1d(round_up(static_cast<std::size_t>(n), 256), 256);
    autograd::record_op("kv_cache_append_f32", {new_k.id(), new_v.id()}, {cache_k.id(), cache_v.id()});
}

namespace {

MultiHeadAttentionGradients grouped_query_attention_backward_staged(const Tensor& q, const Tensor& k, const Tensor& v,
                                                                    const Tensor& grad_out,
                                                                    int n_head, int n_kv_head, bool causal,
                                                                    int64_t query_offset,
                                                                    const GroupedAttentionShape& s) {
    auto dq = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto dk = Tensor::empty(k.backend(), k.shape(), DType::F32);
    auto dv = Tensor::empty(v.backend(), v.shape(), DType::F32);
    const auto rows = static_cast<std::size_t>(s.batch * n_head * s.query_tokens);
    const auto pairs = rows * static_cast<std::size_t>(s.key_tokens);
    auto probs = Tensor::empty(q.backend(), {static_cast<int64_t>(rows), s.key_tokens}, DType::F32);
    auto dp = Tensor::empty(q.backend(), probs.shape(), DType::F32);
    auto ds = Tensor::empty(q.backend(), probs.shape(), DType::F32);
    const int batch_i = static_cast<int>(s.batch);
    const int query_i = static_cast<int>(s.query_tokens);
    const int key_i = static_cast<int>(s.key_tokens);
    const int head_dim_i = static_cast<int>(s.head_dim);
    const int causal_i = causal ? 1 : 0;
    const int offset_i = static_cast<int>(query_offset);
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));

    {
        auto kernel = q.backend().kernels.get("gqa_backward_probs_f32");
        kernel.set_arg(0, q.buffer());
        kernel.set_arg(1, k.buffer());
        kernel.set_arg(2, probs.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, n_kv_head);
        kernel.set_arg(8, head_dim_i);
        kernel.set_arg(9, causal_i);
        kernel.set_arg(10, offset_i);
        kernel.set_arg(11, scale);
        kernel.launch2d(kFlashAttentionWorkgroup, rows, kFlashAttentionWorkgroup, 1);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_dp_f32");
        kernel.set_arg(0, grad_out.buffer());
        kernel.set_arg(1, v.buffer());
        kernel.set_arg(2, dp.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, n_kv_head);
        kernel.set_arg(8, head_dim_i);
        kernel.set_arg(9, causal_i);
        kernel.set_arg(10, offset_i);
        kernel.launch1d(round_up(pairs, 256), 256);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_ds_f32");
        kernel.set_arg(0, probs.buffer());
        kernel.set_arg(1, dp.buffer());
        kernel.set_arg(2, ds.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, causal_i);
        kernel.set_arg(8, offset_i);
        kernel.set_arg(9, scale);
        kernel.launch2d(kFlashAttentionWorkgroup, rows, kFlashAttentionWorkgroup, 1);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_apply_q_f32");
        kernel.set_arg(0, ds.buffer());
        kernel.set_arg(1, k.buffer());
        kernel.set_arg(2, dq.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, n_kv_head);
        kernel.set_arg(8, head_dim_i);
        kernel.set_arg(9, causal_i);
        kernel.set_arg(10, offset_i);
        kernel.launch1d(round_up(static_cast<std::size_t>(q.numel()), 256), 256);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_apply_k_f32");
        kernel.set_arg(0, ds.buffer());
        kernel.set_arg(1, q.buffer());
        kernel.set_arg(2, dk.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, n_kv_head);
        kernel.set_arg(8, head_dim_i);
        kernel.set_arg(9, causal_i);
        kernel.set_arg(10, offset_i);
        kernel.launch1d(round_up(static_cast<std::size_t>(k.numel()), 256), 256);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_apply_v_f32");
        kernel.set_arg(0, probs.buffer());
        kernel.set_arg(1, grad_out.buffer());
        kernel.set_arg(2, dv.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, n_kv_head);
        kernel.set_arg(8, head_dim_i);
        kernel.set_arg(9, causal_i);
        kernel.set_arg(10, offset_i);
        kernel.launch1d(round_up(static_cast<std::size_t>(v.numel()), 256), 256);
    }
    autograd::record_replay_op("gqa_backward_staged_f32",
                               {q.id(), k.id(), v.id(), grad_out.id()},
                               {dq.id(), dk.id(), dv.id()},
                               {probs.id(), dp.id(), ds.id()},
                               []() {});
    return {dq, dk, dv};
}

MultiHeadAttentionGradients grouped_query_attention_backward_masked_staged(const Tensor& q, const Tensor& k, const Tensor& v,
                                                                           const Tensor& mask, const Tensor& grad_out,
                                                                           int n_head, int n_kv_head, bool causal,
                                                                           int64_t query_offset,
                                                                           bool additive_mask,
                                                                           const GroupedAttentionShape& s,
                                                                           const AttentionMaskShape& mask_info) {
    auto dq = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto dk = Tensor::empty(k.backend(), k.shape(), DType::F32);
    auto dv = Tensor::empty(v.backend(), v.shape(), DType::F32);
    const auto rows = static_cast<std::size_t>(s.batch * n_head * s.query_tokens);
    const auto pairs = rows * static_cast<std::size_t>(s.key_tokens);
    auto probs = Tensor::empty(q.backend(), {static_cast<int64_t>(rows), s.key_tokens}, DType::F32);
    auto dp = Tensor::empty(q.backend(), probs.shape(), DType::F32);
    auto ds = Tensor::empty(q.backend(), probs.shape(), DType::F32);
    const int batch_i = static_cast<int>(s.batch);
    const int query_i = static_cast<int>(s.query_tokens);
    const int key_i = static_cast<int>(s.key_tokens);
    const int head_dim_i = static_cast<int>(s.head_dim);
    const int causal_i = causal ? 1 : 0;
    const int offset_i = static_cast<int>(query_offset);
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));

    {
        auto kernel = q.backend().kernels.get("gqa_backward_probs_mask_" + mask_info.suffix);
        kernel.set_arg(0, q.buffer());
        kernel.set_arg(1, k.buffer());
        kernel.set_arg(2, mask.buffer());
        kernel.set_arg(3, probs.buffer());
        kernel.set_arg(4, batch_i);
        kernel.set_arg(5, query_i);
        kernel.set_arg(6, key_i);
        kernel.set_arg(7, n_head);
        kernel.set_arg(8, n_kv_head);
        kernel.set_arg(9, head_dim_i);
        kernel.set_arg(10, causal_i);
        kernel.set_arg(11, offset_i);
        kernel.set_arg(12, mask_info.layout);
        kernel.set_arg(13, additive_mask ? 1 : 0);
        kernel.set_arg(14, scale);
        kernel.launch2d(kFlashAttentionWorkgroup, rows, kFlashAttentionWorkgroup, 1);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_dp_f32");
        kernel.set_arg(0, grad_out.buffer());
        kernel.set_arg(1, v.buffer());
        kernel.set_arg(2, dp.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, n_kv_head);
        kernel.set_arg(8, head_dim_i);
        kernel.set_arg(9, causal_i);
        kernel.set_arg(10, offset_i);
        kernel.launch1d(round_up(pairs, 256), 256);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_ds_f32");
        kernel.set_arg(0, probs.buffer());
        kernel.set_arg(1, dp.buffer());
        kernel.set_arg(2, ds.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, causal_i);
        kernel.set_arg(8, offset_i);
        kernel.set_arg(9, scale);
        kernel.launch2d(kFlashAttentionWorkgroup, rows, kFlashAttentionWorkgroup, 1);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_apply_q_f32");
        kernel.set_arg(0, ds.buffer());
        kernel.set_arg(1, k.buffer());
        kernel.set_arg(2, dq.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, n_kv_head);
        kernel.set_arg(8, head_dim_i);
        kernel.set_arg(9, causal_i);
        kernel.set_arg(10, offset_i);
        kernel.launch1d(round_up(static_cast<std::size_t>(q.numel()), 256), 256);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_apply_k_f32");
        kernel.set_arg(0, ds.buffer());
        kernel.set_arg(1, q.buffer());
        kernel.set_arg(2, dk.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, n_kv_head);
        kernel.set_arg(8, head_dim_i);
        kernel.set_arg(9, causal_i);
        kernel.set_arg(10, offset_i);
        kernel.launch1d(round_up(static_cast<std::size_t>(k.numel()), 256), 256);
    }
    {
        auto kernel = q.backend().kernels.get("gqa_backward_apply_v_f32");
        kernel.set_arg(0, probs.buffer());
        kernel.set_arg(1, grad_out.buffer());
        kernel.set_arg(2, dv.buffer());
        kernel.set_arg(3, batch_i);
        kernel.set_arg(4, query_i);
        kernel.set_arg(5, key_i);
        kernel.set_arg(6, n_head);
        kernel.set_arg(7, n_kv_head);
        kernel.set_arg(8, head_dim_i);
        kernel.set_arg(9, causal_i);
        kernel.set_arg(10, offset_i);
        kernel.launch1d(round_up(static_cast<std::size_t>(v.numel()), 256), 256);
    }
    autograd::record_replay_op("gqa_backward_staged_mask_" + mask_info.suffix,
                               {q.id(), k.id(), v.id(), mask.id(), grad_out.id()},
                               {dq.id(), dk.id(), dv.id()},
                               {probs.id(), dp.id(), ds.id()},
                               []() {});
    return {dq, dk, dv};
}

MultiHeadAttentionGradients grouped_query_attention_backward_fused(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& grad_out,
                                                                   int n_head, int n_kv_head, bool causal,
                                                                   int64_t batch_size, int64_t query_len,
                                                                   int64_t key_len, int64_t query_offset) {
    auto s = validate_grouped_attention(q, k, v, n_head, n_kv_head, batch_size, query_len, key_len, "grouped_query_attention_backward");
    MCL_CHECK(grad_out.dtype() == DType::F32 && grad_out.shape() == q.shape(), "grouped_query_attention_backward grad_out mismatch");
    if (supports_staged_grouped_attention_backward_kernel(q.backend(), s)) {
        return grouped_query_attention_backward_staged(q, k, v, grad_out, n_head, n_kv_head,
                                                       causal, query_offset, s);
    }
    auto dq = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto dk = Tensor::empty(k.backend(), k.shape(), DType::F32);
    auto dv = Tensor::empty(v.backend(), v.shape(), DType::F32);
    auto kernel = q.backend().kernels.get("grouped_query_attention_backward_f32");
    const float scale_value = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
    const std::size_t total = static_cast<std::size_t>(q.numel() + k.numel() + v.numel());
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k.buffer());
    kernel.set_arg(2, v.buffer());
    kernel.set_arg(3, grad_out.buffer());
    kernel.set_arg(4, dq.buffer());
    kernel.set_arg(5, dk.buffer());
    kernel.set_arg(6, dv.buffer());
    kernel.set_arg(7, static_cast<int>(s.batch));
    kernel.set_arg(8, static_cast<int>(s.query_tokens));
    kernel.set_arg(9, static_cast<int>(s.key_tokens));
    kernel.set_arg(10, n_head);
    kernel.set_arg(11, n_kv_head);
    kernel.set_arg(12, static_cast<int>(s.head_dim));
    kernel.set_arg(13, causal ? 1 : 0);
    kernel.set_arg(14, static_cast<int>(query_offset));
    kernel.set_arg(15, scale_value);
    kernel.launch1d(round_up(total, 256), 256);
    autograd::record_op("grouped_query_attention_backward_f32", {q.id(), k.id(), v.id(), grad_out.id()}, {dq.id(), dk.id(), dv.id()});
    return {dq, dk, dv};
}

MultiHeadAttentionGradients grouped_query_attention_backward_masked_fused(const Tensor& q, const Tensor& k, const Tensor& v,
                                                                          const Tensor& mask, const Tensor& grad_out,
                                                                          int n_head, int n_kv_head, bool causal,
                                                                          int64_t batch_size, int64_t query_len,
                                                                          int64_t key_len, int64_t query_offset,
                                                                          bool additive_mask) {
    auto s = validate_grouped_attention(q, k, v, n_head, n_kv_head, batch_size, query_len, key_len, "grouped_query_attention_backward_masked");
    MCL_CHECK(grad_out.dtype() == DType::F32 && grad_out.shape() == q.shape(), "grouped_query_attention_backward_masked grad_out mismatch");
    MCL_CHECK(mask.backend_ptr() == q.backend_ptr(), "grouped_query_attention_backward_masked mask must share backend");
    auto mask_info = validate_attention_mask(mask, s.batch, s.query_tokens, s.key_tokens, additive_mask, "grouped_query_attention_backward_masked");
    if (supports_staged_grouped_attention_backward_kernel(q.backend(), s)) {
        return grouped_query_attention_backward_masked_staged(q, k, v, mask, grad_out, n_head, n_kv_head,
                                                              causal, query_offset, additive_mask, s, mask_info);
    }
    auto dq = Tensor::empty(q.backend(), q.shape(), DType::F32);
    auto dk = Tensor::empty(k.backend(), k.shape(), DType::F32);
    auto dv = Tensor::empty(v.backend(), v.shape(), DType::F32);
    auto kernel = q.backend().kernels.get("grouped_query_attention_backward_mask_" + mask_info.suffix);
    const float scale_value = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
    const std::size_t total = static_cast<std::size_t>(q.numel() + k.numel() + v.numel());
    kernel.set_arg(0, q.buffer());
    kernel.set_arg(1, k.buffer());
    kernel.set_arg(2, v.buffer());
    kernel.set_arg(3, grad_out.buffer());
    kernel.set_arg(4, mask.buffer());
    kernel.set_arg(5, dq.buffer());
    kernel.set_arg(6, dk.buffer());
    kernel.set_arg(7, dv.buffer());
    kernel.set_arg(8, static_cast<int>(s.batch));
    kernel.set_arg(9, static_cast<int>(s.query_tokens));
    kernel.set_arg(10, static_cast<int>(s.key_tokens));
    kernel.set_arg(11, n_head);
    kernel.set_arg(12, n_kv_head);
    kernel.set_arg(13, static_cast<int>(s.head_dim));
    kernel.set_arg(14, causal ? 1 : 0);
    kernel.set_arg(15, static_cast<int>(query_offset));
    kernel.set_arg(16, mask_info.layout);
    kernel.set_arg(17, mask_info.mode);
    kernel.set_arg(18, scale_value);
    kernel.launch1d(round_up(total, 256), 256);
    autograd::record_op("grouped_query_attention_backward_mask_" + mask_info.suffix,
                        {q.id(), k.id(), v.id(), mask.id(), grad_out.id()},
                        {dq.id(), dk.id(), dv.id()});
    return {dq, dk, dv};
}

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
