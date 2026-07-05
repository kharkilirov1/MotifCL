#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace motifcl {

// Minimal Vulkan loader probe for the native-GPU backend line.
//
// This intentionally does not expose Vulkan SDK types. The implementation uses
// the platform Vulkan loader dynamically, so the normal OpenCL build and CI do
// not gain a hard Vulkan SDK/link dependency while the runtime can still detect
// whether a Vulkan GPU path is viable on a user's machine.
struct VulkanPhysicalDeviceInfo {
    std::string name;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t device_type = 0;
    std::uint32_t api_version = 0;
    std::uint32_t driver_version = 0;
};

struct VulkanProbeResult {
    bool loader_found = false;
    bool instance_created = false;
    std::uint32_t api_version = 0;
    std::uint32_t physical_device_count = 0;
    std::string loader_path;
    std::string error;
    std::vector<VulkanPhysicalDeviceInfo> devices;

    bool available() const {
        return loader_found && instance_created && physical_device_count > 0 && error.empty();
    }
};

struct VulkanSmokeComputeResult {
    bool success = false;
    float output = 0.0f;
    std::string device_name;
    std::string error;
};

struct VulkanF32MatmulSmokeResult {
    bool success = false;
    std::vector<float> output;
    std::string device_name;
    std::string error;
};

struct VulkanF32TensorResult {
    bool success = false;
    std::vector<float> output;
    std::string device_name;
    std::string error;
};

struct VulkanU32TensorResult {
    bool success = false;
    std::vector<std::uint32_t> output;
    std::string device_name;
    std::string error;
};

struct VulkanStorageBufferSpec {
    const void* initial_data = nullptr;
    std::size_t nbytes = 0;
};

struct VulkanStorageBufferDispatchResult {
    bool success = false;
    std::string device_name;
    std::string error;
    std::vector<std::vector<std::uint8_t>> outputs;
};

struct VulkanOpResult {
    bool success = false;
    std::string device_name;
    std::string error;
};

class VulkanBuffer {
public:
    VulkanBuffer();
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    VulkanBuffer(VulkanBuffer&&) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&&) noexcept;

    bool valid() const;
    std::size_t nbytes() const;
    void upload(const void* data, std::size_t bytes, std::size_t offset = 0);
    void download(void* data, std::size_t bytes, std::size_t offset = 0) const;

private:
    struct Impl;
    explicit VulkanBuffer(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;

    friend class VulkanRuntime;
};

// Compute-relevant device capabilities read from VkPhysicalDeviceProperties.
struct VulkanDeviceCaps {
    bool timestamps = false;
    double timestamp_period_ns = 0.0;
    std::uint32_t max_shared_memory_bytes = 16384;
    std::uint32_t max_workgroup_invocations = 128;
    std::uint32_t max_push_constant_bytes = 128;
};

// Recording of cached-path dispatches for replay. Holds shared ownership of
// every referenced VulkanBuffer allocation, so replay stays valid after the
// recording tensors go out of scope. Fixed bindings (no rebind support);
// replay re-records all dispatches into one command buffer and submits once.
class VulkanDispatchRecording {
public:
    VulkanDispatchRecording();
    ~VulkanDispatchRecording();
    VulkanDispatchRecording(VulkanDispatchRecording&&) noexcept;
    VulkanDispatchRecording& operator=(VulkanDispatchRecording&&) noexcept;
    VulkanDispatchRecording(const VulkanDispatchRecording&) = delete;
    VulkanDispatchRecording& operator=(const VulkanDispatchRecording&) = delete;

    bool empty() const;
    std::size_t size() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class VulkanRuntime;
};

class VulkanRuntime {
public:
    VulkanRuntime();
    ~VulkanRuntime();

    VulkanRuntime(const VulkanRuntime&) = delete;
    VulkanRuntime& operator=(const VulkanRuntime&) = delete;
    VulkanRuntime(VulkanRuntime&&) noexcept;
    VulkanRuntime& operator=(VulkanRuntime&&) noexcept;

    static VulkanRuntime create();

    bool available() const;
    const std::string& device_name() const;
    const std::string& error() const;
    bool supports_storage_buffer_i8() const;
    const VulkanDeviceCaps& caps() const;

    // Batched recording: while a batch is open, cached-path ops record their
    // dispatches (with compute->compute memory barriers between them) into one
    // primary command buffer; batch_end() submits once and waits on a fence.
    bool batch_begin();
    VulkanOpResult batch_end();
    bool batch_active() const;

    // GPU timing via VK_QUERY_TYPE_TIMESTAMP around the most recent
    // cached-path submission (single dispatch or whole batch).
    void set_gpu_timing_enabled(bool enabled);
    // Microseconds; negative when no timed submission is available.
    double last_gpu_time_us() const;

    // Dispatch capture/replay: while capturing, every cached-path dispatch is
    // recorded (and still executed). Replay re-records the whole sequence
    // into one primary command buffer and submits once. No OpenCL objects
    // are involved; buffers are device-resident Vulkan allocations.
    bool capture_begin();
    VulkanDispatchRecording capture_end();
    bool capture_active() const;
    VulkanOpResult replay(const VulkanDispatchRecording& recording);

    VulkanBuffer create_buffer(std::size_t nbytes, const void* initial_data = nullptr);

    // Cached dispatch path: pipelines / layouts / descriptor sets are cached
    // for the runtime lifetime; scalar arguments travel as push constants.
    // While a batch is open the dispatch is recorded instead of submitted.
    // CONTRACT: `spirv` must point to storage that outlives the runtime and
    // never changes (the pipeline cache is keyed by pointer identity, not by
    // content hash) — embedded static kernel arrays satisfy this.
    VulkanOpResult dispatch_cached(const std::uint32_t* spirv,
                                   std::size_t spirv_word_count,
                                   const std::vector<const VulkanBuffer*>& buffers,
                                   const void* push_constants,
                                   std::uint32_t push_constant_bytes,
                                   std::uint32_t group_count_x,
                                   std::uint32_t group_count_y = 1,
                                   std::uint32_t group_count_z = 1);

    VulkanStorageBufferDispatchResult dispatch_storage_buffers(
        const std::uint32_t* spirv,
        std::size_t spirv_word_count,
        const std::vector<VulkanStorageBufferSpec>& buffer_specs,
        const std::vector<std::size_t>& output_buffer_indices,
        std::uint32_t group_count_x = 1,
        std::uint32_t group_count_y = 1,
        std::uint32_t group_count_z = 1);
    VulkanStorageBufferDispatchResult dispatch_storage_buffers(
        const std::uint32_t* spirv,
        std::size_t spirv_word_count,
        const std::vector<const VulkanBuffer*>& buffers,
        const std::vector<std::size_t>& output_buffer_indices,
        std::uint32_t group_count_x = 1,
        std::uint32_t group_count_y = 1,
        std::uint32_t group_count_z = 1);

private:
    struct Impl;
    explicit VulkanRuntime(std::shared_ptr<Impl> impl);

    std::shared_ptr<Impl> impl_;

    friend class VulkanBuffer;
    friend class VulkanDispatchRecording;
};

VulkanProbeResult probe_vulkan_runtime();
VulkanSmokeComputeResult run_vulkan_smoke_compute();
VulkanSmokeComputeResult run_vulkan_smoke_compute(VulkanRuntime& runtime);
VulkanF32MatmulSmokeResult run_vulkan_f32_matmul(const std::vector<float>& a,
                                                 const std::vector<float>& b,
                                                 std::size_t m,
                                                 std::size_t k,
                                                 std::size_t n);
VulkanF32TensorResult run_vulkan_softmax_rows(const std::vector<float>& x,
                                              std::size_t rows,
                                              std::size_t cols);
VulkanF32TensorResult run_vulkan_rmsnorm(const std::vector<float>& x,
                                         const std::vector<float>& weight,
                                         std::size_t rows,
                                         std::size_t cols,
                                         float eps);
VulkanF32TensorResult run_vulkan_swiglu(const std::vector<float>& packed,
                                        std::size_t rows,
                                        std::size_t hidden);
VulkanF32TensorResult run_vulkan_add(const std::vector<float>& a,
                                     const std::vector<float>& b);
VulkanF32TensorResult run_vulkan_i8_scaled_matmul(const std::vector<std::int8_t>& a,
                                                  const std::vector<std::int8_t>& b,
                                                  std::size_t m,
                                                  std::size_t k,
                                                  std::size_t n,
                                                  float scale_a,
                                                  float scale_b);
VulkanF32TensorResult run_vulkan_grouped_query_attention(const std::vector<float>& q,
                                                         const std::vector<float>& k,
                                                         const std::vector<float>& v,
                                                         std::size_t query_tokens,
                                                         std::size_t key_tokens,
                                                         std::size_t n_head,
                                                         std::size_t n_kv_head,
                                                         std::size_t head_dim,
                                                         float scale);
VulkanOpResult run_vulkan_grouped_query_attention(VulkanRuntime& runtime,
                                                  const VulkanBuffer& q,
                                                  const VulkanBuffer& k,
                                                  const VulkanBuffer& v,
                                                  VulkanBuffer& out,
                                                  std::size_t query_tokens,
                                                  std::size_t key_tokens,
                                                  std::size_t n_head,
                                                  std::size_t n_kv_head,
                                                  std::size_t head_dim,
                                                  float scale);
// Non-causal batch=1 GQA backward (three cached dispatches). probs_scratch
// and ds_scratch must hold n_head*query_tokens*key_tokens floats each.
VulkanOpResult run_vulkan_grouped_query_attention_backward(VulkanRuntime& runtime,
                                                           const VulkanBuffer& q,
                                                           const VulkanBuffer& k,
                                                           const VulkanBuffer& v,
                                                           const VulkanBuffer& grad_out,
                                                           VulkanBuffer& probs_scratch,
                                                           VulkanBuffer& ds_scratch,
                                                           VulkanBuffer& grad_q,
                                                           VulkanBuffer& grad_k,
                                                           VulkanBuffer& grad_v,
                                                           std::size_t query_tokens,
                                                           std::size_t key_tokens,
                                                           std::size_t n_head,
                                                           std::size_t n_kv_head,
                                                           std::size_t head_dim,
                                                           float scale);
VulkanF32TensorResult run_vulkan_compact_counter_backward_input(
    const std::vector<std::uint32_t>& packed_state_words,
    const std::vector<float>& scale,
    const std::vector<float>& grad_out,
    std::size_t batch,
    std::size_t in_features,
    std::size_t out_features,
    std::size_t C);
VulkanU32TensorResult run_vulkan_compact_counter_increment(
    const std::vector<std::uint32_t>& packed_state_words,
    const std::vector<std::uint32_t>& packed_increment_words);
VulkanF32TensorResult run_vulkan_sgd_update(const std::vector<float>& param,
                                            const std::vector<float>& grad,
                                            float lr);
VulkanOpResult run_vulkan_f32_matmul(VulkanRuntime& runtime,
                                     const VulkanBuffer& a,
                                     const VulkanBuffer& b,
                                     VulkanBuffer& c,
                                     std::size_t m,
                                     std::size_t k,
                                     std::size_t n);
VulkanOpResult run_vulkan_f32_matmul_transpose_b(VulkanRuntime& runtime,
                                                 const VulkanBuffer& a,
                                                 const VulkanBuffer& b,
                                                 VulkanBuffer& c,
                                                 std::size_t m,
                                                 std::size_t k,
                                                 std::size_t n);
// C[M,N] = A[K,M]^T * B[K,N] (the dB = A^T * dC backward form).
VulkanOpResult run_vulkan_f32_matmul_transpose_a(VulkanRuntime& runtime,
                                                 const VulkanBuffer& a,
                                                 const VulkanBuffer& b,
                                                 VulkanBuffer& c,
                                                 std::size_t m,
                                                 std::size_t k,
                                                 std::size_t n);
VulkanOpResult run_vulkan_i8_scaled_matmul(VulkanRuntime& runtime,
                                           const VulkanBuffer& a,
                                           const VulkanBuffer& b,
                                           VulkanBuffer& c,
                                           std::size_t m,
                                           std::size_t k,
                                           std::size_t n,
                                           float scale_a,
                                           float scale_b);
VulkanOpResult run_vulkan_softmax_rows(VulkanRuntime& runtime,
                                       const VulkanBuffer& x,
                                       VulkanBuffer& out,
                                       std::size_t rows,
                                       std::size_t cols);
// dx = y * (dy - sum(dy*y)) per row, where y is the softmax output.
VulkanOpResult run_vulkan_softmax_rows_backward(VulkanRuntime& runtime,
                                                const VulkanBuffer& y,
                                                const VulkanBuffer& dy,
                                                VulkanBuffer& dx,
                                                std::size_t rows,
                                                std::size_t cols);
VulkanOpResult run_vulkan_rmsnorm_backward_x(VulkanRuntime& runtime,
                                             const VulkanBuffer& x,
                                             const VulkanBuffer& weight,
                                             const VulkanBuffer& grad_out,
                                             VulkanBuffer& grad_x,
                                             std::size_t rows,
                                             std::size_t cols,
                                             float eps);
// Two-stage: per-row inverse RMS into row_inv_scratch (>= rows floats), then
// a per-column reduction into grad_weight.
VulkanOpResult run_vulkan_rmsnorm_backward_weight(VulkanRuntime& runtime,
                                                  const VulkanBuffer& x,
                                                  const VulkanBuffer& grad_out,
                                                  VulkanBuffer& row_inv_scratch,
                                                  VulkanBuffer& grad_weight,
                                                  std::size_t rows,
                                                  std::size_t cols,
                                                  float eps);
VulkanOpResult run_vulkan_swiglu_backward(VulkanRuntime& runtime,
                                          const VulkanBuffer& packed,
                                          const VulkanBuffer& grad_out,
                                          VulkanBuffer& grad_packed,
                                          std::size_t rows,
                                          std::size_t hidden);
// Mean softmax cross-entropy over rows: partial (>= rows floats) receives the
// per-row losses, out[0] the mean. targets are i32 class ids.
VulkanOpResult run_vulkan_softmax_cross_entropy(VulkanRuntime& runtime,
                                                const VulkanBuffer& logits,
                                                const VulkanBuffer& targets,
                                                VulkanBuffer& partial,
                                                VulkanBuffer& out,
                                                std::size_t rows,
                                                std::size_t cols);
VulkanOpResult run_vulkan_softmax_cross_entropy_backward(VulkanRuntime& runtime,
                                                         const VulkanBuffer& logits,
                                                         const VulkanBuffer& targets,
                                                         const VulkanBuffer& grad_out,
                                                         VulkanBuffer& grad_logits,
                                                         std::size_t rows,
                                                         std::size_t cols);
VulkanOpResult run_vulkan_gelu(VulkanRuntime& runtime,
                               const VulkanBuffer& x,
                               VulkanBuffer& out,
                               std::size_t elements);
VulkanOpResult run_vulkan_gelu_backward(VulkanRuntime& runtime,
                                        const VulkanBuffer& x,
                                        const VulkanBuffer& grad_out,
                                        VulkanBuffer& grad_x,
                                        std::size_t elements);
VulkanOpResult run_vulkan_rmsnorm(VulkanRuntime& runtime,
                                  const VulkanBuffer& x,
                                  const VulkanBuffer& weight,
                                  VulkanBuffer& out,
                                  std::size_t rows,
                                  std::size_t cols,
                                  float eps);
VulkanOpResult run_vulkan_swiglu(VulkanRuntime& runtime,
                                 const VulkanBuffer& packed,
                                 VulkanBuffer& out,
                                 std::size_t rows,
                                 std::size_t hidden);
VulkanOpResult run_vulkan_add(VulkanRuntime& runtime,
                              const VulkanBuffer& a,
                              const VulkanBuffer& b,
                              VulkanBuffer& out,
                              std::size_t elements);
// Elementwise scalar multiply/add: out = x * alpha / out = x + value.
// Needed by scale/mul_scalar/add_scalar ops and the SubBackward/MulBackward/
// DivBackward/ScalarBackward chains on Vulkan (closes the C2 review gap).
VulkanOpResult run_vulkan_mul_scalar(VulkanRuntime& runtime,
                                     const VulkanBuffer& x,
                                     VulkanBuffer& out,
                                     std::size_t elements,
                                     float alpha);
VulkanOpResult run_vulkan_add_scalar(VulkanRuntime& runtime,
                                     const VulkanBuffer& x,
                                     VulkanBuffer& out,
                                     std::size_t elements,
                                     float value);
VulkanOpResult run_vulkan_sub(VulkanRuntime& runtime,
                              const VulkanBuffer& a,
                              const VulkanBuffer& b,
                              VulkanBuffer& out,
                              std::size_t elements);
VulkanOpResult run_vulkan_sgd_update(VulkanRuntime& runtime,
                                      const VulkanBuffer& param,
                                      const VulkanBuffer& grad,
                                      VulkanBuffer& out,
                                      std::size_t elements,
                                      float lr);
VulkanOpResult run_vulkan_compact_counter_decode_weight(VulkanRuntime& runtime,
                                                        const VulkanBuffer& state,
                                                        const VulkanBuffer& scale,
                                                        VulkanBuffer& weight,
                                                        std::size_t in_features,
                                                        std::size_t out_features,
                                                        std::size_t C);
// Fused memory-native counter update (row stats + stochastic tick + scale
// commit), bit-exact vs the OpenCL fused kernels for the same seed. state is
// the packed U8/3-byte CounterStateLinear layout; scratch buffers hold
// out_features floats each. Requires VK_KHR_8bit_storage.
VulkanOpResult run_vulkan_compact_counter_apply_update_fused(VulkanRuntime& runtime,
                                                             VulkanBuffer& state,
                                                             VulkanBuffer& scale,
                                                             VulkanBuffer& v,
                                                             const VulkanBuffer& grad_out,
                                                             const VulkanBuffer& x,
                                                             VulkanBuffer& scale_new_scratch,
                                                             VulkanBuffer& denom_scratch,
                                                             std::size_t C,
                                                             std::size_t in_features,
                                                             std::size_t out_features,
                                                             std::size_t batch,
                                                             float lr,
                                                             float lr_scale,
                                                             float rms_beta,
                                                             float rms_eps,
                                                             std::uint32_t seed);
VulkanOpResult run_vulkan_compact_counter_backward_input_u8(VulkanRuntime& runtime,
                                                            const VulkanBuffer& state,
                                                            const VulkanBuffer& scale,
                                                            const VulkanBuffer& grad_out,
                                                            VulkanBuffer& grad_x,
                                                            std::size_t batch,
                                                            std::size_t in_features,
                                                            std::size_t out_features,
                                                            std::size_t C);
VulkanF32MatmulSmokeResult run_vulkan_f32_m1_matmul(const std::vector<float>& a,
                                                    const std::vector<float>& b,
                                                    std::size_t k,
                                                    std::size_t n);
VulkanOpResult run_vulkan_f32_m1_matmul(VulkanRuntime& runtime,
                                        const VulkanBuffer& a,
                                        const VulkanBuffer& b,
                                        VulkanBuffer& c,
                                        std::size_t k,
                                        std::size_t n);
VulkanF32MatmulSmokeResult run_vulkan_f32_matmul_smoke();
std::string vulkan_version_string(std::uint32_t version);

// === Embedding + position-embedding (Slice E1) ===
// out[token_count * embed_dim] = weight[indices[token], :] (zero on OOB index).
VulkanOpResult run_vulkan_embedding_gather(VulkanRuntime& runtime,
                                           const VulkanBuffer& weight,
                                           const VulkanBuffer& indices,
                                           VulkanBuffer& out,
                                           std::size_t vocab_size,
                                           std::size_t embed_dim,
                                           std::size_t token_count);
// Backward: grad_weight[vocab, embed] = sum over tokens where indices[t]==vocab.
VulkanOpResult run_vulkan_embedding_weight_backward(VulkanRuntime& runtime,
                                                    const VulkanBuffer& indices,
                                                    const VulkanBuffer& grad_out,
                                                    VulkanBuffer& grad_weight,
                                                    std::size_t vocab_size,
                                                    std::size_t embed_dim,
                                                    std::size_t token_count);
// Token + position embedding: out = token_weight[token_ids] + pos_weight[pos].
VulkanOpResult run_vulkan_token_position_embedding(VulkanRuntime& runtime,
                                                    const VulkanBuffer& token_weight,
                                                    const VulkanBuffer& pos_weight,
                                                    const VulkanBuffer& token_ids,
                                                    VulkanBuffer& out,
                                                    std::size_t vocab_size,
                                                    std::size_t seq_len,
                                                    std::size_t embed_dim);
// Backward of pos embedding: grad_position[pos, d] = sum_b grad_out[(b*seq+pos)*embed + d].
// The output table must be zero-initialized by the host (positions >= seq_len
// receive no gradient and stay zero, matching the OpenCL host contract).
VulkanOpResult run_vulkan_position_embedding_backward(VulkanRuntime& runtime,
                                                       const VulkanBuffer& grad_out,
                                                       VulkanBuffer& grad_position,
                                                       std::size_t batch,
                                                       std::size_t seq_len,
                                                       std::size_t embed_dim);

// === RoPE (Slice E2) — interleaved and split-half layouts ===
// rope_f32 is reused for backward via the inverse flag (negate angle), exactly
// as the OpenCL rope_impl helper does.
VulkanOpResult run_vulkan_rope(VulkanRuntime& runtime,
                               const VulkanBuffer& x,
                               VulkanBuffer& out,
                               std::size_t batch,
                               std::size_t tokens,
                               std::size_t channels,
                               std::size_t n_head,
                               std::size_t head_dim,
                               std::size_t rotary_dim,
                               std::size_t token_offset,
                               float theta,
                               bool inverse);
VulkanOpResult run_vulkan_rope_positions(VulkanRuntime& runtime,
                                         const VulkanBuffer& x,
                                         const VulkanBuffer& positions,
                                         VulkanBuffer& out,
                                         std::size_t batch,
                                         std::size_t tokens,
                                         std::size_t channels,
                                         std::size_t n_head,
                                         std::size_t head_dim,
                                         std::size_t rotary_dim,
                                         float theta);
VulkanOpResult run_vulkan_rope_split_half(VulkanRuntime& runtime,
                                          const VulkanBuffer& x,
                                          VulkanBuffer& out,
                                          std::size_t batch,
                                          std::size_t tokens,
                                          std::size_t channels,
                                          std::size_t n_head,
                                          std::size_t head_dim,
                                          std::size_t rotary_dim,
                                          std::size_t token_offset,
                                          float theta,
                                          bool inverse);
VulkanOpResult run_vulkan_rope_positions_split_half(VulkanRuntime& runtime,
                                                    const VulkanBuffer& x,
                                                    const VulkanBuffer& positions,
                                                    VulkanBuffer& out,
                                                    std::size_t batch,
                                                    std::size_t tokens,
                                                    std::size_t channels,
                                                    std::size_t n_head,
                                                    std::size_t head_dim,
                                                    std::size_t rotary_dim,
                                                    float theta);

} // namespace motifcl
