#pragma once

#include <array>
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

// One kernel's accumulated GPU time in the opt-in per-kernel profiler
// (MOTIFCL_VK_PROFILE=1). Times come from a VK timestamp pair recorded either
// side of every dispatch inside a batch (start at COMPUTE_SHADER, end at
// ALL_COMMANDS so it waits for the dispatch to retire), so each entry is real
// GPU execution time including that dispatch's own drain. Compare the sum of
// all entries against last_gpu_time_us() to see what is left over: the
// difference is time the GPU spent outside any dispatch (compute->compute
// barriers, bind gaps).
struct VulkanKernelProfile {
    std::string name;
    std::uint64_t calls = 0;
    double total_us = 0.0;
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
    // Subgroup (vulkan1.1+): when subgroup_arithmetic_compute is true, kernels
    // may use subgroupAdd / subgroupBroadcast to replace shared-memory tree
    // reductions (5 barriers + 5 LDS round-trips per reduction -> zero).
    bool subgroup_arithmetic_compute = false;
    std::uint32_t subgroup_size = 0;
    bool supports_wave64_arithmetic() const {
        return subgroup_arithmetic_compute && subgroup_size == 64;
    }
    // VK_EXT_shader_atomic_float: when true, the embedding_weight_backward
    // scatter kernel (float atomicAdd) is available. GCN4 (RX 580) reports
    // false (no hardware float atomics); newer AMD/NVIDIA/Intel GPUs report
    // true.
    bool supports_atomic_float = false;
    // Whether the register-block matmul variants (mm_f32_*_rb4) are actually
    // faster than the base 16x16 tile on this device. Decided by a one-time
    // startup micro-benchmark rather than a subgroup-support proxy: rb4 wins on
    // newer GPUs but LOSES ~1.5x on GCN4 (RX 580), where the base tile is best.
    bool prefer_rb4_matmul = false;
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
    // Submit the open batch and return without waiting for the GPU. The runtime keeps
    // two submission slots: the next batch_begin() records into the other one, and the
    // slot that is still running is waited for before it is reused (or immediately by
    // batch_wait()). Descriptor sets and buffers referenced by an in-flight submission
    // are held until its fence signals, so neither is recycled early.
    VulkanOpResult batch_end_async();
    // Wait for every deferred submission and return their sets/buffers to the pools.
    VulkanOpResult batch_wait();
    bool batch_active() const;

    // GPU timing via VK_QUERY_TYPE_TIMESTAMP around the most recent
    // cached-path submission (single dispatch or whole batch).
    void set_gpu_timing_enabled(bool enabled);
    // Microseconds; negative when no timed submission is available.
    double last_gpu_time_us() const;
    // Wall time spent in vkQueueSubmit + vkWaitForFences for the last cached-path
    // submission (microseconds, negative when none yet), and the submit count.
    // Together with last_gpu_time_us this separates GPU work from submit/fence
    // latency, which is a first-class cost on the Windows WDDM path.
    double last_submit_time_us() const;
    std::uint64_t submit_count() const;

    // Opt-in per-kernel GPU profiler (enable with MOTIFCL_VK_PROFILE=1).
    // Returns accumulated GPU time per embedded kernel, sorted by total time.
    std::vector<VulkanKernelProfile> kernel_profile() const;
    void reset_kernel_profile();

    // Device-allocation accounting for tensor storage buffers: bytes currently
    // owned by the runtime (in use plus pooled) and the peak of that value.
    // Independent of whether buffers were placed in device-local or host-visible
    // memory, so it is the comparable memory number across both policies.
    std::size_t buffer_bytes_live() const;
    std::size_t buffer_bytes_peak() const;
    std::size_t buffer_allocations() const;
    // Bytes actually asked for, as opposed to bytes reserved. The pool rounds
    // every allocation up to a power of two, so `buffer_bytes_live()` minus
    // `buffer_bytes_requested()` is the VRAM paid for and not used. Reported
    // alongside the live figure so a shape change can be judged by what it
    // costs after rounding, which is what the device runs out of.
    std::size_t buffer_bytes_requested() const;
    std::size_t buffer_bytes_requested_peak() const;

    // Wait for all queued Vulkan GPU commands to finish execution.
    void finish() const;

    // Dispatch capture/replay: while capturing, every cached-path dispatch is
    // recorded (and still executed). Replay re-records the whole sequence
    // into one primary command buffer and submits once. No OpenCL objects
    // are involved; buffers are device-resident Vulkan allocations.
    bool capture_begin();
    VulkanDispatchRecording capture_end();
    bool capture_active() const;
    VulkanOpResult replay(const VulkanDispatchRecording& recording);

    VulkanBuffer create_buffer(std::size_t nbytes, const void* initial_data = nullptr);

    // Serial, command-ordered workspace. Calls with the same slot may alias;
    // consume each result before the next use of that slot. Not tensor storage.
    VulkanBuffer create_scratch_buffer(const std::string& slot, std::size_t nbytes);

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
// General forward path used by causal/windowed/masked GQA and KV-cache decode.
// kv_dtype: 0=f32, 1=row-scaled q8_0, 2=row-scaled q4_0.
// mask_dtype: 0=none, 1=f32, 2=i32, 3=u8.
VulkanOpResult run_vulkan_grouped_query_attention_general(
    VulkanRuntime& runtime,
    const VulkanBuffer& q,
    const VulkanBuffer& k,
    const VulkanBuffer& v,
    const VulkanBuffer* k_scales,
    const VulkanBuffer* v_scales,
    const VulkanBuffer* mask,
    VulkanBuffer& out,
    std::size_t batch,
    std::size_t query_tokens,
    std::size_t key_tokens,
    std::size_t key_stride,
    std::size_t n_head,
    std::size_t n_kv_head,
    std::size_t head_dim,
    std::size_t v_head_dim,
    bool causal,
    std::size_t query_offset,
    std::size_t sliding_window,
    std::uint32_t mask_layout,
    std::uint32_t mask_mode,
    std::uint32_t mask_dtype,
    std::uint32_t kv_dtype,
    float scale);
VulkanOpResult run_vulkan_kv_cache_append_f32(VulkanRuntime& runtime,
                                              const VulkanBuffer& new_k,
                                              const VulkanBuffer& new_v,
                                              VulkanBuffer& cache_k,
                                              VulkanBuffer& cache_v,
                                              std::size_t batch,
                                              std::size_t new_tokens,
                                              std::size_t max_tokens,
                                              std::size_t kv_channels,
                                              std::size_t start_pos);
VulkanOpResult run_vulkan_kv_cache_append_quantized(VulkanRuntime& runtime,
                                                    const VulkanBuffer& new_k,
                                                    const VulkanBuffer& new_v,
                                                    VulkanBuffer& cache_k,
                                                    VulkanBuffer& cache_v,
                                                    VulkanBuffer& k_scales,
                                                    VulkanBuffer& v_scales,
                                                    std::size_t batch,
                                                    std::size_t new_tokens,
                                                    std::size_t max_tokens,
                                                    std::size_t kv_channels,
                                                    std::size_t start_pos,
                                                    std::uint32_t kv_dtype);
// F32 GQA backward (two cached dispatches), partitioned by (batch, head).
// probs_scratch and ds_scratch must each hold
// batch*n_head*query_tokens*key_tokens floats.
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
                                                           std::size_t batch,
                                                           std::size_t query_tokens,
                                                           std::size_t key_tokens,
                                                           std::size_t n_head,
                                                           std::size_t n_kv_head,
                                                           std::size_t head_dim,
                                                           bool causal,
                                                           std::size_t query_offset,
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

// === Quant core (Slice Q1) ===
// Quantize f32 [M,K] -> int8 [M,K] + per-row scales f32 [M].
// One workgroup per row: shared reduction max|x|, then quantize whole row.
VulkanOpResult run_vulkan_quantize_q8_rowwise(VulkanRuntime& runtime,
                                              const VulkanBuffer& in_f32,
                                              VulkanBuffer& out_i8,
                                              VulkanBuffer& out_scales,
                                              std::size_t m,
                                              std::size_t k);
// Dequantize int8 + scales -> f32. mode: 0=scalar (scales[0]), 1=per-row, 2=per-col.
VulkanOpResult run_vulkan_dequantize_q8_scaled(VulkanRuntime& runtime,
                                               const VulkanBuffer& in_i8,
                                               const VulkanBuffer& scales,
                                               VulkanBuffer& out_f32,
                                               std::size_t count,
                                               std::uint32_t mode,
                                               std::size_t rows,
                                               std::size_t cols);
// Dequantize packed Q4_0 + scales -> f32. count is element count (not bytes).
VulkanOpResult run_vulkan_dequantize_q4_scaled(VulkanRuntime& runtime,
                                               const VulkanBuffer& packed,
                                               const VulkanBuffer& scales,
                                               VulkanBuffer& out_f32,
                                               std::size_t count,
                                               std::uint32_t mode,
                                               std::size_t rows,
                                               std::size_t cols);
// C[M,N] = scales_a[row]*scales_b[col] * sum_k(A_i8[row,k] * B_i8[k,col]).
VulkanOpResult run_vulkan_matmul_q8q8_scaled(VulkanRuntime& runtime,
                                             const VulkanBuffer& a_i8,
                                             const VulkanBuffer& a_scales,
                                             const VulkanBuffer& b_i8,
                                             const VulkanBuffer& b_scales,
                                             VulkanBuffer& c_f32,
                                             std::size_t m,
                                             std::size_t k,
                                             std::size_t n);
// C[M,N] = scales_a[row]*scales_b[col] * sum_k(A_i8[row,k] * B_q4[k,col]).
// B is packed Q4_0: nibble pairs span consecutive N values.
VulkanOpResult run_vulkan_matmul_q8q4_scaled(VulkanRuntime& runtime,
                                             const VulkanBuffer& a_i8,
                                             const VulkanBuffer& a_scales,
                                             const VulkanBuffer& b_q4,
                                             const VulkanBuffer& b_scales,
                                             VulkanBuffer& c_f32,
                                             std::size_t m,
                                             std::size_t k,
                                             std::size_t n);
// C[N] = scales_b[col] * sum_k(A[k] * B_q4[k,col]) — M=1 decode path.
// A: f32[K]; B: packed Q4_0[K*N]; scales_b: f32[N]; C: f32[N].
VulkanOpResult run_vulkan_matmul_f32q4_m1(VulkanRuntime& runtime,
                                          const VulkanBuffer& a_f32,
                                          const VulkanBuffer& b_q4,
                                          const VulkanBuffer& b_scales,
                                          VulkanBuffer& c_f32,
                                          std::size_t k,
                                          std::size_t n);
VulkanOpResult run_vulkan_matmul_f32q4_packed_qkv_m1(
    VulkanRuntime& runtime,
    const VulkanBuffer& a_f32,
    const VulkanBuffer& b_q4,
    const VulkanBuffer& b_scales,
    VulkanBuffer& q_out,
    VulkanBuffer& k_out,
    VulkanBuffer& v_out,
    std::size_t in_dim,
    std::size_t q_dim,
    std::size_t kv_dim);
VulkanOpResult run_vulkan_matmul_f32q4_packed_swiglu_m1(
    VulkanRuntime& runtime,
    const VulkanBuffer& a_f32,
    const VulkanBuffer& b_q4,
    const VulkanBuffer& b_scales,
    VulkanBuffer& out,
    std::size_t in_dim,
    std::size_t hidden);
VulkanOpResult run_vulkan_add_bias_rows(VulkanRuntime& runtime,
                                        const VulkanBuffer& x,
                                        const VulkanBuffer& bias,
                                        VulkanBuffer& out,
                                        std::size_t rows,
                                        std::size_t cols);
VulkanOpResult run_vulkan_qk_norm_rope_decode(
    VulkanRuntime& runtime,
    const VulkanBuffer& q,
    const VulkanBuffer& k,
    const VulkanBuffer& q_weight,
    const VulkanBuffer& k_weight,
    VulkanBuffer& q_out,
    VulkanBuffer& k_out,
    std::size_t batch,
    std::size_t n_head,
    std::size_t n_kv_head,
    std::size_t head_dim,
    std::size_t rotary_dim,
    std::size_t token_offset,
    float theta,
    float q_eps,
    float k_eps);
VulkanOpResult run_vulkan_qk_norm_rope_cache_append_decode(
    VulkanRuntime& runtime,
    const VulkanBuffer& q,
    const VulkanBuffer& k,
    const VulkanBuffer& v,
    const VulkanBuffer& q_weight,
    const VulkanBuffer& k_weight,
    VulkanBuffer& q_out,
    VulkanBuffer& cache_k,
    VulkanBuffer& cache_v,
    std::size_t batch,
    std::size_t n_head,
    std::size_t n_kv_head,
    std::size_t head_dim,
    std::size_t rotary_dim,
    std::size_t token_offset,
    std::size_t max_tokens,
    float theta,
    float q_eps,
    float k_eps);
VulkanOpResult run_vulkan_rope_cache_append_decode(
    VulkanRuntime& runtime,
    const VulkanBuffer& q,
    const VulkanBuffer& k,
    const VulkanBuffer& v,
    VulkanBuffer& q_out,
    VulkanBuffer& cache_k,
    VulkanBuffer& cache_v,
    std::size_t batch,
    std::size_t n_head,
    std::size_t n_kv_head,
    std::size_t head_dim,
    std::size_t rotary_dim,
    std::size_t token_offset,
    std::size_t max_tokens,
    float theta);
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
// row_stats holds [max, inverse_sum] per row and remains live through backward.
VulkanOpResult run_vulkan_softmax_cross_entropy_cached(VulkanRuntime& runtime,
                                                      const VulkanBuffer& logits,
                                                      const VulkanBuffer& targets,
                                                      VulkanBuffer& partial,
                                                      VulkanBuffer& row_stats,
                                                      VulkanBuffer& out,
                                                      std::size_t rows,
                                                      std::size_t cols);
VulkanOpResult run_vulkan_softmax_cross_entropy_backward_cached(VulkanRuntime& runtime,
                                                               const VulkanBuffer& logits,
                                                               const VulkanBuffer& targets,
                                                               const VulkanBuffer& grad_out,
                                                               const VulkanBuffer& row_stats,
                                                               VulkanBuffer& grad_logits,
                                                               std::size_t rows,
                                                               std::size_t cols);
VulkanOpResult run_vulkan_rmsnorm_backward_weight_split(VulkanRuntime& runtime,
                                                        const VulkanBuffer& x,
                                                        const VulkanBuffer& grad_out,
                                                        VulkanBuffer& row_inv_scratch,
                                                        VulkanBuffer& partial,
                                                        VulkanBuffer& grad_weight,
                                                        std::size_t rows,
                                                        std::size_t cols,
                                                        float eps,
                                                        std::size_t splits);
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
VulkanOpResult run_vulkan_mul(VulkanRuntime& runtime,
                              const VulkanBuffer& a,
                              const VulkanBuffer& b,
                              VulkanBuffer& out,
                              std::size_t elements);
VulkanOpResult run_vulkan_silu(VulkanRuntime& runtime,
                               const VulkanBuffer& x,
                               VulkanBuffer& out,
                               std::size_t elements);
VulkanOpResult run_vulkan_silu_backward(VulkanRuntime& runtime,
                                        const VulkanBuffer& x,
                                        const VulkanBuffer& grad_out,
                                        VulkanBuffer& grad_x,
                                        std::size_t elements);
VulkanOpResult run_vulkan_sigmoid(VulkanRuntime& runtime,
                                  const VulkanBuffer& x,
                                  VulkanBuffer& out,
                                  std::size_t elements);
VulkanOpResult run_vulkan_sigmoid_backward(VulkanRuntime& runtime,
                                           const VulkanBuffer& y,
                                           const VulkanBuffer& grad_out,
                                           VulkanBuffer& grad_x,
                                           std::size_t elements);
VulkanOpResult run_vulkan_fog_block_product(VulkanRuntime& runtime,
                                            const VulkanBuffer& value,
                                            const VulkanBuffer& addressed,
                                            VulkanBuffer& out,
                                            std::size_t rows,
                                            std::size_t d_model);
VulkanOpResult run_vulkan_fog_block_product_backward(VulkanRuntime& runtime,
                                                     const VulkanBuffer& value,
                                                     const VulkanBuffer& addressed,
                                                     const VulkanBuffer& grad_out,
                                                     VulkanBuffer& grad_value,
                                                     VulkanBuffer& grad_addressed,
                                                     std::size_t rows,
                                                     std::size_t d_model);
VulkanOpResult run_vulkan_fog_hard_route7(VulkanRuntime& runtime,
                                          const VulkanBuffer& logits,
                                          const std::vector<const VulkanBuffer*>& candidates,
                                          VulkanBuffer& out,
                                          std::size_t rows,
                                          std::size_t d_model);
VulkanOpResult run_vulkan_fog_hard_route7_candidate_backward(VulkanRuntime& runtime,
                                                             const VulkanBuffer& logits,
                                                             const VulkanBuffer& grad_out,
                                                             VulkanBuffer& grad_candidate,
                                                             std::size_t rows,
                                                             std::size_t d_model,
                                                             std::uint32_t candidate);
VulkanOpResult run_vulkan_fog_hard_route7_logits_backward(VulkanRuntime& runtime,
                                                          const VulkanBuffer& logits,
                                                          const std::vector<const VulkanBuffer*>& candidates,
                                                          const VulkanBuffer& grad_out,
                                                          VulkanBuffer& grad_logits,
                                                          std::size_t rows,
                                                          std::size_t d_model);
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
VulkanOpResult run_vulkan_adam_update_fast(VulkanRuntime& runtime,
                                           const VulkanBuffer& param,
                                           const VulkanBuffer& grad,
                                           VulkanBuffer& m,
                                           VulkanBuffer& v,
                                           std::size_t elements,
                                           float lr, float beta1, float beta2,
                                           float eps, float weight_decay,
                                           float c1, float c2);
VulkanOpResult run_vulkan_compact_counter_decode_weight(VulkanRuntime& runtime,
                                                        const VulkanBuffer& state,
                                                        const VulkanBuffer& scale,
                                                        VulkanBuffer& weight,
                                                        std::size_t in_features,
                                                        std::size_t out_features,
                                                        std::size_t C);
// Fused decode-in-GEMM forward: y[batch,out] = x[batch,in] * W(state)^T.
// Uses a wave64 rb4 register block and never materializes dense FP32 W.
VulkanOpResult run_vulkan_compact_counter_forward_u8(VulkanRuntime& runtime,
                                                      const VulkanBuffer& x,
                                                      const VulkanBuffer& state,
                                                      const VulkanBuffer& scale,
                                                      VulkanBuffer& out,
                                                      std::size_t batch,
                                                      std::size_t in_features,
                                                      std::size_t out_features,
                                                      std::size_t C);
// Training epilogue: save [batch,out_features] for backward and emit
// SwiGLU [batch,out_features/2] without a separate activation dispatch.
VulkanOpResult run_vulkan_compact_counter_swiglu_u8(VulkanRuntime& runtime,
                                                     const VulkanBuffer& x,
                                                     const VulkanBuffer& state,
                                                     const VulkanBuffer& scale,
                                                     VulkanBuffer& preactivation,
                                                     VulkanBuffer& out,
                                                     std::size_t C,
                                                     std::size_t in_features,
                                                     std::size_t out_features,
                                                     std::size_t batch);
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
// Memory-native counter update: fold the row statistics of a tiled grad_w
// (grad_w = grad_out^T * x in a reusable scratch buffer), commit the row-RMS and
// scale state, and run the finite-state stochastic-rounding tick. One wave64
// workgroup per output row; this pass is bandwidth-only, so the old path's second
// grad_w recompute and its separate row-scale copy are gone.
VulkanOpResult run_vulkan_moe_balance_bias_update(VulkanRuntime& runtime,
                                            const VulkanBuffer& old_bias,
                                            VulkanBuffer& new_bias,
                                            const std::array<float, 7>& loads,
                                            float gamma,
                                            float clamp_value);

// Device-side load-balancing gradient for the MoE router:
//   dl[n,j] = scale * p[n,j] * (f_j - sum_e f_e p[n,e]), p = softmax(logits[n])
// with scale = (E/N) * aux_weight. The host path downloads the [N,7] logits and
// uploads the [N,7] gradient per block per step; this writes the gradient straight
// from the forward's logits buffer. `rows` is the logits row count.
VulkanOpResult run_vulkan_moe_balance_gradient(VulkanRuntime& runtime,
                                               const VulkanBuffer& logits,
                                               VulkanBuffer& grad,
                                               std::size_t rows,
                                               const std::array<float, 7>& loads,
                                               float scale);

VulkanOpResult run_vulkan_compact_counter_update_apply(VulkanRuntime& runtime,
                                                       VulkanBuffer& state,
                                                       VulkanBuffer& scale,
                                                       VulkanBuffer& v,
                                                       const VulkanBuffer& grad_w,
                                                       std::size_t in_features,
                                                       std::size_t out_features,
                                                       std::size_t C,
                                                       float lr,
                                                       float lr_scale,
                                                       float rms_beta,
                                                       float rms_eps,
                                                       std::uint32_t seed);
VulkanOpResult run_vulkan_compact_counter_decay_v(VulkanRuntime& runtime,
                                                  VulkanBuffer& v,
                                                  std::size_t out_features,
                                                  float rms_beta);
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
// Atomic-scatter variant of embedding weight backward: O(tokens*embed) work
// instead of O(vocab*tokens*embed). The grad_weight buffer MUST be zero-filled
// by the caller (run_vulkan_zero_f32) before this dispatch — atomicAdd
// accumulates into the existing value. Returns the scatter dispatch result.
// Driver must support GL_EXT_shader_atomic_float (Vulkan 1.1+ with the
// shaderAtomicFloat feature; most desktop GPUs since 2018 do).
VulkanOpResult run_vulkan_zero_f32(VulkanRuntime& runtime,
                                   VulkanBuffer& out,
                                   std::size_t elements);
VulkanOpResult run_vulkan_embedding_weight_backward_scatter(VulkanRuntime& runtime,
                                                            const VulkanBuffer& indices,
                                                            const VulkanBuffer& grad_out,
                                                            VulkanBuffer& grad_weight,
                                                            std::size_t vocab_size,
                                                            std::size_t embed_dim,
                                                            std::size_t token_count);
// Portable compare-and-swap variant of the scatter: same O(tokens*embed) work
// and identical result, but emulates the float atomicAdd with an integer
// atomicCompSwap loop (core Vulkan 1.0) instead of VK_EXT_shader_atomic_float.
// Correct on GPUs whose native float atomics are absent or driver-broken
// (notably GCN4 / Radeon RX 580). The grad_weight buffer MUST be zero-filled by
// the caller (run_vulkan_zero_f32) before this dispatch.
VulkanOpResult run_vulkan_embedding_weight_backward_scatter_cas(VulkanRuntime& runtime,
                                                                const VulkanBuffer& indices,
                                                                const VulkanBuffer& grad_out,
                                                                VulkanBuffer& grad_weight,
                                                                std::size_t vocab_size,
                                                                std::size_t embed_dim,
                                                                std::size_t token_count);
// Reproducible embedding-weight backward. Pass 1 counts tokens per row with
// integer atomics (order-independent) and writes each single-token row; pass 2
// re-sums the rows that received duplicate token ids by walking the ids in
// ascending order. The result therefore does not depend on thread scheduling,
// unlike the default float compare-and-swap scatter, whose retry order changes the
// last bits of every duplicated row (and, through Adam, of the whole checkpoint).
// The output buffer is zero-filled internally. Requires embed_dim % 4 == 0.
VulkanOpResult run_vulkan_embedding_weight_backward_deterministic(VulkanRuntime& runtime,
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
// Fused softmax-cross-entropy + linear-head gradients. dlogits is never materialised:
// both GEMMs recompute it from the logits operand, the per-row (max, inv_sum)
// statistics and the row target while staging their operand. Every value and operation
// matches the standalone cached CE-backward kernel, so the results are bit-identical to
// multiplying a materialised dlogits. Constants and layouts are those of the
// double-buffered 64x64 kernel the head already used.
//   grad_input  [rows, in]      = dlogits[rows, classes] * weight[out, classes]^T
VulkanOpResult run_vulkan_matmul_ce_nt(VulkanRuntime& runtime,
                                       const VulkanBuffer& logits,
                                       const VulkanBuffer& weight,
                                       const VulkanBuffer& row_stats,
                                       const VulkanBuffer& targets,
                                       const VulkanBuffer& grad_out,
                                       VulkanBuffer& out,
                                       std::size_t rows,
                                       std::size_t classes,
                                       std::size_t in_features);
//   grad_weight [out, classes]  = x[rows, in]^T * dlogits[rows, classes]
VulkanOpResult run_vulkan_matmul_ce_tn(VulkanRuntime& runtime,
                                       const VulkanBuffer& x,
                                       const VulkanBuffer& logits,
                                       const VulkanBuffer& row_stats,
                                       const VulkanBuffer& targets,
                                       const VulkanBuffer& grad_out,
                                       VulkanBuffer& out,
                                       std::size_t in_features,
                                       std::size_t rows,
                                       std::size_t classes);
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

// ---------------------------------------------------------------------------
// Sparse MoE7 forward (mission Phase C).
//
// Computes only the expert that each token's router argmax selects, instead of
// running all seven experts and discarding six outputs. The routing decision
// stays on the GPU: `moe_route_top1_f32` writes per-expert token counts and
// compact assignment lists, and every expert dispatch is bounded by its own
// count, so no routing result is ever read back to the host.
//
// Semantics are the dense ones (nn::MoE7FFN): hard top-1 over seven experts,
// router bias shifts selection only, SwiGLU gate*sigmoid(gate)*up, output of the
// winner plus the shared expert. `shared_out` is NOT added here — the caller
// merges it exactly as the dense path does, so the merge stays in one place.
//
// Scratch buffers (all device buffers owned by the caller, reused across calls):
//   logits   [tokens, 7]     f32  biased router logits
//   counts   [8]             u32  per-expert token counts (must be zero before
//                                 the call: the routing kernel atomicAdds)
//   list     [7 * tokens]    u32  compact assignment lists, capacity = tokens
//   packed   [tokens, 2*h]   f32  expert up-projection output
//   act      [tokens, h]     f32  expert SwiGLU activation
//   expert_out [tokens, dim] f32  scattered winning-expert output
// `capacity` is `tokens` by construction: every token is assigned to exactly
// one expert, so no list can overflow and no token is dropped.
struct VulkanMoE7SparseSpec {
    std::size_t tokens = 0;
    std::size_t dim = 0;
    std::size_t hidden = 0;
    std::size_t C = 0;
    const VulkanBuffer* x = nullptr;
    const VulkanBuffer* router_w = nullptr;
    const VulkanBuffer* router_b = nullptr;
    std::array<const VulkanBuffer*, 7> up_state{};
    std::array<const VulkanBuffer*, 7> up_scale{};
    std::array<const VulkanBuffer*, 7> down_state{};
    std::array<const VulkanBuffer*, 7> down_scale{};
    VulkanBuffer* logits = nullptr;
    VulkanBuffer* counts = nullptr;
    VulkanBuffer* list = nullptr;
    VulkanBuffer* packed = nullptr;
    VulkanBuffer* act = nullptr;
    VulkanBuffer* expert_out = nullptr;
};

VulkanOpResult run_vulkan_moe7_sparse_forward(VulkanRuntime& runtime,
                                              const VulkanMoE7SparseSpec& spec);

} // namespace motifcl
