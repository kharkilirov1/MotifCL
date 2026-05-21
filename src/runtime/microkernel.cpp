#include <motifcl/runtime/microkernel.hpp>

#include <motifcl/core/logging.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace motifcl {
namespace {

std::string normalize_backend_name(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    auto first = std::find_if(value.begin(), value.end(), not_space);
    auto last = std::find_if(value.rbegin(), value.rend(), not_space).base();
    if (first >= last) return {};
    value = std::string(first, last);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::size_t domain_index(MicrokernelDomain domain) {
    switch (domain) {
    case MicrokernelDomain::Matmul: return 0;
    case MicrokernelDomain::Attention: return 1;
    case MicrokernelDomain::Quant: return 2;
    case MicrokernelDomain::Norm: return 3;
    case MicrokernelDomain::Activation: return 4;
    }
    return 0;
}

std::mutex& warning_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::array<bool, 5>& warning_emitted() {
    static std::array<bool, 5> emitted = {false, false, false, false, false};
    return emitted;
}

void warn_fallback_once(const MicrokernelSelection& selection) {
    if (!selection.fallback_to_opencl || selection.fallback_reason.empty()) return;
    const auto index = domain_index(selection.domain);
    std::lock_guard<std::mutex> lock(warning_mutex());
    auto& emitted = warning_emitted();
    if (emitted[index]) return;
    emitted[index] = true;
    Logger::warn(selection.fallback_reason);
}

MicrokernelDescriptor descriptor(MicrokernelDomain domain,
                                 MicrokernelBackendKind backend,
                                 const char* name,
                                 const char* contract) {
    MicrokernelDescriptor d;
    d.domain = domain;
    d.backend = backend;
    d.stability = microkernel_backend_stability(backend);
    d.name = name;
    d.contract = contract;
    return d;
}

template <typename Backend>
Backend selected_domain_backend(MicrokernelDomain domain) {
    const auto selection = select_microkernel_backend(domain);
    warn_fallback_once(selection);
    Backend backend;
    backend.kind = selection.effective;
    backend.stability = microkernel_backend_stability(selection.effective);
    backend.kernels = microkernel_descriptors(domain, selection.effective);
    return backend;
}

} // namespace

MicrokernelStability microkernel_backend_stability(MicrokernelBackendKind kind) {
    return microkernel_backend_is_experimental(kind)
        ? MicrokernelStability::Experimental
        : MicrokernelStability::Stable;
}

bool microkernel_backend_is_experimental(MicrokernelBackendKind kind) {
    return kind == MicrokernelBackendKind::Native ||
           kind == MicrokernelBackendKind::Asm ||
           kind == MicrokernelBackendKind::Vulkan;
}

const char* microkernel_backend_name(MicrokernelBackendKind kind) {
    switch (kind) {
    case MicrokernelBackendKind::OpenCL: return "opencl";
    case MicrokernelBackendKind::Native: return "native";
    case MicrokernelBackendKind::Asm: return "asm";
    case MicrokernelBackendKind::Vulkan: return "vulkan";
    }
    return "unknown";
}

const char* microkernel_domain_name(MicrokernelDomain domain) {
    switch (domain) {
    case MicrokernelDomain::Matmul: return "matmul";
    case MicrokernelDomain::Attention: return "attention";
    case MicrokernelDomain::Quant: return "quant";
    case MicrokernelDomain::Norm: return "norm";
    case MicrokernelDomain::Activation: return "activation";
    }
    return "unknown";
}

const char* microkernel_env_name(MicrokernelDomain domain) {
    switch (domain) {
    case MicrokernelDomain::Matmul: return "MOTIFCL_MATMUL_BACKEND";
    case MicrokernelDomain::Attention: return "MOTIFCL_ATTENTION_BACKEND";
    case MicrokernelDomain::Quant: return "MOTIFCL_QUANT_BACKEND";
    case MicrokernelDomain::Norm: return "MOTIFCL_NORM_BACKEND";
    case MicrokernelDomain::Activation: return "MOTIFCL_ACTIVATION_BACKEND";
    }
    return "MOTIFCL_BACKEND";
}

MicrokernelBackendKind microkernel_backend_from_name(const std::string& value, bool* recognized) {
    const std::string normalized = normalize_backend_name(value);
    if (recognized) *recognized = true;
    if (normalized.empty() || normalized == "default" || normalized == "opencl" || normalized == "cl") {
        return MicrokernelBackendKind::OpenCL;
    }
    if (normalized == "native" || normalized == "cpu") return MicrokernelBackendKind::Native;
    if (normalized == "asm" || normalized == "assembly") return MicrokernelBackendKind::Asm;
    if (normalized == "vulkan" || normalized == "vk") return MicrokernelBackendKind::Vulkan;
    if (recognized) *recognized = false;
    return MicrokernelBackendKind::OpenCL;
}

std::vector<MicrokernelDescriptor> microkernel_descriptors(MicrokernelDomain domain,
                                                           MicrokernelBackendKind backend) {
    if (backend == MicrokernelBackendKind::Native && domain == MicrokernelDomain::Matmul) {
        return {
            descriptor(domain, backend, "native.matmul.f32_m1",
                       "opt-in host native F32 M=1 decode matmul; rank-2 same-backend tensors, no autograd, bounded M/N/K, OpenCL fallback for unsupported shapes"),
            descriptor(domain, backend, "native.matmul.f32_q4_0_m1",
                       "opt-in host native F32 x packed-Q4_0 M=1 decode matmul; scalar quant scale only, no autograd, OpenCL fallback for external scale tensors"),
        };
    }
    if (backend == MicrokernelBackendKind::Vulkan && domain == MicrokernelDomain::Matmul) {
        return {
            descriptor(domain, backend, "vulkan.matmul.f32_m1",
                       "opt-in native-GPU Vulkan F32 M=1 decode matmul; rank-2 same-backend tensors, no autograd, bounded K/N specialization, OpenCL fallback when Vulkan compute is unavailable"),
            descriptor(domain, backend, "vulkan.matmul.f32",
                       "opt-in native-GPU Vulkan F32 matmul; rank-2 same-backend tensors, no autograd, bounded specialized K, exact-dispatch M/N, OpenCL fallback when Vulkan compute is unavailable"),
        };
    }
    if (backend == MicrokernelBackendKind::Vulkan && domain == MicrokernelDomain::Attention) {
        return {
            descriptor(domain, backend, "vulkan.attention.softmax_rows_f32",
                       "opt-in native-GPU Vulkan F32 row softmax; rank-2 tensors, bounded row/column specialization, OpenCL fallback when Vulkan compute is unavailable"),
        };
    }
    if (backend == MicrokernelBackendKind::Vulkan && domain == MicrokernelDomain::Norm) {
        return {
            descriptor(domain, backend, "vulkan.norm.rmsnorm_f32",
                       "opt-in native-GPU Vulkan F32 RMSNorm; rank-2 input plus rank-1 weight, no autograd, bounded rows/cols specialization, OpenCL fallback when Vulkan compute is unavailable"),
        };
    }
    if (backend == MicrokernelBackendKind::Vulkan && domain == MicrokernelDomain::Activation) {
        return {
            descriptor(domain, backend, "vulkan.activation.swiglu_f32",
                       "opt-in native-GPU Vulkan F32 SwiGLU; rank-2 packed [rows,2*hidden], no autograd, bounded rows/hidden specialization, OpenCL fallback when Vulkan compute is unavailable"),
        };
    }
    if (backend != MicrokernelBackendKind::OpenCL) return {};

    switch (domain) {
    case MicrokernelDomain::Matmul:
        return {
            descriptor(domain, backend, "opencl.matmul.f32",
                       "rank-2 same-backend f32 tensors, bounded M/N/K int args, selected tile/workgroup within device limits"),
            descriptor(domain, backend, "opencl.matmul.quantized",
                       "rank-2 quant tensors, packed dtype/layout checks, quant scale tensors cover rows/cols/blocks before launch"),
            descriptor(domain, backend, "opencl.matmul.f16_accum_f32",
                       "f16-only no-autograd path with f32 accumulation and bounded matrix dimensions"),
        };
    case MicrokernelDomain::Attention:
        return {
            descriptor(domain, backend, "opencl.attention.grouped_query",
                       "q/k/v rank, batch, head, kv-head, mask, and sequence lengths validated before launch"),
            descriptor(domain, backend, "opencl.attention.kv_cache",
                       "cache/page table shapes, offsets, row counts, dtype, and quant scale coverage validated before launch"),
            descriptor(domain, backend, "opencl.attention.backward",
                       "gradient tensor shapes and staged kernel resource limits validated before launch"),
        };
    case MicrokernelDomain::Quant:
        return {
            descriptor(domain, backend, "opencl.quantize",
                       "f32 source, explicit quant dtype, finite scalar/axis/block scales, and bounded work-item counts"),
            descriptor(domain, backend, "opencl.dequantize",
                       "quant source dtype, scalar/metadata scale validity, and scale tensor coverage validated before launch"),
        };
    case MicrokernelDomain::Norm:
        return {
            descriptor(domain, backend, "opencl.norm.rmsnorm",
                       "rank-2 f32 input, rank-1 f32 weight, same-backend tensors, finite eps, bounded row/column dimensions before launch"),
            descriptor(domain, backend, "opencl.norm.layernorm",
                       "rank-2 f32 input, rank-1 f32 weight/bias, same-backend tensors, finite eps, bounded row/column dimensions before launch"),
        };
    case MicrokernelDomain::Activation:
        return {
            descriptor(domain, backend, "opencl.activation.unary",
                       "f32 activation tensors with bounded element counts and same-backend output before launch"),
            descriptor(domain, backend, "opencl.activation.swiglu",
                       "rank-2 f32 packed [rows,2*hidden] tensor with bounded rows/hidden before launch"),
        };
    }
    return {};
}

bool microkernel_backend_available(MicrokernelDomain domain, MicrokernelBackendKind backend) {
    return !microkernel_descriptors(domain, backend).empty();
}

MicrokernelSelection select_microkernel_backend_from_value(MicrokernelDomain domain, const std::string& value) {
    MicrokernelSelection selection;
    selection.domain = domain;
    selection.requested_name = normalize_backend_name(value);
    bool recognized = true;
    selection.requested = microkernel_backend_from_name(value, &recognized);
    selection.stability = microkernel_backend_stability(selection.requested);

    if (!recognized) {
        selection.effective = MicrokernelBackendKind::OpenCL;
        selection.stability = MicrokernelStability::Stable;
        selection.fallback_to_opencl = true;
        selection.fallback_reason = std::string("Unknown ") + microkernel_domain_name(domain) +
            " microkernel backend '" + selection.requested_name + "'; falling back to opencl. Set " +
            microkernel_env_name(domain) + "=opencl|native|asm|vulkan.";
        return selection;
    }

    if (microkernel_backend_available(domain, selection.requested)) {
        selection.effective = selection.requested;
        selection.stability = microkernel_backend_stability(selection.effective);
        return selection;
    }

    selection.effective = MicrokernelBackendKind::OpenCL;
    selection.fallback_to_opencl = true;
    selection.fallback_reason = std::string("Experimental ") + microkernel_backend_name(selection.requested) +
        " " + microkernel_domain_name(domain) +
        " microkernel backend is not implemented in this build; falling back to opencl.";
    return selection;
}

MicrokernelSelection select_microkernel_backend(MicrokernelDomain domain) {
    const char* value = std::getenv(microkernel_env_name(domain));
    return select_microkernel_backend_from_value(domain, value ? value : "opencl");
}

MatmulBackend selected_matmul_backend() {
    return selected_domain_backend<MatmulBackend>(MicrokernelDomain::Matmul);
}

AttentionBackend selected_attention_backend() {
    return selected_domain_backend<AttentionBackend>(MicrokernelDomain::Attention);
}

QuantBackend selected_quant_backend() {
    return selected_domain_backend<QuantBackend>(MicrokernelDomain::Quant);
}

NormBackend selected_norm_backend() {
    return selected_domain_backend<NormBackend>(MicrokernelDomain::Norm);
}

ActivationBackend selected_activation_backend() {
    return selected_domain_backend<ActivationBackend>(MicrokernelDomain::Activation);
}

bool microkernel_runtime_uses_opencl(MicrokernelDomain domain) {
    const auto selection = select_microkernel_backend(domain);
    warn_fallback_once(selection);
    return selection.effective == MicrokernelBackendKind::OpenCL;
}

} // namespace motifcl
