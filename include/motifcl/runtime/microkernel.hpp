#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace motifcl {

enum class MicrokernelDomain {
    Matmul,
    Attention,
    Quant,
};

enum class MicrokernelBackendKind {
    OpenCL,
    Native,
    Asm,
};

enum class MicrokernelStability {
    Stable,
    Experimental,
};

struct MicrokernelDescriptor {
    MicrokernelDomain domain = MicrokernelDomain::Matmul;
    MicrokernelBackendKind backend = MicrokernelBackendKind::OpenCL;
    MicrokernelStability stability = MicrokernelStability::Stable;
    std::string name;
    std::string contract;
};

struct MatmulBackend {
    MicrokernelBackendKind kind = MicrokernelBackendKind::OpenCL;
    MicrokernelStability stability = MicrokernelStability::Stable;
    std::vector<MicrokernelDescriptor> kernels;
};

struct AttentionBackend {
    MicrokernelBackendKind kind = MicrokernelBackendKind::OpenCL;
    MicrokernelStability stability = MicrokernelStability::Stable;
    std::vector<MicrokernelDescriptor> kernels;
};

struct QuantBackend {
    MicrokernelBackendKind kind = MicrokernelBackendKind::OpenCL;
    MicrokernelStability stability = MicrokernelStability::Stable;
    std::vector<MicrokernelDescriptor> kernels;
};

struct MicrokernelSelection {
    MicrokernelDomain domain = MicrokernelDomain::Matmul;
    MicrokernelBackendKind requested = MicrokernelBackendKind::OpenCL;
    MicrokernelBackendKind effective = MicrokernelBackendKind::OpenCL;
    MicrokernelStability stability = MicrokernelStability::Stable;
    bool fallback_to_opencl = false;
    std::string requested_name;
    std::string fallback_reason;
};

MicrokernelStability microkernel_backend_stability(MicrokernelBackendKind kind);
bool microkernel_backend_is_experimental(MicrokernelBackendKind kind);
const char* microkernel_backend_name(MicrokernelBackendKind kind);
const char* microkernel_domain_name(MicrokernelDomain domain);
const char* microkernel_env_name(MicrokernelDomain domain);
MicrokernelBackendKind microkernel_backend_from_name(const std::string& value, bool* recognized = nullptr);

MicrokernelSelection select_microkernel_backend_from_value(MicrokernelDomain domain, const std::string& value);
MicrokernelSelection select_microkernel_backend(MicrokernelDomain domain);

// Returns true for the currently compiled runtime. Experimental native/ASM
// requests are explicit opt-ins, but until those implementations exist this
// function emits one warning per domain and keeps OpenCL as the safe fallback.
bool microkernel_runtime_uses_opencl(MicrokernelDomain domain);

} // namespace motifcl
