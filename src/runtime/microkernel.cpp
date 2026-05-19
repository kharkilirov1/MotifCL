#include <motifcl/runtime/microkernel.hpp>

#include <motifcl/core/logging.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <string>

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
    }
    return 0;
}

std::mutex& warning_mutex() {
    static std::mutex mutex;
    return mutex;
}

std::array<bool, 3>& warning_emitted() {
    static std::array<bool, 3> emitted = {false, false, false};
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

} // namespace

MicrokernelStability microkernel_backend_stability(MicrokernelBackendKind kind) {
    return microkernel_backend_is_experimental(kind)
        ? MicrokernelStability::Experimental
        : MicrokernelStability::Stable;
}

bool microkernel_backend_is_experimental(MicrokernelBackendKind kind) {
    return kind == MicrokernelBackendKind::Native || kind == MicrokernelBackendKind::Asm;
}

const char* microkernel_backend_name(MicrokernelBackendKind kind) {
    switch (kind) {
    case MicrokernelBackendKind::OpenCL: return "opencl";
    case MicrokernelBackendKind::Native: return "native";
    case MicrokernelBackendKind::Asm: return "asm";
    }
    return "unknown";
}

const char* microkernel_domain_name(MicrokernelDomain domain) {
    switch (domain) {
    case MicrokernelDomain::Matmul: return "matmul";
    case MicrokernelDomain::Attention: return "attention";
    case MicrokernelDomain::Quant: return "quant";
    }
    return "unknown";
}

const char* microkernel_env_name(MicrokernelDomain domain) {
    switch (domain) {
    case MicrokernelDomain::Matmul: return "MOTIFCL_MATMUL_BACKEND";
    case MicrokernelDomain::Attention: return "MOTIFCL_ATTENTION_BACKEND";
    case MicrokernelDomain::Quant: return "MOTIFCL_QUANT_BACKEND";
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
    if (recognized) *recognized = false;
    return MicrokernelBackendKind::OpenCL;
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
            microkernel_env_name(domain) + "=opencl|native|asm.";
        return selection;
    }

    if (selection.requested == MicrokernelBackendKind::OpenCL) {
        selection.effective = MicrokernelBackendKind::OpenCL;
        selection.stability = MicrokernelStability::Stable;
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

bool microkernel_runtime_uses_opencl(MicrokernelDomain domain) {
    const auto selection = select_microkernel_backend(domain);
    warn_fallback_once(selection);
    return selection.effective == MicrokernelBackendKind::OpenCL;
}

} // namespace motifcl
