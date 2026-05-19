#include <iostream>
#include <string>

#include <motifcl/runtime/microkernel.hpp>

int main() {
    using namespace motifcl;

    auto expect = [](bool condition, const char* message) {
        if (!condition) {
            std::cerr << message << '\n';
            return false;
        }
        return true;
    };

    bool ok = true;

    ok &= expect(std::string(microkernel_backend_name(MicrokernelBackendKind::OpenCL)) == "opencl",
                 "OpenCL backend name mismatch");
    ok &= expect(std::string(microkernel_backend_name(MicrokernelBackendKind::Native)) == "native",
                 "native backend name mismatch");
    ok &= expect(std::string(microkernel_backend_name(MicrokernelBackendKind::Asm)) == "asm",
                 "asm backend name mismatch");

    ok &= expect(!microkernel_backend_is_experimental(MicrokernelBackendKind::OpenCL),
                 "OpenCL must be stable");
    ok &= expect(microkernel_backend_is_experimental(MicrokernelBackendKind::Native),
                 "native backend must be experimental");
    ok &= expect(microkernel_backend_is_experimental(MicrokernelBackendKind::Asm),
                 "asm backend must be experimental");

    ok &= expect(std::string(microkernel_domain_name(MicrokernelDomain::Matmul)) == "matmul",
                 "matmul domain name mismatch");
    ok &= expect(std::string(microkernel_domain_name(MicrokernelDomain::Attention)) == "attention",
                 "attention domain name mismatch");
    ok &= expect(std::string(microkernel_domain_name(MicrokernelDomain::Quant)) == "quant",
                 "quant domain name mismatch");

    ok &= expect(std::string(microkernel_env_name(MicrokernelDomain::Matmul)) == "MOTIFCL_MATMUL_BACKEND",
                 "matmul env name mismatch");
    ok &= expect(std::string(microkernel_env_name(MicrokernelDomain::Attention)) == "MOTIFCL_ATTENTION_BACKEND",
                 "attention env name mismatch");
    ok &= expect(std::string(microkernel_env_name(MicrokernelDomain::Quant)) == "MOTIFCL_QUANT_BACKEND",
                 "quant env name mismatch");

    bool recognized = false;
    ok &= expect(microkernel_backend_from_name(" OPENCL ", &recognized) == MicrokernelBackendKind::OpenCL && recognized,
                 "failed to parse opencl backend");
    ok &= expect(microkernel_backend_from_name("Native", &recognized) == MicrokernelBackendKind::Native && recognized,
                 "failed to parse native backend");
    ok &= expect(microkernel_backend_from_name("assembly", &recognized) == MicrokernelBackendKind::Asm && recognized,
                 "failed to parse assembly backend");
    ok &= expect(microkernel_backend_from_name("definitely-not-real", &recognized) == MicrokernelBackendKind::OpenCL && !recognized,
                 "unknown backend must be unrecognized OpenCL fallback");

    const auto opencl = select_microkernel_backend_from_value(MicrokernelDomain::Matmul, "opencl");
    ok &= expect(opencl.requested == MicrokernelBackendKind::OpenCL &&
                     opencl.effective == MicrokernelBackendKind::OpenCL &&
                     !opencl.fallback_to_opencl &&
                     opencl.fallback_reason.empty(),
                 "opencl selection should not fallback");

    const auto native = select_microkernel_backend_from_value(MicrokernelDomain::Attention, "native");
    ok &= expect(native.requested == MicrokernelBackendKind::Native &&
                     native.effective == MicrokernelBackendKind::OpenCL &&
                     native.stability == MicrokernelStability::Experimental &&
                     native.fallback_to_opencl &&
                     !native.fallback_reason.empty(),
                 "native selection must explicitly fallback to OpenCL");

    const auto asm_backend = select_microkernel_backend_from_value(MicrokernelDomain::Quant, "asm");
    ok &= expect(asm_backend.requested == MicrokernelBackendKind::Asm &&
                     asm_backend.effective == MicrokernelBackendKind::OpenCL &&
                     asm_backend.stability == MicrokernelStability::Experimental &&
                     asm_backend.fallback_to_opencl &&
                     !asm_backend.fallback_reason.empty(),
                 "asm selection must explicitly fallback to OpenCL");

    const auto unknown = select_microkernel_backend_from_value(MicrokernelDomain::Matmul, " mystery ");
    ok &= expect(unknown.requested == MicrokernelBackendKind::OpenCL &&
                     unknown.effective == MicrokernelBackendKind::OpenCL &&
                     unknown.stability == MicrokernelStability::Stable &&
                     unknown.fallback_to_opencl &&
                     unknown.requested_name == "mystery" &&
                     !unknown.fallback_reason.empty(),
                 "unknown selection must be controlled OpenCL fallback");

    return ok ? 0 : 1;
}
