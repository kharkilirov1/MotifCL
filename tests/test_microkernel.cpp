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

    auto expect_descriptors = [&](MicrokernelDomain domain, MicrokernelBackendKind backend, const char* message) {
        const auto descriptors = microkernel_descriptors(domain, backend);
        ok &= expect(!descriptors.empty(), message);
        for (const auto& descriptor : descriptors) {
            ok &= expect(descriptor.domain == domain, "descriptor domain mismatch");
            ok &= expect(descriptor.backend == backend, "descriptor backend mismatch");
            ok &= expect(!descriptor.name.empty(), "descriptor name must not be empty");
            ok &= expect(!descriptor.contract.empty(), "descriptor contract must not be empty");
            ok &= expect(descriptor.stability == microkernel_backend_stability(backend),
                         "descriptor stability mismatch");
        }
    };
    expect_descriptors(MicrokernelDomain::Matmul, MicrokernelBackendKind::OpenCL,
                       "OpenCL matmul descriptor registry must not be empty");
    expect_descriptors(MicrokernelDomain::Attention, MicrokernelBackendKind::OpenCL,
                       "OpenCL attention descriptor registry must not be empty");
    expect_descriptors(MicrokernelDomain::Quant, MicrokernelBackendKind::OpenCL,
                       "OpenCL quant descriptor registry must not be empty");

    ok &= expect(microkernel_backend_available(MicrokernelDomain::Matmul, MicrokernelBackendKind::OpenCL),
                 "OpenCL matmul backend must be available");
    ok &= expect(!microkernel_backend_available(MicrokernelDomain::Matmul, MicrokernelBackendKind::Native),
                 "native matmul backend must stay unavailable until implemented");
    ok &= expect(!microkernel_backend_available(MicrokernelDomain::Attention, MicrokernelBackendKind::Asm),
                 "asm attention backend must stay unavailable until implemented");

    const auto selected_matmul = selected_matmul_backend();
    ok &= expect(selected_matmul.kind == MicrokernelBackendKind::OpenCL &&
                     selected_matmul.stability == MicrokernelStability::Stable &&
                     !selected_matmul.kernels.empty(),
                 "selected matmul backend must resolve to registered OpenCL kernels");
    const auto selected_attention = selected_attention_backend();
    ok &= expect(selected_attention.kind == MicrokernelBackendKind::OpenCL &&
                     selected_attention.stability == MicrokernelStability::Stable &&
                     !selected_attention.kernels.empty(),
                 "selected attention backend must resolve to registered OpenCL kernels");
    const auto selected_quant = selected_quant_backend();
    ok &= expect(selected_quant.kind == MicrokernelBackendKind::OpenCL &&
                     selected_quant.stability == MicrokernelStability::Stable &&
                     !selected_quant.kernels.empty(),
                 "selected quant backend must resolve to registered OpenCL kernels");

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
