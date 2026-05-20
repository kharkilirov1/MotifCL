#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/runtime/microkernel.hpp>

#include "test_utils.hpp"

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

} // namespace

int main() {
    try {
        set_env("MOTIFCL_MATMUL_BACKEND", "native");

        const auto selection = motifcl::select_microkernel_backend(motifcl::MicrokernelDomain::Matmul);
        if (selection.effective != motifcl::MicrokernelBackendKind::Native) {
            std::cerr << "native matmul backend was not selected\n";
            return 1;
        }
        const auto native = motifcl::selected_matmul_backend();
        if (native.kind != motifcl::MicrokernelBackendKind::Native || native.kernels.empty()) {
            std::cerr << "native matmul backend has no registered kernels\n";
            return 2;
        }

        auto backend = motifcl::Backend::create_opencl();

        constexpr int K = 31;
        constexpr int N = 17;
        std::vector<float> a(K);
        std::vector<float> b(K * N);
        for (int k = 0; k < K; ++k) a[k] = static_cast<float>((k % 9) - 4) * 0.03125f;
        for (int i = 0; i < K * N; ++i) b[i] = static_cast<float>((i % 13) - 6) * 0.015625f;

        auto A = motifcl::Tensor::from_cpu(backend, {1, K}, motifcl::DType::F32, a.data());
        auto B = motifcl::Tensor::from_cpu(backend, {K, N}, motifcl::DType::F32, b.data());

        motifcl::autograd::begin_graph_capture();
        auto C = motifcl::matmul(A, B);
        auto graph = motifcl::autograd::end_graph_capture();
        if (graph.empty() || graph.nodes()[0].op != "matmul_native_f32_m1") {
            std::cerr << "native matmul path was not used\n";
            return 3;
        }

        const auto c = C.to_vector<float>();
        for (int col = 0; col < N; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) expected += a[k] * b[k * N + col];
            if (std::fabs(c[col] - expected) > 1e-5f) {
                std::cerr << "native matmul mismatch at col " << col << ": got "
                          << c[col] << " expected " << expected << '\n';
                return 4;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
