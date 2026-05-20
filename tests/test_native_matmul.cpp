#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/runtime/microkernel.hpp>
#include <motifcl/runtime/native_matmul.hpp>

#include "test_utils.hpp"

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void pack_q4_0(std::vector<std::uint8_t>& packed, int idx, int q) {
    const int code = q + 8;
    if ((idx & 1) == 0) {
        packed[static_cast<std::size_t>(idx >> 1)] =
            static_cast<std::uint8_t>((packed[static_cast<std::size_t>(idx >> 1)] & 0xf0u) | code);
    } else {
        packed[static_cast<std::size_t>(idx >> 1)] =
            static_cast<std::uint8_t>((packed[static_cast<std::size_t>(idx >> 1)] & 0x0fu) | (code << 4));
    }
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

        const std::string kernel_name = motifcl::native::matmul_f32_m1_kernel_name();
        if (kernel_name.empty()) {
            std::cerr << "native matmul kernel name must not be empty\n";
            return 3;
        }

        std::vector<float> scalar_core(N, 0.0f);
        motifcl::native::matmul_f32_m1_scalar(a.data(), b.data(), scalar_core.data(), K, N);

        std::vector<float> native_core(N, 0.0f);
        motifcl::native::matmul_f32_m1(a.data(), b.data(), native_core.data(), K, N);
        for (int col = 0; col < N; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) expected += a[k] * b[k * N + col];
            if (std::fabs(scalar_core[col] - expected) > 1e-6f) {
                std::cerr << "native scalar core mismatch at col " << col << ": got "
                          << scalar_core[col] << " expected " << expected << '\n';
                return 3;
            }
            if (std::fabs(native_core[col] - scalar_core[col]) > 1e-6f) {
                std::cerr << "native dispatch mismatch at col " << col << ": got "
                          << native_core[col] << " scalar " << scalar_core[col] << '\n';
                return 4;
            }
        }

        auto A = motifcl::Tensor::from_cpu(backend, {1, K}, motifcl::DType::F32, a.data());
        auto B = motifcl::Tensor::from_cpu(backend, {K, N}, motifcl::DType::F32, b.data());

        motifcl::autograd::begin_graph_capture();
        auto C = motifcl::matmul(A, B);
        auto graph = motifcl::autograd::end_graph_capture();
        if (graph.empty() || graph.nodes()[0].op != "matmul_native_f32_m1") {
            std::cerr << "native matmul path was not used\n";
            return 5;
        }

        const auto c = C.to_vector<float>();
        for (int col = 0; col < N; ++col) {
            float expected = 0.0f;
            for (int k = 0; k < K; ++k) expected += a[k] * b[k * N + col];
            if (std::fabs(c[col] - expected) > 1e-5f) {
                std::cerr << "native matmul mismatch at col " << col << ": got "
                          << c[col] << " expected " << expected << '\n';
                return 6;
            }
        }

        constexpr int QK = 33;
        constexpr int QN = 19;
        constexpr float q_scale = 0.03125f;
        std::vector<float> qa(QK);
        std::vector<std::uint8_t> qb(static_cast<std::size_t>((QK * QN + 1) / 2), 0);
        std::vector<float> q_expected(QN, 0.0f);
        for (int k = 0; k < QK; ++k) qa[k] = static_cast<float>((k % 11) - 5) * 0.015625f;
        for (int k = 0; k < QK; ++k) {
            for (int col = 0; col < QN; ++col) {
                const int idx = k * QN + col;
                const int q = ((k * 7 + col * 3) % 16) - 8;
                pack_q4_0(qb, idx, q);
                q_expected[col] += qa[k] * static_cast<float>(q) * q_scale;
            }
        }

        const std::string q4_kernel_name = motifcl::native::matmul_f32_q4_0_m1_kernel_name();
        if (q4_kernel_name.empty()) {
            std::cerr << "native q4 matmul kernel name must not be empty\n";
            return 7;
        }

        std::vector<float> q_scalar(QN, 0.0f);
        motifcl::native::matmul_f32_q4_0_m1_scalar(qa.data(), qb.data(), q_scalar.data(), QK, QN, q_scale);
        std::vector<float> q_native(QN, 0.0f);
        motifcl::native::matmul_f32_q4_0_m1(qa.data(), qb.data(), q_native.data(), QK, QN, q_scale);
        for (int col = 0; col < QN; ++col) {
            if (std::fabs(q_scalar[col] - q_expected[col]) > 1e-6f) {
                std::cerr << "native q4 scalar mismatch at col " << col << ": got "
                          << q_scalar[col] << " expected " << q_expected[col] << '\n';
                return 8;
            }
            if (std::fabs(q_native[col] - q_scalar[col]) > 1e-6f) {
                std::cerr << "native q4 dispatch mismatch at col " << col << ": got "
                          << q_native[col] << " scalar " << q_scalar[col] << '\n';
                return 9;
            }
        }

        auto QA = motifcl::Tensor::from_cpu(backend, {1, QK}, motifcl::DType::F32, qa.data());
        auto QB = motifcl::Tensor::from_cpu(backend, {QK, QN}, motifcl::DType::Q4_0, qb.data());
        QB._set_quant_scale(q_scale);

        motifcl::autograd::begin_graph_capture();
        auto QC = motifcl::matmul(QA, QB);
        auto qgraph = motifcl::autograd::end_graph_capture();
        if (qgraph.empty() || qgraph.nodes()[0].op != "matmul_native_f32_q4_0_m1") {
            std::cerr << "native q4 matmul path was not used\n";
            return 10;
        }

        const auto qc = QC.to_vector<float>();
        for (int col = 0; col < QN; ++col) {
            if (std::fabs(qc[col] - q_expected[col]) > 1e-5f) {
                std::cerr << "native q4 tensor mismatch at col " << col << ": got "
                          << qc[col] << " expected " << q_expected[col] << '\n';
                return 11;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
