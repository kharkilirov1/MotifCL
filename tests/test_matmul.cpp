#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> a = {1, 2, 3, 4, 5, 6};
        std::vector<float> b = {7, 8, 9, 10, 11, 12};
        std::vector<float> ref = {58, 64, 139, 154};
        auto A = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, a.data());
        auto B = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32, b.data());
        auto C = motifcl::matmul(A, B).to_vector<float>();
        for (int i = 0; i < 4; ++i) if (std::fabs(C[i] - ref[i]) > 1e-4f) return 1;
        motifcl::autograd::begin_graph_capture();
        auto Captured = motifcl::matmul(A, B);
        (void)Captured;
        auto graph = motifcl::autograd::end_graph_capture();
        if (graph.empty() || graph.nodes()[0].op != "matmul_register_block4_f32") return 1;
        for (int tile : {4, 8, 16}) {
            auto V = motifcl::matmul_tiled_variant(A, B, tile).to_vector<float>();
            for (int i = 0; i < 4; ++i) if (std::fabs(V[i] - ref[i]) > 1e-4f) return 1;
        }

        constexpr int M = 33;
        constexpr int K = 17;
        constexpr int N = 35;
        std::vector<float> big_a(M * K);
        std::vector<float> big_b(K * N);
        for (int i = 0; i < M * K; ++i) big_a[i] = static_cast<float>((i % 11) - 5) * 0.125f;
        for (int i = 0; i < K * N; ++i) big_b[i] = static_cast<float>((i % 7) - 3) * 0.2f;
        auto BigA = motifcl::Tensor::from_cpu(backend, {M, K}, motifcl::DType::F32, big_a.data());
        auto BigB = motifcl::Tensor::from_cpu(backend, {K, N}, motifcl::DType::F32, big_b.data());
        auto BigC = motifcl::matmul(BigA, BigB).to_vector<float>();
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                float expected = 0.0f;
                for (int k = 0; k < K; ++k) expected += big_a[r * K + k] * big_b[k * N + c];
                if (std::fabs(BigC[r * N + c] - expected) > 1e-4f) return 1;
            }
        }
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
