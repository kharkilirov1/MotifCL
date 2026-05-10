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

        constexpr int KM = 5;
        constexpr int KK = 512;
        constexpr int KN = 7;
        std::vector<float> k512_a(KM * KK);
        std::vector<float> k512_b(KK * KN);
        for (int i = 0; i < KM * KK; ++i) k512_a[i] = static_cast<float>((i % 13) - 6) * 0.01f;
        for (int i = 0; i < KK * KN; ++i) k512_b[i] = static_cast<float>((i % 17) - 8) * 0.015f;
        auto K512A = motifcl::Tensor::from_cpu(backend, {KM, KK}, motifcl::DType::F32, k512_a.data());
        auto K512B = motifcl::Tensor::from_cpu(backend, {KK, KN}, motifcl::DType::F32, k512_b.data());
        auto K512C = motifcl::matmul(K512A, K512B).to_vector<float>();
        for (int r = 0; r < KM; ++r) {
            for (int c = 0; c < KN; ++c) {
                float expected = 0.0f;
                for (int k = 0; k < KK; ++k) expected += k512_a[r * KK + k] * k512_b[k * KN + c];
                if (std::fabs(K512C[r * KN + c] - expected) > 2e-4f) return 2;
            }
        }

        std::vector<float> k512_at(KK * KM);
        std::vector<float> k512_bt(KN * KK);
        for (int r = 0; r < KM; ++r) {
            for (int k = 0; k < KK; ++k) k512_at[k * KM + r] = k512_a[r * KK + k];
        }
        for (int k = 0; k < KK; ++k) {
            for (int c = 0; c < KN; ++c) k512_bt[c * KK + k] = k512_b[k * KN + c];
        }
        auto K512AT = motifcl::Tensor::from_cpu(backend, {KK, KM}, motifcl::DType::F32, k512_at.data());
        auto K512BT = motifcl::Tensor::from_cpu(backend, {KN, KK}, motifcl::DType::F32, k512_bt.data());
        auto K512CTA = motifcl::matmul_transpose_a(K512AT, K512B).to_vector<float>();
        auto K512CTB = motifcl::matmul_transpose_b(K512A, K512BT).to_vector<float>();
        for (int i = 0; i < KM * KN; ++i) {
            if (std::fabs(K512CTA[i] - K512C[i]) > 2e-4f) return 3;
            if (std::fabs(K512CTB[i] - K512C[i]) > 2e-4f) return 4;
        }

        constexpr int TM = 64;
        constexpr int TN = 256;
        std::vector<float> tuned_a(TM * KK);
        std::vector<float> tuned_b(KK * TN);
        for (int i = 0; i < TM * KK; ++i) tuned_a[i] = static_cast<float>((i % 19) - 9) * 0.007f;
        for (int i = 0; i < KK * TN; ++i) tuned_b[i] = static_cast<float>((i % 23) - 11) * 0.006f;
        auto TunedA = motifcl::Tensor::from_cpu(backend, {TM, KK}, motifcl::DType::F32, tuned_a.data());
        auto TunedB = motifcl::Tensor::from_cpu(backend, {KK, TN}, motifcl::DType::F32, tuned_b.data());
        motifcl::autograd::begin_graph_capture();
        auto TunedC = motifcl::matmul(TunedA, TunedB);
        auto tuned_graph = motifcl::autograd::end_graph_capture();
        if (tuned_graph.empty() || tuned_graph.nodes()[0].op != "matmul_tuned_f32_t16") return 5;
        auto tuned_c = TunedC.to_vector<float>();
        for (int r = 0; r < TM; ++r) {
            for (int c = 0; c < TN; ++c) {
                float expected = 0.0f;
                for (int k = 0; k < KK; ++k) expected += tuned_a[r * KK + k] * tuned_b[k * TN + c];
                if (std::fabs(tuned_c[r * TN + c] - expected) > 4e-4f) return 6;
            }
        }
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
