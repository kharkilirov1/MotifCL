#include <cmath>
#include <algorithm>
#include <cstdint>
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

        constexpr int CK = 64;
        constexpr int CN = 5;
        constexpr int CB = 32;
        std::vector<float> col_a(CK);
        std::vector<float> col_b(CK * CN);
        for (int i = 0; i < CK; ++i) col_a[i] = static_cast<float>((i % 29) - 14) * 0.013f;
        for (int i = 0; i < CK * CN; ++i) col_b[i] = static_cast<float>((i % 31) - 15) * 0.017f;
        const int blocks_per_col = (CK + CB - 1) / CB;
        std::vector<std::uint8_t> col_q(static_cast<std::size_t>(CK * CN + 1) / 2, 0);
        std::vector<float> col_scales(CN * blocks_per_col, 1.0f);
        std::vector<float> col_expected(CN, 0.0f);
        for (int c = 0; c < CN; ++c) {
            for (int kb = 0; kb < blocks_per_col; ++kb) {
                const int begin = kb * CB;
                const int end = std::min(CK, begin + CB);
                float max_abs = 0.0f;
                for (int k = begin; k < end; ++k) {
                    max_abs = std::max(max_abs, std::fabs(col_b[k * CN + c]));
                }
                const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
                col_scales[c * blocks_per_col + kb] = scale;
                for (int k = begin; k < end; ++k) {
                    int q = static_cast<int>(std::lrint(col_b[k * CN + c] / scale));
                    q = std::max(-7, std::min(7, q));
                    const auto code = static_cast<std::uint8_t>((q + 8) & 0x0f);
                    const int pos = c * CK + k;
                    if ((pos & 1) == 0) col_q[static_cast<std::size_t>(pos >> 1)] =
                        static_cast<std::uint8_t>((col_q[static_cast<std::size_t>(pos >> 1)] & 0xf0u) | code);
                    else col_q[static_cast<std::size_t>(pos >> 1)] =
                        static_cast<std::uint8_t>((col_q[static_cast<std::size_t>(pos >> 1)] & 0x0fu) | (code << 4));
                    col_expected[c] += col_a[k] * static_cast<float>(q) * scale;
                }
            }
        }
        auto ColA = motifcl::Tensor::from_cpu(backend, {1, CK}, motifcl::DType::F32, col_a.data());
        auto ColB = motifcl::Tensor::from_cpu(backend, {CK, CN}, motifcl::DType::Q4_0_COL, col_q.data());
        auto ColS = motifcl::Tensor::from_cpu(backend, {static_cast<int64_t>(col_scales.size())}, motifcl::DType::F32, col_scales.data());
        ColB._set_quant_scales(ColS, 3, CB);
        auto ColC = motifcl::matmul(ColA, ColB).to_vector<float>();
        for (int c = 0; c < CN; ++c) {
            if (std::fabs(ColC[c] - col_expected[c]) > 2e-4f) return 7;
        }

        constexpr int TN8 = 13;
        std::vector<float> tile_b(CK * TN8);
        for (int i = 0; i < CK * TN8; ++i) tile_b[i] = static_cast<float>((i % 37) - 18) * 0.011f;
        std::vector<std::uint8_t> tile_q(static_cast<std::size_t>(CK * TN8 + 1) / 2, 0);
        std::vector<float> tile_scales(TN8 * blocks_per_col, 1.0f);
        std::vector<float> tile_expected(TN8, 0.0f);
        constexpr int TCOL = 8;
        for (int c0 = 0; c0 < TN8; c0 += TCOL) {
            const int cols_in_tile = std::min(TCOL, TN8 - c0);
            const int tile = c0 / TCOL;
            for (int kb = 0; kb < blocks_per_col; ++kb) {
                const int begin = kb * CB;
                for (int tc = 0; tc < cols_in_tile; ++tc) {
                    const int c = c0 + tc;
                    float max_abs = 0.0f;
                    for (int kk = 0; kk < CB; ++kk) {
                        const int k = begin + kk;
                        max_abs = std::max(max_abs, std::fabs(tile_b[k * TN8 + c]));
                    }
                    const float scale = max_abs > 0.0f ? max_abs / 7.0f : 1.0f;
                    tile_scales[tile * blocks_per_col * TCOL + kb * cols_in_tile + tc] = scale;
                }
                for (int kk = 0; kk < CB; ++kk) {
                    const int k = begin + kk;
                    for (int tc = 0; tc < cols_in_tile; ++tc) {
                        const int c = c0 + tc;
                        const float scale = tile_scales[tile * blocks_per_col * TCOL + kb * cols_in_tile + tc];
                        int q = static_cast<int>(std::lrint(tile_b[k * TN8 + c] / scale));
                        q = std::max(-7, std::min(7, q));
                        const auto code = static_cast<std::uint8_t>((q + 8) & 0x0f);
                        const int pos = tile * blocks_per_col * CB * TCOL + (kb * CB + kk) * cols_in_tile + tc;
                        if ((pos & 1) == 0) tile_q[static_cast<std::size_t>(pos >> 1)] =
                            static_cast<std::uint8_t>((tile_q[static_cast<std::size_t>(pos >> 1)] & 0xf0u) | code);
                        else tile_q[static_cast<std::size_t>(pos >> 1)] =
                            static_cast<std::uint8_t>((tile_q[static_cast<std::size_t>(pos >> 1)] & 0x0fu) | (code << 4));
                        tile_expected[c] += col_a[k] * static_cast<float>(q) * scale;
                    }
                }
            }
        }
        auto TileB = motifcl::Tensor::from_cpu(backend, {CK, TN8}, motifcl::DType::Q4_0_COL, tile_q.data());
        auto TileS = motifcl::Tensor::from_cpu(backend, {static_cast<int64_t>(tile_scales.size())}, motifcl::DType::F32, tile_scales.data());
        TileB._set_quant_scales(TileS, 4, CB);
        auto TileC = motifcl::matmul(ColA, TileB).to_vector<float>();
        for (int c = 0; c < TN8; ++c) {
            if (std::fabs(TileC[c] - tile_expected[c]) > 2e-4f) return 8;
        }
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
