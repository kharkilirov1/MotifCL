#include <cmath>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        const auto int_dot_mode = backend.int_dot_mode();
        if (int_dot_mode != "cl_khr_integer_dot_product" &&
            int_dot_mode != "cl_arm_integer_dot_product" &&
            int_dot_mode != "cl_intel_subgroups" &&
            int_dot_mode != "vendor_dot4_unrolled" &&
            int_dot_mode != "scalar_fallback") return 1;
        if (int_dot_mode != "scalar_fallback" && !backend.supports_integer_dot()) return 1;

        std::vector<float> values = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
        auto X = motifcl::Tensor::from_cpu(backend, {5}, motifcl::DType::F32, values.data());
        auto Q = motifcl::quantize_q8_symmetric(X, 0.01f);
        if (Q.dtype() != motifcl::DType::Q8_0) return 1;
        if (std::fabs(Q.quant_scale() - 0.01f) > 1e-8f) return 1;
        auto D = motifcl::dequantize_q8(Q).to_vector<float>();
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (std::fabs(D[i] - values[i]) > 0.011f) return 1;
        }

        auto Q4 = motifcl::quantize_q4_symmetric(X, 0.25f);
        if (Q4.dtype() != motifcl::DType::Q4_0) return 1;
        if (Q4.nbytes() != 3) return 1;
        if (std::fabs(Q4.quant_scale() - 0.25f) > 1e-8f) return 1;
        auto D4 = motifcl::dequantize_q4(Q4).to_vector<float>();
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (std::fabs(D4[i] - values[i]) > 0.126f) return 1;
        }

        std::vector<float> a = {1.0f, 2.0f, -3.0f, 4.0f, 5.0f, -6.0f};
        std::vector<float> b = {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
        auto A = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, a.data());
        auto B = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32, b.data());
        auto AQ = motifcl::quantize_q8_symmetric(A, 0.1f);
        auto BQ = motifcl::quantize_q8_symmetric(B, 0.1f);
        auto C = motifcl::matmul(AQ, BQ).to_vector<float>();
        std::vector<float> ref = {22.0f, -12.0f, 49.0f, -24.0f};
        for (std::size_t i = 0; i < ref.size(); ++i) {
            if (std::fabs(C[i] - ref[i]) > 0.15f) return 1;
        }

        std::vector<float> dot_a = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f};
        std::vector<float> dot_b = {1.0f, 2.0f, -3.0f, -4.0f, 5.0f, 6.0f, -7.0f, -8.0f};
        auto DA = motifcl::Tensor::from_cpu(backend, {2, 4}, motifcl::DType::F32, dot_a.data());
        auto DB = motifcl::Tensor::from_cpu(backend, {4, 2}, motifcl::DType::F32, dot_b.data());
        auto DQ8A = motifcl::quantize_q8_symmetric(DA, 1.0f);
        auto DQ8B = motifcl::quantize_q8_symmetric(DB, 1.0f);
        auto DC = motifcl::matmul(DQ8A, DQ8B).to_vector<float>();
        std::vector<float> dot_ref = {50.0f, 60.0f, 114.0f, 140.0f};
        for (std::size_t i = 0; i < dot_ref.size(); ++i) {
            if (std::fabs(DC[i] - dot_ref[i]) > 1e-4f) return 1;
        }

        auto AQ4 = motifcl::quantize_q4_symmetric(A, 1.0f);
        auto BQ4 = motifcl::quantize_q4_symmetric(B, 1.0f);
        auto C4 = motifcl::matmul(AQ4, BQ4).to_vector<float>();
        for (std::size_t i = 0; i < ref.size(); ++i) {
            if (std::fabs(C4[i] - ref[i]) > 0.15f) return 1;
        }

        auto C84 = motifcl::matmul(AQ, BQ4).to_vector<float>();
        auto C48 = motifcl::matmul(AQ4, BQ).to_vector<float>();
        for (std::size_t i = 0; i < ref.size(); ++i) {
            if (std::fabs(C84[i] - ref[i]) > 0.15f) return 1;
            if (std::fabs(C48[i] - ref[i]) > 0.15f) return 1;
        }

        std::vector<float> a_exact = {1.0f, 2.0f, -7.0f, 4.0f, 6.0f, -14.0f};
        std::vector<float> b_exact = {1.0f, -2.0f, 3.0f, 4.0f, -7.0f, 14.0f};
        auto A2 = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, a_exact.data());
        auto B2 = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32, b_exact.data());
        std::vector<float> ref_exact = {56.0f, -92.0f, 120.0f, -180.0f};

        auto A2Q4Rows = motifcl::quantize_q4_symmetric_rows(A2);
        auto B2Q4Cols = motifcl::quantize_q4_symmetric_cols(B2);
        if (!A2Q4Rows.has_quant_scales() || A2Q4Rows.quant_scale_axis() != 0) return 1;
        if (!B2Q4Cols.has_quant_scales() || B2Q4Cols.quant_scale_axis() != 1) return 1;
        auto A2D4 = motifcl::dequantize_q4(A2Q4Rows).to_vector<float>();
        auto B2D4 = motifcl::dequantize_q4(B2Q4Cols).to_vector<float>();
        for (std::size_t i = 0; i < a_exact.size(); ++i) {
            if (std::fabs(A2D4[i] - a_exact[i]) > 1e-4f) return 1;
            if (std::fabs(B2D4[i] - b_exact[i]) > 1e-4f) return 1;
        }
        auto CAxis = motifcl::matmul(A2Q4Rows, B2Q4Cols).to_vector<float>();
        for (std::size_t i = 0; i < ref_exact.size(); ++i) {
            if (std::fabs(CAxis[i] - ref_exact[i]) > 0.15f) return 1;
        }

        auto A2Q8Rows = motifcl::quantize_q8_symmetric_rows(A2);
        auto B2Q4ColsMixed = motifcl::quantize_q4_symmetric_cols(B2);
        auto CAxisMixed = motifcl::matmul(A2Q8Rows, B2Q4ColsMixed).to_vector<float>();
        for (std::size_t i = 0; i < ref_exact.size(); ++i) {
            if (std::fabs(CAxisMixed[i] - ref_exact[i]) > 0.35f) return 1;
        }

        std::vector<float> b_block = {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, 6.0f};
        auto B3 = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32, b_block.data());
        auto ABlock = motifcl::quantize_q4_symmetric_blocks(A2, 3);
        auto BScalar = motifcl::quantize_q4_symmetric(B3, 1.0f);
        if (!ABlock.has_quant_scales() || ABlock.quant_scale_axis() != 2 || ABlock.quant_block_size() != 3) return 1;
        auto CBlock = motifcl::matmul(ABlock, BScalar).to_vector<float>();
        std::vector<float> ref_block = {42.0f, -36.0f, 92.0f, -68.0f};
        for (std::size_t i = 0; i < ref_block.size(); ++i) {
            if (std::fabs(CBlock[i] - ref_block[i]) > 0.15f) return 1;
        }
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
