#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> x = {1, 2, 3, 1, 1, 1};
        auto X = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32, x.data());
        auto Y = motifcl::softmax_rows(X).to_vector<float>();
        float row0 = Y[0] + Y[1] + Y[2];
        float row1 = Y[3] + Y[4] + Y[5];
        if (std::fabs(row0 - 1.0f) > 1e-4f || std::fabs(row1 - 1.0f) > 1e-4f) return 1;

        auto RS = motifcl::rowwise_sum(X).to_vector<float>();
        auto RM = motifcl::rowwise_max(X).to_vector<float>();
        if (std::fabs(RS[0] - 6.0f) > 1e-4f || std::fabs(RS[1] - 3.0f) > 1e-4f) return 1;
        if (std::fabs(RM[0] - 3.0f) > 1e-4f || std::fabs(RM[1] - 1.0f) > 1e-4f) return 1;

        constexpr int Rows = 3;
        constexpr int Cols = 513;
        std::vector<float> big(Rows * Cols);
        for (int i = 0; i < Rows * Cols; ++i) big[i] = static_cast<float>((i % 19) - 9) * 0.1f;
        auto Big = motifcl::Tensor::from_cpu(backend, {Rows, Cols}, motifcl::DType::F32, big.data());
        auto BigS = motifcl::rowwise_sum(Big).to_vector<float>();
        auto BigM = motifcl::rowwise_max(Big).to_vector<float>();
        motifcl::autograd::begin_graph_capture();
        auto captured_sum = motifcl::rowwise_sum(Big);
        (void)captured_sum;
        auto graph = motifcl::autograd::end_graph_capture();
        if (backend.device_info().max_work_group_size >= 256 &&
            (graph.empty() || graph.nodes()[0].op != "rowwise_sum_wg_f32")) return 1;
        for (int r = 0; r < Rows; ++r) {
            float s = 0.0f;
            float m = -1.0e30f;
            for (int c = 0; c < Cols; ++c) {
                float v = big[r * Cols + c];
                s += v;
                m = std::max(m, v);
            }
            if (std::fabs(BigS[r] - s) > 1e-4f) return 1;
            if (std::fabs(BigM[r] - m) > 1e-4f) return 1;
        }
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
