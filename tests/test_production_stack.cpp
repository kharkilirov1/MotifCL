#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include <motifcl/motifcl.hpp>

#include "test_utils.hpp"

namespace {

bool near(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

void require_vec_near(const std::vector<float>& got, const std::vector<float>& expected, float eps = 1e-4f) {
    if (got.size() != expected.size()) {
        throw std::runtime_error("vector size mismatch");
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!near(got[i], expected[i], eps)) {
            throw std::runtime_error("vector value mismatch at " + std::to_string(i) +
                                     ": got " + std::to_string(got[i]) +
                                     " expected " + std::to_string(expected[i]));
        }
    }
}

} // namespace

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();

        motifcl::manual_seed(42);
        auto r1 = motifcl::Tensor::randn(backend, {2, 2}, 1.0f).to_vector<float>();
        motifcl::manual_seed(42);
        auto r2 = motifcl::Tensor::randn(backend, {2, 2}, 1.0f).to_vector<float>();
        require_vec_near(r1, r2);

        auto a = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32,
            std::vector<float>{1, 2, 3, 4, 5, 6}.data());
        auto b = motifcl::Tensor::from_cpu(backend, {3}, motifcl::DType::F32,
            std::vector<float>{10, 20, 30}.data());
        require_vec_near(motifcl::add(a, b).to_vector<float>(), {11, 22, 33, 14, 25, 36});
        require_vec_near(motifcl::mul(a, b).to_vector<float>(), {10, 40, 90, 40, 100, 180});
        require_vec_near(motifcl::slice_rows(a, 1, 2).to_vector<float>(), {4, 5, 6});
        if (!near(motifcl::sum_all(a).item(), 21.0f) || !near(motifcl::mean_all(a).item(), 3.5f)) {
            throw std::runtime_error("sum_all/mean_all mismatch");
        }

        auto gelu_bias = motifcl::Tensor::from_cpu(backend, {3}, motifcl::DType::F32,
            std::vector<float>{0.5f, -0.5f, 1.0f}.data());
        auto fused_gelu = motifcl::add_bias_gelu_rows(a, gelu_bias).to_vector<float>();
        auto gelu_ref = [](float v) {
            float t = 0.7978845608028654f * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + std::tanh(t));
        };
        require_vec_near(fused_gelu, {gelu_ref(1.5f), gelu_ref(1.5f), gelu_ref(4.0f),
                                      gelu_ref(4.5f), gelu_ref(4.5f), gelu_ref(7.0f)}, 1e-4f);

        if (motifcl::backend_supports_fp16(backend)) {
            auto ma = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32,
                std::vector<float>{1, 2, 3, 4, 5, 6}.data());
            auto mb = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32,
                std::vector<float>{7, 8, 9, 10, 11, 12}.data());
            auto ma_f16 = motifcl::cast_f32_to_f16(ma);
            require_vec_near(motifcl::cast_f16_to_f32(ma_f16).to_vector<float>(),
                             {1, 2, 3, 4, 5, 6}, 5e-3f);
            require_vec_near(motifcl::matmul_f16_accum_f32(ma_f16, motifcl::cast_f32_to_f16(mb)).to_vector<float>(),
                             {58, 64, 139, 154}, 5e-2f);
        }

        auto mask = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::I32,
            std::vector<std::int32_t>{0, 1, 0, 1, 0, 1}.data());
        require_vec_near(motifcl::masked_fill(a, mask, -99.0f).to_vector<float>(), {1, -99, 3, -99, 5, -99});

        motifcl::manual_seed(7);
        auto d1 = motifcl::dropout(a, 0.25f, true).to_vector<float>();
        motifcl::manual_seed(7);
        auto d2 = motifcl::dropout(a, 0.25f, true).to_vector<float>();
        require_vec_near(d1, d2);

        auto ag = motifcl::Tensor::from_cpu(backend, {2, 3}, motifcl::DType::F32,
            std::vector<float>{1, 2, 3, 4, 5, 6}.data());
        auto bg = motifcl::Tensor::from_cpu(backend, {3}, motifcl::DType::F32,
            std::vector<float>{10, 20, 30}.data());
        ag.set_requires_grad(true);
        bg.set_requires_grad(true);
        auto loss = motifcl::sum_all(motifcl::add(ag, bg));
        loss.backward();
        require_vec_near(ag.grad()->to_vector<float>(), {1, 1, 1, 1, 1, 1});
        require_vec_near(bg.grad()->to_vector<float>(), {2, 2, 2});

        const auto tensor_path = (std::filesystem::current_path() / "production_tensor.mclt").string();
        motifcl::save_tensor(a, tensor_path);
        require_vec_near(motifcl::load_tensor(backend, tensor_path).to_vector<float>(), a.to_vector<float>());
        std::filesystem::remove(tensor_path);

        motifcl::nn::Sequential model({
            std::make_shared<motifcl::nn::Linear>(backend, 3, 4),
            std::make_shared<motifcl::nn::GELU>(),
            std::make_shared<motifcl::nn::Linear>(backend, 4, 2)
        });
        const auto ckpt_path = (std::filesystem::current_path() / "production_params.mclp").string();
        motifcl::save_parameters(model.parameters(), ckpt_path);
        motifcl::load_parameters(model.parameters(), backend, ckpt_path);
        std::filesystem::remove(ckpt_path);

        motifcl::optim::SGD sgd(model.parameters(), 1e-2f);
        motifcl::train::StepLR step_lr(1e-2f, 2, 0.5f);
        sgd.set_lr(step_lr.lr_at(3));
        if (!near(sgd.lr(), 5e-3f)) throw std::runtime_error("StepLR/SGD lr mismatch");

        auto cap_a = motifcl::Tensor::ones(backend, {2, 2});
        motifcl::autograd::GraphCaptureGuard guard;
        auto cap_b = motifcl::add(cap_a, cap_a);
        (void)cap_b;
        auto graph = guard.finish();
        auto runtime_specs = graph.tensor_specs_list();
        auto runtime_plan = graph.compile_runtime_plan(runtime_specs);
        if (!runtime_plan.compatible || runtime_plan.bindings.empty()) {
            throw std::runtime_error("graph runtime plan did not compile");
        }

        std::cout << "production stack checks passed\n";
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
