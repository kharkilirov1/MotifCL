#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> ph = {1, 2, 3, 4};
        std::vector<float> gh = {0.1f, -0.2f, 0.3f, -0.4f};
        auto P = motifcl::Tensor::from_cpu(backend, {4}, motifcl::DType::F32, ph.data());
        auto G = motifcl::Tensor::from_cpu(backend, {4}, motifcl::DType::F32, gh.data());
        auto M = motifcl::Tensor::zeros(backend, {4});
        auto V = motifcl::Tensor::zeros(backend, {4});
        motifcl::adam_update(P, G, M, V, 1e-3f, 0.9f, 0.999f, 1e-8f, 1);
        auto out = P.to_vector<float>();
        for (float v : out) if (!std::isfinite(v)) return 1;

        std::vector<motifcl::nn::Parameter> params;
        std::vector<motifcl::nn::Parameter*> ptrs;
        params.reserve(5);
        ptrs.reserve(5);
        for (int i = 0; i < 5; ++i) {
            std::vector<float> value = {static_cast<float>(i + 1)};
            params.emplace_back(motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::F32, value.data()));
            ptrs.push_back(&params.back());
            std::vector<float> grad = {1.0f};
            auto G1 = motifcl::Tensor::from_cpu(backend, {1}, motifcl::DType::F32, grad.data());
            params.back().data._accumulate_grad(G1);
        }
        motifcl::optim::Adam opt(ptrs, 1e-2f, 0.9f, 0.999f, 1e-8f, 0.1f);
        if (std::fabs(opt.weight_decay() - 0.1f) > 1e-8f) return 2;
        opt.step();
        for (int i = 0; i < 5; ++i) {
            auto value = params[static_cast<std::size_t>(i)].data.to_vector<float>();
            const float p0 = static_cast<float>(i + 1);
            const float expected = p0 - 1e-2f - 1e-2f * 0.1f * p0;
            if (std::fabs(value[0] - expected) > 1e-4f) return 3;
        }

        std::vector<float> mp_w = {1.0f, 2.0f};
        auto MPW = motifcl::Tensor::from_cpu(backend, {2}, motifcl::DType::F32, mp_w.data());
        motifcl::nn::Parameter mp_param(MPW);
        std::vector<float> mp_grad = {1.0f, 1.0f};
        auto MPG = motifcl::Tensor::from_cpu(backend, {2}, motifcl::DType::F32, mp_grad.data());
        mp_param.data._set_grad(MPG);
        motifcl::optim::DynamicLossScaler scaler(8.0f, 2.0f, 0.5f, 2);
        scaler.unscale_({&mp_param});
        auto unscaled = mp_param.data.grad()->to_vector<float>();
        for (float value : unscaled) {
            if (std::fabs(value - 0.125f) > 1e-6f) return 4;
        }

        mp_param.data._set_grad(MPG);
        motifcl::optim::MixedPrecisionAdam mp_opt({&mp_param}, 0.1f);
        if (!mp_opt.step_scaled(scaler)) return 5;
        auto mp_after = mp_param.data.to_vector<float>();
        if (std::fabs(mp_after[0] - 0.9f) > 1e-4f) return 6;
        if (std::fabs(mp_after[1] - 1.9f) > 1e-4f) return 7;

        std::vector<float> inf_grad = {std::numeric_limits<float>::infinity(), 0.0f};
        auto InfG = motifcl::Tensor::from_cpu(backend, {2}, motifcl::DType::F32, inf_grad.data());
        mp_param.data._set_grad(InfG);
        if (!scaler.has_overflow({&mp_param})) return 8;
        float before_scale = scaler.scale();
        auto update = scaler.update(true);
        if (!update.found_inf || !(update.scale < before_scale)) return 9;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
