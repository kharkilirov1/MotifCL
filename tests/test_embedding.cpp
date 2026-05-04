#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <motifcl/motifcl.hpp>
#include "test_utils.hpp"

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        std::vector<float> weights = {1,2, 3,4, 5,6};
        std::vector<std::int32_t> ids = {2, 0, 1};
        auto W = motifcl::Tensor::from_cpu(backend, {3, 2}, motifcl::DType::F32, weights.data());
        auto I = motifcl::Tensor::from_cpu(backend, {3}, motifcl::DType::I32, ids.data());
        auto Y = motifcl::nn::token_position_embedding(I, W, motifcl::Tensor::zeros(backend, {3, 2})).to_vector<float>();
        std::vector<float> ref = {5,6, 1,2, 3,4};
        for (std::size_t i = 0; i < ref.size(); ++i) if (std::fabs(Y[i] - ref[i]) > 1e-6f) return 1;

        W.set_requires_grad(true);
        auto P = motifcl::Tensor::zeros(backend, {3, 2});
        P.set_requires_grad(true);
        auto E = motifcl::nn::token_position_embedding(I, W, P);
        auto loss = motifcl::mse_loss(E, motifcl::Tensor::zeros(backend, {3, 2}));
        loss.backward();
        if (!W.grad() || !P.grad()) return 1;
        for (float v : W.grad()->to_vector<float>()) if (!std::isfinite(v)) return 1;
        for (float v : P.grad()->to_vector<float>()) if (!std::isfinite(v)) return 1;

        motifcl::nn::Embedding emb(backend, 4, 2);
        auto EY = emb.forward(I);
        auto eloss = motifcl::mse_loss(EY, motifcl::Tensor::zeros(backend, {3, 2}));
        eloss.backward();
        if (!emb.weight.grad()) return 1;
        for (float v : emb.weight.grad()->to_vector<float>()) if (!std::isfinite(v)) return 1;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
