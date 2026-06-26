// Memory truth gate for the finite-state counter backward.
//
// The persistent state is compact by construction (packed 6-bit), but the *training
// peak* is only memory-native if the backward never (a) carries a decoded dense
// weight per layer across forward->backward, nor (b) materializes a dense [out,in]
// weight-gradient. This test fails the moment either leak reappears, so the
// "0.75 byte/weight training peak" claim cannot silently regress via one line.
//
// It checks three things on a multi-layer stack driven by a RAW input (no
// requires_grad), which also exercises that a counter layer updates itself even
// when its input asks for no gradient:
//   1. peak live dense [out,in] weights across the whole backward <= 1
//   2. dense [out,in] grad_w allocations == 0   (fused path forms grad_w in-kernel)
//   3. the stack actually learns (loss strictly decreases) through that path
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/ops/loss.hpp>
#include <motifcl/autograd/node.hpp>

#include "test_utils.hpp"

using motifcl::nn::CounterStateLinear;

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();

        const int N = 64, d = 16, depth = 3, C = 11, steps = 200;
        const float teacher_scale = 0.25f, lr = 0.005f;

        // Ternary teacher (single linear map) so the stack has something to fit.
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> tern(-1, 1);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<int> tw(d * d);
        for (int& w : tw) w = tern(rng);
        std::vector<float> xh(N * d);
        for (float& xi : xh) xi = gauss(rng);
        std::vector<float> yh(N * d, 0.0f);
        for (int r = 0; r < N; ++r)
            for (int o = 0; o < d; ++o) {
                float a = 0.0f;
                for (int i = 0; i < d; ++i) a += xh[r * d + i] * (teacher_scale * (float)tw[o * d + i]);
                yh[r * d + o] = a;
            }

        // RAW input: deliberately NOT marking requires_grad. With the old node this
        // would have skipped the update entirely; the layer must still self-update.
        auto x = motifcl::Tensor::from_cpu(backend, {N, d}, motifcl::DType::F32, xh.data());
        auto y = motifcl::Tensor::from_cpu(backend, {N, d}, motifcl::DType::F32, yh.data());

        const float base = std::sqrt(3.0f / (2.0f * (float)d));
        std::vector<std::unique_ptr<CounterStateLinear>> layers;
        layers.reserve(depth);
        for (int l = 0; l < depth; ++l)
            layers.push_back(std::make_unique<CounterStateLinear>(
                backend, d, d, C, lr, 0.0f, teacher_scale / base,
                0.9f, 1e-3f, 1234u + static_cast<std::uint32_t>(l)));

        auto run_forward = [&]() {
            motifcl::Tensor h = x;
            for (auto& layer : layers) h = layer->forward(h);
            return h;
        };

        // --- measured step: reset counters, one forward + backward over the stack ---
        CounterStateLinear::reset_memory_counters();
        {
            auto pred = run_forward();
            auto loss = motifcl::mse_loss(pred, y);
            loss.backward();
        }
        const int peak_w = CounterStateLinear::peak_live_dense_weight();
        const int gradw = CounterStateLinear::dense_gradw_allocations();

        double loss0 = 0.0;
        {
            motifcl::autograd::set_enabled(false);
            auto pred = run_forward();
            auto ph = pred.to_vector<float>();
            motifcl::autograd::set_enabled(true);
            for (int k = 0; k < N * d; ++k) { double e = ph[k] - yh[k]; loss0 += e * e; }
            loss0 /= (double)(N * d);
        }

        // --- learning check: keep training, loss must drop through the fused path ---
        for (int s = 0; s < steps; ++s) {
            auto pred = run_forward();
            auto loss = motifcl::mse_loss(pred, y);
            loss.backward();
        }
        double loss1 = 0.0;
        {
            motifcl::autograd::set_enabled(false);
            auto pred = run_forward();
            auto ph = pred.to_vector<float>();
            motifcl::autograd::set_enabled(true);
            for (int k = 0; k < N * d; ++k) { double e = ph[k] - yh[k]; loss1 += e * e; }
            loss1 /= (double)(N * d);
        }

        bool ok = true;
        if (peak_w > 1) {
            std::cerr << "MEMORY LEAK: peak live dense [out,in] weights = " << peak_w
                      << " (> 1) -- a weight is being saved per layer across backward\n";
            ok = false;
        }
        if (gradw != 0) {
            std::cerr << "MEMORY LEAK: dense [out,in] grad_w allocations = " << gradw
                      << " (!= 0) -- the backward is materializing a full weight-gradient\n";
            ok = false;
        }
        if (!(loss1 < loss0)) {
            std::cerr << "raw-input stack did not learn: loss " << loss0 << " -> " << loss1
                      << " (counter layers must self-update even when input needs no grad)\n";
            ok = false;
        }

        if (!ok) return 1;
        std::cout << "memory truth gate PASS: peak_w=" << peak_w << " grad_w_allocs=" << gradw
                  << " loss " << loss0 << " -> " << loss1 << "\n";
        return 0;
    } catch (const std::exception& e) {
        return motifcl_test::handle_exception(e);
    }
}
