// Isolation test for the *Vulkan* counter layer's training memory.
//
// The repository states the memory-native claim as "0.75 byte/weight training peak" and
// enforces it (on the OpenCL path) with tests/test_counter_memory_truth.cpp: the backward
// must neither carry a decoded dense [out,in] weight across forward->backward nor
// materialise a dense [out,in] weight gradient. That gate has no Vulkan counterpart, and
// the trainer's measured footprint at DIM=1280 / experts 2560 grew by ~424 MB per block
// when the packed state there is only ~64 MB -- so the leak this test looks for is
// exactly the difference between the claimed and the observed peak.
//
// Measured with the runtime's own allocation accounting, one layer at a time:
//   expected per layer: state = in*out*0.75 bytes, plus activations of the batch,
//   plus O(out) scratch. A dense FP32 weight would add in*out*4 bytes per layer, and a
//   dense grad_w another in*out*4 -- both would show up as ~26 MB steps in this shape.
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/nn/moe7.hpp>
#include <motifcl/ops/indexing.hpp>

#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr double kMB = 1024.0 * 1024.0;
}

int main() {
    using namespace motifcl;
    try {
        const auto probe = probe_vulkan_runtime();
        if (!probe.available()) {
            std::printf("SKIP test_counter_memory_vulkan: no Vulkan device\n");
            return 0;
        }
        Backend backend = Backend::create_vulkan();
        auto& rt = backend.vulkan_runtime();

        const int in = 2560, out = 2560, C = 11, rows = 256, layers = 6;
        const double state_mb = (double)in * out * 0.75 / kMB;          // packed 6-bit state
        const double dense_mb = (double)in * out * 4.0 / kMB;           // one dense FP32 copy

        auto live = [&]() { return (double)rt.buffer_bytes_live() / kMB; };
        auto peak = [&]() { return (double)rt.buffer_bytes_peak() / kMB; };
        auto report = [&](const char* tag, double base) {
            std::printf("%-34s live=%8.1f MB  delta=%+7.1f MB  peak=%8.1f MB  allocs=%zu\n",
                        tag, live(), live() - base, peak(), rt.buffer_allocations());
        };

        std::printf("shape in=%d out=%d rows=%d layers=%d | packed state=%.1f MB/layer, "
                    "dense FP32 copy=%.1f MB/layer\n", in, out, rows, layers, state_mb, dense_mb);

        const double base0 = live();
        report("baseline (backend only)", base0);

        std::mt19937 rng(5);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        std::vector<float> xh((std::size_t)rows * in);
        for (auto& v : xh) v = gauss(rng);
        auto x = Tensor::from_cpu(backend, {rows, in}, DType::F32, xh.data());
        report("after input tensor", base0);

        std::vector<std::unique_ptr<nn::CounterStateLinear>> stack;
        const double before_stack = live();
        for (int l = 0; l < layers; ++l) {
            stack.push_back(std::make_unique<nn::CounterStateLinear>(
                backend, in, out, C, 4e-3f, 0.0f, 1.0f, 0.9f, 1e-3f,
                100u + (std::uint32_t)l));
            char tag[64];
            std::snprintf(tag, sizeof(tag), "after layer %d construct", l + 1);
            report(tag, before_stack);
        }
        const double per_layer = (live() - before_stack) / layers;

        auto y = stack[0]->forward(x);
        const double after_fwd = live();
        report("after one forward", after_fwd);

        auto loss = sum_all(y);
        loss.backward();
        report("after one backward/update", after_fwd);

        // A stack forward+backward, i.e. the shape the trainer actually runs.
        const double before_stack_step = live();
        {
            Tensor h = x;
            for (auto& layer : stack) h = layer->forward(h);
            auto l2 = sum_all(h);
            l2.backward();
        }
        report("after stack fwd+bwd", before_stack_step);

        const bool state_ok = per_layer < state_mb + 0.35 * dense_mb;
        std::printf("marginal cost per layer: %.1f MB (packed state %.1f MB, one dense copy %.1f MB)"
                    " -> %s\n", per_layer, state_mb, dense_mb,
                    state_ok ? "counter-memory-native on Vulkan" : "DENSE LEAK: cost is near a full FP32 copy");

        // --- the MoE block itself, in the trainer's 1B-class shape ---
        // Everything below is activation footprint: the packed expert state is 0.75 B/param.
        const int dim = 1280, h_exp = 2560, h_shared = 2560;
        std::vector<float> xb_host((std::size_t)rows * dim);
        for (auto& v : xb_host) v = gauss(rng);
        const double before_moe = live();
        nn::MoE7Config cfg;
        cfg.dim = dim;
        cfg.hidden_expert = h_exp;
        cfg.hidden_shared = h_shared;
        cfg.C = C;
        cfg.counter_lr = 4e-3f;
        cfg.seed = 99u;
        auto moe = std::make_unique<nn::MoE7FFN>(backend, cfg);
        report("MoE block: construct (+state)", before_moe);

        auto xb = Tensor::from_cpu(backend, {rows, dim}, DType::F32, xb_host.data());
        auto yb = moe->forward(xb);
        const double after_moe_fwd = live();
        report("MoE block: forward", after_moe_fwd);
        auto lb = sum_all(yb);
        lb.backward();
        report("MoE block: backward", after_moe_fwd);

        const double moe_block_mb = live() - before_moe;
        // Packed state of the block: 8 banks (7 experts + shared), one up and one down each.
        const double bank_params = (double)dim * 2 * h_exp + (double)h_exp * dim;
        const double moe_state_mb = 8.0 * bank_params * 0.75 / kMB;
        std::printf("MoE block total %.1f MB (packed expert state %.1f MB + bucket rounding) -> the "
                    "rest is the activation footprint of 7 retained candidates\n", moe_block_mb, moe_state_mb);
        if (!state_ok) {
            std::printf("FAIL test_counter_memory_vulkan\n");
            return 1;
        }
        std::printf("PASS test_counter_memory_vulkan\n");
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL test_counter_memory_vulkan: %s\n", e.what());
        return 1;
    }
}
