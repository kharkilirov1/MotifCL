#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int verify_workspace_growth() {
    using namespace motifcl;
    auto runtime = VulkanRuntime::create();
    if (!runtime.available()) return 77;
    const auto bytes_before = runtime.buffer_bytes_live();
    const auto allocations_before = runtime.buffer_allocations();
    std::vector<VulkanBuffer> retained;
    retained.reserve(65);
    require(runtime.batch_begin(), "workspace growth batch begin failed");
    for (std::size_t i = 0; i < 65; ++i) {
        const std::size_t requested = 512 * (256 + 4 * i) * sizeof(float);
        retained.push_back(runtime.create_scratch_buffer("growth", requested));
        require(retained.back().nbytes() >= requested,
                "growing workspace returned an undersized logical capacity");
    }
    const auto bytes_delta = runtime.buffer_bytes_live() - bytes_before;
    const auto allocations_delta = runtime.buffer_allocations() - allocations_before;
    const auto submitted = runtime.batch_end();
    require(submitted.success, "workspace growth empty batch end failed");
    // Keep all handles alive, just as recorded dispatches retain their buffers.
    // Only the first 512 KiB and the next 1 MiB bucket may be allocated; the
    // other 63 requests must reuse the latter capacity, not allocate it again.
    constexpr std::size_t limit = 512 * 1024 + 1024 * 1024 + 64 * 1024;
    std::cout << "workspace growth requests=" << retained.size()
              << " owned_delta=" << bytes_delta << " limit=" << limit
              << " allocations=" << allocations_delta << '\n';
    require(bytes_delta <= limit,
            "growing workspace retains a separate allocation per same-bucket request");
    require(allocations_delta == 2,
            "growing workspace must allocate exactly two power-of-two capacities");
    std::cout << "PASS growing workspace capacity reuse with 65 live handles\n";
    return 0;
}

void verify_validation(motifcl::Backend& backend) {
    using namespace motifcl;
    auto& runtime = backend.vulkan_runtime();
    constexpr int rows = 2, in = 8, out = 4;
    nn::CounterStateLinear layer(backend, in, out, 3, 1.0f, 0.05f,
                                1.0f, 0.9f, 1.0e-6f, 777u);
    const std::vector<float> x_host(rows * in, 0.5f), grad_host(rows * out, 0.5f);
    // These are integer tensors, but their bit patterns are valid finite F32.
    // A raw-buffer GEMM without the dtype guard would silently consume 0.5f.
    const std::vector<std::int32_t> x_int_host(rows * in, 0x3f000000);
    const std::vector<std::int32_t> grad_int_host(rows * out, 0x3f000000);
    auto x = Tensor::from_cpu(backend, {rows, in}, DType::F32, x_host.data());
    auto grad = Tensor::from_cpu(backend, {rows, out}, DType::F32, grad_host.data());
    auto x_int = Tensor::from_cpu(backend, {rows, in}, DType::I32, x_int_host.data());
    auto grad_int = Tensor::from_cpu(backend, {rows, out}, DType::I32, grad_int_host.data());
    const auto state_before = layer.state.to_vector<std::uint8_t>();
    const auto scale_before = layer.scale.to_vector<float>();
    const auto v_before = layer.v.to_vector<float>();

    auto reject_dtype = [&](const Tensor& go, const Tensor& activation,
                             const char* label) {
        require(go.backend_ptr() == &backend && activation.backend_ptr() == &backend,
                "dtype fixture must use the layer backend");
        require(go.shape() == Shape{rows, out} && activation.shape() == Shape{rows, in},
                "dtype fixture must have valid counter shapes");
        runtime.reset_kernel_profile();
        const auto submits_before = runtime.submit_count();
        bool rejected = false;
        try {
            layer.apply_update_backward(go, activation, 777u);
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error(std::string(label) + " was accepted (dtype guard missing)");
        require(runtime.submit_count() == submits_before,
                "invalid counter dtype submitted work before rejection");
        for (const auto& entry : runtime.kernel_profile()) {
            require(entry.calls == 0,
                    "invalid counter dtype dispatched a matmul/update kernel before rejection");
        }
        require(layer.state.to_vector<std::uint8_t>() == state_before,
                "invalid counter dtype changed packed state");
        require(layer.scale.to_vector<float>() == scale_before,
                "invalid counter dtype changed scale");
        require(layer.v.to_vector<float>() == v_before,
                "invalid counter dtype changed RMS state");
        std::cout << "  rejected " << label << " without dispatch/state mutation\n";
    };
    // Keep the I32 grad case first: the phase-1 raw-helper regression must
    // fail here, rather than being hidden by the separately repaired C guard.
    for (const char* tiled : {"1", "0"}) {
        set_env("MOTIFCL_VK_COUNTER_TILED", tiled);
        reject_dtype(grad_int, x, "I32 grad_out/F32 x");
        reject_dtype(grad, x_int, "F32 grad_out/I32 x");
    }
    for (int C : {0, -1, 12, std::numeric_limits<int>::max()}) {
        runtime.reset_kernel_profile();
        const auto submits_before = runtime.submit_count();
        const auto allocations_before = runtime.buffer_allocations();
        bool rejected = false;
        try {
            nn::CounterStateLinear invalid(backend, in, out, C);
        } catch (const std::exception&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("invalid counter C accepted: " + std::to_string(C));
        require(runtime.submit_count() == submits_before &&
                runtime.buffer_allocations() == allocations_before,
                "invalid C allocated/dispatched before rejection");
        std::cout << "  rejected constructor C=" << C << '\n';
    }
    // A blanket rejection is not a valid fix: finite F32 inputs and the upper
    // valid C boundary must still execute an actual, non-empty counter update.
    nn::CounterStateLinear valid(backend, in, out, 11, 1.0f, 0.05f,
                                1.0f, 0.9f, 1.0e-6f, 777u);
    const auto valid_v_before = valid.v.to_vector<float>();
    runtime.reset_kernel_profile();
    valid.apply_update_backward(grad, x, 777u);
    bool updated = false;
    for (const auto& entry : runtime.kernel_profile()) {
        updated |= entry.calls > 0 && entry.name.find("counter_") == 0;
    }
    require(valid.v.to_vector<float>() != valid_v_before,
            "valid F32/C11 counter control did not update RMS state");
    if (runtime.caps().timestamps) {
        require(updated, "valid F32/C11 counter control did not engage an update kernel");
    } else {
        std::cout << "  SKIP valid F32/C11 kernel-profile subcheck: device has no timestamps; "
                     "RMS state delta remains required\n";
    }
    set_env("MOTIFCL_VK_COUNTER_TILED", "1");
}
}

int main(int argc, char** argv) {
    using namespace motifcl;
    set_env("MOTIFCL_VK_PROFILE", "1");
    if (!probe_vulkan_runtime().available()) return 77;
    try {
        if (argc == 2 && std::string(argv[1]) == "--workspace-growth-only") {
            return verify_workspace_growth();
        }
        Backend backend = Backend::create_vulkan();
        auto& runtime = backend.vulkan_runtime();
        if (!runtime.supports_storage_buffer_i8()) return 77;
        verify_validation(backend);
        if (argc == 2 && std::string(argv[1]) == "--validation-only") {
            std::cout << "PASS counter dtype/C validation and valid F32/C11 control\n";
            return 0;
        }
        const int growth = verify_workspace_growth();
        if (growth != 0) return growth;
        if (!runtime.caps().subgroup_arithmetic_compute || runtime.caps().subgroup_size != 64) return 77;
        constexpr int count = 4, rows = 37, width = 512;
        constexpr std::uint32_t seed = 777;
        std::array<std::unique_ptr<nn::CounterStateLinear>, count> layers, reference;
        for (int i = 0; i < count; ++i) {
            const auto init = static_cast<std::uint32_t>(1234 + i);
            layers[i] = std::make_unique<nn::CounterStateLinear>(backend, width, width, 3,
                0.01f, 0.001f, 1.0f, 0.9f, 1e-6f, init);
            reference[i] = std::make_unique<nn::CounterStateLinear>(backend, width, width, 3,
                0.01f, 0.001f, 1.0f, 0.9f, 1e-6f, init);
        }
        std::vector<float> xh(rows * width), gh(rows * width);
        for (std::size_t i = 0; i < xh.size(); ++i) {
            xh[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.013f;
            gh[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.017f;
        }
        auto x = Tensor::from_cpu(backend, {rows, width}, DType::F32, xh.data());
        auto grad = Tensor::from_cpu(backend, {rows, width}, DType::F32, gh.data());
        auto scale_new = Tensor::empty(backend, {width}, DType::F32);
        auto denom = Tensor::empty(backend, {width}, DType::F32);
        const auto base = runtime.buffer_bytes_live();
        constexpr std::size_t scratch_bytes = width * width * sizeof(float);
        std::size_t first_peak = 0;
        VulkanDispatchRecording recording;
        for (int step = 0; step < 3; ++step) {
            if (step < 2) {
                if (step == 1) require(runtime.capture_begin(), "counter capture begin failed");
                require(runtime.batch_begin(), "counter memory batch begin failed");
                for (auto& layer : layers) layer->apply_update_backward(grad, x, seed);
                const auto delta = runtime.buffer_bytes_live() - base;
                std::cout << "batch step=" << step << " owned_scratch_delta=" << delta << '\n';
                require(delta <= scratch_bytes + 64 * 1024,
                        "dense grad_w scratch grows with the number of layers in a batch");
                if (step == 0) first_peak = runtime.buffer_bytes_peak();
                if (step == 1) recording = runtime.capture_end();
                const auto submit = runtime.batch_end();
                require(submit.success, "counter memory batch submit failed");
            } else {
                require(!recording.empty(), "counter capture recorded no work");
                const auto replay = runtime.replay(recording);
                require(replay.success, "counter workspace replay failed");
            }
            for (int i = 0; i < count; ++i) {
                const auto old = run_vulkan_compact_counter_apply_update_fused(runtime,
                    reference[i]->state.storage().vulkan_buffer,
                    reference[i]->scale.storage().vulkan_buffer,
                    reference[i]->v.storage().vulkan_buffer,
                    grad.storage().vulkan_buffer, x.storage().vulkan_buffer,
                    scale_new.storage().vulkan_buffer, denom.storage().vulkan_buffer,
                    3, width, width, rows, 0.01f, 0.001f, 0.9f, 1e-6f, seed);
                require(old.success, "independent old counter update failed");
                require(layers[i]->state.to_vector<std::uint8_t>() == reference[i]->state.to_vector<std::uint8_t>(),
                        "workspace reuse changed counter state");
                require(layers[i]->scale.to_vector<float>() == reference[i]->scale.to_vector<float>(),
                        "workspace reuse changed counter scale");
                require(layers[i]->v.to_vector<float>() == reference[i]->v.to_vector<float>(),
                        "workspace reuse changed counter RMS state");
            }
            require(runtime.buffer_bytes_peak() <= first_peak + 64 * 1024,
                    "counter workspace allocations grow on repeated steps/capture");
        }
        std::cout << "PASS test_vulkan_counter_memory_batch: bounded scratch, exact old parity, capture/replay\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_counter_memory_batch: " << e.what() << '\n';
        return 1;
    }
}
