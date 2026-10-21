#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/runtime/vulkan_backend.hpp>

#ifdef MOTIFCL_COUNTER_SWIGLU_AUTOGRAD_TEST
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

template <typename T>
void require_exact(const std::vector<T>& actual, const std::vector<T>& expected,
                   const std::string& label) {
    if (actual.size() != expected.size()) throw std::runtime_error(label + " size mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(double(actual[i])) || !std::isfinite(double(expected[i])) ||
                actual[i] != expected[i]) {
            throw std::runtime_error(label + " exact mismatch at " + std::to_string(i));
        }
    }
}

float require_close(const std::vector<float>& actual, const std::vector<float>& expected,
                    float atol, float rtol, const std::string& label) {
    if (actual.size() != expected.size()) throw std::runtime_error(label + " size mismatch");
    float maximum = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error(label + " non-finite value at " + std::to_string(i));
        }
        const float error = std::fabs(actual[i] - expected[i]);
        if (!(error <= atol + rtol * std::fabs(expected[i]))) {
            throw std::runtime_error(label + " error at " + std::to_string(i) + ": " + std::to_string(error));
        }
        maximum = std::max(maximum, error);
    }
    return maximum;
}

void fill_deterministic(std::vector<float>& values, std::uint32_t seed, float scale) {
    for (float& value : values) {
        seed = seed * 1664525u + 1013904223u;
        value = (float((seed >> 8) & 0xffffu) / 32767.5f - 1.0f) * scale;
    }
}

template <typename T>
std::size_t changed_count(const std::vector<T>& before, const std::vector<T>& after) {
    require(before.size() == after.size(), "changed_count size mismatch");
    std::size_t count = 0;
    for (std::size_t i = 0; i < before.size(); ++i) count += before[i] != after[i];
    return count;
}

std::uint64_t profile_calls(const motifcl::VulkanRuntime& runtime, const std::string& name) {
    for (const auto& entry : runtime.kernel_profile()) if (entry.name == name) return entry.calls;
    return 0;
}

std::vector<float> download(const motifcl::VulkanBuffer& buffer, std::size_t count) {
    std::vector<float> values(count);
    buffer.download(values.data(), count * sizeof(float));
    return values;
}

struct Fixture {
    std::vector<std::uint8_t> state;
    std::vector<float> scales;
    std::vector<float> x;
    std::vector<float> preactivation;
    std::vector<float> output;
};

Fixture make_fixture(std::size_t rows, std::size_t in, std::size_t out, std::size_t C) {
    Fixture fixture;
    fixture.state.resize(out * (in / 4) * 3);
    fixture.scales.resize(out);
    fixture.x.resize(rows * in);
    fixture.preactivation.resize(rows * out);
    fixture.output.resize(rows * (out / 2));
    fill_deterministic(fixture.x, 0x135719u, 0.75f);
    // The CPU oracle starts from independently chosen ternary weights, not a
    // second copy of the shader's packed-state decoding implementation.
    std::vector<float> weights(out * in);
    const std::size_t levels = 2 * C - 1;
    for (std::size_t o = 0; o < out; ++o) {
        fixture.scales[o] = 0.04f + float(o % 13) * 0.0075f;
        for (std::size_t group = 0; group < in / 4; ++group) {
            std::uint32_t word = 0;
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const std::size_t k = group * 4 + lane;
                const int ternary = int((o * 19 + k * 7 + k / 7) % 3) - 1;
                const std::size_t counter_offset = (o * 11 + k * 5) % levels;
                const std::uint32_t code = std::uint32_t(std::size_t(ternary + 1) * levels + counter_offset);
                word |= code << (lane * 6);
                weights[o * in + k] = fixture.scales[o] * float(ternary);
            }
            const std::size_t base = (o * (in / 4) + group) * 3;
            fixture.state[base] = std::uint8_t(word);
            fixture.state[base + 1] = std::uint8_t(word >> 8);
            fixture.state[base + 2] = std::uint8_t(word >> 16);
        }
    }
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t o = 0; o < out; ++o) {
            double sum = 0.0;
            for (std::size_t k = 0; k < in; ++k) sum += double(fixture.x[r * in + k]) * weights[o * in + k];
            fixture.preactivation[r * out + o] = float(sum);
        }
        for (std::size_t h = 0; h < out / 2; ++h) {
            const double gate = fixture.preactivation[r * out + h];
            const double up = fixture.preactivation[r * out + out / 2 + h];
            fixture.output[r * (out / 2) + h] = float(gate / (1.0 + std::exp(-gate)) * up);
        }
    }
    return fixture;
}

void test_forward(motifcl::VulkanRuntime& runtime, std::size_t rows, std::size_t in,
                  std::size_t out, std::size_t C) {
    const auto fixture = make_fixture(rows, in, out, C);
    constexpr std::size_t guard = 32;
    constexpr float sentinel = -9876.5f;
    auto guarded_x = fixture.x;
    guarded_x.resize(guarded_x.size() + guard, std::numeric_limits<float>::quiet_NaN());
    auto guarded_scale = fixture.scales;
    guarded_scale.resize(guarded_scale.size() + guard, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> initial_pre(fixture.preactivation.size() + guard, sentinel);
    std::vector<float> initial_out(fixture.output.size() + guard, sentinel);
    auto x = runtime.create_buffer(guarded_x.size() * sizeof(float), guarded_x.data());
    auto state = runtime.create_buffer(fixture.state.size(), fixture.state.data());
    auto scale = runtime.create_buffer(guarded_scale.size() * sizeof(float), guarded_scale.data());
    auto pre = runtime.create_buffer(initial_pre.size() * sizeof(float), initial_pre.data());
    auto output = runtime.create_buffer(initial_out.size() * sizeof(float), initial_out.data());
    runtime.reset_kernel_profile();
    const auto run = motifcl::run_vulkan_compact_counter_swiglu_u8(
        runtime, x, state, scale, pre, output, C, in, out, rows);
    require(run.success, "counter SwiGLU runtime helper: " + run.error);
    const auto pre_all = download(pre, initial_pre.size());
    const auto out_all = download(output, initial_out.size());
    const std::vector<float> actual_pre(pre_all.begin(), pre_all.begin() + fixture.preactivation.size());
    const std::vector<float> actual_out(out_all.begin(), out_all.begin() + fixture.output.size());
    const std::string label = std::to_string(rows) + "x" + std::to_string(in) + "x" + std::to_string(out);
    const auto pre_error = require_close(actual_pre, fixture.preactivation, 2e-5f, 5e-6f, label + " CPU preactivation");
    const auto out_error = require_close(actual_out, fixture.output, 3e-5f, 1e-5f, label + " CPU SwiGLU");
    for (std::size_t i = fixture.preactivation.size(); i < pre_all.size(); ++i) {
        require(pre_all[i] == sentinel, label + " preactivation guard overwritten");
    }
    for (std::size_t i = fixture.output.size(); i < out_all.size(); ++i) {
        require(out_all[i] == sentinel, label + " SwiGLU output guard overwritten");
    }
    require(runtime.caps().timestamps ? profile_calls(runtime, "counter_swiglu_db") == 1 : true,
            "counter_swiglu_db did not actually engage");
    require(runtime.caps().timestamps ? profile_calls(runtime, "counter_gemm_db") == 0 &&
            profile_calls(runtime, "swiglu_f32") == 0 : true, "standalone fusion silently used legacy kernels");

    auto legacy_pre = runtime.create_buffer(fixture.preactivation.size() * sizeof(float));
    auto legacy_out = runtime.create_buffer(fixture.output.size() * sizeof(float));
    require(runtime.batch_begin(), "legacy comparison batch_begin failed");
    require(motifcl::run_vulkan_compact_counter_forward_u8(runtime, x, state, scale, legacy_pre, rows, in, out, C).success,
            "legacy counter GEMM failed");
    require(motifcl::run_vulkan_swiglu(runtime, legacy_pre, legacy_out, rows, out / 2).success,
            "legacy SwiGLU failed");
    require(runtime.batch_end().success, "legacy comparison batch_end failed");
    // Per-output FMA order and the activation expression are unchanged by the
    // fusion. Do not weaken these checks to hide an optimizer/state drift.
    require_exact(actual_pre, download(legacy_pre, fixture.preactivation.size()), label + " legacy preactivation");
    require_exact(actual_out, download(legacy_out, fixture.output.size()), label + " legacy SwiGLU");
    std::cout << "counter_swiglu " << label << " C=" << C << " pre_max_abs=" << pre_error
              << " out_max_abs=" << out_error << " exact_legacy=1 guards=1\n";
}

void test_validation(motifcl::VulkanRuntime& runtime) {
    const auto fixture = make_fixture(2, 4, 8, 3);
    // Ample buffers ensure a malformed alignment is rejected for its shape,
    // not accidentally concealed by an unrelated buffer-size check.
    auto x = runtime.create_buffer(4096);
    auto state = runtime.create_buffer(4096);
    auto scale = runtime.create_buffer(4096);
    auto pre = runtime.create_buffer(4096);
    auto output = runtime.create_buffer(4096);
    for (const auto& dims : {std::vector<std::size_t>{0, 4, 8, 2}, {12, 4, 8, 2},
                           {3, 0, 8, 2}, {3, 5, 8, 2}, {3, 4, 0, 2},
                           {3, 4, 10, 2}, {3, 4, 12, 2}, {3, 4, 8, 0}}) {
        const auto result = motifcl::run_vulkan_compact_counter_swiglu_u8(
            runtime, x, state, scale, pre, output, dims[0], dims[1], dims[2], dims[3]);
        require(!result.success && !result.error.empty(), "counter SwiGLU accepted malformed dimensions");
    }
    auto short_pre = runtime.create_buffer(fixture.preactivation.size() * sizeof(float) - sizeof(float));
    auto short_out = runtime.create_buffer(fixture.output.size() * sizeof(float) - sizeof(float));
    require(!motifcl::run_vulkan_compact_counter_swiglu_u8(
        runtime, x, state, scale, short_pre, output, 3, 4, 8, 2).success,
        "counter SwiGLU accepted short preactivation buffer");
    require(!motifcl::run_vulkan_compact_counter_swiglu_u8(
        runtime, x, state, scale, pre, short_out, 3, 4, 8, 2).success,
        "counter SwiGLU accepted short output buffer");
}

#ifdef MOTIFCL_COUNTER_SWIGLU_AUTOGRAD_TEST
void test_autograd(motifcl::Backend& backend, int rows, int in, int out, bool need_grad = true) {
    using namespace motifcl;
    constexpr int C = 3;
    constexpr std::uint32_t seed = 777u;
    nn::CounterStateLinear fused(backend, in, out, C, 1.0f, 0.05f, 1.0f, 0.9f, 1e-6f, seed);
    nn::CounterStateLinear legacy(backend, in, out, C, 1.0f, 0.05f, 1.0f, 0.9f, 1e-6f, seed);
    std::vector<float> x_host(std::size_t(rows) * in);
    std::vector<float> dy_host(std::size_t(rows) * (out / 2));
    fill_deterministic(x_host, 0x1234u, 0.8f);
    fill_deterministic(dy_host, 0x5678u, 0.9f);
    auto xf = Tensor::from_cpu(backend, {rows, in}, DType::F32, x_host.data());
    auto xl = Tensor::from_cpu(backend, {rows, in}, DType::F32, x_host.data());
    auto dy = Tensor::from_cpu(backend, {rows, out / 2}, DType::F32, dy_host.data());
    if (need_grad) {
        xf.set_requires_grad();
        xl.set_requires_grad();
    }
    auto& runtime = backend.vulkan_runtime();
    for (int step = 0; step < 2; ++step) {
        const auto state_before = fused.state.to_vector<std::uint8_t>();
        const auto scale_before = fused.scale.to_vector<float>();
        const auto v_before = fused.v.to_vector<float>();
        require_exact(state_before, legacy.state.to_vector<std::uint8_t>(), "initial state");
        require_exact(scale_before, legacy.scale.to_vector<float>(), "initial scales");
        require_exact(v_before, legacy.v.to_vector<float>(), "initial RMS");
        xf.zero_grad();
        xl.zero_grad();
        set_env("MOTIFCL_VK_COUNTER_SWIGLU", "1");
        runtime.reset_kernel_profile();
        require(runtime.batch_begin(), "fused autograd batch_begin failed");
        auto yf = fused.forward_swiglu(xf);
        require(yf.requires_grad(), "counter SwiGLU training dropped its update node");
        yf.backward(dy);
        require(runtime.batch_end().success, "fused autograd batch_end failed");
        if (runtime.caps().timestamps) {
            require(profile_calls(runtime, "counter_swiglu_db") == 1, "high-level fused path did not engage");
            require(profile_calls(runtime, "swiglu_f32") == 0, "fused path retained separate SwiGLU forward");
        }
        set_env("MOTIFCL_VK_COUNTER_SWIGLU", "0");
        runtime.reset_kernel_profile();
        require(runtime.batch_begin(), "legacy autograd batch_begin failed");
        auto yl = legacy.forward_swiglu(xl);
        yl.backward(dy);
        require(runtime.batch_end().success, "legacy autograd batch_end failed");
        if (runtime.caps().timestamps) {
            require(profile_calls(runtime, "counter_swiglu_db") == 0, "forced legacy still fused");
            require(profile_calls(runtime, "swiglu_f32") == 1, "forced legacy SwiGLU did not engage");
        }
        require_exact(yf.to_vector<float>(), yl.to_vector<float>(), "SwiGLU output");
        if (need_grad) {
            require(bool(xf.grad()) && bool(xl.grad()), "counter SwiGLU missing dX");
            require_exact(xf.grad()->to_vector<float>(), xl.grad()->to_vector<float>(), "counter SwiGLU dX");
        } else {
            require(!xf.grad() && !xl.grad(), "raw input unexpectedly received dX");
        }
        const auto new_state = fused.state.to_vector<std::uint8_t>();
        const auto new_scale = fused.scale.to_vector<float>();
        const auto new_v = fused.v.to_vector<float>();
        require_exact(new_state, legacy.state.to_vector<std::uint8_t>(), "counter SwiGLU packed state");
        require_exact(new_scale, legacy.scale.to_vector<float>(), "counter SwiGLU scales");
        require_exact(new_v, legacy.v.to_vector<float>(), "counter SwiGLU RMS");
        const auto ds = changed_count(state_before, new_state);
        const auto dc = changed_count(scale_before, new_scale);
        const auto dv = changed_count(v_before, new_v);
        require(ds != 0 && dc != 0 && dv != 0, "counter SwiGLU parity had no real update delta");
        std::cout << "counter_swiglu autograd " << rows << 'x' << in << 'x' << out
                  << " step=" << step << " input_grad=" << need_grad
                  << " exact(y/state/scale/v)=1 dX=" << (need_grad ? "exact" : "not_requested")
                  << " changed=" << ds << '/' << dc << '/' << dv << '\n';
    }
    for (bool eval : {false, true}) {
        fused.set_training(!eval);
        set_env("MOTIFCL_VK_COUNTER_SWIGLU", "1");
        const auto check_inference = [&] {
            auto expected = swiglu(fused.forward(xf));
            runtime.reset_kernel_profile();
            auto actual = fused.forward_swiglu(xf);
            require_exact(actual.to_vector<float>(), expected.to_vector<float>(), "NoGrad/eval fallback output");
            require(!actual.requires_grad() && !actual._grad_fn(), "NoGrad/eval fallback kept an update node");
            if (runtime.caps().timestamps) {
                require(profile_calls(runtime, "counter_swiglu_db") == 0,
                        "training-only fusion engaged under NoGrad/eval");
            }
        };
        if (eval) {
            check_inference();
        } else {
            autograd::NoGradGuard no_grad;
            check_inference();
        }
    }
}
#endif

void test_negative_comparator_controls() {
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        bool rejected = false;
        try { require_close({bad}, {0.0f}, 1.0f, 1.0f, "negative control"); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "counter SwiGLU comparator accepted non-finite output");
    }
}

} // namespace

int main() {
    try {
        test_negative_comparator_controls();
        set_env("MOTIFCL_VK_PROFILE", "1");
        const auto probe = motifcl::probe_vulkan_runtime();
        if (!probe.available()) {
            std::cout << "SKIP test_vulkan_counter_swiglu: " << probe.error << '\n';
            return 77;
        }
        auto runtime = motifcl::VulkanRuntime::create();
        require(runtime.available(), "Vulkan runtime: " + runtime.error());
        if (!runtime.supports_storage_buffer_i8()) {
            std::cout << "SKIP test_vulkan_counter_swiglu: storage_buffer_i8 unavailable\n";
            return 77;
        }
        test_forward(runtime, 1, 4, 8, 1);
        test_forward(runtime, 37, 68, 136, 3);
        test_forward(runtime, 256, 288, 384, 11);
        test_validation(runtime);
#ifdef MOTIFCL_COUNTER_SWIGLU_AUTOGRAD_TEST
        auto backend = motifcl::Backend::create_vulkan();
        test_autograd(backend, 37, 68, 136);
        test_autograd(backend, 256, 288, 384);
        test_autograd(backend, 37, 68, 136, false);
#endif
        if (!runtime.caps().timestamps) std::cout << "SKIP counter_swiglu profile engagement: no timestamps\n";
        std::cout << "PASS test_vulkan_counter_swiglu: independent CPU oracle, exact legacy, tails, guards, validation\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_counter_swiglu: " << e.what() << '\n';
        return 1;
    }
}
