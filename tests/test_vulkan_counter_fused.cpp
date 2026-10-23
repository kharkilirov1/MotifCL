#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/autograd/node.hpp>
#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

#include "vulkan_p0_test_utils.hpp"

namespace {

template <typename T>
void require_exact(const std::vector<T>& actual,
                   const std::vector<T>& expected,
                   const std::string& label) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(label + " exact mismatch at index " +
                                     std::to_string(i));
        }
    }
}

template <typename T>
std::size_t changed_count(const std::vector<T>& before,
                          const std::vector<T>& after) {
    std::size_t count = 0;
    for (std::size_t i = 0; i < before.size(); ++i) {
        if (before[i] != after[i]) ++count;
    }
    return count;
}

std::uint32_t packed_word_group(const std::vector<std::uint32_t>& words,
                                 std::size_t group, int mutation = 0) {
    const std::size_t base = group * 3;
    const unsigned shift = static_cast<unsigned>((base & 3) * (mutation == 2 ? 6 : 8));
    std::uint32_t packed = words.at(base >> 2) >> shift;
    if (shift > (mutation == 4 ? 16u : 8u) && mutation != 1) {
        packed |= words.at((base >> 2) + 1) << (32 - shift);
    }
    if (mutation == 3) packed ^= 1u;
    return packed;
}

void verify_packed_word_decode_controls() {
    std::array<std::size_t, 4> mutant_mismatches{};
    std::size_t checked_codes = 0;
    for (int C : {1, 3, 11}) {
        const int levels = 2 * C - 1;
        // in=68 gives 51-byte rows, so row starts cycle through all four
        // byte alignments. out%4 keeps the whole storage word-aligned.
        for (const auto& dims : std::vector<std::array<int, 2>>{
                 {4, 4}, {12, 4}, {68, 36}, {288, 384}}) {
            const std::size_t groups = static_cast<std::size_t>(dims[0] / 4 * dims[1]);
            std::vector<std::uint8_t> bytes(groups * 3, 0);
            std::vector<unsigned> codes(groups * 4);
            // Independent bit-stream encoder, rather than the shader's group
            // shifts. Every reachable code is covered, including C=11/code62.
            for (std::size_t i = 0; i < codes.size(); ++i) {
                codes[i] = static_cast<unsigned>(i % (3 * levels));
                for (unsigned bit = 0; bit < 6; ++bit) {
                    const std::size_t offset = i * 6 + bit;
                    bytes[offset / 8] |= static_cast<std::uint8_t>(
                        ((codes[i] >> bit) & 1u) << (offset % 8));
                }
            }
            std::vector<std::uint32_t> words(bytes.size() / 4, 0);
            for (std::size_t i = 0; i < bytes.size(); ++i) {
                words[i / 4] |= static_cast<std::uint32_t>(bytes[i]) << ((i % 4) * 8);
            }
            for (std::size_t group = 0; group < groups; ++group) {
                const auto packed = packed_word_group(words, group);
                const unsigned b0 = bytes[group * 3];
                const unsigned b1 = bytes[group * 3 + 1];
                const unsigned b2 = bytes[group * 3 + 2];
                const std::array<unsigned, 4> byte_codes{
                    b0 & 63u, ((b0 >> 6) | (b1 << 2)) & 63u,
                    ((b1 >> 4) | (b2 << 4)) & 63u, (b2 >> 2) & 63u};
                for (unsigned k = 0; k < 4; ++k) {
                    const unsigned expected = codes[group * 4 + k];
                    const unsigned actual = (packed >> (6 * k)) & 63u;
                    if (actual != expected || byte_codes[k] != expected) {
                        throw std::runtime_error("packed word/byte code mismatch at group " +
                            std::to_string(group) + " lane " + std::to_string(k));
                    }
                    const int ternary = actual < static_cast<unsigned>(levels) ? -1
                        : (actual < static_cast<unsigned>(2 * levels) ? 0 : 1);
                    if (ternary != static_cast<int>(expected / levels) - 1) {
                        throw std::runtime_error("packed word ternary decode mismatch");
                    }
                    for (int mutant = 1; mutant <= 4; ++mutant) {
                        const auto bad = (packed_word_group(words, group, mutant) >> (6 * k)) & 63u;
                        if (bad != expected) ++mutant_mismatches[mutant - 1];
                    }
                    ++checked_codes;
                }
            }
            // The last group starts at byte offset %4==1, requiring exactly
            // the final uint word, never a speculative read past the buffer.
            if (((groups - 1) * 3) % 4 != 1) {
                throw std::runtime_error("packed word tail fixture is not aligned as required");
            }
        }
    }
    for (std::size_t mismatches : mutant_mismatches) {
        if (mismatches == 0) throw std::runtime_error("packed word decode mutant escaped");
    }
    std::cout << "  packed word/byte exact codes=" << checked_codes
              << " C=1,3,11; mutant mismatches=" << mutant_mismatches[0] << '/'
              << mutant_mismatches[1] << '/' << mutant_mismatches[2] << '/'
              << mutant_mismatches[3] << '\n';
}

std::vector<float> reference_grad_x(const std::vector<float>& grad_out,
                                    const std::vector<float>& weight,
                                    int rows,
                                    int in_features,
                                    int out_features) {
    std::vector<float> grad_x(static_cast<std::size_t>(rows * in_features), 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int i = 0; i < in_features; ++i) {
            float acc = 0.0f;
            for (int o = 0; o < out_features; ++o) {
                acc += grad_out[static_cast<std::size_t>(r * out_features + o)] *
                       weight[static_cast<std::size_t>(o * in_features + i)];
            }
            grad_x[static_cast<std::size_t>(r * in_features + i)] = acc;
        }
    }
    return grad_x;
}

void run_case(motifcl::Backend& backend, int rows, int in_features,
              int out_features, bool force_legacy, bool word_load) {
    using namespace motifcl;
    using namespace motifcl_vulkan_p0_test;
    set_env("MOTIFCL_VK_COUNTER_TILED", force_legacy ? "0" : "1");
    set_env("MOTIFCL_VK_COUNTER_WORD_LOAD", word_load ? "1" : "0");
    constexpr int C = 3;
    constexpr std::uint32_t seed = 777u;

    std::vector<float> x_host(static_cast<std::size_t>(rows * in_features));
    std::vector<float> grad_out_host(static_cast<std::size_t>(rows * out_features));
    fill_deterministic(x_host, 0x1234u, 0.8f);
    fill_deterministic(grad_out_host, 0x5678u, 0.9f);

    nn::CounterStateLinear fast(
        backend, in_features, out_features, C, 1.0f, 0.05f,
        1.0f, 0.9f, 1.0e-6f, seed);
    nn::CounterStateLinear reference(
        backend, in_features, out_features, C, 1.0f, 0.05f,
        1.0f, 0.9f, 1.0e-6f, seed);

    require_exact(fast.state.to_vector<std::uint8_t>(),
                  reference.state.to_vector<std::uint8_t>(),
                  "initial packed state");
    require_exact(fast.scale.to_vector<float>(), reference.scale.to_vector<float>(),
                  "initial scale");

    auto x_fast = Tensor::from_cpu(
        backend, {rows, in_features}, DType::F32, x_host.data());
    auto x_reference = Tensor::from_cpu(
        backend, {rows, in_features}, DType::F32, x_host.data());
    auto grad_out = Tensor::from_cpu(
        backend, {rows, out_features}, DType::F32, grad_out_host.data());
    x_fast.set_requires_grad(true);
    auto old_scale = Tensor::empty(backend, {out_features}, DType::F32);
    auto old_denom = Tensor::empty(backend, {out_features}, DType::F32);

    // A second update also checks non-zero persistent RMS and changed packed
    // state. The reference never calls apply_update_backward: that would
    // exercise the same new tiled implementation as the system under test.
    for (std::uint32_t step = 0; step < 2; ++step) {
        const auto state_before = reference.state.to_vector<std::uint8_t>();
        const auto scale_before = reference.scale.to_vector<float>();
        const auto v_before = reference.v.to_vector<float>();
        Tensor dense_weight;
        Tensor y_reference;
        {
            autograd::NoGradGuard no_grad;
            dense_weight = reference.decode_weight();
            y_reference = matmul_transpose_b(x_reference, dense_weight);
        }
        const auto weight_host = dense_weight.to_vector<float>();

        backend.vulkan_runtime().reset_kernel_profile();
        x_fast.zero_grad();
        autograd::begin_graph_capture();
        auto y_fast = fast.forward(x_fast);
        const auto forward_graph = autograd::end_graph_capture();
        // The counter forward may run the decode-inside-GEMM kernel (rb4) or the
        // MOTIFCL_COUNTER_DECODE_DENSE decode-then-dense-GEMM path; both must
        // still record themselves in the captured graph.
        if (!graph_has_op(forward_graph, "compact_counter_forward_vulkan_f32_rb4") &&
            !graph_has_op(forward_graph, "compact_counter_forward_vulkan_f32_decode_dense")) {
            throw std::runtime_error(
                "counter forward is missing Vulkan graph metadata");
        }
        require_close(y_fast.to_vector<float>(), y_reference.to_vector<float>(),
                      2.0e-5f, "counter fused forward");

        y_fast.backward(grad_out);
        if (!x_fast.grad()) {
            throw std::runtime_error("counter fused backward did not produce grad_x");
        }
        require_close(
            x_fast.grad()->to_vector<float>(),
            reference_grad_x(grad_out_host, weight_host, rows, in_features,
                             out_features),
            2.0e-5f, "counter fused grad_x");

        // The profiled call belongs to fast.backward, not to the old oracle
        // invoked below. This catches accidentally forcing both sides to legacy.
        const bool can_profile = backend.vulkan_runtime().caps().timestamps;
        const bool expect_tiled = !force_legacy &&
            backend.vulkan_runtime().caps().subgroup_arithmetic_compute &&
            backend.vulkan_runtime().caps().subgroup_size == 64;
        bool saw_tiled = false;
        bool saw_legacy = false;
        std::uint64_t db_calls = 0;
        std::uint64_t word_db_calls = 0;
        std::uint64_t scalar_grad_x_calls = 0;
        std::uint64_t decode_calls = 0;
        for (const auto& profile : backend.vulkan_runtime().kernel_profile()) {
            if (profile.calls == 0) continue;
            saw_tiled |= profile.name == "counter_update_apply_f32";
            saw_legacy |= profile.name == "counter_apply_update_fused_f32";
            if (profile.name == "counter_gemm_db") db_calls += profile.calls;
            if (profile.name == "counter_gemm_word_db") word_db_calls += profile.calls;
            if (profile.name == "counter_decode_weight_f32") decode_calls += profile.calls;
            if (profile.name == "counter_backward_input_f32_rb4") {
                scalar_grad_x_calls += profile.calls;
            }
        }
        // MOTIFCL_COUNTER_DECODE_DENSE replaces each projection with a decode plus
        // a dense FP32 GEMM. One decode per projection is the equivalent signal
        // that a real Vulkan projection kernel ran, which is what this assertion
        // is about; the kernel identity is not the property under test.
        const char* dense_env = std::getenv("MOTIFCL_COUNTER_DECODE_DENSE");
        const bool dense_path = !(dense_env && *dense_env == '0');
        // Only rewrite the expectation when the dense path actually engaged: the
        // module also gates it on the row count, and these shapes are small.
        const bool dense_engaged = dense_path && decode_calls > 0;
        if (dense_engaged) {
            db_calls = decode_calls;
            word_db_calls = 0;
        }
        if (can_profile && (saw_tiled != expect_tiled || saw_legacy == expect_tiled)) {
            throw std::runtime_error("counter update selected an unexpected profiled path");
        }
        const bool expect_word = !dense_engaged && word_load && in_features % 4 == 0 && out_features % 4 == 0;
        const std::uint64_t expected_db = out_features % 4 == 0 ? 2u : 1u;
        if (can_profile && (word_db_calls != (expect_word ? expected_db : 0u) ||
            db_calls != (expect_word ? 0u : expected_db) ||
            scalar_grad_x_calls != (out_features % 4 == 0 ? 0u : 1u))) {
            throw std::runtime_error("counter forward/grad_x profiled kernels did not engage");
        }
        if (!can_profile && step == 0) {
            std::cout << "  SKIP counter kernel-profile subcheck: device has no timestamps; "
                         "numerical/state parity remains required\n";
        }

        const auto old_run = run_vulkan_compact_counter_apply_update_fused(
            backend.vulkan_runtime(), reference.state.storage().vulkan_buffer,
            reference.scale.storage().vulkan_buffer,
            reference.v.storage().vulkan_buffer, grad_out.storage().vulkan_buffer,
            x_reference.storage().vulkan_buffer, old_scale.storage().vulkan_buffer,
            old_denom.storage().vulkan_buffer, C, in_features, out_features, rows,
            1.0f, 0.05f, 0.9f, 1.0e-6f, seed + step);
        if (!old_run.success) {
            throw std::runtime_error("independent old counter update failed: " + old_run.error);
        }
        backend.finish();

        const auto expected_state = reference.state.to_vector<std::uint8_t>();
        const auto expected_scale = reference.scale.to_vector<float>();
        const auto expected_v = reference.v.to_vector<float>();
        require_close(fast.scale.to_vector<float>(), expected_scale, 0.0f,
                      "finite exact counter scale");
        require_close(fast.v.to_vector<float>(), expected_v, 0.0f,
                      "finite exact counter RMS");
        require_exact(fast.state.to_vector<std::uint8_t>(), expected_state,
                      "counter state transition");
        require_exact(fast.scale.to_vector<float>(), expected_scale,
                      "counter scale transition");
        require_exact(fast.v.to_vector<float>(), expected_v,
                      "counter RMS transition");

        const auto state_delta = changed_count(state_before, expected_state);
        const auto scale_delta = changed_count(scale_before, expected_scale);
        const auto v_delta = changed_count(v_before, expected_v);
        if (state_delta == 0 || scale_delta == 0 || v_delta == 0) {
            throw std::runtime_error(
                "counter update witness produced no packed-state/scale/RMS delta");
        }
        std::cout << "  counter rows=" << rows << " in=" << in_features
                  << " out=" << out_features << " step=" << step + 1
                  << " path=" << (!can_profile ? "unverified(no timestamps)"
                      : (saw_tiled ? "counter_update_apply_f32" : "counter_apply_update_fused_f32"))
                  << " forced_legacy=" << force_legacy
                  << " projection=" << (!can_profile ? "unverified(no timestamps)"
                      : (expect_word ? "counter_gemm_word_db" : "counter_gemm_db"))
                  << " changed(state/scale/v)=" << state_delta << '/'
                  << scale_delta << '/' << v_delta << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    using namespace motifcl;
    using namespace motifcl_vulkan_p0_test;
    try {
        verify_require_close_controls();
        verify_packed_word_decode_controls();
    } catch (const std::exception& e) {
        std::cerr << "FAIL packed counter CPU controls: " << e.what() << '\n';
        return 1;
    }
    if (argc == 2 && std::string(argv[1]) == "--packed-decode-only") {
        std::cout << "PASS packed counter byte/word decode controls\n";
        return 0;
    }
    set_env("MOTIFCL_VK_PROFILE", "1");
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::cout << "SKIP test_vulkan_counter_fused: " << probe.error << '\n';
        return 77;
    }
    try {
        Backend backend = Backend::create_vulkan();
        if (!backend.vulkan_runtime().supports_storage_buffer_i8()) {
            std::cout << "SKIP test_vulkan_counter_fused: storage_buffer_i8 unavailable\n";
            return 77;
        }
        for (bool force_legacy : {false, true}) {
            for (bool word_load : {false, true}) {
                run_case(backend, 37, 68, 35, force_legacy, word_load);
                run_case(backend, 37, 68, 36, force_legacy, word_load);
                run_case(backend, 257, 288, 384, force_legacy, word_load);
            }
        }
        std::cout
            << "PASS test_vulkan_counter_fused: forward, grad_x, independent exact old update; "
               "3 shapes x 2 update paths x 2 projection modes x 2 steps\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL test_vulkan_counter_fused: " << e.what() << '\n';
        return 1;
    }
}
