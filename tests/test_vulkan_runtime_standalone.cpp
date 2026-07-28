#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <motifcl/runtime/vulkan_backend.hpp>

namespace {

bool close_enough(float a, float b, float tol = 1.0e-5f) {
    return std::fabs(a - b) <= tol;
}

std::uint32_t pack_counter_word(const std::vector<std::uint32_t>& codes) {
    std::uint32_t word = 0;
    for (std::size_t i = 0; i < codes.size(); ++i) word |= (codes[i] & 0x3fu) << (i * 6);
    return word;
}

} // namespace

int main() {
    using namespace motifcl;

    bool ok = true;
    auto expect = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << message << '\n';
            ok = false;
        }
    };

    const auto probe = probe_vulkan_runtime();
    expect(probe.physical_device_count == 0 || probe.instance_created,
           "Vulkan probe cannot report physical devices without an instance");
    expect(probe.error.empty() || !probe.available(),
           "Vulkan probe with an error must not report available=true");
    expect(!vulkan_version_string(0).empty(),
           "Vulkan version formatting must be total");

    const auto invalid_m1 = run_vulkan_f32_m1_matmul({1.0f, 2.0f}, {1.0f, 2.0f, 3.0f}, 2, 2);
    expect(!invalid_m1.success, "standalone Vulkan M=1 matmul must reject malformed B size");
    expect(!invalid_m1.error.empty(), "standalone Vulkan M=1 matmul validation failure must explain why");

    const auto invalid_general = run_vulkan_f32_matmul({1.0f, 2.0f}, {1.0f, 2.0f, 3.0f}, 2, 2, 2);
    expect(!invalid_general.success, "standalone Vulkan general matmul must reject malformed A size");
    expect(!invalid_general.error.empty(), "standalone Vulkan general matmul validation failure must explain why");

    const auto invalid_softmax = run_vulkan_softmax_rows({1.0f, 2.0f, 3.0f}, 2, 2);
    expect(!invalid_softmax.success, "standalone Vulkan softmax rows must reject malformed input size");
    expect(!invalid_softmax.error.empty(), "standalone Vulkan softmax rows validation failure must explain why");

    const auto invalid_rms =
        run_vulkan_rmsnorm({1.0f, 2.0f, 3.0f}, {1.0f, 1.0f}, 2, 2, 1.0e-6f);
    expect(!invalid_rms.success, "standalone Vulkan RMSNorm must reject malformed input size");
    expect(!invalid_rms.error.empty(), "standalone Vulkan RMSNorm validation failure must explain why");

    const auto invalid_rms_eps =
        run_vulkan_rmsnorm({1.0f, 2.0f}, {1.0f, 1.0f}, 1, 2, std::numeric_limits<float>::infinity());
    expect(!invalid_rms_eps.success, "standalone Vulkan RMSNorm must reject non-finite eps");
    expect(!invalid_rms_eps.error.empty(), "standalone Vulkan RMSNorm eps validation failure must explain why");

    const auto invalid_swiglu = run_vulkan_swiglu({1.0f, 2.0f, 3.0f}, 1, 2);
    expect(!invalid_swiglu.success, "standalone Vulkan SwiGLU must reject malformed packed input size");
    expect(!invalid_swiglu.error.empty(), "standalone Vulkan SwiGLU validation failure must explain why");

    const auto invalid_add = run_vulkan_add({1.0f, 2.0f}, {1.0f});
    expect(!invalid_add.success, "standalone Vulkan add must reject mismatched input sizes");
    expect(!invalid_add.error.empty(), "standalone Vulkan add validation failure must explain why");

    // sub validation happens inside run_vulkan_sub(runtime, ...) device-resident
    // dispatch (no standalone vector-API overload by design — see note in
    // vulkan_backend.cpp). The device-resident parity block below covers
    // mismatched-size rejection through that path.

    const auto invalid_q8 = run_vulkan_i8_scaled_matmul({1, 2}, {1, 2, 3}, 1, 2, 2, 1.0f, 1.0f);
    expect(!invalid_q8.success, "standalone Vulkan i8 matmul must reject malformed B size");
    expect(!invalid_q8.error.empty(), "standalone Vulkan i8 matmul validation failure must explain why");

    const auto invalid_gqa =
        run_vulkan_grouped_query_attention({1.0f}, {1.0f}, {1.0f}, 1, 1, 3, 2, 1, 1.0f);
    expect(!invalid_gqa.success, "standalone Vulkan GQA must reject invalid head grouping");
    expect(!invalid_gqa.error.empty(), "standalone Vulkan GQA validation failure must explain why");

    const auto invalid_counter =
        run_vulkan_compact_counter_backward_input({0u}, {1.0f}, {1.0f}, 1, 3, 1, 3);
    expect(!invalid_counter.success, "standalone Vulkan compact-counter backward must reject non-packed input width");
    expect(!invalid_counter.error.empty(), "standalone Vulkan compact-counter validation failure must explain why");

    const auto invalid_sgd = run_vulkan_sgd_update({1.0f, 2.0f}, {1.0f}, 0.1f);
    expect(!invalid_sgd.success, "standalone Vulkan SGD update must reject mismatched sizes");
    expect(!invalid_sgd.error.empty(), "standalone Vulkan SGD validation failure must explain why");

    if (!probe.available()) {
        std::cout << "Standalone Vulkan runtime compute skipped: " << probe.error << '\n';
        return ok ? 0 : 1;
    }

    const bool require_compute = std::getenv("MOTIFCL_REQUIRE_VULKAN_COMPUTE") != nullptr;
    const auto smoke = run_vulkan_smoke_compute();
    if (!smoke.success && !require_compute) {
        std::cout << "Standalone Vulkan runtime compute skipped: " << smoke.error << '\n';
        return ok ? 0 : 1;
    }
    expect(smoke.success, "standalone Vulkan smoke compute must succeed when strict compute is required");
    expect(smoke.error.empty(), "standalone Vulkan smoke compute success must not carry an error");
    expect(smoke.output == 42.0f, "standalone Vulkan smoke compute must write the shader output");

    auto runtime = VulkanRuntime::create();
    expect(runtime.available(), "persistent Vulkan runtime must initialize when smoke compute is available");
    if (runtime.available()) {
        const auto persistent_smoke_1 = run_vulkan_smoke_compute(runtime);
        const auto persistent_smoke_2 = run_vulkan_smoke_compute(runtime);
        expect(persistent_smoke_1.success && persistent_smoke_2.success,
               "persistent Vulkan runtime must dispatch multiple smoke workloads");
        expect(persistent_smoke_1.output == 42.0f && persistent_smoke_2.output == 42.0f,
               "persistent Vulkan runtime smoke outputs must match");
        expect(!runtime.device_name().empty() || !persistent_smoke_1.device_name.empty(),
               "persistent Vulkan runtime must expose the selected device name when available");

        const std::vector<float> device_add_a = {1.0f, -2.0f, 3.5f, 4.0f};
        const std::vector<float> device_add_b = {4.0f, 5.0f, -0.5f, -1.0f};
        auto add_a_buffer = runtime.create_buffer(device_add_a.size() * sizeof(float), device_add_a.data());
        auto add_b_buffer = runtime.create_buffer(device_add_b.size() * sizeof(float), device_add_b.data());
        auto add_out_buffer = runtime.create_buffer(device_add_a.size() * sizeof(float));
        const auto add_dispatch = run_vulkan_add(runtime, add_a_buffer, add_b_buffer, add_out_buffer,
                                                 device_add_a.size());
        expect(add_dispatch.success, "persistent Vulkan runtime must dispatch device-resident add");
        expect(add_dispatch.error.empty(), "persistent Vulkan add success must not carry an error");
        std::vector<float> device_add_out(device_add_a.size(), 0.0f);
        add_out_buffer.download(device_add_out.data(), device_add_out.size() * sizeof(float));
        for (std::size_t i = 0; i < device_add_out.size(); ++i) {
            expect(close_enough(device_add_out[i], device_add_a[i] + device_add_b[i], 1.0e-6f),
                   "persistent Vulkan add output mismatch");
        }

        // sub device-resident path (mirror of add above); required for reversible inverse coupling.
        const std::vector<float> device_sub_a = {1.0f, -2.0f, 3.5f, 4.0f};
        const std::vector<float> device_sub_b = {4.0f, 5.0f, -0.5f, -1.0f};
        auto sub_a_buffer = runtime.create_buffer(device_sub_a.size() * sizeof(float), device_sub_a.data());
        auto sub_b_buffer = runtime.create_buffer(device_sub_b.size() * sizeof(float), device_sub_b.data());
        auto sub_out_buffer = runtime.create_buffer(device_sub_a.size() * sizeof(float));
        const auto sub_dispatch = run_vulkan_sub(runtime, sub_a_buffer, sub_b_buffer, sub_out_buffer,
                                                 device_sub_a.size());
        expect(sub_dispatch.success, "persistent Vulkan runtime must dispatch device-resident sub");
        expect(sub_dispatch.error.empty(), "persistent Vulkan sub success must not carry an error");
        std::vector<float> device_sub_out(device_sub_a.size(), 0.0f);
        sub_out_buffer.download(device_sub_out.data(), device_sub_out.size() * sizeof(float));
        for (std::size_t i = 0; i < device_sub_out.size(); ++i) {
            expect(close_enough(device_sub_out[i], device_sub_a[i] - device_sub_b[i], 1.0e-6f),
                   "persistent Vulkan sub output mismatch");
        }

        // === Slice E1: embedding gather, device-resident OpenCL-free witness ===
        // weight[V=4, D=3], indices[T=4] -> out[T*D].
        {
            const std::size_t V = 4, D = 3, T = 4;
            const std::vector<float> weight = {
                1.0f, 2.0f, 3.0f,      // vocab 0
                4.0f, 5.0f, 6.0f,      // vocab 1
                -1.0f, -2.0f, -3.0f,   // vocab 2
                0.5f, 0.25f, 0.0f,     // vocab 3
            };
            const std::vector<std::int32_t> indices = {3, 0, 1, 2};
            const std::vector<float> expected = {
                0.5f, 0.25f, 0.0f,
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                -1.0f, -2.0f, -3.0f,
            };
            auto w_buf = runtime.create_buffer(weight.size() * sizeof(float), weight.data());
            auto i_buf = runtime.create_buffer(indices.size() * sizeof(std::int32_t), indices.data());
            auto o_buf = runtime.create_buffer(expected.size() * sizeof(float));
            const auto r = run_vulkan_embedding_gather(runtime, w_buf, i_buf, o_buf, V, D, T);
            expect(r.success, "device-resident Vulkan embedding gather must succeed");
            expect(r.error.empty(), "Vulkan embedding gather success must not carry an error");
            std::vector<float> out(expected.size(), 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool gather_ok = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && gather_ok; ++i)
                if (!close_enough(out[i], expected[i], 1e-6f)) gather_ok = false;
            expect(gather_ok, "Vulkan embedding gather output mismatch vs CPU reference");
        }

        // === Slice E1: embedding weight backward, device-resident witness ===
        {
            const std::size_t V = 3, D = 2, T = 3;
            const std::vector<std::int32_t> indices = {0, 2, 0};
            const std::vector<float> grad_out = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            // Reference: grad_w[v, d] = sum_t (idx[t]==v) * grad_out[t*D+d]
            std::vector<float> expected(V * D, 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                for (std::size_t d = 0; d < D; ++d) {
                    expected[indices[t] * D + d] += grad_out[t * D + d];
                }
            }
            auto i_buf = runtime.create_buffer(indices.size() * sizeof(std::int32_t), indices.data());
            auto g_buf = runtime.create_buffer(grad_out.size() * sizeof(float), grad_out.data());
            auto o_buf = runtime.create_buffer(V * D * sizeof(float));
            const auto r = run_vulkan_embedding_weight_backward(runtime, i_buf, g_buf, o_buf, V, D, T);
            expect(r.success, "device-resident Vulkan embedding weight backward must succeed");
            expect(r.error.empty(), "Vulkan embedding weight backward success must not carry an error");
            std::vector<float> out(V * D, 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool bw_ok = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && bw_ok; ++i)
                if (!close_enough(out[i], expected[i], 1e-6f)) bw_ok = false;
            expect(bw_ok, "Vulkan embedding weight backward output mismatch vs CPU reference");
        }

        // === Slice K: atomic-scatter embedding weight backward ===
        // O(tokens*embed) instead of O(vocab*tokens*embed). Larger vocab than
        // the brute-force test above so the scatter heuristic actually fires.
        // The native-atomicAdd variant only runs where caps.supports_atomic_float
        // survived the startup smoke check (GCN4 / RX 580 fails it — its driver's
        // float atomics are broken — so this block is skipped there and the CAS
        // variant below is exercised instead).
        if (runtime.caps().supports_atomic_float) {
            const std::size_t V = 64, D = 8, T = 6;
            // Use sparse indices that touch only a few vocab rows to verify
            // atomic accumulation across multiple tokens mapping to the same row.
            const std::vector<std::int32_t> indices = {5, 5, 10, 5, 60, 0};
            std::vector<float> go(T * D);
            for (std::size_t i = 0; i < go.size(); ++i) go[i] = 0.1f * static_cast<float>(i % 7);
            // Reference: grad_w[v, d] = sum_t (idx[t]==v) * go[t*D + d].
            std::vector<float> expected(V * D, 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                for (std::size_t d = 0; d < D; ++d) {
                    expected[indices[t] * D + d] += go[t * D + d];
                }
            }
            auto i_buf = runtime.create_buffer(indices.size() * sizeof(std::int32_t), indices.data());
            auto g_buf = runtime.create_buffer(go.size() * sizeof(float), go.data());
            auto o_buf = runtime.create_buffer(V * D * sizeof(float));
            // Zero-fill grad_weight first (atomicAdd accumulates).
            const auto zr = run_vulkan_zero_f32(runtime, o_buf, V * D);
            expect(zr.success, "device-resident Vulkan zero_f32 must succeed");
            expect(zr.error.empty(), "Vulkan zero_f32 success must not carry an error");
            const auto r = run_vulkan_embedding_weight_backward_scatter(runtime, i_buf, g_buf, o_buf,
                                                                         V, D, T);
            expect(r.success, "device-resident Vulkan embedding weight backward scatter must succeed");
            expect(r.error.empty(), "Vulkan embedding scatter success must not carry an error");
            std::vector<float> out(V * D, 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool scatter_ok = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && scatter_ok; ++i) {
                if (!close_enough(out[i], expected[i], 1e-6f)) scatter_ok = false;
            }
            expect(scatter_ok, "Vulkan embedding weight backward scatter output mismatch vs CPU reference");
        }

        // === Slice K (portable): compare-and-swap scatter ===
        // Runs on EVERY Vulkan device (integer atomicCompSwap is core Vulkan 1.0),
        // so this is the path that actually executes on GCN4 / RX 580. Same sparse
        // pattern as above, including the row (5) touched by three tokens, which
        // the native float atomicAdd corrupts on RX 580 but the CAS loop handles
        // correctly.
        {
            const std::size_t V = 64, D = 8, T = 6;
            const std::vector<std::int32_t> indices = {5, 5, 10, 5, 60, 0};
            std::vector<float> go(T * D);
            for (std::size_t i = 0; i < go.size(); ++i) go[i] = 0.1f * static_cast<float>(i % 7);
            std::vector<float> expected(V * D, 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                for (std::size_t d = 0; d < D; ++d) {
                    expected[indices[t] * D + d] += go[t * D + d];
                }
            }
            auto i_buf = runtime.create_buffer(indices.size() * sizeof(std::int32_t), indices.data());
            auto g_buf = runtime.create_buffer(go.size() * sizeof(float), go.data());
            auto o_buf = runtime.create_buffer(V * D * sizeof(float));
            const auto zr = run_vulkan_zero_f32(runtime, o_buf, V * D);
            expect(zr.success, "device-resident Vulkan zero_f32 (cas) must succeed");
            const auto r = run_vulkan_embedding_weight_backward_scatter_cas(runtime, i_buf, g_buf, o_buf,
                                                                            V, D, T);
            expect(r.success, "device-resident Vulkan embedding weight backward scatter_cas must succeed");
            expect(r.error.empty(), "Vulkan embedding scatter_cas success must not carry an error");
            std::vector<float> out(V * D, 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool cas_ok = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && cas_ok; ++i) {
                if (!close_enough(out[i], expected[i], 1e-6f)) cas_ok = false;
            }
            expect(cas_ok, "Vulkan embedding weight backward scatter_cas output mismatch vs CPU reference");
        }

        // === Slice E1: token+position embedding forward + position backward ===
        {
            const std::size_t V = 2, S = 2, D = 2;
            const std::size_t token_count = 3;  // B=1.5 -> not, use B=1, T=3 conceptually
            // Use B*T=3 tokens, seq_len=2 means pos wraps: token_linear%seq.
            const std::vector<float> tw = {10.0f, 11.0f, 20.0f, 21.0f};
            const std::vector<float> pw = {0.1f, 0.2f, 0.3f, 0.4f};
            const std::vector<std::int32_t> ids = {0, 1, 0};
            // Expected out[token_linear, d] = tw[ids[tl]*D+d] + pw[(tl%S)*D+d]
            std::vector<float> expected(token_count * D);
            for (std::size_t tl = 0; tl < token_count; ++tl) {
                for (std::size_t d = 0; d < D; ++d) {
                    expected[tl * D + d] = tw[ids[tl] * D + d] + pw[(tl % S) * D + d];
                }
            }
            auto tw_buf = runtime.create_buffer(tw.size() * sizeof(float), tw.data());
            auto pw_buf = runtime.create_buffer(pw.size() * sizeof(float), pw.data());
            auto id_buf = runtime.create_buffer(ids.size() * sizeof(std::int32_t), ids.data());
            auto o_buf = runtime.create_buffer(expected.size() * sizeof(float));
            const auto fr = run_vulkan_token_position_embedding(runtime, tw_buf, pw_buf, id_buf, o_buf,
                                                                 V, S, D);
            expect(fr.success, "device-resident Vulkan token+position embedding must succeed");
            expect(fr.error.empty(), "Vulkan token+position embedding success must not carry an error");
            std::vector<float> out(expected.size(), 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool fwd_ok = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && fwd_ok; ++i)
                if (!close_enough(out[i], expected[i], 1e-6f)) fwd_ok = false;
            expect(fwd_ok, "Vulkan token+position embedding output mismatch vs CPU reference");

            // Position backward: batch=2 over the same grad_out table (so B*T=4).
            const std::size_t batch = 2, seq = 2;
            const std::vector<float> go = {0.5f, 0.5f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f};
            std::vector<float> expected_pw(seq * D, 0.0f);
            for (std::size_t b = 0; b < batch; ++b) {
                for (std::size_t pos = 0; pos < seq; ++pos) {
                    for (std::size_t d = 0; d < D; ++d) {
                        expected_pw[pos * D + d] += go[(b * seq + pos) * D + d];
                    }
                }
            }
            auto go_buf = runtime.create_buffer(go.size() * sizeof(float), go.data());
            auto gpw_buf = runtime.create_buffer(seq * D * sizeof(float));
            // Pre-zero the table to match the host contract (positions >= seq_len stay 0).
            std::vector<float> zeros(seq * D, 0.0f);
            gpw_buf.upload(zeros.data(), zeros.size() * sizeof(float));
            const auto br = run_vulkan_position_embedding_backward(runtime, go_buf, gpw_buf,
                                                                     batch, seq, D);
            expect(br.success, "device-resident Vulkan position embedding backward must succeed");
            expect(br.error.empty(), "Vulkan position embedding backward success must not carry an error");
            std::vector<float> pw_out(seq * D, 0.0f);
            gpw_buf.download(pw_out.data(), pw_out.size() * sizeof(float));
            bool pw_ok = pw_out.size() == expected_pw.size();
            for (std::size_t i = 0; i < pw_out.size() && pw_ok; ++i)
                if (!close_enough(pw_out[i], expected_pw[i], 1e-6f)) pw_ok = false;
            expect(pw_ok, "Vulkan position embedding backward output mismatch vs CPU reference");
        }

        // === Slice E2: rope (interleaved) forward + inverse, device-resident ===
        {
            const std::size_t B = 1, T = 2, n_head = 1, head_dim = 4;
            const std::size_t channels = n_head * head_dim;
            const std::size_t rows = B * T;
            const float theta = 10000.0f;
            const std::vector<float> xv = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f};
            // Reference: angle uses position t (0,1).
            std::vector<float> expected(rows * channels);
            for (std::size_t t = 0; t < T; ++t) {
                for (std::size_t pair = 0; pair < head_dim / 2; ++pair) {
                    const std::size_t pair_d = 2 * pair;
                    const float exponent = static_cast<float>(pair_d) / static_cast<float>(head_dim);
                    const float angle = static_cast<float>(t) / std::pow(theta, exponent);
                    const float cs = std::cos(angle), sn = std::sin(angle);
                    const float even = xv[t * channels + pair_d];
                    const float odd = xv[t * channels + pair_d + 1];
                    expected[t * channels + pair_d] = even * cs - odd * sn;
                    expected[t * channels + pair_d + 1] = even * sn + odd * cs;
                }
            }
            auto x_buf = runtime.create_buffer(xv.size() * sizeof(float), xv.data());
            auto o_buf = runtime.create_buffer(expected.size() * sizeof(float));
            const auto fr = run_vulkan_rope(runtime, x_buf, o_buf, B, T, channels, n_head, head_dim,
                                             /*rotary_dim=*/0, /*token_offset=*/0, theta, /*inverse=*/false);
            expect(fr.success, "device-resident Vulkan rope forward must succeed");
            expect(fr.error.empty(), "Vulkan rope forward success must not carry an error");
            std::vector<float> out(expected.size(), 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool fwd_ok = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && fwd_ok; ++i)
                if (!close_enough(out[i], expected[i], 1e-5f)) fwd_ok = false;
            expect(fwd_ok, "Vulkan rope forward output mismatch vs CPU reference");

            // Inverse: feed rope(out) backward must recover xv (rotations cancel).
            auto o2_buf = runtime.create_buffer(xv.size() * sizeof(float));
            const auto ir = run_vulkan_rope(runtime, o_buf, o2_buf, B, T, channels, n_head, head_dim,
                                             0, 0, theta, /*inverse=*/true);
            expect(ir.success, "device-resident Vulkan rope inverse must succeed");
            std::vector<float> roundtrip(xv.size(), 0.0f);
            o2_buf.download(roundtrip.data(), roundtrip.size() * sizeof(float));
            bool inv_ok = roundtrip.size() == xv.size();
            for (std::size_t i = 0; i < roundtrip.size() && inv_ok; ++i)
                if (!close_enough(roundtrip[i], xv[i], 1e-4f)) inv_ok = false;
            expect(inv_ok, "Vulkan rope inverse must recover the original input");
        }

        // === Slice E2: rope_split_half forward, device-resident witness ===
        {
            const std::size_t B = 1, T = 2, n_head = 1, head_dim = 4;
            const std::size_t channels = n_head * head_dim;
            const std::size_t rows = B * T;
            const float theta = 10000.0f;
            const std::vector<float> xv = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 1.0f, 1.0f, 1.0f};
            const std::size_t half_dim = head_dim / 2;
            std::vector<float> expected(rows * channels, 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                for (std::size_t pair = 0; pair < half_dim; ++pair) {
                    const float exponent = static_cast<float>(2 * pair) / static_cast<float>(head_dim);
                    const float angle = static_cast<float>(t) / std::pow(theta, exponent);
                    const float cs = std::cos(angle), sn = std::sin(angle);
                    const float first = xv[t * channels + pair];
                    const float second = xv[t * channels + half_dim + pair];
                    expected[t * channels + pair] = first * cs - second * sn;
                    expected[t * channels + half_dim + pair] = second * cs + first * sn;
                }
            }
            auto x_buf = runtime.create_buffer(xv.size() * sizeof(float), xv.data());
            auto o_buf = runtime.create_buffer(expected.size() * sizeof(float));
            const auto r = run_vulkan_rope_split_half(runtime, x_buf, o_buf, B, T, channels, n_head,
                                                      head_dim, 0, 0, theta, false);
            expect(r.success, "device-resident Vulkan rope_split_half must succeed");
            std::vector<float> out(expected.size(), 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool ok_sh = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && ok_sh; ++i)
                if (!close_enough(out[i], expected[i], 1e-5f)) ok_sh = false;
            expect(ok_sh, "Vulkan rope_split_half output mismatch vs CPU reference");
        }

        // === Slice E2: rope_positions forward, device-resident witness ===
        {
            const std::size_t B = 1, T = 2, n_head = 1, head_dim = 4;
            const std::size_t channels = n_head * head_dim;
            const std::size_t rows = B * T;
            const float theta = 10000.0f;
            const std::vector<float> xv = {1.0f, 0.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f};
            const std::vector<std::int32_t> pos = {0, 5};
            std::vector<float> expected(rows * channels, 0.0f);
            for (std::size_t t = 0; t < T; ++t) {
                for (std::size_t pair = 0; pair < head_dim / 2; ++pair) {
                    const std::size_t pair_d = 2 * pair;
                    const float exponent = static_cast<float>(pair_d) / static_cast<float>(head_dim);
                    const float angle = static_cast<float>(pos[t]) / std::pow(theta, exponent);
                    const float cs = std::cos(angle), sn = std::sin(angle);
                    const float even = xv[t * channels + pair_d];
                    const float odd = xv[t * channels + pair_d + 1];
                    expected[t * channels + pair_d] = even * cs - odd * sn;
                    expected[t * channels + pair_d + 1] = even * sn + odd * cs;
                }
            }
            auto x_buf = runtime.create_buffer(xv.size() * sizeof(float), xv.data());
            auto p_buf = runtime.create_buffer(pos.size() * sizeof(std::int32_t), pos.data());
            auto o_buf = runtime.create_buffer(expected.size() * sizeof(float));
            const auto r = run_vulkan_rope_positions(runtime, x_buf, p_buf, o_buf, B, T, channels,
                                                      n_head, head_dim, 0, theta);
            expect(r.success, "device-resident Vulkan rope_positions must succeed");
            std::vector<float> out(expected.size(), 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool ok_rp = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && ok_rp; ++i)
                if (!close_enough(out[i], expected[i], 1e-5f)) ok_rp = false;
            expect(ok_rp, "Vulkan rope_positions output mismatch vs CPU reference");
        }

        // === Slice E2: rope_positions_split_half forward, device-resident witness ===
        {
            const std::size_t B = 1, T = 1, n_head = 1, head_dim = 4;
            const std::size_t channels = n_head * head_dim;
            const std::size_t rows = B * T;
            const float theta = 10000.0f;
            const std::vector<float> xv = {1.0f, 2.0f, 3.0f, 4.0f};
            const std::vector<std::int32_t> pos = {3};
            const std::size_t half_dim = head_dim / 2;
            std::vector<float> expected(rows * channels, 0.0f);
            for (std::size_t pair = 0; pair < half_dim; ++pair) {
                const float exponent = static_cast<float>(2 * pair) / static_cast<float>(head_dim);
                const float angle = static_cast<float>(pos[0]) / std::pow(theta, exponent);
                const float cs = std::cos(angle), sn = std::sin(angle);
                const float first = xv[pair];
                const float second = xv[half_dim + pair];
                expected[pair] = first * cs - second * sn;
                expected[half_dim + pair] = second * cs + first * sn;
            }
            auto x_buf = runtime.create_buffer(xv.size() * sizeof(float), xv.data());
            auto p_buf = runtime.create_buffer(pos.size() * sizeof(std::int32_t), pos.data());
            auto o_buf = runtime.create_buffer(expected.size() * sizeof(float));
            const auto r = run_vulkan_rope_positions_split_half(runtime, x_buf, p_buf, o_buf, B, T,
                                                                 channels, n_head, head_dim, 0, theta);
            expect(r.success, "device-resident Vulkan rope_positions_split_half must succeed");
            std::vector<float> out(expected.size(), 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool ok_rps = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && ok_rps; ++i)
                if (!close_enough(out[i], expected[i], 1e-5f)) ok_rps = false;
            expect(ok_rps, "Vulkan rope_positions_split_half output mismatch vs CPU reference");
        }

        // === Slice C2 fix: mul_scalar / add_scalar device-resident witnesses ===
        // These elementwise-scalar kernels close the SubBackward/MulBackward/
        // DivBackward/ScalarBackward/DropoutBackward chain on Vulkan.
        {
            const std::vector<float> xv = {1.0f, -2.0f, 3.5f, 4.0f, 0.25f, -0.5f};
            const float alpha = -2.0f;
            std::vector<float> expected(xv.size());
            for (std::size_t i = 0; i < xv.size(); ++i) expected[i] = xv[i] * alpha;
            auto x_buf = runtime.create_buffer(xv.size() * sizeof(float), xv.data());
            auto o_buf = runtime.create_buffer(expected.size() * sizeof(float));
            const auto r = run_vulkan_mul_scalar(runtime, x_buf, o_buf, xv.size(), alpha);
            expect(r.success, "device-resident Vulkan mul_scalar must succeed");
            expect(r.error.empty(), "Vulkan mul_scalar success must not carry an error");
            std::vector<float> out(expected.size(), 0.0f);
            o_buf.download(out.data(), out.size() * sizeof(float));
            bool mul_ok = out.size() == expected.size();
            for (std::size_t i = 0; i < out.size() && mul_ok; ++i)
                if (!close_enough(out[i], expected[i], 1e-6f)) mul_ok = false;
            expect(mul_ok, "Vulkan mul_scalar output mismatch vs CPU reference");

            // add_scalar
            const float value = 0.75f;
            for (std::size_t i = 0; i < xv.size(); ++i) expected[i] = xv[i] + value;
            auto o2_buf = runtime.create_buffer(expected.size() * sizeof(float));
            const auto r2 = run_vulkan_add_scalar(runtime, x_buf, o2_buf, xv.size(), value);
            expect(r2.success, "device-resident Vulkan add_scalar must succeed");
            std::vector<float> out2(expected.size(), 0.0f);
            o2_buf.download(out2.data(), out2.size() * sizeof(float));
            bool add_ok = out2.size() == expected.size();
            for (std::size_t i = 0; i < out2.size() && add_ok; ++i)
                if (!close_enough(out2[i], expected[i], 1e-6f)) add_ok = false;
            expect(add_ok, "Vulkan add_scalar output mismatch vs CPU reference");

            // Non-finite alpha must be rejected (validation gap M-class fix).
            const auto r3 = run_vulkan_mul_scalar(runtime, x_buf, o_buf, xv.size(),
                                                   std::numeric_limits<float>::infinity());
            expect(!r3.success, "Vulkan mul_scalar must reject non-finite alpha");
            const auto r4 = run_vulkan_add_scalar(runtime, x_buf, o2_buf, xv.size(),
                                                    std::numeric_limits<float>::quiet_NaN());
            expect(!r4.success, "Vulkan add_scalar must reject NaN value");
        }

        const std::vector<float> device_sgd_param = {1.0f, 2.0f, -3.0f, 4.0f};
        const std::vector<float> device_sgd_grad = {0.5f, -1.0f, 2.0f, -0.25f};
        auto sgd_param_buffer = runtime.create_buffer(device_sgd_param.size() * sizeof(float),
                                                      device_sgd_param.data());
        auto sgd_grad_buffer = runtime.create_buffer(device_sgd_grad.size() * sizeof(float),
                                                     device_sgd_grad.data());
        auto sgd_out_buffer = runtime.create_buffer(device_sgd_param.size() * sizeof(float));
        const auto sgd_dispatch = run_vulkan_sgd_update(runtime, sgd_param_buffer, sgd_grad_buffer,
                                                        sgd_out_buffer, device_sgd_param.size(), 0.1f);
        expect(sgd_dispatch.success, "persistent Vulkan runtime must dispatch device-resident SGD update");
        expect(sgd_dispatch.error.empty(), "persistent Vulkan SGD update success must not carry an error");
        std::vector<float> device_sgd_out(device_sgd_param.size(), 0.0f);
        sgd_out_buffer.download(device_sgd_out.data(), device_sgd_out.size() * sizeof(float));
        for (std::size_t i = 0; i < device_sgd_out.size(); ++i) {
            expect(close_enough(device_sgd_out[i], device_sgd_param[i] - 0.1f * device_sgd_grad[i], 1.0e-6f),
                   "persistent Vulkan SGD update output mismatch");
        }

        const std::vector<float> device_tb_a = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        };
        const std::vector<float> device_tb_b = {
            1.0f, 3.0f, 5.0f,
            2.0f, 4.0f, 6.0f,
        };
        auto tb_a_buffer = runtime.create_buffer(device_tb_a.size() * sizeof(float), device_tb_a.data());
        auto tb_b_buffer = runtime.create_buffer(device_tb_b.size() * sizeof(float), device_tb_b.data());
        auto tb_out_buffer = runtime.create_buffer(4 * sizeof(float));
        const auto tb_dispatch = run_vulkan_f32_matmul_transpose_b(runtime, tb_a_buffer, tb_b_buffer,
                                                                   tb_out_buffer, 2, 3, 2);
        expect(tb_dispatch.success, "persistent Vulkan runtime must dispatch device-resident transpose-B matmul");
        expect(tb_dispatch.error.empty(), "persistent Vulkan transpose-B matmul success must not carry an error");
        std::vector<float> device_tb_out(4, 0.0f);
        tb_out_buffer.download(device_tb_out.data(), device_tb_out.size() * sizeof(float));
        expect(close_enough(device_tb_out[0], 22.0f) && close_enough(device_tb_out[1], 28.0f) &&
                   close_enough(device_tb_out[2], 49.0f) && close_enough(device_tb_out[3], 64.0f),
               "persistent Vulkan transpose-B matmul output mismatch");

        if (runtime.supports_storage_buffer_i8()) {
            const std::size_t C = 3;
            const std::uint32_t lv = static_cast<std::uint32_t>(2 * C - 1);
            auto code_u8 = [&](int t) {
                return static_cast<std::uint32_t>((t + 1) * static_cast<int>(lv) + static_cast<int>(C - 1));
            };
            const std::vector<std::uint32_t> counter_words = {
                pack_counter_word({code_u8(1), code_u8(0), code_u8(-1), code_u8(1)}),
                pack_counter_word({code_u8(0), code_u8(1), code_u8(1), code_u8(-1)}),
            };
            std::vector<std::uint8_t> counter_bytes;
            counter_bytes.reserve(counter_words.size() * 3);
            for (const auto word : counter_words) {
                counter_bytes.push_back(static_cast<std::uint8_t>(word & 0xffu));
                counter_bytes.push_back(static_cast<std::uint8_t>((word >> 8u) & 0xffu));
                counter_bytes.push_back(static_cast<std::uint8_t>((word >> 16u) & 0xffu));
            }
            const std::vector<float> counter_scale_u8 = {0.5f, 2.0f};
            auto counter_state_buffer = runtime.create_buffer(counter_bytes.size(), counter_bytes.data());
            auto counter_scale_buffer = runtime.create_buffer(counter_scale_u8.size() * sizeof(float),
                                                              counter_scale_u8.data());
            auto counter_weight_buffer = runtime.create_buffer(8 * sizeof(float));
            const auto counter_decode_dispatch = run_vulkan_compact_counter_decode_weight(
                runtime, counter_state_buffer, counter_scale_buffer, counter_weight_buffer, 4, 2, C);
            expect(counter_decode_dispatch.success,
                   "persistent Vulkan runtime must dispatch production U8 compact-counter decode");
            expect(counter_decode_dispatch.error.empty(),
                   "persistent Vulkan compact-counter decode success must not carry an error");
            std::vector<float> counter_weight_out(8, 0.0f);
            counter_weight_buffer.download(counter_weight_out.data(), counter_weight_out.size() * sizeof(float));
            const std::vector<float> counter_weight_expected = {
                0.5f, 0.0f, -0.5f, 0.5f,
                0.0f, 2.0f, 2.0f, -2.0f,
            };
            for (std::size_t i = 0; i < counter_weight_expected.size(); ++i) {
                expect(close_enough(counter_weight_out[i], counter_weight_expected[i]),
                       "persistent Vulkan compact-counter decode output mismatch");
            }

            const std::vector<float> counter_grad_out_u8 = {1.0f, 2.0f, -1.0f, 0.5f};
            auto counter_grad_out_buffer = runtime.create_buffer(counter_grad_out_u8.size() * sizeof(float),
                                                                 counter_grad_out_u8.data());
            auto counter_grad_x_buffer = runtime.create_buffer(8 * sizeof(float));
            const auto counter_bwd_dispatch = run_vulkan_compact_counter_backward_input_u8(
                runtime, counter_state_buffer, counter_scale_buffer, counter_grad_out_buffer,
                counter_grad_x_buffer, 2, 4, 2, C);
            expect(counter_bwd_dispatch.success,
                   "persistent Vulkan runtime must dispatch production U8 compact-counter backward-input");
            expect(counter_bwd_dispatch.error.empty(),
                   "persistent Vulkan compact-counter backward-input success must not carry an error");
            std::vector<float> counter_grad_x_out(8, 0.0f);
            counter_grad_x_buffer.download(counter_grad_x_out.data(), counter_grad_x_out.size() * sizeof(float));
            const std::vector<int> tern = {1, 0, -1, 1, 0, 1, 1, -1};
            for (std::size_t r = 0; r < 2; ++r) {
                for (std::size_t i = 0; i < 4; ++i) {
                    const float expected =
                        counter_grad_out_u8[r * 2 + 0] * counter_scale_u8[0] * static_cast<float>(tern[i]) +
                        counter_grad_out_u8[r * 2 + 1] * counter_scale_u8[1] * static_cast<float>(tern[4 + i]);
                    expect(close_enough(counter_grad_x_out[r * 4 + i], expected),
                           "persistent Vulkan compact-counter backward-input output mismatch");
                }
            }
        }

        // ---- OpenCL-free multi-tile parity for the cached tiled kernels ----
        // Shapes deliberately cross the 16x16 tile (non-multiples of 16, >1
        // workgroup, >1 K-tile) so cross-tile accumulation and boundary
        // guards are asserted, not just the single-tile case. Tolerance 1e-3
        // absolute (F32 accumulation over K<=40 with values in [-1,1]).
        {
            auto fill = [](std::vector<float>& v, std::uint32_t seed) {
                std::uint32_t s = seed | 1u;
                for (auto& x : v) {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    x = static_cast<float>(static_cast<std::int32_t>(s % 2001) - 1000) / 1000.0f;
                }
            };
            const std::size_t M = 48, K = 40, N = 33;
            std::vector<float> a(M * K), b(K * N), bt(N * K), at(K * M);
            fill(a, 0x1001u);
            fill(b, 0x2002u);
            fill(bt, 0x3003u);
            fill(at, 0x4004u);

            auto a_buf = runtime.create_buffer(a.size() * sizeof(float), a.data());
            auto b_buf = runtime.create_buffer(b.size() * sizeof(float), b.data());
            auto bt_buf = runtime.create_buffer(bt.size() * sizeof(float), bt.data());
            auto at_buf = runtime.create_buffer(at.size() * sizeof(float), at.data());
            auto c_buf = runtime.create_buffer(M * N * sizeof(float));

            const auto nn = run_vulkan_f32_matmul(runtime, a_buf, b_buf, c_buf, M, K, N);
            expect(nn.success, "standalone multi-tile NN matmul must dispatch");
            if (nn.success) {
                std::vector<float> c(M * N, 0.0f);
                c_buf.download(c.data(), c.size() * sizeof(float));
                bool match = true;
                for (std::size_t m = 0; m < M && match; ++m) {
                    for (std::size_t n = 0; n < N; ++n) {
                        float ref = 0.0f;
                        for (std::size_t k = 0; k < K; ++k) ref += a[m * K + k] * b[k * N + n];
                        if (!close_enough(c[m * N + n], ref, 1e-3f)) {
                            match = false;
                            break;
                        }
                    }
                }
                expect(match, "standalone multi-tile NN matmul parity mismatch vs CPU reference");
            }

            const auto nt = run_vulkan_f32_matmul_transpose_b(runtime, a_buf, bt_buf, c_buf, M, K, N);
            expect(nt.success, "standalone multi-tile NT matmul must dispatch");
            if (nt.success) {
                std::vector<float> c(M * N, 0.0f);
                c_buf.download(c.data(), c.size() * sizeof(float));
                bool match = true;
                for (std::size_t m = 0; m < M && match; ++m) {
                    for (std::size_t n = 0; n < N; ++n) {
                        float ref = 0.0f;
                        for (std::size_t k = 0; k < K; ++k) ref += a[m * K + k] * bt[n * K + k];
                        if (!close_enough(c[m * N + n], ref, 1e-3f)) {
                            match = false;
                            break;
                        }
                    }
                }
                expect(match, "standalone multi-tile NT matmul parity mismatch vs CPU reference");
            }

            const auto tn = run_vulkan_f32_matmul_transpose_a(runtime, at_buf, b_buf, c_buf, M, K, N);
            expect(tn.success, "standalone multi-tile TN matmul must dispatch");
            if (tn.success) {
                std::vector<float> c(M * N, 0.0f);
                c_buf.download(c.data(), c.size() * sizeof(float));
                bool match = true;
                for (std::size_t m = 0; m < M && match; ++m) {
                    for (std::size_t n = 0; n < N; ++n) {
                        float ref = 0.0f;
                        for (std::size_t k = 0; k < K; ++k) ref += at[k * M + m] * b[k * N + n];
                        if (!close_enough(c[m * N + n], ref, 1e-3f)) {
                            match = false;
                            break;
                        }
                    }
                }
                expect(match, "standalone multi-tile TN matmul parity mismatch vs CPU reference");
            }

            // M=1 transpose-B decode form (wave-per-output reduction kernel).
            const std::size_t dK = 130, dN = 70;
            std::vector<float> da(dK), db(dN * dK);
            fill(da, 0x5005u);
            fill(db, 0x6006u);
            auto da_buf = runtime.create_buffer(da.size() * sizeof(float), da.data());
            auto db_buf = runtime.create_buffer(db.size() * sizeof(float), db.data());
            auto dc_buf = runtime.create_buffer(dN * sizeof(float));
            const auto m1nt = run_vulkan_f32_matmul_transpose_b(runtime, da_buf, db_buf, dc_buf, 1, dK, dN);
            expect(m1nt.success, "standalone M=1 NT matmul must dispatch");
            if (m1nt.success) {
                std::vector<float> c(dN, 0.0f);
                dc_buf.download(c.data(), c.size() * sizeof(float));
                bool match = true;
                for (std::size_t n = 0; n < dN && match; ++n) {
                    float ref = 0.0f;
                    for (std::size_t k = 0; k < dK; ++k) ref += da[k] * db[n * dK + k];
                    if (!close_enough(c[n], ref, 1e-3f)) match = false;
                }
                expect(match, "standalone M=1 NT matmul parity mismatch vs CPU reference");
            }
        }

        // ---- OpenCL-free parity for the wave-per-row backward kernels ----
        {
            auto fill = [](std::vector<float>& v, std::uint32_t seed) {
                std::uint32_t s = seed | 1u;
                for (auto& x : v) {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    x = static_cast<float>(static_cast<std::int32_t>(s % 2001) - 1000) / 1000.0f;
                }
            };
            const std::size_t R = 5, C = 100;
            const float eps = 1e-5f;
            std::vector<float> x(R * C), w(C), dy(R * C);
            fill(x, 0x7007u);
            fill(w, 0x8008u);
            fill(dy, 0x9009u);
            auto x_buf = runtime.create_buffer(x.size() * sizeof(float), x.data());
            auto w_buf = runtime.create_buffer(w.size() * sizeof(float), w.data());
            auto dy_buf = runtime.create_buffer(dy.size() * sizeof(float), dy.data());
            auto out_buf = runtime.create_buffer(R * C * sizeof(float));
            auto inv_buf = runtime.create_buffer(R * sizeof(float));
            auto dw_buf = runtime.create_buffer(C * sizeof(float));

            std::vector<float> inv_r(R);
            for (std::size_t r = 0; r < R; ++r) {
                float ss = 0.0f;
                for (std::size_t c = 0; c < C; ++c) ss += x[r * C + c] * x[r * C + c];
                inv_r[r] = 1.0f / std::sqrt(ss / static_cast<float>(C) + eps);
            }

            const auto bwd_x = run_vulkan_rmsnorm_backward_x(runtime, x_buf, w_buf, dy_buf, out_buf, R, C, eps);
            expect(bwd_x.success, "standalone rmsnorm backward_x must dispatch");
            if (bwd_x.success) {
                std::vector<float> dx(R * C, 0.0f);
                out_buf.download(dx.data(), dx.size() * sizeof(float));
                bool match = true;
                for (std::size_t r = 0; r < R && match; ++r) {
                    float dot = 0.0f;
                    for (std::size_t c = 0; c < C; ++c) dot += dy[r * C + c] * w[c] * x[r * C + c];
                    for (std::size_t c = 0; c < C; ++c) {
                        const float inv = inv_r[r];
                        const float ref = dy[r * C + c] * w[c] * inv -
                                          x[r * C + c] * inv * inv * inv * dot / static_cast<float>(C);
                        if (!close_enough(dx[r * C + c], ref, 1e-4f)) {
                            match = false;
                            break;
                        }
                    }
                }
                expect(match, "standalone rmsnorm backward_x parity mismatch vs CPU reference");
            }

            const auto bwd_w = run_vulkan_rmsnorm_backward_weight(runtime, x_buf, dy_buf, inv_buf, dw_buf, R, C, eps);
            expect(bwd_w.success, "standalone rmsnorm backward_weight must dispatch");
            if (bwd_w.success) {
                std::vector<float> dw(C, 0.0f);
                dw_buf.download(dw.data(), dw.size() * sizeof(float));
                bool match = true;
                for (std::size_t c = 0; c < C && match; ++c) {
                    float ref = 0.0f;
                    for (std::size_t r = 0; r < R; ++r) ref += dy[r * C + c] * x[r * C + c] * inv_r[r];
                    if (!close_enough(dw[c], ref, 1e-4f)) match = false;
                }
                expect(match, "standalone rmsnorm backward_weight parity mismatch vs CPU reference");
            }

            const auto smax_bwd = run_vulkan_softmax_rows_backward(runtime, x_buf, dy_buf, out_buf, R, C);
            expect(smax_bwd.success, "standalone softmax backward must dispatch");
            if (smax_bwd.success) {
                std::vector<float> dx(R * C, 0.0f);
                out_buf.download(dx.data(), dx.size() * sizeof(float));
                bool match = true;
                for (std::size_t r = 0; r < R && match; ++r) {
                    float dot = 0.0f;
                    for (std::size_t c = 0; c < C; ++c) dot += dy[r * C + c] * x[r * C + c];
                    for (std::size_t c = 0; c < C; ++c) {
                        const float ref = x[r * C + c] * (dy[r * C + c] - dot);
                        if (!close_enough(dx[r * C + c], ref, 1e-4f)) {
                            match = false;
                            break;
                        }
                    }
                }
                expect(match, "standalone softmax backward parity mismatch vs CPU reference");
            }

            // SwiGLU backward: packed [R, 2*H].
            const std::size_t H = 40;
            std::vector<float> packed(R * 2 * H), sg(R * H);
            fill(packed, 0xa00au);
            fill(sg, 0xb00bu);
            auto packed_buf = runtime.create_buffer(packed.size() * sizeof(float), packed.data());
            auto sg_buf = runtime.create_buffer(sg.size() * sizeof(float), sg.data());
            auto dpacked_buf = runtime.create_buffer(packed.size() * sizeof(float));
            const auto sw_bwd = run_vulkan_swiglu_backward(runtime, packed_buf, sg_buf, dpacked_buf, R, H);
            expect(sw_bwd.success, "standalone swiglu backward must dispatch");
            if (sw_bwd.success) {
                std::vector<float> dp(packed.size(), 0.0f);
                dpacked_buf.download(dp.data(), dp.size() * sizeof(float));
                bool match = true;
                for (std::size_t r = 0; r < R && match; ++r) {
                    for (std::size_t c = 0; c < H; ++c) {
                        const float gate = packed[r * 2 * H + c];
                        const float up = packed[r * 2 * H + H + c];
                        const float go = sg[r * H + c];
                        const float sig = 1.0f / (1.0f + std::exp(-gate));
                        const float dgate = go * up * (sig + gate * sig * (1.0f - sig));
                        const float dup = go * gate * sig;
                        if (!close_enough(dp[r * 2 * H + c], dgate, 1e-4f) ||
                            !close_enough(dp[r * 2 * H + H + c], dup, 1e-4f)) {
                            match = false;
                            break;
                        }
                    }
                }
                expect(match, "standalone swiglu backward parity mismatch vs CPU reference");
            }

            // cols < wave (32 < 64): idle reduction lanes must contribute
            // identity values, matching the per-head head_dim=32 norm shape.
            {
                const std::size_t r32 = 3, c32 = 32;
                std::vector<float> x32(r32 * c32), w32(c32), dy32(r32 * c32);
                fill(x32, 0xc00cu);
                fill(w32, 0xd00du);
                fill(dy32, 0xe00eu);
                auto x32_buf = runtime.create_buffer(x32.size() * sizeof(float), x32.data());
                auto w32_buf = runtime.create_buffer(w32.size() * sizeof(float), w32.data());
                auto dy32_buf = runtime.create_buffer(dy32.size() * sizeof(float), dy32.data());
                auto dx32_buf = runtime.create_buffer(x32.size() * sizeof(float));
                const auto r = run_vulkan_rmsnorm_backward_x(runtime, x32_buf, w32_buf, dy32_buf, dx32_buf,
                                                             r32, c32, eps);
                expect(r.success, "standalone rmsnorm backward_x cols<64 must dispatch");
                if (r.success) {
                    std::vector<float> dx32(x32.size(), 0.0f);
                    dx32_buf.download(dx32.data(), dx32.size() * sizeof(float));
                    bool match = true;
                    for (std::size_t row = 0; row < r32 && match; ++row) {
                        float ss = 0.0f, dot = 0.0f;
                        for (std::size_t c = 0; c < c32; ++c) {
                            ss += x32[row * c32 + c] * x32[row * c32 + c];
                            dot += dy32[row * c32 + c] * w32[c] * x32[row * c32 + c];
                        }
                        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(c32) + eps);
                        for (std::size_t c = 0; c < c32; ++c) {
                            const float ref = dy32[row * c32 + c] * w32[c] * inv -
                                              x32[row * c32 + c] * inv * inv * inv * dot /
                                                  static_cast<float>(c32);
                            if (!close_enough(dx32[row * c32 + c], ref, 1e-4f)) {
                                match = false;
                                break;
                            }
                        }
                    }
                    expect(match, "standalone rmsnorm backward_x cols<64 parity mismatch");
                }
            }

            // GELU fwd + bwd.
            const auto gelu_fwd = run_vulkan_gelu(runtime, x_buf, out_buf, R * C);
            expect(gelu_fwd.success, "standalone gelu forward must dispatch");
            if (gelu_fwd.success) {
                std::vector<float> y(R * C, 0.0f);
                out_buf.download(y.data(), y.size() * sizeof(float));
                bool match = true;
                const float cst = 0.7978845608028654f;
                for (std::size_t i = 0; i < R * C && match; ++i) {
                    const float v = x[i];
                    const float t = cst * (v + 0.044715f * v * v * v);
                    const float ref = 0.5f * v * (1.0f + std::tanh(t));
                    if (!close_enough(y[i], ref, 1e-4f)) match = false;
                }
                expect(match, "standalone gelu forward parity mismatch vs CPU reference");
            }
            const auto gelu_bwd = run_vulkan_gelu_backward(runtime, x_buf, dy_buf, out_buf, R * C);
            expect(gelu_bwd.success, "standalone gelu backward must dispatch");
            if (gelu_bwd.success) {
                std::vector<float> dx(R * C, 0.0f);
                out_buf.download(dx.data(), dx.size() * sizeof(float));
                bool match = true;
                const float cst = 0.7978845608028654f;
                for (std::size_t i = 0; i < R * C && match; ++i) {
                    const float v = x[i];
                    const float inner = cst * (v + 0.044715f * v * v * v);
                    const float th = std::tanh(inner);
                    const float sech2 = 1.0f - th * th;
                    const float inner_grad = cst * (1.0f + 3.0f * 0.044715f * v * v);
                    const float ref = dy[i] * (0.5f * (1.0f + th) + 0.5f * v * sech2 * inner_grad);
                    if (!close_enough(dx[i], ref, 1e-4f)) match = false;
                }
                expect(match, "standalone gelu backward parity mismatch vs CPU reference");
            }
        }

        // ---- dispatch capture/replay + steady-state timing (invariants
        // 6/7/9): a captured matmul->add chain replays into ONE command
        // buffer per replay; 1000 replays must stay flat (no per-call
        // pipeline/descriptor growth), asserted as second-half wall time
        // <= 1.5x first-half.
        {
            const std::size_t N = 32;
            std::vector<float> a(N * N), b(N * N), d(N * N);
            for (std::size_t i = 0; i < a.size(); ++i) {
                a[i] = 0.01f * static_cast<float>(i % 37) - 0.2f;
                b[i] = 0.02f * static_cast<float>(i % 29) - 0.3f;
                d[i] = 0.03f * static_cast<float>(i % 17) - 0.1f;
            }
            auto a_buf = runtime.create_buffer(a.size() * sizeof(float), a.data());
            auto b_buf = runtime.create_buffer(b.size() * sizeof(float), b.data());
            auto c_buf = runtime.create_buffer(N * N * sizeof(float));
            auto d_buf = runtime.create_buffer(d.size() * sizeof(float), d.data());
            auto e_buf = runtime.create_buffer(N * N * sizeof(float));

            expect(runtime.capture_begin(), "dispatch capture must start");
            auto mm = run_vulkan_f32_matmul(runtime, a_buf, b_buf, c_buf, N, N, N);
            auto ad = run_vulkan_add(runtime, c_buf, d_buf, e_buf, N * N);
            auto recording = runtime.capture_end();
            expect(mm.success && ad.success, "captured dispatches must execute during capture");
            expect(recording.size() == 2, "capture must record both dispatches");

            std::vector<float> eager(N * N, 0.0f);
            e_buf.download(eager.data(), eager.size() * sizeof(float));

            // Scribble over outputs, then replay and compare.
            std::vector<float> zeros(N * N, -123.0f);
            c_buf.upload(zeros.data(), zeros.size() * sizeof(float));
            e_buf.upload(zeros.data(), zeros.size() * sizeof(float));
            const auto replay_result = runtime.replay(recording);
            expect(replay_result.success, "captured recording must replay");
            std::vector<float> replayed(N * N, 0.0f);
            e_buf.download(replayed.data(), replayed.size() * sizeof(float));
            bool match = true;
            for (std::size_t i = 0; i < replayed.size(); ++i) {
                if (!close_enough(replayed[i], eager[i], 1e-5f)) {
                    match = false;
                    break;
                }
            }
            expect(match, "replayed outputs must match eager outputs");

            const int iters = 1000;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters / 2; ++i) {
                const auto r = runtime.replay(recording);
                if (!r.success) {
                    expect(false, "steady-state replay must not fail");
                    break;
                }
            }
            const auto t1 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters / 2; ++i) {
                const auto r = runtime.replay(recording);
                if (!r.success) {
                    expect(false, "steady-state replay must not fail");
                    break;
                }
            }
            const auto t2 = std::chrono::steady_clock::now();
            const double first_half = std::chrono::duration<double>(t1 - t0).count();
            const double second_half = std::chrono::duration<double>(t2 - t1).count();
            expect(second_half <= first_half * 1.5 + 0.05,
                   "1000-replay steady state must stay flat (no per-call pipeline/descriptor growth)");
            std::cout << "replay 1000x: first half " << first_half << " s, second half " << second_half
                      << " s\n";
        }
    }

    const std::vector<float> a = {1.0f, 2.0f, 3.0f};
    const std::vector<float> b = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const auto m1 = run_vulkan_f32_m1_matmul(a, b, 3, 2);
    expect(m1.success, "standalone Vulkan M=1 matmul must succeed when Vulkan compute is available");
    expect(m1.error.empty(), "standalone Vulkan M=1 matmul success must not carry an error");
    expect(m1.output.size() == 2, "standalone Vulkan M=1 matmul must return N outputs");
    if (m1.output.size() == 2) {
        expect(close_enough(m1.output[0], 22.0f) && close_enough(m1.output[1], 28.0f),
               "standalone Vulkan M=1 matmul output mismatch");
    }

    const std::vector<float> a2 = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    const auto matmul = run_vulkan_f32_matmul(a2, b, 2, 3, 2);
    expect(matmul.success, "standalone Vulkan general matmul must succeed when Vulkan compute is available");
    expect(matmul.error.empty(), "standalone Vulkan general matmul success must not carry an error");
    expect(matmul.output.size() == 4, "standalone Vulkan general matmul must return M*N outputs");
    if (matmul.output.size() == 4) {
        expect(close_enough(matmul.output[0], 22.0f) && close_enough(matmul.output[1], 28.0f) &&
                   close_enough(matmul.output[2], 49.0f) && close_enough(matmul.output[3], 64.0f),
               "standalone Vulkan general matmul output mismatch");
    }

    const std::vector<float> add_a = {1.0f, -2.0f, 3.5f, 4.0f};
    const std::vector<float> add_b = {4.0f, 5.0f, -0.5f, -1.0f};
    const auto add = run_vulkan_add(add_a, add_b);
    expect(add.success, "standalone Vulkan add must succeed when Vulkan compute is available");
    expect(add.error.empty(), "standalone Vulkan add success must not carry an error");
    expect(add.output.size() == add_a.size(), "standalone Vulkan add must return input-sized output");
    if (add.output.size() == add_a.size()) {
        for (std::size_t i = 0; i < add_a.size(); ++i) {
            expect(close_enough(add.output[i], add_a[i] + add_b[i], 1.0e-6f),
                   "standalone Vulkan add output mismatch");
        }
    }

    // Standalone vector-path for sub is intentionally omitted: device-resident
    // parity already covers sub through run_vulkan_sub(runtime, ...) above, and
    // the standalone vector-API path is a host-staging shim not on the
    // memory-native training hot path (per PORT_PROMPT invariant 4).

    const std::vector<std::int8_t> q8_a = {1, -2, 3, 4, 0, 2};
    const std::vector<std::int8_t> q8_b = {
        1, 2,
        -1, 3,
        4, -2,
    };
    const auto q8 = run_vulkan_i8_scaled_matmul(q8_a, q8_b, 2, 3, 2, 0.5f, 0.25f);
    expect(q8.success, "standalone Vulkan i8 scaled matmul must succeed when Vulkan compute is available");
    expect(q8.error.empty(), "standalone Vulkan i8 scaled matmul success must not carry an error");
    expect(q8.output.size() == 4, "standalone Vulkan i8 scaled matmul must return M*N outputs");
    if (q8.output.size() == 4) {
        const std::vector<float> expected = {
            (1.0f * 1.0f + -2.0f * -1.0f + 3.0f * 4.0f) * 0.125f,
            (1.0f * 2.0f + -2.0f * 3.0f + 3.0f * -2.0f) * 0.125f,
            (4.0f * 1.0f + 0.0f * -1.0f + 2.0f * 4.0f) * 0.125f,
            (4.0f * 2.0f + 0.0f * 3.0f + 2.0f * -2.0f) * 0.125f,
        };
        for (std::size_t i = 0; i < expected.size(); ++i) {
            expect(close_enough(q8.output[i], expected[i]), "standalone Vulkan i8 scaled matmul output mismatch");
        }
    }

    const std::size_t qt = 2, kt = 3, nh = 2, nkh = 1, hd = 2;
    const std::vector<float> gqa_q = {
        1.0f, 0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, -1.0f, 0.5f,
    };
    const std::vector<float> gqa_k = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };
    const std::vector<float> gqa_v = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
    };
    const auto gqa = run_vulkan_grouped_query_attention(gqa_q, gqa_k, gqa_v, qt, kt, nh, nkh, hd, 1.0f);
    expect(gqa.success, "standalone Vulkan GQA must succeed when Vulkan compute is available");
    expect(gqa.error.empty(), "standalone Vulkan GQA success must not carry an error");
    expect(gqa.output.size() == gqa_q.size(), "standalone Vulkan GQA must return query-sized output");
    if (gqa.output.size() == gqa_q.size()) {
        std::vector<float> expected(gqa_q.size(), 0.0f);
        for (std::size_t tq = 0; tq < qt; ++tq) {
            for (std::size_t h = 0; h < nh; ++h) {
                std::vector<float> scores(kt, 0.0f);
                for (std::size_t tk = 0; tk < kt; ++tk) {
                    for (std::size_t d = 0; d < hd; ++d) {
                        scores[tk] += gqa_q[(tq * nh + h) * hd + d] * gqa_k[tk * hd + d];
                    }
                }
                const float mx = std::max(scores[0], std::max(scores[1], scores[2]));
                float denom = 0.0f;
                for (float score : scores) denom += std::exp(score - mx);
                for (std::size_t d = 0; d < hd; ++d) {
                    float value = 0.0f;
                    for (std::size_t tk = 0; tk < kt; ++tk) {
                        value += (std::exp(scores[tk] - mx) / denom) * gqa_v[tk * hd + d];
                    }
                    expected[(tq * nh + h) * hd + d] = value;
                }
            }
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            expect(close_enough(gqa.output[i], expected[i], 2.0e-5f), "standalone Vulkan GQA output mismatch");
        }
    }

    const std::size_t C = 3;
    const std::uint32_t lv = static_cast<std::uint32_t>(2 * C - 1);
    auto code = [&](int t) {
        return static_cast<std::uint32_t>((t + 1) * static_cast<int>(lv) + static_cast<int>(C - 1));
    };
    const std::vector<std::uint32_t> counter_state = {
        pack_counter_word({code(1), code(0), code(-1), code(1)}),
        pack_counter_word({code(0), code(1), code(1), code(-1)}),
    };
    const std::vector<float> counter_scale = {0.5f, 2.0f};
    const std::vector<float> counter_grad_out = {1.0f, 2.0f, -1.0f, 0.5f};
    const auto counter_gx =
        run_vulkan_compact_counter_backward_input(counter_state, counter_scale, counter_grad_out, 2, 4, 2, C);
    expect(counter_gx.success, "standalone Vulkan compact-counter backward must succeed when Vulkan compute is available");
    expect(counter_gx.error.empty(), "standalone Vulkan compact-counter backward success must not carry an error");
    expect(counter_gx.output.size() == 8, "standalone Vulkan compact-counter backward must return batch*in outputs");
    if (counter_gx.output.size() == 8) {
        const std::vector<int> tern = {1, 0, -1, 1, 0, 1, 1, -1};
        for (std::size_t r = 0; r < 2; ++r) {
            for (std::size_t i = 0; i < 4; ++i) {
                const float expected =
                    counter_grad_out[r * 2 + 0] * counter_scale[0] * static_cast<float>(tern[i]) +
                    counter_grad_out[r * 2 + 1] * counter_scale[1] * static_cast<float>(tern[4 + i]);
                expect(close_enough(counter_gx.output[r * 4 + i], expected),
                       "standalone Vulkan compact-counter backward output mismatch");
            }
        }
    }

    const std::vector<std::uint32_t> inc_words = {pack_counter_word({1, 0, 1, 0})};
    const std::vector<std::uint32_t> inc_state = {pack_counter_word({10, 20, 30, 40})};
    const auto inc = run_vulkan_compact_counter_increment(inc_state, inc_words);
    expect(inc.success, "standalone Vulkan compact-counter increment must succeed when Vulkan compute is available");
    expect(inc.output.size() == 1, "standalone Vulkan compact-counter increment must return one word");
    if (inc.output.size() == 1) {
        expect(inc.output[0] == pack_counter_word({11, 20, 31, 40}),
               "standalone Vulkan compact-counter increment output mismatch");
    }

    const auto sgd = run_vulkan_sgd_update({1.0f, 2.0f, -3.0f}, {0.5f, -1.0f, 2.0f}, 0.1f);
    expect(sgd.success, "standalone Vulkan SGD update must succeed when Vulkan compute is available");
    expect(sgd.output.size() == 3, "standalone Vulkan SGD update must return input-sized output");
    if (sgd.output.size() == 3) {
        expect(close_enough(sgd.output[0], 0.95f) && close_enough(sgd.output[1], 2.1f) &&
                   close_enough(sgd.output[2], -3.2f),
               "standalone Vulkan SGD update output mismatch");
    }

    // === Slice Q1: quant core parity witnesses ===

    // Host reference emulation
    auto host_quantize_q8_rowwise = [](const std::vector<float>& x, std::size_t M, std::size_t K)
        -> std::pair<std::vector<std::int8_t>, std::vector<float>> {
        std::vector<std::int8_t> out(M * K);
        std::vector<float> scales(M);
        for (std::size_t row = 0; row < M; ++row) {
            float max_abs = 0.0f;
            for (std::size_t c = 0; c < K; ++c)
                max_abs = std::max(max_abs, std::fabs(x[row * K + c]));
            float scale = (max_abs <= 0.0f) ? 1.0f : (max_abs / 127.0f);
            scales[row] = scale;
            for (std::size_t c = 0; c < K; ++c) {
                int q = static_cast<int>(std::nearbyint(x[row * K + c] / scale));
                q = std::min(127, std::max(-127, q));
                out[row * K + c] = static_cast<std::int8_t>(q);
            }
        }
        return {out, scales};
    };

    auto host_dequant_q8 = [](const std::vector<std::int8_t>& x,
                               const std::vector<float>& scales,
                               std::uint32_t mode, std::size_t rows, std::size_t cols)
        -> std::vector<float> {
        (void)rows;
        std::vector<float> out(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            float scale;
            if (mode == 1) scale = scales[i / cols];
            else if (mode == 2) scale = scales[i - (i / cols) * cols];
            else scale = scales[0];
            out[i] = static_cast<float>(static_cast<int>(x[i])) * scale;
        }
        return out;
    };

    auto host_dequant_q4 = [](const std::vector<std::uint8_t>& packed,
                               const std::vector<float>& scales,
                               std::uint32_t mode, std::size_t rows, std::size_t cols,
                               std::size_t count)
        -> std::vector<float> {
        (void)rows;
        std::vector<float> out(count);
        for (std::size_t i = 0; i < count; ++i) {
            std::uint8_t byte = packed[i >> 1];
            std::uint8_t code = (i & 1) ? ((byte >> 4) & 15) : (byte & 15);
            int q = static_cast<int>(code) - 8;
            float scale;
            if (mode == 1) scale = scales[i / cols];
            else if (mode == 2) scale = scales[i - (i / cols) * cols];
            else scale = scales[0];
            out[i] = static_cast<float>(q) * scale;
        }
        return out;
    };

    auto host_mm_q8q8 = [](const std::vector<std::int8_t>& a,
                            const std::vector<float>& scales_a,
                            const std::vector<std::int8_t>& b,
                            const std::vector<float>& scales_b,
                            std::size_t M, std::size_t K, std::size_t N)
        -> std::vector<float> {
        std::vector<float> c(M * N, 0.0f);
        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t col = 0; col < N; ++col) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < K; ++k)
                    acc += static_cast<float>(static_cast<int>(a[row * K + k])) *
                           static_cast<float>(static_cast<int>(b[k * N + col]));
                c[row * N + col] = acc * scales_a[row] * scales_b[col];
            }
        }
        return c;
    };

    auto host_mm_q8q4 = [](const std::vector<std::int8_t>& a,
                            const std::vector<float>& scales_a,
                            const std::vector<std::uint8_t>& b_packed,
                            const std::vector<float>& scales_b,
                            std::size_t M, std::size_t K, std::size_t N)
        -> std::vector<float> {
        // Unpack b inline: index is (k*N + n), byte at idx>>1, nibble from LSB/high
        auto q4_load = [&](std::size_t k, std::size_t n_val) -> int {
            std::size_t idx = k * N + n_val;
            std::uint8_t byte = b_packed[idx >> 1];
            std::uint8_t code = (idx & 1) ? ((byte >> 4) & 15) : (byte & 15);
            return static_cast<int>(code) - 8;
        };
        std::vector<float> c(M * N, 0.0f);
        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t col = 0; col < N; ++col) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < K; ++k)
                    acc += static_cast<float>(static_cast<int>(a[row * K + k])) *
                           static_cast<float>(q4_load(k, col));
                c[row * N + col] = acc * scales_a[row] * scales_b[col];
            }
        }
        return c;
    };

    // quantize parity: bit-identical bytes and scales
    {
        const std::size_t M = 3, K = 47;
        std::vector<float> x(M * K);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = std::sin(static_cast<float>(i) * 0.37f) * 3.0f;
        auto [host_i8, host_scales] = host_quantize_q8_rowwise(x, M, K);
        auto in_buf = runtime.create_buffer(x.size() * sizeof(float), x.data());
        auto out_buf = runtime.create_buffer(host_i8.size() * sizeof(std::int8_t));
        auto scale_buf = runtime.create_buffer(host_scales.size() * sizeof(float));
        const auto qr = run_vulkan_quantize_q8_rowwise(runtime, in_buf, out_buf, scale_buf, M, K);
        expect(qr.success, "device-resident Vulkan quantize q8 rowwise must succeed");
        expect(qr.error.empty(), "Vulkan quantize q8 rowwise success must not carry an error");
        std::vector<std::int8_t> dev_i8(host_i8.size());
        out_buf.download(dev_i8.data(), dev_i8.size() * sizeof(std::int8_t));
        std::vector<float> dev_scales(host_scales.size());
        scale_buf.download(dev_scales.data(), dev_scales.size() * sizeof(float));
        expect(dev_i8 == host_i8, "Vulkan quantize q8 rowwise int8 output must be bit-identical to host");
        bool scales_ok = dev_scales.size() == host_scales.size();
        for (std::size_t i = 0; i < dev_scales.size() && scales_ok; ++i)
            if (!close_enough(dev_scales[i], host_scales[i], 1e-7f)) scales_ok = false;
        expect(scales_ok, "Vulkan quantize q8 rowwise scales must match host");
    }

    // dequant q8 parity
    {
        constexpr std::size_t count = 53, rows = 7, cols = 10;
        const std::uint32_t mode = 1;
        std::vector<std::int8_t> q(count);
        std::vector<float> scales(rows);
        for (std::size_t i = 0; i < count; ++i) q[i] = static_cast<std::int8_t>((i % 257) - 128);
        for (std::size_t i = 0; i < rows; ++i) scales[i] = 0.5f + 0.1f * static_cast<float>(i);
        const auto expected = host_dequant_q8(q, scales, mode, rows, cols);
        auto q_buf = runtime.create_buffer(q.size() * sizeof(std::int8_t), q.data());
        auto s_buf = runtime.create_buffer(scales.size() * sizeof(float), scales.data());
        auto o_buf = runtime.create_buffer(expected.size() * sizeof(float));
        const auto r = run_vulkan_dequantize_q8_scaled(runtime, q_buf, s_buf, o_buf, count, mode, rows, cols);
        expect(r.success, "device-resident Vulkan dequantize q8 must succeed");
        std::vector<float> out(expected.size());
        o_buf.download(out.data(), out.size() * sizeof(float));
        bool dq_ok = out.size() == expected.size();
        for (std::size_t i = 0; i < out.size() && dq_ok; ++i)
            if (!close_enough(out[i], expected[i], 1e-7f)) dq_ok = false;
        expect(dq_ok, "Vulkan dequantize q8 output mismatch vs host");
    }

    // dequant q4 parity
    {
        constexpr std::size_t count = 53, rows = 7, cols = 10;
        const std::uint32_t mode = 1;
        const std::size_t packed_bytes = (count + 1) / 2;
        std::vector<std::uint8_t> packed(packed_bytes);
        std::vector<float> scales(rows);
        // Fill packed with known Q4_0 values
        for (std::size_t i = 0; i < count; ++i) {
            int q = static_cast<int>((i % 15) - 7); // range [-7,7]
            std::uint8_t code = static_cast<std::uint8_t>(q + 8);
            if (i & 1) packed[i >> 1] |= (code << 4);
            else       packed[i >> 1] = code;
        }
        for (std::size_t i = 0; i < rows; ++i) scales[i] = 0.5f + 0.1f * static_cast<float>(i);
        const auto expected = host_dequant_q4(packed, scales, mode, rows, cols, count);
        auto p_buf = runtime.create_buffer(packed.size(), packed.data());
        auto s_buf = runtime.create_buffer(scales.size() * sizeof(float), scales.data());
        auto o_buf = runtime.create_buffer(expected.size() * sizeof(float));
        const auto r = run_vulkan_dequantize_q4_scaled(runtime, p_buf, s_buf, o_buf, count, mode, rows, cols);
        expect(r.success, "device-resident Vulkan dequantize q4 must succeed");
        std::vector<float> out(expected.size());
        o_buf.download(out.data(), out.size() * sizeof(float));
        bool dq4_ok = out.size() == expected.size();
        for (std::size_t i = 0; i < out.size() && dq4_ok; ++i)
            if (!close_enough(out[i], expected[i], 1e-7f)) dq4_ok = false;
        expect(dq4_ok, "Vulkan dequantize q4 output mismatch vs host");
    }

    // mm_q8q8 parity: shapes crossing tiles
    auto test_mm_q8q8 = [&](std::size_t M, std::size_t K, std::size_t N) {
        std::vector<std::int8_t> a(M * K);
        std::vector<std::int8_t> b(K * N);
        std::vector<float> scales_a(M), scales_b(N);
        for (std::size_t i = 0; i < a.size(); ++i)
            a[i] = static_cast<std::int8_t>((static_cast<int>(i * 7 + 3) % 257) - 128);
        for (std::size_t i = 0; i < b.size(); ++i)
            b[i] = static_cast<std::int8_t>((static_cast<int>(i * 11 + 5) % 257) - 128);
        for (std::size_t i = 0; i < M; ++i) scales_a[i] = 0.3f / 127.0f * (1.0f + 0.1f * static_cast<float>(i));
        for (std::size_t i = 0; i < N; ++i) scales_b[i] = 0.5f / 127.0f * (1.0f + 0.1f * static_cast<float>(i));
        const auto expected = host_mm_q8q8(a, scales_a, b, scales_b, M, K, N);
        auto a_buf = runtime.create_buffer(a.size() * sizeof(std::int8_t), a.data());
        auto b_buf = runtime.create_buffer(b.size() * sizeof(std::int8_t), b.data());
        auto sa_buf = runtime.create_buffer(scales_a.size() * sizeof(float), scales_a.data());
        auto sb_buf = runtime.create_buffer(scales_b.size() * sizeof(float), scales_b.data());
        auto c_buf = runtime.create_buffer(expected.size() * sizeof(float));
        const auto r = run_vulkan_matmul_q8q8_scaled(runtime, a_buf, sa_buf, b_buf, sb_buf, c_buf, M, K, N);
        expect(r.success, "device-resident Vulkan q8q8 matmul must succeed");
        std::vector<float> out(expected.size());
        c_buf.download(out.data(), out.size() * sizeof(float));
        const float base_tol = 1.0e-4f;
        bool mm_ok = out.size() == expected.size();
        for (std::size_t i = 0; i < out.size() && mm_ok; ++i) {
            float ref = expected[i];
            float tol = base_tol * std::max(1.0f, std::fabs(ref));
            if (std::fabs(out[i] - ref) > tol) mm_ok = false;
        }
        return mm_ok;
    };
    expect(test_mm_q8q8(33, 47, 65), "Vulkan q8q8 matmul (33,47,65) output mismatch vs host");
    expect(test_mm_q8q8(1, 128, 96), "Vulkan q8q8 matmul (1,128,96) output mismatch vs host");

    // mm_q8q4 parity: same shapes, odd N = 65 covers packed tail
    auto test_mm_q8q4 = [&](std::size_t M, std::size_t K, std::size_t N) {
        std::vector<std::int8_t> a(M * K);
        std::vector<float> scales_a(M), scales_b(N);
        for (std::size_t i = 0; i < a.size(); ++i)
            a[i] = static_cast<std::int8_t>((static_cast<int>(i * 7 + 3) % 257) - 128);
        for (std::size_t i = 0; i < M; ++i) scales_a[i] = 0.3f / 127.0f * (1.0f + 0.1f * static_cast<float>(i));
        for (std::size_t i = 0; i < N; ++i) scales_b[i] = 0.5f / 127.0f * (1.0f + 0.1f * static_cast<float>(i));
        // Build packed Q4_0 B
        const auto total_b = K * N;
        const auto packed_bytes = (total_b + 1) / 2;
        std::vector<std::uint8_t> b_packed(packed_bytes, 0);
        for (std::size_t idx = 0; idx < total_b; ++idx) {
            int q = static_cast<int>((idx * 11 + 5) % 15) - 7; // range [-7,7]
            std::uint8_t code = static_cast<std::uint8_t>(q + 8);
            if (idx & 1) b_packed[idx >> 1] |= (code << 4);
            else         b_packed[idx >> 1] = code;
        }
        const auto expected = host_mm_q8q4(a, scales_a, b_packed, scales_b, M, K, N);
        auto a_buf = runtime.create_buffer(a.size() * sizeof(std::int8_t), a.data());
        auto b_buf = runtime.create_buffer(b_packed.size(), b_packed.data());
        auto sa_buf = runtime.create_buffer(scales_a.size() * sizeof(float), scales_a.data());
        auto sb_buf = runtime.create_buffer(scales_b.size() * sizeof(float), scales_b.data());
        auto c_buf = runtime.create_buffer(expected.size() * sizeof(float));
        const auto r = run_vulkan_matmul_q8q4_scaled(runtime, a_buf, sa_buf, b_buf, sb_buf, c_buf, M, K, N);
        expect(r.success, "device-resident Vulkan q8q4 matmul must succeed");
        std::vector<float> out(expected.size());
        c_buf.download(out.data(), out.size() * sizeof(float));
        const float base_tol = 1.0e-4f;
        bool mm_ok = out.size() == expected.size();
        for (std::size_t i = 0; i < out.size() && mm_ok; ++i) {
            float ref = expected[i];
            float tol = base_tol * std::max(1.0f, std::fabs(ref));
            if (std::fabs(out[i] - ref) > tol) mm_ok = false;
        }
        return mm_ok;
    };
    expect(test_mm_q8q4(33, 47, 65), "Vulkan q8q4 matmul (33,47,65) output mismatch vs host");
    expect(test_mm_q8q4(1, 128, 96), "Vulkan q8q4 matmul (1,128,96) output mismatch vs host");

    // mm_f32q4_m1 parity: F32 x Q4_0 decode, shapes cross tiles
    auto host_mm_f32q4_m1 = [&](const std::vector<float>& a,
                                 const std::vector<std::uint8_t>& b_packed,
                                 const std::vector<float>& scales_b,
                                 std::size_t K, std::size_t N)
        -> std::vector<float> {
        auto q4_load = [&](std::size_t k, std::size_t col) -> int {
            std::size_t idx = k * N + col;
            std::uint8_t byte = b_packed[idx >> 1];
            std::uint8_t code = (idx & 1) ? ((byte >> 4) & 15) : (byte & 15);
            return static_cast<int>(code) - 8;
        };
        std::vector<float> c(N, 0.0f);
        for (std::size_t col = 0; col < N; ++col) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k)
                acc += a[k] * static_cast<float>(q4_load(k, col));
            c[col] = acc * scales_b[col];
        }
        return c;
    };
    auto test_mm_f32q4_m1 = [&](std::size_t K, std::size_t N) {
        std::vector<float> a(K);
        std::vector<float> scales_b(N);
        for (std::size_t i = 0; i < K; ++i)
            a[i] = std::sin(static_cast<float>(i) * 0.73f) * 2.0f;
        for (std::size_t i = 0; i < N; ++i)
            scales_b[i] = 0.5f / 7.0f * (1.0f + 0.1f * static_cast<float>(i));
        const auto total_b = K * N;
        const auto packed_bytes = (total_b + 1) / 2;
        std::vector<std::uint8_t> b_packed(packed_bytes, 0);
        for (std::size_t idx = 0; idx < total_b; ++idx) {
            int q = static_cast<int>((idx * 11 + 5) % 15) - 7;
            std::uint8_t code = static_cast<std::uint8_t>(q + 8);
            if (idx & 1) b_packed[idx >> 1] |= (code << 4);
            else         b_packed[idx >> 1] = code;
        }
        const auto expected = host_mm_f32q4_m1(a, b_packed, scales_b, K, N);
        auto a_buf = runtime.create_buffer(a.size() * sizeof(float), a.data());
        auto b_buf = runtime.create_buffer(b_packed.size(), b_packed.data());
        auto sb_buf = runtime.create_buffer(scales_b.size() * sizeof(float), scales_b.data());
        auto c_buf = runtime.create_buffer(expected.size() * sizeof(float));
        const auto r = run_vulkan_matmul_f32q4_m1(runtime, a_buf, b_buf, sb_buf, c_buf, K, N);
        expect(r.success, "device-resident Vulkan f32q4 M=1 matmul must succeed");
        std::vector<float> out(expected.size());
        c_buf.download(out.data(), out.size() * sizeof(float));
        const float base_tol = 1.0e-4f;
        bool mm_ok = out.size() == expected.size();
        for (std::size_t i = 0; i < out.size() && mm_ok; ++i) {
            float ref = expected[i];
            float tol = base_tol * std::max(1.0f, std::fabs(ref));
            if (std::fabs(out[i] - ref) > tol) mm_ok = false;
        }
        return mm_ok;
    };
    // Test shapes: odd K+N (non-multiple of 64, covers tail), K > 64 (stride loop),
    // K large enough for multiple stride iterations (K=128), N=96 (2 workgroups).
    expect(test_mm_f32q4_m1(47, 65), "Vulkan f32q4 M=1 matmul (47,65) output mismatch vs host");
    expect(test_mm_f32q4_m1(128, 96), "Vulkan f32q4 M=1 matmul (128,96) output mismatch vs host");
    expect(test_mm_f32q4_m1(1024, 256), "Vulkan f32q4 M=1 matmul (1024,256) output mismatch vs host");

    // Negative tests: incorrect buffer sizes
    {
        auto small_buf = runtime.create_buffer(4);
        auto ok_buf = runtime.create_buffer(256 * sizeof(float));
        auto scale_buf = runtime.create_buffer(4 * sizeof(float));
        const auto r1 = run_vulkan_quantize_q8_rowwise(runtime, small_buf, ok_buf, scale_buf, 4, 4);
        expect(!r1.success, "Vulkan quantize q8 must reject too-small input buffer");
        expect(!r1.error.empty(), "Vulkan quantize q8 validation failure must explain why");

        const auto r2 = run_vulkan_dequantize_q8_scaled(runtime, small_buf, scale_buf, ok_buf, 100, 0, 1, 1);
        expect(!r2.success, "Vulkan dequantize q8 must reject too-small int8 buffer");

        const auto r3 = run_vulkan_dequantize_q4_scaled(runtime, small_buf, scale_buf, ok_buf, 100, 0, 1, 1);
        expect(!r3.success, "Vulkan dequantize q4 must reject too-small packed buffer");

        const auto r4 = run_vulkan_matmul_q8q8_scaled(runtime, small_buf, scale_buf, ok_buf, scale_buf, ok_buf, 100, 100, 100);
        expect(!r4.success, "Vulkan q8q8 matmul must reject too-small A buffer");

        const auto r5 = run_vulkan_matmul_q8q4_scaled(runtime, small_buf, scale_buf, ok_buf, scale_buf, ok_buf, 100, 100, 100);
        expect(!r5.success, "Vulkan q8q4 matmul must reject too-small A buffer");
    }

    return ok ? 0 : 1;
}
