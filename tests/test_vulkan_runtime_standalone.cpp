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

    return ok ? 0 : 1;
}
