#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/runtime/microkernel.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

#include "test_utils.hpp"

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
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

    const auto result = probe_vulkan_runtime();
    expect(result.physical_device_count == 0 || result.instance_created,
           "Vulkan probe cannot report physical devices without an instance");
    expect(result.error.empty() || !result.available(),
           "Vulkan probe with an error must not report available=true");
    expect(!vulkan_version_string(0).empty(),
           "Vulkan version formatting must be total");

    const auto selection = select_microkernel_backend_from_value(MicrokernelDomain::Matmul, "vulkan");
    expect(selection.requested == MicrokernelBackendKind::Vulkan,
           "Vulkan backend selection must preserve requested backend");
    expect(selection.effective == MicrokernelBackendKind::Vulkan,
           "Vulkan matmul backend must select the implemented M=1 F32 descriptor");
    expect(!selection.fallback_to_opencl && selection.fallback_reason.empty(),
           "Vulkan matmul backend selection must not fallback once descriptor exists");

    const auto invalid_matmul = run_vulkan_f32_m1_matmul({1.0f, 2.0f}, {1.0f, 2.0f, 3.0f}, 2, 2);
    expect(!invalid_matmul.success, "Vulkan parameterized matmul must reject malformed B size");
    expect(!invalid_matmul.error.empty(), "Vulkan parameterized matmul validation failure must explain why");
    const auto invalid_general_matmul =
        run_vulkan_f32_matmul({1.0f, 2.0f}, {1.0f, 2.0f, 3.0f}, 2, 2, 2);
    expect(!invalid_general_matmul.success, "Vulkan general matmul must reject malformed A size");
    expect(!invalid_general_matmul.error.empty(), "Vulkan general matmul validation failure must explain why");
    const auto invalid_softmax = run_vulkan_softmax_rows({1.0f, 2.0f, 3.0f}, 2, 2);
    expect(!invalid_softmax.success, "Vulkan softmax rows must reject malformed input size");
    expect(!invalid_softmax.error.empty(), "Vulkan softmax rows validation failure must explain why");
    const auto invalid_rmsnorm =
        run_vulkan_rmsnorm({1.0f, 2.0f, 3.0f}, {1.0f, 1.0f}, 2, 2, 1.0e-6f);
    expect(!invalid_rmsnorm.success, "Vulkan RMSNorm must reject malformed input size");
    expect(!invalid_rmsnorm.error.empty(), "Vulkan RMSNorm validation failure must explain why");
    const auto invalid_rmsnorm_eps =
        run_vulkan_rmsnorm({1.0f, 2.0f}, {1.0f, 1.0f}, 1, 2, std::numeric_limits<float>::infinity());
    expect(!invalid_rmsnorm_eps.success, "Vulkan RMSNorm must reject non-finite eps");
    expect(!invalid_rmsnorm_eps.error.empty(), "Vulkan RMSNorm eps validation failure must explain why");
    const auto invalid_swiglu = run_vulkan_swiglu({1.0f, 2.0f, 3.0f}, 1, 2);
    expect(!invalid_swiglu.success, "Vulkan SwiGLU must reject malformed packed input size");
    expect(!invalid_swiglu.error.empty(), "Vulkan SwiGLU validation failure must explain why");
    const auto invalid_add = run_vulkan_add({1.0f, 2.0f}, {1.0f});
    expect(!invalid_add.success, "Vulkan add must reject mismatched input sizes");
    expect(!invalid_add.error.empty(), "Vulkan add validation failure must explain why");

    if (result.available()) {
        const bool require_vulkan_compute = std::getenv("MOTIFCL_REQUIRE_VULKAN_COMPUTE") != nullptr;
        const auto compute = run_vulkan_smoke_compute();
        if (!compute.success && !require_vulkan_compute) {
            std::cout << "Vulkan compute smoke skipped: " << compute.error << '\n';
            return ok ? 0 : 1;
        }
        expect(compute.success, "Vulkan smoke compute must succeed when strict Vulkan compute is required");
        expect(compute.error.empty(), "Vulkan smoke compute success must not carry an error");
        expect(compute.output == 42.0f, "Vulkan smoke compute must write the shader output");

        const auto fixed_matmul = run_vulkan_f32_matmul_smoke();
        expect(fixed_matmul.success, "Vulkan f32 matmul smoke must succeed when a Vulkan device is available");
        expect(fixed_matmul.error.empty(), "Vulkan f32 matmul smoke success must not carry an error");
        expect(fixed_matmul.output.size() == 4, "Vulkan f32 matmul smoke must return four output values");
        if (fixed_matmul.output.size() == 4) {
            expect(fixed_matmul.output[0] == 90.0f && fixed_matmul.output[1] == 100.0f &&
                       fixed_matmul.output[2] == 110.0f && fixed_matmul.output[3] == 120.0f,
                   "Vulkan f32 matmul smoke output mismatch");
        }

        const std::vector<float> a = {1.0f, 2.0f, 3.0f};
        const std::vector<float> b = {
            1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
        };
        const auto dynamic_matmul = run_vulkan_f32_m1_matmul(a, b, 3, 2);
        expect(dynamic_matmul.success,
               "Vulkan parameterized f32 M=1 matmul must succeed when a Vulkan device is available");
        expect(dynamic_matmul.error.empty(),
               "Vulkan parameterized f32 M=1 matmul success must not carry an error");
        expect(dynamic_matmul.output.size() == 2,
               "Vulkan parameterized f32 M=1 matmul must return N output values");
        if (dynamic_matmul.output.size() == 2) {
            expect(dynamic_matmul.output[0] == 22.0f && dynamic_matmul.output[1] == 28.0f,
                   "Vulkan parameterized f32 M=1 matmul output mismatch");
        }

        const std::vector<float> a2 = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        };
        const auto general_matmul = run_vulkan_f32_matmul(a2, b, 2, 3, 2);
        expect(general_matmul.success,
               "Vulkan general f32 matmul must succeed when a Vulkan device is available");
        expect(general_matmul.error.empty(),
               "Vulkan general f32 matmul success must not carry an error");
        expect(general_matmul.output.size() == 4,
               "Vulkan general f32 matmul must return M*N output values");
        if (general_matmul.output.size() == 4) {
            expect(general_matmul.output[0] == 22.0f && general_matmul.output[1] == 28.0f &&
                       general_matmul.output[2] == 49.0f && general_matmul.output[3] == 64.0f,
                   "Vulkan general f32 matmul output mismatch");
        }

        const std::vector<float> softmax_input = {
            1.0f, 2.0f, 3.0f,
            -1.0f, -1.0f, -1.0f,
        };
        const auto softmax = run_vulkan_softmax_rows(softmax_input, 2, 3);
        expect(softmax.success,
               "Vulkan softmax rows must succeed when a Vulkan device is available");
        expect(softmax.error.empty(),
               "Vulkan softmax rows success must not carry an error");
        expect(softmax.output.size() == softmax_input.size(),
               "Vulkan softmax rows must return rows*cols output values");
        if (softmax.output.size() == softmax_input.size()) {
            const float denom = std::exp(1.0f - 3.0f) + std::exp(2.0f - 3.0f) + 1.0f;
            expect(std::fabs(softmax.output[0] - std::exp(1.0f - 3.0f) / denom) <= 1e-5f &&
                       std::fabs(softmax.output[1] - std::exp(2.0f - 3.0f) / denom) <= 1e-5f &&
                       std::fabs(softmax.output[2] - 1.0f / denom) <= 1e-5f,
                   "Vulkan softmax rows first row output mismatch");
            expect(std::fabs(softmax.output[3] - (1.0f / 3.0f)) <= 1e-5f &&
                       std::fabs(softmax.output[4] - (1.0f / 3.0f)) <= 1e-5f &&
                       std::fabs(softmax.output[5] - (1.0f / 3.0f)) <= 1e-5f,
                   "Vulkan softmax rows uniform row output mismatch");
        }

        const std::vector<float> rms_x = {
            1.0f, 2.0f, 3.0f, 4.0f,
            -2.0f, 0.0f, 2.0f, 4.0f,
        };
        const std::vector<float> rms_w = {1.0f, 0.5f, 1.5f, 2.0f};
        const auto rmsnorm_result = run_vulkan_rmsnorm(rms_x, rms_w, 2, 4, 1.0e-6f);
        expect(rmsnorm_result.success,
               "Vulkan RMSNorm must succeed when a Vulkan device is available");
        expect(rmsnorm_result.error.empty(),
               "Vulkan RMSNorm success must not carry an error");
        expect(rmsnorm_result.output.size() == rms_x.size(),
               "Vulkan RMSNorm must return rows*cols output values");
        if (rmsnorm_result.output.size() == rms_x.size()) {
            for (std::size_t r = 0; r < 2; ++r) {
                float ss = 0.0f;
                for (std::size_t c = 0; c < 4; ++c) ss += rms_x[r * 4 + c] * rms_x[r * 4 + c];
                const float inv = 1.0f / std::sqrt(ss / 4.0f + 1.0e-6f);
                for (std::size_t c = 0; c < 4; ++c) {
                    const float expected = rms_x[r * 4 + c] * inv * rms_w[c];
                    expect(std::fabs(rmsnorm_result.output[r * 4 + c] - expected) <= 1e-5f,
                           "Vulkan RMSNorm output mismatch");
                }
            }
        }

        const std::vector<float> swiglu_input = {
            1.0f, -2.0f, 0.5f, 4.0f,
            -1.5f, 3.0f, -2.0f, 0.25f,
        };
        const auto swiglu_result = run_vulkan_swiglu(swiglu_input, 2, 2);
        expect(swiglu_result.success,
               "Vulkan SwiGLU must succeed when a Vulkan device is available");
        expect(swiglu_result.error.empty(),
               "Vulkan SwiGLU success must not carry an error");
        expect(swiglu_result.output.size() == 4,
               "Vulkan SwiGLU must return rows*hidden output values");
        if (swiglu_result.output.size() == 4) {
            for (std::size_t r = 0; r < 2; ++r) {
                for (std::size_t c = 0; c < 2; ++c) {
                    const float gate = swiglu_input[r * 4 + c];
                    const float up = swiglu_input[r * 4 + 2 + c];
                    const float expected = gate / (1.0f + std::exp(-gate)) * up;
                    expect(std::fabs(swiglu_result.output[r * 2 + c] - expected) <= 1e-5f,
                           "Vulkan SwiGLU output mismatch");
                }
            }
        }

        const std::vector<float> add_a = {1.0f, -2.0f, 3.5f, 4.0f};
        const std::vector<float> add_b = {4.0f, 5.0f, -0.5f, -1.0f};
        const auto add_result = run_vulkan_add(add_a, add_b);
        expect(add_result.success,
               "Vulkan add must succeed when a Vulkan device is available");
        expect(add_result.error.empty(),
               "Vulkan add success must not carry an error");
        expect(add_result.output.size() == add_a.size(),
               "Vulkan add must return input-sized output");
        if (add_result.output.size() == add_a.size()) {
            for (std::size_t i = 0; i < add_a.size(); ++i) {
                expect(std::fabs(add_result.output[i] - (add_a[i] + add_b[i])) <= 1e-6f,
                       "Vulkan add output mismatch");
            }
        }

        try {
            auto vk_backend = Backend::create_vulkan();
            auto VA = Tensor::from_cpu(vk_backend, {1, 3}, DType::F32, a.data());
            auto VB = Tensor::from_cpu(vk_backend, {3, 2}, DType::F32, b.data());
            auto VC = matmul(VA, VB);
            const auto vc = VC.to_vector<float>();
            expect(vc.size() == 2, "Vulkan-native Tensor matmul must return N output values");
            if (vc.size() == 2) {
                expect(std::fabs(vc[0] - 22.0f) <= 1e-5f && std::fabs(vc[1] - 28.0f) <= 1e-5f,
                       "Vulkan-native Tensor matmul output mismatch");
            }

            const std::vector<std::int8_t> q8_a = {
                1, -2, 3,
                4, 0, 2,
            };
            const std::vector<std::int8_t> q8_b = {
                2, -1,
                -3, 4,
                1, 2,
            };
            auto VQ8A = Tensor::from_cpu(vk_backend, {2, 3}, DType::Q8_0, q8_a.data());
            auto VQ8B = Tensor::from_cpu(vk_backend, {3, 2}, DType::Q8_0, q8_b.data());
            VQ8A._set_quant_scale(0.5f);
            VQ8B._set_quant_scale(0.25f);
            auto VQ8Y = matmul(VQ8A, VQ8B);
            const auto vq8y = VQ8Y.to_vector<float>();
            const std::vector<float> q8_expected = {
                (1.0f * 2.0f + -2.0f * -3.0f + 3.0f * 1.0f) * 0.125f,
                (1.0f * -1.0f + -2.0f * 4.0f + 3.0f * 2.0f) * 0.125f,
                (4.0f * 2.0f + 0.0f * -3.0f + 2.0f * 1.0f) * 0.125f,
                (4.0f * -1.0f + 0.0f * 4.0f + 2.0f * 2.0f) * 0.125f,
            };
            expect(vq8y.size() == q8_expected.size(), "Vulkan-native Tensor Q8 matmul must return M*N values");
            if (vq8y.size() == q8_expected.size()) {
                for (std::size_t i = 0; i < vq8y.size(); ++i) {
                    expect(std::fabs(vq8y[i] - q8_expected[i]) <= 1e-5f,
                           "Vulkan-native Tensor Q8 matmul output mismatch");
                }
            }

            auto VSX = Tensor::from_cpu(vk_backend, {2, 3}, DType::F32, softmax_input.data());
            auto VSY = softmax_rows(VSX);
            const auto vsy = VSY.to_vector<float>();
            expect(vsy.size() == softmax_input.size(), "Vulkan-native Tensor softmax must return rows*cols values");
            if (vsy.size() == softmax_input.size()) {
                expect(std::fabs(vsy[0] - softmax.output[0]) <= 1e-5f &&
                           std::fabs(vsy[1] - softmax.output[1]) <= 1e-5f &&
                           std::fabs(vsy[2] - softmax.output[2]) <= 1e-5f,
                       "Vulkan-native Tensor softmax output mismatch");
            }

            auto VRX = Tensor::from_cpu(vk_backend, {2, 4}, DType::F32, rms_x.data());
            auto VRW = Tensor::from_cpu(vk_backend, {4}, DType::F32, rms_w.data());
            auto VRY = rmsnorm(VRX, VRW);
            const auto vry = VRY.to_vector<float>();
            expect(vry.size() == rms_x.size(), "Vulkan-native Tensor RMSNorm must return rows*cols values");
            if (vry.size() == rms_x.size()) {
                expect(std::fabs(vry[0] - rmsnorm_result.output[0]) <= 1e-5f &&
                           std::fabs(vry[3] - rmsnorm_result.output[3]) <= 1e-5f &&
                           std::fabs(vry[4] - rmsnorm_result.output[4]) <= 1e-5f,
                       "Vulkan-native Tensor RMSNorm output mismatch");
            }

            auto VSwiX = Tensor::from_cpu(vk_backend, {2, 4}, DType::F32, swiglu_input.data());
            auto VSwiY = swiglu(VSwiX);
            const auto vswiy = VSwiY.to_vector<float>();
            expect(vswiy.size() == swiglu_result.output.size(),
                   "Vulkan-native Tensor SwiGLU must return rows*hidden values");
            if (vswiy.size() == swiglu_result.output.size()) {
                expect(std::fabs(vswiy[0] - swiglu_result.output[0]) <= 1e-5f &&
                           std::fabs(vswiy[1] - swiglu_result.output[1]) <= 1e-5f &&
                           std::fabs(vswiy[2] - swiglu_result.output[2]) <= 1e-5f,
                       "Vulkan-native Tensor SwiGLU output mismatch");
            }

            auto VAddA = Tensor::from_cpu(vk_backend, {2, 2}, DType::F32, add_a.data());
            auto VAddB = Tensor::from_cpu(vk_backend, {2, 2}, DType::F32, add_b.data());
            auto VAddY = add(VAddA, VAddB);
            const auto vadd_y = VAddY.to_vector<float>();
            expect(vadd_y.size() == add_result.output.size(),
                   "Vulkan-native Tensor add must return input-sized output");
            if (vadd_y.size() == add_result.output.size()) {
                expect(std::fabs(vadd_y[0] - add_result.output[0]) <= 1e-6f &&
                           std::fabs(vadd_y[1] - add_result.output[1]) <= 1e-6f &&
                           std::fabs(vadd_y[2] - add_result.output[2]) <= 1e-6f,
                       "Vulkan-native Tensor add output mismatch");
            }

            auto VSgdParam = Tensor::from_cpu(vk_backend, {2, 2}, DType::F32, add_a.data());
            auto VSgdGrad = Tensor::from_cpu(vk_backend, {2, 2}, DType::F32, add_b.data());
            sgd_update(VSgdParam, VSgdGrad, 0.1f);
            const auto vsgd = VSgdParam.to_vector<float>();
            expect(vsgd.size() == add_a.size(), "Vulkan-native Tensor SGD update must preserve param size");
            if (vsgd.size() == add_a.size()) {
                for (std::size_t i = 0; i < add_a.size(); ++i) {
                    expect(std::fabs(vsgd[i] - (add_a[i] - 0.1f * add_b[i])) <= 1e-6f,
                           "Vulkan-native Tensor SGD update output mismatch");
                }
            }

            // matmul backward parity: dA = dC * B^T (NT kernel), dB = A^T * dC
            // (TN kernel), checked against CPU references. Shapes cross the
            // 16x16 tile (multiple workgroups, multiple K-tiles, non-multiples
            // of 16) so cross-tile accumulation is asserted. Tolerance 1e-4
            // absolute (F32 accumulation over K=40 with |values| <= ~1).
            {
                const std::size_t M = 33, K = 40, N = 18;
                std::vector<float> mm_a(M * K);
                std::vector<float> mm_b(K * N);
                std::vector<float> mm_g(M * N);
                for (std::size_t i = 0; i < mm_a.size(); ++i) mm_a[i] = 0.25f * static_cast<float>(i % 7) - 0.5f;
                for (std::size_t i = 0; i < mm_b.size(); ++i) mm_b[i] = 0.125f * static_cast<float>(i % 5) - 0.25f;
                for (std::size_t i = 0; i < mm_g.size(); ++i) mm_g[i] = 0.5f * static_cast<float>(i % 3) - 0.5f;

                auto MA = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(M), static_cast<int64_t>(K)},
                                           DType::F32, mm_a.data());
                auto MB = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(K), static_cast<int64_t>(N)},
                                           DType::F32, mm_b.data());
                MA.set_requires_grad(true);
                MB.set_requires_grad(true);
                auto MC = matmul(MA, MB);
                auto MG = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(M), static_cast<int64_t>(N)},
                                           DType::F32, mm_g.data());
                MC.backward(MG);
                expect(MA.grad() != nullptr && MB.grad() != nullptr,
                       "Vulkan matmul backward must populate both input gradients");
                if (MA.grad() && MB.grad()) {
                    const auto ga = MA.grad()->to_vector<float>();
                    const auto gb = MB.grad()->to_vector<float>();
                    bool ga_ok = ga.size() == M * K;
                    bool gb_ok = gb.size() == K * N;
                    for (std::size_t m = 0; m < M && ga_ok; ++m) {
                        for (std::size_t k2 = 0; k2 < K; ++k2) {
                            float ref = 0.0f;
                            for (std::size_t n2 = 0; n2 < N; ++n2) ref += mm_g[m * N + n2] * mm_b[k2 * N + n2];
                            if (std::fabs(ga[m * K + k2] - ref) > 1e-4f) ga_ok = false;
                        }
                    }
                    for (std::size_t k2 = 0; k2 < K && gb_ok; ++k2) {
                        for (std::size_t n2 = 0; n2 < N; ++n2) {
                            float ref = 0.0f;
                            for (std::size_t m = 0; m < M; ++m) ref += mm_a[m * K + k2] * mm_g[m * N + n2];
                            if (std::fabs(gb[k2 * N + n2] - ref) > 1e-4f) gb_ok = false;
                        }
                    }
                    expect(ga_ok, "Vulkan matmul backward dA parity mismatch vs CPU reference");
                    expect(gb_ok, "Vulkan matmul backward dB parity mismatch vs CPU reference");
                }

                // M=1 decode form exercises the wave-per-output NT kernel in backward.
                auto M1A = Tensor::from_cpu(vk_backend, {1, static_cast<int64_t>(K)}, DType::F32, mm_a.data());
                auto M1B = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(K), static_cast<int64_t>(N)},
                                            DType::F32, mm_b.data());
                M1A.set_requires_grad(true);
                M1B.set_requires_grad(true);
                auto M1C = matmul(M1A, M1B);
                auto M1G = Tensor::from_cpu(vk_backend, {1, static_cast<int64_t>(N)}, DType::F32, mm_g.data());
                M1C.backward(M1G);
                expect(M1A.grad() != nullptr && M1B.grad() != nullptr,
                       "Vulkan M=1 matmul backward must populate both input gradients");
                if (M1A.grad() && M1B.grad()) {
                    const auto ga = M1A.grad()->to_vector<float>();
                    const auto gb = M1B.grad()->to_vector<float>();
                    bool ok = ga.size() == K;
                    bool gb_ok = gb.size() == K * N;
                    for (std::size_t k2 = 0; k2 < K && ok; ++k2) {
                        float ref = 0.0f;
                        for (std::size_t n2 = 0; n2 < N; ++n2) ref += mm_g[n2] * mm_b[k2 * N + n2];
                        if (std::fabs(ga[k2] - ref) > 1e-4f) ok = false;
                    }
                    // dB = A^T[K,1] * dC[1,N] (outer product through the TN kernel).
                    for (std::size_t k2 = 0; k2 < K && gb_ok; ++k2) {
                        for (std::size_t n2 = 0; n2 < N; ++n2) {
                            const float ref = mm_a[k2] * mm_g[n2];
                            if (std::fabs(gb[k2 * N + n2] - ref) > 1e-4f) gb_ok = false;
                        }
                    }
                    expect(ok, "Vulkan M=1 matmul backward dA parity mismatch vs CPU reference");
                    expect(gb_ok, "Vulkan M=1 matmul backward dB parity mismatch vs CPU reference");
                }
            }

            // RMSNorm backward parity (dx wave-per-row kernel + two-stage dw)
            // vs CPU reference; tolerance 1e-4 absolute.
            {
                const std::size_t R = 3, C = 8;
                const float eps = 1e-5f;
                std::vector<float> nx(R * C), nw(C), ng(R * C);
                for (std::size_t i = 0; i < nx.size(); ++i) nx[i] = 0.2f * static_cast<float>(i % 9) - 0.8f;
                for (std::size_t i = 0; i < nw.size(); ++i) nw[i] = 0.5f + 0.1f * static_cast<float>(i % 4);
                for (std::size_t i = 0; i < ng.size(); ++i) ng[i] = 0.3f * static_cast<float>(i % 5) - 0.6f;

                auto NX = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(R), static_cast<int64_t>(C)},
                                           DType::F32, nx.data());
                auto NW = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(C)}, DType::F32, nw.data());
                NX.set_requires_grad(true);
                NW.set_requires_grad(true);
                auto NY = rmsnorm(NX, NW, eps);
                auto NG = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(R), static_cast<int64_t>(C)},
                                           DType::F32, ng.data());
                NY.backward(NG);
                expect(NX.grad() != nullptr && NW.grad() != nullptr,
                       "Vulkan rmsnorm backward must populate x and weight gradients");
                if (NX.grad() && NW.grad()) {
                    const auto gx = NX.grad()->to_vector<float>();
                    const auto gw = NW.grad()->to_vector<float>();
                    bool gx_ok = gx.size() == R * C;
                    bool gw_ok = gw.size() == C;
                    std::vector<float> inv_r(R);
                    for (std::size_t r = 0; r < R; ++r) {
                        float ss = 0.0f;
                        for (std::size_t c = 0; c < C; ++c) ss += nx[r * C + c] * nx[r * C + c];
                        inv_r[r] = 1.0f / std::sqrt(ss / static_cast<float>(C) + eps);
                    }
                    for (std::size_t r = 0; r < R && gx_ok; ++r) {
                        float dot = 0.0f;
                        for (std::size_t c = 0; c < C; ++c) dot += ng[r * C + c] * nw[c] * nx[r * C + c];
                        for (std::size_t c = 0; c < C; ++c) {
                            const float inv = inv_r[r];
                            const float ref = ng[r * C + c] * nw[c] * inv -
                                              nx[r * C + c] * inv * inv * inv * dot / static_cast<float>(C);
                            if (std::fabs(gx[r * C + c] - ref) > 1e-4f) gx_ok = false;
                        }
                    }
                    for (std::size_t c = 0; c < C && gw_ok; ++c) {
                        float ref = 0.0f;
                        for (std::size_t r = 0; r < R; ++r) ref += ng[r * C + c] * nx[r * C + c] * inv_r[r];
                        if (std::fabs(gw[c] - ref) > 1e-4f) gw_ok = false;
                    }
                    expect(gx_ok, "Vulkan rmsnorm backward dx parity mismatch vs CPU reference");
                    expect(gw_ok, "Vulkan rmsnorm backward dw parity mismatch vs CPU reference");
                }
            }

            // SwiGLU backward parity vs CPU reference; tolerance 1e-4.
            {
                const std::size_t R = 2, H = 4;
                std::vector<float> sp(R * 2 * H), sg(R * H);
                for (std::size_t i = 0; i < sp.size(); ++i) sp[i] = 0.3f * static_cast<float>(i % 7) - 0.9f;
                for (std::size_t i = 0; i < sg.size(); ++i) sg[i] = 0.4f * static_cast<float>(i % 3) - 0.4f;
                auto SP = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(R), static_cast<int64_t>(2 * H)},
                                           DType::F32, sp.data());
                SP.set_requires_grad(true);
                auto SY = swiglu(SP);
                auto SG = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(R), static_cast<int64_t>(H)},
                                           DType::F32, sg.data());
                SY.backward(SG);
                expect(SP.grad() != nullptr, "Vulkan swiglu backward must populate packed gradient");
                if (SP.grad()) {
                    const auto gp = SP.grad()->to_vector<float>();
                    bool ok = gp.size() == R * 2 * H;
                    for (std::size_t r = 0; r < R && ok; ++r) {
                        for (std::size_t c = 0; c < H; ++c) {
                            const float gate = sp[r * 2 * H + c];
                            const float up = sp[r * 2 * H + H + c];
                            const float go = sg[r * H + c];
                            const float sig = 1.0f / (1.0f + std::exp(-gate));
                            const float dgate = go * up * (sig + gate * sig * (1.0f - sig));
                            const float dup = go * gate * sig;
                            if (std::fabs(gp[r * 2 * H + c] - dgate) > 1e-4f) ok = false;
                            if (std::fabs(gp[r * 2 * H + H + c] - dup) > 1e-4f) ok = false;
                        }
                    }
                    expect(ok, "Vulkan swiglu backward parity mismatch vs CPU reference");
                }
            }

            // GELU forward + backward parity (tanh approximation); tol 1e-4.
            {
                const std::size_t N = 7;
                std::vector<float> gxv(N), ggv(N);
                for (std::size_t i = 0; i < N; ++i) {
                    gxv[i] = 0.7f * static_cast<float>(i) - 2.0f;
                    ggv[i] = 0.25f * static_cast<float>(i % 4) - 0.25f;
                }
                auto GX = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(N)}, DType::F32, gxv.data());
                GX.set_requires_grad(true);
                auto GY = gelu(GX);
                const auto gy = GY.to_vector<float>();
                auto GG = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(N)}, DType::F32, ggv.data());
                GY.backward(GG);
                bool fwd_ok = gy.size() == N;
                bool bwd_ok = GX.grad() != nullptr;
                const float cst = 0.7978845608028654f;
                for (std::size_t i = 0; i < N && fwd_ok; ++i) {
                    const float v = gxv[i];
                    const float t = cst * (v + 0.044715f * v * v * v);
                    const float ref = 0.5f * v * (1.0f + std::tanh(t));
                    if (std::fabs(gy[i] - ref) > 1e-4f) fwd_ok = false;
                }
                if (bwd_ok) {
                    const auto gd = GX.grad()->to_vector<float>();
                    for (std::size_t i = 0; i < N && bwd_ok; ++i) {
                        const float v = gxv[i];
                        const float inner = cst * (v + 0.044715f * v * v * v);
                        const float th = std::tanh(inner);
                        const float sech2 = 1.0f - th * th;
                        const float inner_grad = cst * (1.0f + 3.0f * 0.044715f * v * v);
                        const float ref = ggv[i] * (0.5f * (1.0f + th) + 0.5f * v * sech2 * inner_grad);
                        if (std::fabs(gd[i] - ref) > 1e-4f) bwd_ok = false;
                    }
                }
                expect(fwd_ok, "Vulkan gelu forward parity mismatch vs CPU reference");
                expect(bwd_ok, "Vulkan gelu backward parity mismatch vs CPU reference");
            }

            // softmax rows backward (runtime-level kernel) vs CPU reference.
            {
                const std::size_t R = 2, C = 100;
                std::vector<float> sy(R * C), sdy(R * C);
                for (std::size_t r = 0; r < R; ++r) {
                    float sum = 0.0f;
                    for (std::size_t c = 0; c < C; ++c) {
                        sy[r * C + c] = std::exp(0.01f * static_cast<float>((c * 7 + r * 3) % 23));
                        sum += sy[r * C + c];
                    }
                    for (std::size_t c = 0; c < C; ++c) sy[r * C + c] /= sum;
                }
                for (std::size_t i = 0; i < sdy.size(); ++i) sdy[i] = 0.2f * static_cast<float>(i % 6) - 0.5f;
                auto& vk_runtime = vk_backend.vulkan_runtime();
                auto YB = vk_runtime.create_buffer(sy.size() * sizeof(float), sy.data());
                auto DYB = vk_runtime.create_buffer(sdy.size() * sizeof(float), sdy.data());
                auto DXB = vk_runtime.create_buffer(sy.size() * sizeof(float));
                const auto sres = run_vulkan_softmax_rows_backward(vk_runtime, YB, DYB, DXB, R, C);
                expect(sres.success, "Vulkan softmax backward dispatch must succeed");
                if (sres.success) {
                    std::vector<float> sdx(R * C, 0.0f);
                    DXB.download(sdx.data(), sdx.size() * sizeof(float));
                    bool ok = true;
                    for (std::size_t r = 0; r < R && ok; ++r) {
                        float dot = 0.0f;
                        for (std::size_t c = 0; c < C; ++c) dot += sdy[r * C + c] * sy[r * C + c];
                        for (std::size_t c = 0; c < C; ++c) {
                            const float ref = sy[r * C + c] * (sdy[r * C + c] - dot);
                            if (std::fabs(sdx[r * C + c] - ref) > 1e-5f) ok = false;
                        }
                    }
                    expect(ok, "Vulkan softmax backward parity mismatch vs CPU reference");
                }
            }

            nn::CounterStateLinear counter_layer(vk_backend, 4, 2, 3, 0.01f, 0.0f, 1.0f, 0.9f, 1.0e-8f, 123u);
            counter_layer.set_training(false);
            const auto counter_state = counter_layer.state.to_vector<std::uint8_t>();
            const auto counter_scale = counter_layer.scale.to_vector<float>();
            auto decode_counter_t = [&](std::size_t row, std::size_t feature) -> float {
                const std::size_t gpr = 1;
                const std::size_t group = feature / 4;
                const std::size_t lane = feature % 4;
                const std::size_t base = (row * gpr + group) * 3;
                const std::uint32_t word = static_cast<std::uint32_t>(counter_state[base]) |
                                           (static_cast<std::uint32_t>(counter_state[base + 1]) << 8u) |
                                           (static_cast<std::uint32_t>(counter_state[base + 2]) << 16u);
                const std::uint32_t code = (word >> (lane * 6u)) & 0x3fu;
                return static_cast<float>(code / 5u) - 1.0f;
            };

            const auto counter_weight = counter_layer.decode_weight().to_vector<float>();
            expect(counter_weight.size() == 8,
                   "Vulkan CounterStateLinear decode_weight must return out*in values");
            if (counter_weight.size() == 8 && counter_scale.size() == 2) {
                for (std::size_t row = 0; row < 2; ++row) {
                    for (std::size_t col = 0; col < 4; ++col) {
                        const float expected = counter_scale[row] * decode_counter_t(row, col);
                        expect(std::fabs(counter_weight[row * 4 + col] - expected) <= 1e-6f,
                               "Vulkan CounterStateLinear decode_weight output mismatch");
                    }
                }
            }

            const std::vector<float> counter_x = {
                1.0f, 2.0f, -1.0f, 0.5f,
                -0.25f, 0.75f, 1.5f, -2.0f,
            };
            auto CounterX = Tensor::from_cpu(vk_backend, {2, 4}, DType::F32, counter_x.data());
            const auto counter_y = counter_layer.forward(CounterX).to_vector<float>();
            expect(counter_y.size() == 4,
                   "Vulkan CounterStateLinear inference forward must return batch*out values");
            if (counter_y.size() == 4 && counter_weight.size() == 8) {
                for (std::size_t r = 0; r < 2; ++r) {
                    for (std::size_t o = 0; o < 2; ++o) {
                        float expected = 0.0f;
                        for (std::size_t i = 0; i < 4; ++i) {
                            expected += counter_x[r * 4 + i] * counter_weight[o * 4 + i];
                        }
                        expect(std::fabs(counter_y[r * 2 + o] - expected) <= 1e-5f,
                               "Vulkan CounterStateLinear inference forward output mismatch");
                    }
                }
            }

            const std::vector<float> counter_grad_out = {
                1.0f, -0.5f,
                0.25f, 2.0f,
            };
            auto CounterGradOut = Tensor::from_cpu(vk_backend, {2, 2}, DType::F32, counter_grad_out.data());
            const auto counter_grad_x = counter_layer.backward_input_from_state(CounterGradOut).to_vector<float>();
            expect(counter_grad_x.size() == counter_x.size(),
                   "Vulkan CounterStateLinear backward_input must return batch*in values");
            if (counter_grad_x.size() == counter_x.size() && counter_scale.size() == 2) {
                for (std::size_t r = 0; r < 2; ++r) {
                    for (std::size_t i = 0; i < 4; ++i) {
                        float expected = 0.0f;
                        for (std::size_t o = 0; o < 2; ++o) {
                            expected += counter_grad_out[r * 2 + o] * counter_scale[o] *
                                        decode_counter_t(o, i);
                        }
                        expect(std::fabs(counter_grad_x[r * 4 + i] - expected) <= 1e-5f,
                               "Vulkan CounterStateLinear backward_input output mismatch");
                    }
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
            const auto gqa_ref = run_vulkan_grouped_query_attention(gqa_q, gqa_k, gqa_v, qt, kt, nh, nkh, hd, 1.0f);
            expect(gqa_ref.success, "Vulkan GQA reference dispatch must succeed");
            auto VGQ = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(qt), static_cast<int64_t>(nh * hd)},
                                        DType::F32, gqa_q.data());
            auto VGK = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(kt), static_cast<int64_t>(nkh * hd)},
                                        DType::F32, gqa_k.data());
            auto VGV = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(kt), static_cast<int64_t>(nkh * hd)},
                                        DType::F32, gqa_v.data());
            auto VGY = grouped_query_attention(VGQ, VGK, VGV, static_cast<int>(nh), static_cast<int>(nkh),
                                               false, 1, static_cast<int64_t>(qt), static_cast<int64_t>(kt), 0, 1.0f);
            const auto vgy = VGY.to_vector<float>();
            expect(vgy.size() == gqa_ref.output.size(), "Vulkan-native Tensor GQA must return query-sized output");
            if (vgy.size() == gqa_ref.output.size()) {
                for (std::size_t i = 0; i < vgy.size(); ++i) {
                    expect(std::fabs(vgy[i] - gqa_ref.output[i]) <= 2.0e-5f,
                           "Vulkan-native Tensor GQA output mismatch");
                }
            }

            // Softmax cross-entropy forward + backward parity vs CPU
            // reference (mean over rows; ignore out-of-range targets).
            // Tolerance 1e-4 absolute.
            {
                const std::size_t R = 6, C = 37;
                std::vector<float> logits(R * C);
                std::vector<std::int32_t> tgt(R);
                std::vector<float> ce_g = {0.7f};
                for (std::size_t i = 0; i < logits.size(); ++i) {
                    logits[i] = 0.15f * static_cast<float>((i * 13 + 5) % 29) - 2.0f;
                }
                for (std::size_t r = 0; r < R; ++r) tgt[r] = static_cast<std::int32_t>((r * 11 + 2) % C);
                auto LG = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(R), static_cast<int64_t>(C)},
                                           DType::F32, logits.data());
                auto TG = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(R)}, DType::I32, tgt.data());
                LG.set_requires_grad(true);
                auto LOSS = softmax_cross_entropy(LG, TG);
                const float loss_v = LOSS.item();
                auto CEG = Tensor::from_cpu(vk_backend, {}, DType::F32, ce_g.data());
                LOSS.backward(CEG);

                float ref_loss = 0.0f;
                std::vector<float> ref_grad(R * C, 0.0f);
                for (std::size_t r = 0; r < R; ++r) {
                    float maxv = -3.4e38f;
                    for (std::size_t c = 0; c < C; ++c) maxv = std::max(maxv, logits[r * C + c]);
                    float sum = 0.0f;
                    for (std::size_t c = 0; c < C; ++c) sum += std::exp(logits[r * C + c] - maxv);
                    ref_loss += std::log(sum) + maxv - logits[r * C + static_cast<std::size_t>(tgt[r])];
                    for (std::size_t c = 0; c < C; ++c) {
                        const float prob = std::exp(logits[r * C + c] - maxv) / sum;
                        const float y = (static_cast<std::int32_t>(c) == tgt[r]) ? 1.0f : 0.0f;
                        ref_grad[r * C + c] = ce_g[0] / static_cast<float>(R) * (prob - y);
                    }
                }
                ref_loss /= static_cast<float>(R);
                expect(std::fabs(loss_v - ref_loss) <= 1e-4f,
                       "Vulkan softmax_cross_entropy forward parity mismatch vs CPU reference");
                expect(LG.grad() != nullptr, "Vulkan softmax_cross_entropy backward must populate grad");
                if (LG.grad()) {
                    const auto gl = LG.grad()->to_vector<float>();
                    bool ok = gl.size() == R * C;
                    for (std::size_t i = 0; i < gl.size() && ok; ++i) {
                        if (std::fabs(gl[i] - ref_grad[i]) > 1e-4f) ok = false;
                    }
                    expect(ok, "Vulkan softmax_cross_entropy backward parity mismatch vs CPU reference");
                }
            }

            // Compact-counter fused state update parity: identical seed on
            // the OpenCL and Vulkan backends must produce bit-identical state
            // bytes and matching scale/v rows (the SR tick is hash-seeded,
            // not driver-random).
            try {
                auto cl_backend = Backend::create_opencl();
                const int cin = 8, cout = 2, cC = 3;
                const std::uint32_t update_seed = 777u;
                nn::CounterStateLinear cl_layer(cl_backend, cin, cout, cC, 0.05f, 0.01f, 1.0f, 0.9f, 1.0e-8f,
                                                42u);
                nn::CounterStateLinear vk_layer(vk_backend, cin, cout, cC, 0.05f, 0.01f, 1.0f, 0.9f, 1.0e-8f,
                                                42u);
                const auto state0_cl = cl_layer.state.to_vector<std::uint8_t>();
                const auto state0_vk = vk_layer.state.to_vector<std::uint8_t>();
                expect(state0_cl == state0_vk,
                       "counter layers with the same seed must start from identical state");

                std::vector<float> cx(2 * cin), cgo(2 * cout);
                for (std::size_t i = 0; i < cx.size(); ++i) cx[i] = 0.2f * static_cast<float>(i % 5) - 0.4f;
                for (std::size_t i = 0; i < cgo.size(); ++i) cgo[i] = 0.3f * static_cast<float>(i % 3) - 0.3f;
                auto CLX = Tensor::from_cpu(cl_backend, {2, cin}, DType::F32, cx.data());
                auto CLG = Tensor::from_cpu(cl_backend, {2, cout}, DType::F32, cgo.data());
                auto VKX = Tensor::from_cpu(vk_backend, {2, cin}, DType::F32, cx.data());
                auto VKG = Tensor::from_cpu(vk_backend, {2, cout}, DType::F32, cgo.data());

                cl_layer.apply_update_backward(CLG, CLX, update_seed);
                vk_layer.apply_update_backward(VKG, VKX, update_seed);
                cl_backend.finish();

                const auto state_cl = cl_layer.state.to_vector<std::uint8_t>();
                const auto state_vk = vk_layer.state.to_vector<std::uint8_t>();
                expect(state_cl == state_vk,
                       "Vulkan fused counter update state bytes must match OpenCL for the same seed");
                const auto scale_cl = cl_layer.scale.to_vector<float>();
                const auto scale_vk = vk_layer.scale.to_vector<float>();
                const auto v_cl = cl_layer.v.to_vector<float>();
                const auto v_vk = vk_layer.v.to_vector<float>();
                bool rows_ok = scale_cl.size() == scale_vk.size() && v_cl.size() == v_vk.size();
                for (std::size_t i = 0; rows_ok && i < scale_cl.size(); ++i) {
                    if (std::fabs(scale_cl[i] - scale_vk[i]) > 1e-6f) rows_ok = false;
                    if (std::fabs(v_cl[i] - v_vk[i]) > 1e-6f) rows_ok = false;
                }
                expect(rows_ok, "Vulkan fused counter update scale/v rows must match OpenCL");
            } catch (const std::exception& e) {
                std::cout << "counter fused update parity skipped (OpenCL unavailable): " << e.what()
                          << '\n';
            }

            // GQA backward parity (non-causal, batch=1, default 1/sqrt(hd)
            // scale) against a full CPU attention-backward reference.
            // kt=70 crosses the 64-lane stride; nh=4/nkh=2 exercises grouped
            // KV accumulation. Tolerance 1e-4 absolute.
            {
                const std::size_t bqt = 5, bkt = 70, bnh = 4, bnkh = 2, bhd = 16;
                const float scale = 1.0f / std::sqrt(static_cast<float>(bhd));
                std::vector<float> bq(bqt * bnh * bhd), bk(bkt * bnkh * bhd), bv(bkt * bnkh * bhd);
                std::vector<float> bg(bqt * bnh * bhd);
                auto fill = [](std::vector<float>& v, std::uint32_t seed) {
                    std::uint32_t s = seed | 1u;
                    for (auto& x : v) {
                        s ^= s << 13;
                        s ^= s >> 17;
                        s ^= s << 5;
                        x = static_cast<float>(static_cast<std::int32_t>(s % 1001) - 500) / 500.0f;
                    }
                };
                fill(bq, 0x1201u);
                fill(bk, 0x2302u);
                fill(bv, 0x3403u);
                fill(bg, 0x4504u);

                auto BQ = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(bqt), static_cast<int64_t>(bnh * bhd)},
                                           DType::F32, bq.data());
                auto BK = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(bkt), static_cast<int64_t>(bnkh * bhd)},
                                           DType::F32, bk.data());
                auto BV = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(bkt), static_cast<int64_t>(bnkh * bhd)},
                                           DType::F32, bv.data());
                BQ.set_requires_grad(true);
                BK.set_requires_grad(true);
                BV.set_requires_grad(true);
                auto BY = grouped_query_attention(BQ, BK, BV, static_cast<int>(bnh), static_cast<int>(bnkh),
                                                  false, 1, static_cast<int64_t>(bqt), static_cast<int64_t>(bkt),
                                                  0, 0.0f);
                auto BG = Tensor::from_cpu(vk_backend, {static_cast<int64_t>(bqt), static_cast<int64_t>(bnh * bhd)},
                                           DType::F32, bg.data());
                BY.backward(BG);
                expect(BQ.grad() != nullptr && BK.grad() != nullptr && BV.grad() != nullptr,
                       "Vulkan GQA backward must populate q/k/v gradients");
                if (BQ.grad() && BK.grad() && BV.grad()) {
                    const auto gq = BQ.grad()->to_vector<float>();
                    const auto gk = BK.grad()->to_vector<float>();
                    const auto gv = BV.grad()->to_vector<float>();
                    std::vector<float> ref_dq(bq.size(), 0.0f), ref_dk(bk.size(), 0.0f), ref_dv(bv.size(), 0.0f);
                    const std::size_t qs = bnh * bhd;
                    const std::size_t kvs = bnkh * bhd;
                    const std::size_t gs = bnh / bnkh;
                    for (std::size_t h = 0; h < bnh; ++h) {
                        const std::size_t g = h / gs;
                        for (std::size_t tq2 = 0; tq2 < bqt; ++tq2) {
                            std::vector<float> sc(bkt);
                            float maxv = -3.4e38f;
                            for (std::size_t tk2 = 0; tk2 < bkt; ++tk2) {
                                float dot = 0.0f;
                                for (std::size_t d = 0; d < bhd; ++d) {
                                    dot += bq[tq2 * qs + h * bhd + d] * bk[tk2 * kvs + g * bhd + d];
                                }
                                sc[tk2] = dot * scale;
                                maxv = std::max(maxv, sc[tk2]);
                            }
                            float sum = 0.0f;
                            for (auto& s2 : sc) {
                                s2 = std::exp(s2 - maxv);
                                sum += s2;
                            }
                            for (auto& s2 : sc) s2 /= sum;
                            std::vector<float> dp(bkt, 0.0f);
                            float dot_dp_p = 0.0f;
                            for (std::size_t tk2 = 0; tk2 < bkt; ++tk2) {
                                for (std::size_t d = 0; d < bhd; ++d) {
                                    dp[tk2] += bg[tq2 * qs + h * bhd + d] * bv[tk2 * kvs + g * bhd + d];
                                }
                                dot_dp_p += dp[tk2] * sc[tk2];
                            }
                            for (std::size_t tk2 = 0; tk2 < bkt; ++tk2) {
                                const float ds2 = sc[tk2] * (dp[tk2] - dot_dp_p);
                                for (std::size_t d = 0; d < bhd; ++d) {
                                    ref_dq[tq2 * qs + h * bhd + d] += scale * ds2 * bk[tk2 * kvs + g * bhd + d];
                                    ref_dk[tk2 * kvs + g * bhd + d] += scale * ds2 * bq[tq2 * qs + h * bhd + d];
                                    ref_dv[tk2 * kvs + g * bhd + d] += sc[tk2] * bg[tq2 * qs + h * bhd + d];
                                }
                            }
                        }
                    }
                    auto max_err = [](const std::vector<float>& a2, const std::vector<float>& b2) {
                        float m = 0.0f;
                        for (std::size_t i = 0; i < a2.size(); ++i) m = std::max(m, std::fabs(a2[i] - b2[i]));
                        return m;
                    };
                    expect(gq.size() == ref_dq.size() && max_err(gq, ref_dq) <= 1e-4f,
                           "Vulkan GQA backward dQ parity mismatch vs CPU reference");
                    expect(gk.size() == ref_dk.size() && max_err(gk, ref_dk) <= 1e-4f,
                           "Vulkan GQA backward dK parity mismatch vs CPU reference");
                    expect(gv.size() == ref_dv.size() && max_err(gv, ref_dv) <= 1e-4f,
                           "Vulkan GQA backward dV parity mismatch vs CPU reference");
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Vulkan-native Tensor smoke failed: " << e.what() << '\n';
            ok = false;
        }

        set_env("MOTIFCL_MATMUL_BACKEND", "vulkan");
        try {
            auto backend = Backend::create_opencl();
            auto A = Tensor::from_cpu(backend, {1, 3}, DType::F32, a.data());
            auto B = Tensor::from_cpu(backend, {3, 2}, DType::F32, b.data());
            autograd::begin_graph_capture();
            auto C = matmul(A, B);
            auto graph = autograd::end_graph_capture();
            const auto c = C.to_vector<float>();
            expect(c.size() == 2, "Vulkan matmul dispatch must return N output values");
            if (c.size() == 2) {
                expect(std::fabs(c[0] - 22.0f) <= 1e-5f && std::fabs(c[1] - 28.0f) <= 1e-5f,
                       "Vulkan matmul dispatch output mismatch");
            }
            expect(!graph.empty() && graph.nodes()[0].op == "matmul_vulkan_f32_m1",
                   "Vulkan matmul dispatch must record the Vulkan M=1 op");

            auto A2 = Tensor::from_cpu(backend, {2, 3}, DType::F32, a2.data());
            autograd::begin_graph_capture();
            auto C2 = matmul(A2, B);
            auto graph2 = autograd::end_graph_capture();
            const auto c2 = C2.to_vector<float>();
            expect(c2.size() == 4, "Vulkan general matmul dispatch must return M*N output values");
            if (c2.size() == 4) {
                expect(std::fabs(c2[0] - 22.0f) <= 1e-5f && std::fabs(c2[1] - 28.0f) <= 1e-5f &&
                           std::fabs(c2[2] - 49.0f) <= 1e-5f && std::fabs(c2[3] - 64.0f) <= 1e-5f,
                       "Vulkan general matmul dispatch output mismatch");
            }
            expect(!graph2.empty() && graph2.nodes()[0].op == "matmul_vulkan_f32",
                   "Vulkan general matmul dispatch must record the Vulkan F32 op");

            set_env("MOTIFCL_ATTENTION_BACKEND", "vulkan");
            auto SX = Tensor::from_cpu(backend, {2, 3}, DType::F32, softmax_input.data());
            autograd::begin_graph_capture();
            auto SY = softmax_rows(SX);
            auto softmax_graph = autograd::end_graph_capture();
            const auto sy = SY.to_vector<float>();
            expect(sy.size() == softmax_input.size(), "Vulkan softmax dispatch must return rows*cols values");
            if (sy.size() == softmax_input.size()) {
                expect(std::fabs(sy[0] - softmax.output[0]) <= 1e-5f &&
                           std::fabs(sy[1] - softmax.output[1]) <= 1e-5f &&
                           std::fabs(sy[2] - softmax.output[2]) <= 1e-5f &&
                           std::fabs(sy[3] - softmax.output[3]) <= 1e-5f,
                       "Vulkan softmax dispatch output mismatch");
            }
            expect(!softmax_graph.empty() && softmax_graph.nodes()[0].op == "softmax_rows_vulkan_f32",
                   "Vulkan softmax dispatch must record the Vulkan softmax op");
            set_env("MOTIFCL_ATTENTION_BACKEND", "opencl");

            set_env("MOTIFCL_NORM_BACKEND", "vulkan");
            auto RX = Tensor::from_cpu(backend, {2, 4}, DType::F32, rms_x.data());
            auto RW = Tensor::from_cpu(backend, {4}, DType::F32, rms_w.data());
            autograd::begin_graph_capture();
            auto RY = rmsnorm(RX, RW);
            auto rms_graph = autograd::end_graph_capture();
            const auto ry = RY.to_vector<float>();
            expect(ry.size() == rms_x.size(), "Vulkan RMSNorm dispatch must return rows*cols values");
            if (ry.size() == rms_x.size()) {
                expect(std::fabs(ry[0] - rmsnorm_result.output[0]) <= 1e-5f &&
                           std::fabs(ry[3] - rmsnorm_result.output[3]) <= 1e-5f &&
                           std::fabs(ry[4] - rmsnorm_result.output[4]) <= 1e-5f,
                       "Vulkan RMSNorm dispatch output mismatch");
            }
            expect(!rms_graph.empty() && rms_graph.nodes()[0].op == "rmsnorm_vulkan_f32",
                   "Vulkan RMSNorm dispatch must record the Vulkan RMSNorm op");
            set_env("MOTIFCL_NORM_BACKEND", "opencl");

            set_env("MOTIFCL_ACTIVATION_BACKEND", "vulkan");
            auto SwiX = Tensor::from_cpu(backend, {2, 4}, DType::F32, swiglu_input.data());
            autograd::begin_graph_capture();
            auto SwiY = swiglu(SwiX);
            auto swiglu_graph = autograd::end_graph_capture();
            const auto swiy = SwiY.to_vector<float>();
            expect(swiy.size() == swiglu_result.output.size(), "Vulkan SwiGLU dispatch must return rows*hidden values");
            if (swiy.size() == swiglu_result.output.size()) {
                expect(std::fabs(swiy[0] - swiglu_result.output[0]) <= 1e-5f &&
                           std::fabs(swiy[1] - swiglu_result.output[1]) <= 1e-5f &&
                           std::fabs(swiy[2] - swiglu_result.output[2]) <= 1e-5f,
                       "Vulkan SwiGLU dispatch output mismatch");
            }
            expect(!swiglu_graph.empty() && swiglu_graph.nodes()[0].op == "swiglu_vulkan_f32",
                   "Vulkan SwiGLU dispatch must record the Vulkan SwiGLU op");
            set_env("MOTIFCL_ACTIVATION_BACKEND", "opencl");

            set_env("MOTIFCL_ELEMENTWISE_BACKEND", "vulkan");
            auto AddA = Tensor::from_cpu(backend, {2, 2}, DType::F32, add_a.data());
            auto AddB = Tensor::from_cpu(backend, {2, 2}, DType::F32, add_b.data());
            autograd::begin_graph_capture();
            auto AddY = add(AddA, AddB);
            auto add_graph = autograd::end_graph_capture();
            const auto add_y = AddY.to_vector<float>();
            expect(add_y.size() == add_result.output.size(), "Vulkan add dispatch must return input-sized output");
            if (add_y.size() == add_result.output.size()) {
                expect(std::fabs(add_y[0] - add_result.output[0]) <= 1e-6f &&
                           std::fabs(add_y[1] - add_result.output[1]) <= 1e-6f &&
                           std::fabs(add_y[2] - add_result.output[2]) <= 1e-6f,
                       "Vulkan add dispatch output mismatch");
            }
            expect(!add_graph.empty() && add_graph.nodes()[0].op == "add_vulkan_f32",
                   "Vulkan add dispatch must record the Vulkan add op");
            set_env("MOTIFCL_ELEMENTWISE_BACKEND", "opencl");
        } catch (const std::exception& e) {
            if (motifcl_test::is_opencl_unavailable(e)) {
                std::cout << "Vulkan matmul dispatch smoke skipped because OpenCL tensors are unavailable: "
                          << e.what() << '\n';
            } else {
                std::cerr << "Vulkan matmul dispatch smoke failed: " << e.what() << '\n';
                ok = false;
            }
        }
        set_env("MOTIFCL_MATMUL_BACKEND", "opencl");
        set_env("MOTIFCL_ATTENTION_BACKEND", "opencl");
        set_env("MOTIFCL_NORM_BACKEND", "opencl");
        set_env("MOTIFCL_ACTIVATION_BACKEND", "opencl");
        set_env("MOTIFCL_ELEMENTWISE_BACKEND", "opencl");
    }

    return ok ? 0 : 1;
}
