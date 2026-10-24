// Fused head loss witness: head_ce_fused must reproduce both the value and the
// gradients of softmax_cross_entropy(matmul(xf, weight), targets) bit for bit, because
// it recomputes dlogits inside the two head GEMMs instead of materialising it.
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>

namespace {

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

bool report(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    return ok;
}

bool bit_equal(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
    }
    return true;
}

std::vector<float> fill(std::size_t n, std::uint32_t seed, float scale) {
    std::vector<float> v(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        v[i] = (float((state >> 8) & 0xFFFFu) / 32768.0f - 1.0f) * scale;
    }
    return v;
}

} // namespace

int main() {
    using namespace motifcl;
    // head_ce_fused needs rows/classes multiples of 64 and in_features a multiple of four.
    const std::int64_t rows = 64;
    const std::int64_t in = 32;
    const std::int64_t classes = 128;
    set_env("MOTIFCL_VK_PROFILE", "1");
    Backend backend = Backend::create_vulkan();
    if (!backend.vulkan_runtime().caps().timestamps) {
        std::printf("SKIP head_ce_fused witness: device has no timestamps\n");
        return 0;
    }

    const auto xf_host = fill(std::size_t(rows * in), 0x51u, 0.8f);
    const auto w_host = fill(std::size_t(in * classes), 0x52u, 0.9f);
    std::vector<std::int32_t> targets(static_cast<std::size_t>(rows));
    for (std::int64_t r = 0; r < rows; ++r) targets[std::size_t(r)] = std::int32_t((r * 37) % classes);

    // Reference (double precision) for loss and both gradients.
    std::vector<double> ref_logits(std::size_t(rows * classes));
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < classes; ++c) {
            double acc = 0.0;
            for (std::int64_t i = 0; i < in; ++i) {
                acc += double(xf_host[std::size_t(r * in + i)]) * double(w_host[std::size_t(i * classes + c)]);
            }
            ref_logits[std::size_t(r * classes + c)] = acc;
        }
    }
    std::vector<double> ref_p(std::size_t(rows * classes));
    double ref_loss = 0.0;
    for (std::int64_t r = 0; r < rows; ++r) {
        double maxv = -1e30;
        for (std::int64_t c = 0; c < classes; ++c) maxv = std::max(maxv, ref_logits[std::size_t(r * classes + c)]);
        double sum = 0.0;
        for (std::int64_t c = 0; c < classes; ++c) {
            const double e = std::exp(ref_logits[std::size_t(r * classes + c)] - maxv);
            ref_p[std::size_t(r * classes + c)] = e;
            sum += e;
        }
        for (std::int64_t c = 0; c < classes; ++c) ref_p[std::size_t(r * classes + c)] /= sum;
        ref_loss -= std::log(ref_p[std::size_t(r * classes + targets[std::size_t(r)])]);
    }
    ref_loss /= double(rows);
    std::vector<double> ref_dlogits(std::size_t(rows * classes));
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t c = 0; c < classes; ++c) {
            double v = ref_p[std::size_t(r * classes + c)];
            if (c == targets[std::size_t(r)]) v -= 1.0;
            ref_dlogits[std::size_t(r * classes + c)] = v / double(rows);
        }
    }
    std::vector<double> ref_dxf(std::size_t(rows * in), 0.0);
    for (std::int64_t r = 0; r < rows; ++r) {
        for (std::int64_t i = 0; i < in; ++i) {
            double acc = 0.0;
            for (std::int64_t c = 0; c < classes; ++c) {
                acc += ref_dlogits[std::size_t(r * classes + c)] * double(w_host[std::size_t(i * classes + c)]);
            }
            ref_dxf[std::size_t(r * in + i)] = acc;
        }
    }
    std::vector<double> ref_dw(std::size_t(in * classes), 0.0);
    for (std::int64_t i = 0; i < in; ++i) {
        for (std::int64_t c = 0; c < classes; ++c) {
            double acc = 0.0;
            for (std::int64_t r = 0; r < rows; ++r) {
                acc += double(xf_host[std::size_t(r * in + i)]) * ref_dlogits[std::size_t(r * classes + c)];
            }
            ref_dw[std::size_t(i * classes + c)] = acc;
        }
    }

    auto run_arm = [&](bool fused, std::vector<float>& loss, std::vector<float>& dxf,
                       std::vector<float>& dw) {
        auto xf = Tensor::from_cpu(backend, Shape{rows, in}, DType::F32, xf_host.data());
        auto w = Tensor::from_cpu(backend, Shape{in, classes}, DType::F32, w_host.data());
        auto tgt = Tensor::from_cpu(backend, Shape{rows}, DType::I32, targets.data());
        xf.set_requires_grad(true);
        w.set_requires_grad(true);
        Tensor out = fused ? head_ce_fused(xf, w, tgt)
                           : softmax_cross_entropy(matmul(xf, w), tgt);
        out.backward();
        loss = out.to_vector<float>();
        dxf = xf.grad()->to_vector<float>();
        dw = w.grad()->to_vector<float>();
    };

    std::vector<float> loss_ref, dxf_ref, dw_ref, loss_fused, dxf_fused, dw_fused;
    run_arm(false, loss_ref, dxf_ref, dw_ref);
    run_arm(true, loss_fused, dxf_fused, dw_fused);

    bool ok = true;
    ok &= bit_equal(loss_ref, loss_fused) && report(bit_equal(loss_ref, loss_fused), "loss bit-identical");
    ok &= bit_equal(dxf_ref, dxf_fused) &&
          report(bit_equal(dxf_ref, dxf_fused), "grad_input bit-identical");
    ok &= bit_equal(dw_ref, dw_fused) && report(bit_equal(dw_ref, dw_fused), "grad_weight bit-identical");

    auto close = [](const std::vector<float>& got, const std::vector<double>& ref, double tol) {
        if (got.size() != ref.size()) return false;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (std::abs(double(got[i]) - ref[i]) > tol) return false;
        }
        return true;
    };
    ok &= close(loss_fused, {ref_loss}, 2.0e-5) && report(close(loss_fused, {ref_loss}, 2.0e-5), "loss matches CPU reference");
    ok &= close(dxf_fused, ref_dxf, 2.0e-5) &&
          report(close(dxf_fused, ref_dxf, 2.0e-5), "grad_input matches CPU reference");
    ok &= close(dw_fused, ref_dw, 2.0e-5) &&
          report(close(dw_fused, ref_dw, 2.0e-5), "grad_weight matches CPU reference");

    if (!ok) {
        std::printf("FAIL head_ce_fused witness\n");
        return 1;
    }
    std::printf("PASS head_ce_fused witness (loss, grad_input, grad_weight)\n");
    return 0;
}
