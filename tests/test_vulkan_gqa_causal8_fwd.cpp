// Forward causal-8 witness (VulkanTensor witness, NOT claimed OpenCL-free).
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <motifcl/motifcl.hpp>

namespace {
void set_env(const char* n, const char* v) {
#ifdef _WIN32
    _putenv_s(n, v);
#else
    setenv(n, v, 1);
#endif
}
void report(bool ok, const std::string& m) { std::printf("%s %s\n", ok ? "ok  " : "FAIL", m.c_str()); std::fflush(stdout); }
void require_size(std::size_t got, std::size_t expect, const char* what) {
    if (got != expect) throw std::runtime_error(std::string("input size mismatch for ") + what);
}
int calls(motifcl::VulkanRuntime& rt, const char* needle) {
    for (const auto& row : rt.kernel_profile())
        if (row.name.find(needle) != std::string::npos && row.calls) return int(row.calls);
    return 0;
}
struct Pack { motifcl::Tensor q, k, v; };
Pack make_pack(motifcl::Backend& be, std::int64_t b, std::int64_t t, std::int64_t nh, std::int64_t nkh, std::int64_t hd) {
    using namespace motifcl;
    const std::size_t qn = std::size_t(b * t * nh * hd), kn = std::size_t(b * t * nkh * hd);
    std::vector<float> qv(qn), kv(kn), vv(kn);
    for (std::size_t i = 0; i < qn; ++i) qv[i] = -0.3f + 0.001f * float((i * 13u) % 977u);
    for (std::size_t i = 0; i < kn; ++i) { kv[i] = -0.2f + 0.001f * float((i * 29u) % 977u); vv[i] = -0.1f + 0.001f * float((i * 41u) % 977u); }
    require_size(qv.size(), qn, "q");
    return {Tensor::from_cpu(be, Shape{b * t, nh * hd}, DType::F32, qv.data()),
            Tensor::from_cpu(be, Shape{b * t, nkh * hd}, DType::F32, kv.data()),
            Tensor::from_cpu(be, Shape{b * t, nkh * hd}, DType::F32, vv.data())};
}
struct Res { std::vector<float> y; int c8 = 0; int c4 = 0; };
Res run_plain(motifcl::VulkanRuntime& rt, const Pack& p, std::int64_t b, std::int64_t t, std::int64_t nh, std::int64_t nkh, const char* rows) {
    set_env("MOTIFCL_VK_GQA_FWD_ROWS", rows);
    rt.reset_kernel_profile();
    auto y = motifcl::grouped_query_attention(p.q, p.k, p.v, int(nh), int(nkh), true, b, t, t, 0, 0.0f);
    auto out = y.to_vector<float>();
    return {out, calls(rt, "gqa_fwd_causal8"), calls(rt, "gqa_fwd_causal4_f32")};
}
bool micro(motifcl::VulkanRuntime& rt, const Pack& p, std::int64_t b, std::int64_t t, std::int64_t nh, std::int64_t nkh) {
    // Known-answer statistics check: catches the phase-8 best-of error.
    {
        std::vector<double> ka = {1.0, 2.0, 3.0, 100.0};
        auto mean = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s / double(v.size()); };
        auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end());
            const std::size_t n = v.size(); return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); };
        const bool stat_ok = std::abs(mean(ka) - 26.5) < 1e-9 && std::abs(median(ka) - 2.5) < 1e-9 &&
                             median(ka) > 1.0;
        std::printf("%s known-answer stats: mean=%.1f median=%.1f (not min=1)\n", stat_ok ? "ok  " : "FAIL",
                    mean(ka), median(ka));
        if (!stat_ok) return false;
    }
    auto arm = [&](const char* rows, int iters, double* out_med, double* out_mean, std::string* names) {
        std::vector<double> s;
        for (int i = 0; i < iters; ++i) {
            set_env("MOTIFCL_VK_GQA_FWD_ROWS", rows);
            rt.reset_kernel_profile();
            (void)motifcl::grouped_query_attention(p.q, p.k, p.v, int(nh), int(nkh), true, b, t, t, 0, 0.0f);
            const double us = rt.last_gpu_time_us();
            if (!(us > 0.0) || !std::isfinite(us)) { std::printf("FAIL invalid GPU time (%g)\n", us); return false; }
            s.push_back(us * 1e-3);
            if (i == 0) {
                const char* want = std::string(rows) == "8" ? "gqa_fwd_causal8" : "gqa_fwd_causal4_f32";
                if (calls(rt, want) <= 0) { std::printf("FAIL measured dispatch did not select %s\n", want); return false; }
                *names = want;
            }
        }
        std::sort(s.begin(), s.end());
        const std::size_t n = s.size();
        *out_med = n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
        double sum = 0; for (double x : s) sum += x;
        *out_mean = sum / double(n);
        return true;
    };
    for (int i = 0; i < 100; ++i) { set_env("MOTIFCL_VK_GQA_FWD_ROWS", "4");
        (void)motifcl::grouped_query_attention(p.q, p.k, p.v, int(nh), int(nkh), true, b, t, t, 0, 0.0f); }
    for (int i = 0; i < 100; ++i) { set_env("MOTIFCL_VK_GQA_FWD_ROWS", "8");
        (void)motifcl::grouped_query_attention(p.q, p.k, p.v, int(nh), int(nkh), true, b, t, t, 0, 0.0f); }
    std::printf("warmup: 100 dispatches per variant discarded (both variants)\n"); std::fflush(stdout);
    const int blocks = 4, iters = 30;
    double old_med_sum = 0, new_med_sum = 0; int counted = 0;
    for (int blk = 0; blk < blocks; ++blk) {
        const bool old_first = (blk % 2) == 0;
        double o_med = 0, o_mean = 0, n_med = 0, n_mean = 0; std::string on, nn;
        if (old_first) {
            if (!arm("4", iters, &o_med, &o_mean, &on) || !arm("8", iters, &n_med, &n_mean, &nn)) return false;
        } else {
            if (!arm("8", iters, &n_med, &n_mean, &nn) || !arm("4", iters, &o_med, &o_mean, &on)) return false;
        }
        const double eff = (n_med / o_med - 1.0) * 100.0;
        std::printf("block %d: old(%s) med=%.4f mean=%.4f new(%s) med=%.4f mean=%.4f paired=%+.2f%%\n",
                    blk + 1, on.c_str(), o_med, o_mean, nn.c_str(), n_med, n_mean, eff); std::fflush(stdout);
        old_med_sum += o_med; new_med_sum += n_med; ++counted;
    }
    const double total = (new_med_sum / old_med_sum - 1.0) * 100.0;
    std::printf("micro B8/T256/H6/KV6/D48: aggregate old_med=%.4f new_med=%.4f delta=%+.2f%% (no min/best-of)\n",
                old_med_sum / counted, new_med_sum / counted, total); std::fflush(stdout);
    return true;
}}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    set_env("MOTIFCL_VK_PROFILE", "1");
    using namespace motifcl;
    auto probe = probe_vulkan_runtime();
    if (!probe.available()) { std::printf("skip: %s\n", probe.error.c_str()); return 77; }
    bool ok = true;
    auto expect = [&](bool cond, const std::string& m) { report(cond, m); ok = ok && cond; };
    try {
        Backend be = Backend::create_vulkan();
        auto& rt = be.vulkan_runtime();
        rt.set_gpu_timing_enabled(true);
        bool threw = false;
        try { require_size(10, 20, "q"); } catch (const std::exception&) { threw = true; }
        expect(threw, "size-mismatch negative control throws deterministically");
        struct Case { std::int64_t b, t, nh, nkh; const char* tag; };
        const std::vector<Case> cases = {{8, 256, 6, 6, "B8/T256/H6/KV6/D48"}, {2, 253, 6, 6, "tail T253"}, {2, 256, 6, 2, "GQA H6/KV2"}};
        for (const auto& c : cases) {
            auto pack = make_pack(be, c.b, c.t, c.nh, c.nkh, 48);
            auto old4 = run_plain(rt, pack, c.b, c.t, c.nh, c.nkh, "4");
            auto new8 = run_plain(rt, pack, c.b, c.t, c.nh, c.nkh, "8");
            expect(old4.c4 > 0, std::string(c.tag) + ": forced-old selects gqa_fwd_causal4_f32");
            expect(new8.c8 > 0, std::string(c.tag) + ": gqa_fwd_causal8 selected (calls=" + std::to_string(new8.c8) + ")");
            const bool same = old4.y.size() == new8.y.size() && !old4.y.empty() &&
                              std::memcmp(old4.y.data(), new8.y.data(), old4.y.size() * sizeof(float)) == 0;
            expect(same, std::string(c.tag) + ": y byte-identical rows8 vs rows4");
        }
        auto fb = make_pack(be, 2, 128, 6, 6, 48);
        set_env("MOTIFCL_VK_GQA_FWD_ROWS", "8");
        rt.reset_kernel_profile();
        (void)grouped_query_attention(fb.q, fb.k, fb.v, 6, 6, false, 2, 128, 128, 0, 0.0f);
        expect(calls(rt, "gqa_fwd_causal8") == 0, "non-causal fallback does not select rows8");
        rt.reset_kernel_profile();
        (void)grouped_query_attention_windowed(fb.q, fb.k, fb.v, 6, 6, 32, true, 2, 128, 128, 0, 0.0f);
        expect(calls(rt, "gqa_fwd_causal8") == 0, "sliding-window fallback does not select rows8");
        auto pack8 = make_pack(be, 8, 256, 6, 6, 48);
        if (!micro(rt, pack8, 8, 256, 6, 6)) ok = false;
    } catch (const std::exception& e) {
        std::printf("FAIL exception: %s\n", e.what());
        return 1;
    }
    if (!ok) return 1;
    std::printf("PASS causal8 forward witness\n");
    return 0;
}
