// OpenCL-free witness for the deterministic embedding-weight backward.
//
// The default scatter accumulates duplicate token rows with a float
// compare-and-swap loop, so its last bits depend on thread scheduling (that is
// why two identical training runs produce different checkpoints). The
// deterministic path (owner/count pass + ordered gather) is compared here
// against an independent CPU implementation and checked for byte-identical
// repeated execution, which is the property the trainer relies on.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <motifcl/runtime/vulkan_backend.hpp>

namespace {
bool ok = true;
void expect(bool cond, const char* msg) {
    std::printf("%s %s\n", cond ? "ok  " : "FAIL", msg);
    std::fflush(stdout);
    if (!cond) ok = false;
}

void fill(std::vector<float>& v, std::uint32_t seed, float scale) {
    std::uint32_t state = seed;
    for (float& x : v) {
        state = state * 1664525u + 1013904223u;
        x = (static_cast<float>((state >> 8) & 0xffffu) / 32767.5f - 1.0f) * scale;
    }
}

std::vector<float> cpu_reference(const std::vector<std::int32_t>& ids, const std::vector<float>& grad,
                                 std::size_t vocab, std::size_t embed) {
    std::vector<float> out(vocab * embed, 0.0f);
    std::vector<float> acc(embed, 0.0f);
    const std::size_t tokens = ids.size();
    for (std::size_t row = 0; row < vocab; ++row) {
        // Sum in ascending token order, which is what the kernel does.
        bool any = false;
        for (std::size_t t = 0; t < tokens; ++t) {
            if (static_cast<std::size_t>(ids[t]) != row) continue;
            if (!any) {
                std::fill(acc.begin(), acc.end(), 0.0f);
                any = true;
            }
            for (std::size_t q = 0; q < embed; ++q) acc[q] += grad[t * embed + q];
        }
        if (!any) continue;
        for (std::size_t q = 0; q < embed; ++q) out[row * embed + q] = acc[q];
    }
    return out;
}

bool run_case(motifcl::VulkanRuntime& rt, std::size_t vocab, std::size_t embed,
              const std::vector<std::int32_t>& ids, std::uint32_t seed, const char* label) {
    using namespace motifcl;
    const std::size_t tokens = ids.size();
    std::vector<float> grad(tokens * embed);
    fill(grad, seed, 0.5f);
    auto id_buf = rt.create_buffer(tokens * sizeof(std::int32_t), ids.data());
    auto grad_buf = rt.create_buffer(grad.size() * sizeof(float), grad.data());
    auto out_buf = rt.create_buffer(vocab * embed * sizeof(float));
    const auto run = run_vulkan_embedding_weight_backward_deterministic(rt, id_buf, grad_buf, out_buf, vocab, embed,
                                                                  tokens);
    if (!run.success) {
        std::printf("FAIL %s: dispatch failed: %s\n", label, run.error.c_str());
        return false;
    }
    std::vector<float> got(vocab * embed);
    out_buf.download(got.data(), got.size() * sizeof(float));

    const auto want = cpu_reference(ids, grad, vocab, embed);
    double max_abs = 0.0;
    std::size_t worst = 0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double diff = std::fabs(static_cast<double>(got[i]) - static_cast<double>(want[i]));
        if (diff > max_abs) {
            max_abs = diff;
            worst = i;
        }
    }
    const bool close = std::isfinite(max_abs) && max_abs < 1.0e-6;
    if (!close) {
        std::printf("  %s worst row=%zu col=%zu gpu=%.9g cpu=%.9g\n", label, worst / embed, worst % embed, got[worst],
                    want[worst]);
        std::size_t mismatches = 0, nonzero_gpu = 0, nonzero_cpu = 0;
        for (std::size_t i = 0; i < want.size(); ++i) {
            if (got[i] != want[i]) ++mismatches;
            if (got[i] != 0.0f) ++nonzero_gpu;
            if (want[i] != 0.0f) ++nonzero_cpu;
        }
        std::printf("  %s mismatches=%zu nonzero gpu=%zu cpu=%zu expected_dup_rows?\n", label, mismatches,
                    nonzero_gpu, nonzero_cpu);
    }
    expect(close, (std::string(label) + ": matches independent CPU formula").c_str());
    std::printf("  %s vocab=%zu embed=%zu tokens=%zu max_abs=%.3e\n", label, vocab, embed, tokens, max_abs);

    // Re-run over whatever is already in the buffer: the kernel writes every row,
    // so the bytes must be identical and independent of the previous contents.
    const auto run2 = run_vulkan_embedding_weight_backward_deterministic(rt, id_buf, grad_buf, out_buf, vocab, embed,
                                                                   tokens);
    expect(run2.success, "second dispatch");
    std::vector<float> again(vocab * embed);
    out_buf.download(again.data(), again.size() * sizeof(float));
    expect(std::memcmp(got.data(), again.data(), got.size() * sizeof(float)) == 0,
           (std::string(label) + ": repeated execution is byte-identical").c_str());
    return close;
}
} // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    using namespace motifcl;
    const auto probe = probe_vulkan_runtime();
    if (!probe.available()) {
        std::printf("SKIP: no Vulkan runtime (%s)\n", probe.error.c_str());
        return 77;
    }
    VulkanRuntime rt = VulkanRuntime::create();
    rt.set_gpu_timing_enabled(true);

    // Unique ids, a few duplicates, and a pathological case where every token hits
    // the same row (the ordered-gather path, not the copy fast path).
    std::vector<std::int32_t> unique_ids(512);
    for (std::size_t i = 0; i < unique_ids.size(); ++i) unique_ids[i] = static_cast<std::int32_t>((i * 37) % 4000);
    std::vector<std::int32_t> dup_ids(1000);
    for (std::size_t i = 0; i < dup_ids.size(); ++i) dup_ids[i] = static_cast<std::int32_t>((i * 11) % 97);
    std::vector<std::int32_t> same_ids(300, 42);
    std::vector<std::int32_t> trainer_ids(2048);
    for (std::size_t i = 0; i < trainer_ids.size(); ++i) trainer_ids[i] = static_cast<std::int32_t>((i * 2651u) % 32000u);

    run_case(rt, 4000, 8, unique_ids, 0x1234u, "unique ids");
    run_case(rt, 97, 16, dup_ids, 0x5678u, "duplicate ids");
    run_case(rt, 64, 4, same_ids, 0x9abcu, "all ids equal");
    run_case(rt, 32000, 288, trainer_ids, 0xdef0u, "trainer shape");

    // Invalid arguments: rejected before dispatch, output untouched.
    const std::array<std::int32_t, 4> ids = {0, 1, 2, 3};
    auto id_buf = rt.create_buffer(ids.size() * sizeof(std::int32_t), ids.data());
    auto grad_buf = rt.create_buffer(16 * sizeof(float));
    auto tiny_buf = rt.create_buffer(2 * sizeof(float));
    auto out_buf = rt.create_buffer(8 * 4 * sizeof(float));
    std::vector<float> sentinel(8 * 4, -3.0f);
    out_buf.upload(sentinel.data(), sentinel.size() * sizeof(float));
    const std::vector<float> before = sentinel;
    struct Bad {
        const char* name;
        bool rejected;
    };
    const std::array<Bad, 5> bads = {{
        {"tokens=0 rejected",
         !run_vulkan_embedding_weight_backward_deterministic(rt, id_buf, grad_buf, out_buf, 8, 4, 0).success},
        {"embed%4 rejected",
         !run_vulkan_embedding_weight_backward_deterministic(rt, id_buf, grad_buf, out_buf, 8, 5, 4).success},
        {"undersized indices rejected",
         !run_vulkan_embedding_weight_backward_deterministic(rt, tiny_buf, grad_buf, out_buf, 8, 4, 4).success},
        {"undersized output rejected",
         !run_vulkan_embedding_weight_backward_deterministic(rt, id_buf, grad_buf, grad_buf, 8, 4, 4).success},
        {"vocab overflow rejected",
         !run_vulkan_embedding_weight_backward_deterministic(
              rt, id_buf, grad_buf, out_buf, std::size_t{1} << 31, 4, 4)
              .success},
    }};
    for (const auto& bad : bads) expect(bad.rejected, bad.name);
    std::vector<float> after(8 * 4);
    out_buf.download(after.data(), after.size() * sizeof(float));
    expect(after == before, "invalid arguments mutated nothing (bytes snapshot)");

    if (!ok) return 1;
    std::printf("PASS deterministic embedding backward witness (4 fixtures, repeat-runs, invalid-argument controls)\n");
    return 0;
}
