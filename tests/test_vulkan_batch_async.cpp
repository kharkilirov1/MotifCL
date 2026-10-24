// Deferred batch submission witness: batch_end_async() must be equivalent to batch_end()
// for anything recorded afterwards, including the case where the second batch allocates
// buffers whose size matches memory freed while the first batch is still in flight.
//
// The runtime holds two submission slots, each with its own command buffer, fence,
// descriptor sets and kept-alive buffers. This test exists so that the parts a wrong
// implementation breaks first - submission order, descriptor reuse and buffer reuse - are
// checked rather than assumed.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>

namespace {

bool report(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    std::fflush(stdout);
    return ok;
}

std::vector<float> fill(std::size_t n, std::uint32_t seed) {
    std::vector<float> v(n);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        v[i] = float((state >> 8) & 0xFFFFu) / 65536.0f - 0.5f;
    }
    return v;
}

bool close(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

} // namespace

int main() {
    using namespace motifcl;
    Backend backend = Backend::create_vulkan();
    if (!backend.vulkan_runtime().caps().timestamps) {
        std::printf("SKIP deferred-batch witness: device has no timestamps\n");
        return 0;
    }
    auto& rt = backend.vulkan_runtime();
    // Shapes taken from the trainer (the Vulkan GEMM path is picky about them).
    const std::int64_t m = 512;
    const std::int64_t k = 288;
    const std::int64_t n = 192;
    const auto a_host = fill(std::size_t(m * k), 0x11u);
    const auto b_host = fill(std::size_t(k * n), 0x12u);
    bool ok = true;
    {
        auto a = Tensor::from_cpu(backend, Shape{m, k}, DType::F32, a_host.data());
        auto b = Tensor::from_cpu(backend, Shape{k, n}, DType::F32, b_host.data());
        auto bc = Tensor::from_cpu(backend, Shape{n, n}, DType::F32, b_host.data());
        auto first = matmul(a, b);
        auto second = matmul(first, bc);
        const auto ref_first = first.to_vector<float>();
        const auto ref_second = second.to_vector<float>();
        ok &= report(ref_first.size() == std::size_t(m * n), "synchronous reference produced output");

        // Deferred: batch 1 produces the intermediate, batch 2 consumes it. Submission
        // order must make the second see the first's result although the host never waited.
        auto a2 = Tensor::from_cpu(backend, Shape{m, k}, DType::F32, a_host.data());
        auto b2 = Tensor::from_cpu(backend, Shape{k, n}, DType::F32, b_host.data());
        auto bc2 = Tensor::from_cpu(backend, Shape{n, n}, DType::F32, b_host.data());
        Tensor mid, out;
        {
            if (!rt.batch_begin()) {
                std::printf("FAIL deferred-batch: batch_begin (first) failed: %s\n", rt.error().c_str());
                return 1;
            }
            mid = matmul(a2, b2);
            const auto submitted = rt.batch_end_async();
            if (!submitted.success) {
                std::printf("FAIL deferred-batch: first batch_end_async failed: %s\n", submitted.error.c_str());
                return 1;
            }
            // Free and re-acquire a buffer of the same size while the first batch runs: a
            // pool that recycled it immediately would have the GPU read memory the host is
            // rewriting.
            {
                auto churn = Tensor::empty(backend, Shape{m, n}, DType::F32);
                churn.to_vector<float>();
            }
            if (!rt.batch_begin()) {
                std::printf("FAIL deferred-batch: batch_begin (second) failed: %s\n", rt.error().c_str());
                return 1;
            }
            out = matmul(mid, bc2);
            const auto submitted2 = rt.batch_end_async();
            if (!submitted2.success) {
                std::printf("FAIL deferred-batch: second batch_end_async failed: %s\n", submitted2.error.c_str());
                return 1;
            }
            const auto waited = rt.batch_wait();
            if (!waited.success) {
                std::printf("FAIL deferred-batch: batch_wait failed: %s\n", waited.error.c_str());
                return 1;
            }
        }
        const auto out_host = out.to_vector<float>();
        const auto mid_host = mid.to_vector<float>();
        ok &= report(close(mid_host, ref_first, 1.0e-5f), "deferred first matmul matches");
        ok &= report(close(out_host, ref_second, 1.0e-5f),
                     "deferred chain matches the synchronous chain (submission order kept)");

        // A synchronous batch after deferred ones must still work (slot recycling).
        auto tail = matmul(a2, b2);
        ok &= report(close(tail.to_vector<float>(), ref_first, 1.0e-5f),
                     "synchronous batch after deferred submissions matches");

        // Elementwise path through a deferred slot.
        auto x = Tensor::from_cpu(backend, Shape{m, n}, DType::F32, ref_first.data());
        auto y = Tensor::from_cpu(backend, Shape{m, n}, DType::F32, ref_second.data());
        const auto ref_sum = add(x, y).to_vector<float>();
        Tensor sum;
        {
            if (!rt.batch_begin()) {
                std::printf("FAIL deferred-batch: batch_begin (add) failed\n");
                return 1;
            }
            sum = add(x, y);
            if (!rt.batch_end_async().success) {
                std::printf("FAIL deferred-batch: add submission failed\n");
                return 1;
            }
            if (!rt.batch_wait().success) {
                std::printf("FAIL deferred-batch: add wait failed\n");
                return 1;
            }
        }
        ok &= report(close(sum.to_vector<float>(), ref_sum, 1.0e-6f), "deferred elementwise add matches");
    }

    if (!ok) {
        std::printf("FAIL deferred-batch witness\n");
        return 1;
    }
    std::printf("PASS deferred-batch witness (ordering, buffer reuse, slot recycling)\n");
    return 0;
}
