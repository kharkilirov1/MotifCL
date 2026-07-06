// Vulkan-vs-OpenCL per-op performance harness for the Vulkan port.
//
// For every registered (op, shape) case this runs the Vulkan cached-dispatch
// path (wall time around submit+wait, plus VK_QUERY_TYPE_TIMESTAMP GPU time)
// and the OpenCL Tensor path (wall time around op+finish) on the same device,
// then writes reports/vulkan-perf/<op>.json and reports/vulkan-perf/SUMMARY.md.
//
// Methodology: median of --runs (default 50) after --warmup (default 5) runs.
// The PASS ratio compares wall-time medians (both sides measured the same
// way); GPU timestamp time is recorded alongside for analysis.
//
// Usage (from the repo root so reports/ lands in the tree):
//   build/port-vk/benchmarks/bench_vulkan_perf.exe [--runs N] [--warmup N]
//       [--filter substr] [--out reports/vulkan-perf]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <motifcl/motifcl.hpp>
#include <motifcl/autograd/node.hpp>
#include <motifcl/nn/compact_counter.hpp>
#include <motifcl/runtime/vulkan_backend.hpp>

namespace {

struct Stats {
    double median_us = -1.0;
    double min_us = -1.0;
    double p99_us = -1.0;
};

Stats summarize(std::vector<double> samples) {
    Stats out;
    if (samples.empty()) return out;
    std::sort(samples.begin(), samples.end());
    out.min_us = samples.front();
    out.median_us = samples[samples.size() / 2];
    const std::size_t p99_index =
        static_cast<std::size_t>(std::ceil(0.99 * static_cast<double>(samples.size() - 1)));
    out.p99_us = samples[std::min(p99_index, samples.size() - 1)];
    return out;
}

struct CaseResult {
    std::string op;
    std::string shape;
    std::string dtype = "f32";
    double target_ratio = 0.0;
    double work_flops = 0.0;  // when >0, throughput reported as GFLOP/s
    double work_bytes = 0.0;  // else GB/s
    Stats vk_wall;
    Stats vk_gpu;
    double cl_wall_median_us = -1.0;
    bool vk_ok = false;
    bool cl_ok = false;
    std::string error;

    double ratio() const {
        if (!vk_ok || !cl_ok || vk_wall.median_us <= 0.0 || cl_wall_median_us <= 0.0) return -1.0;
        return cl_wall_median_us / vk_wall.median_us;
    }
    bool pass() const { return vk_ok && cl_ok && ratio() >= target_ratio; }
    // PASS/FAIL only when an OpenCL baseline exists; otherwise the honest
    // record is NO_BASELINE (never invented).
    const char* verdict() const {
        if (!vk_ok) return "ERROR";
        if (!cl_ok) return "NO_BASELINE";
        return pass() ? "PASS" : "FAIL";
    }
    double achieved_throughput() const {
        if (!vk_ok || vk_wall.median_us <= 0.0) return 0.0;
        if (work_flops > 0.0) return work_flops / (vk_wall.median_us * 1.0e3);  // GFLOP/s
        return work_bytes / (vk_wall.median_us * 1.0e3);                        // GB/s
    }
};

struct BenchCase {
    std::string op;
    std::string shape;
    double target_ratio;
    double work_flops;
    double work_bytes;
    // Returns empty string on success. Runs one Vulkan iteration.
    std::function<std::string()> vulkan_iter;
    // Runs one OpenCL iteration (op + finish). Null when no baseline exists.
    std::function<void()> opencl_iter;
};

int g_runs = 50;
int g_warmup = 5;

CaseResult run_case(const BenchCase& bench, motifcl::VulkanRuntime& runtime, bool opencl_available) {
    CaseResult result;
    result.op = bench.op;
    result.shape = bench.shape;
    result.target_ratio = bench.target_ratio;
    result.work_flops = bench.work_flops;
    result.work_bytes = bench.work_bytes;

    runtime.set_gpu_timing_enabled(true);
    std::vector<double> wall;
    std::vector<double> gpu;
    wall.reserve(g_runs);
    gpu.reserve(g_runs);
    bool failed = false;
    for (int i = 0; i < g_warmup + g_runs && !failed; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto error = bench.vulkan_iter();
        const auto t1 = std::chrono::steady_clock::now();
        if (!error.empty()) {
            result.error = error;
            failed = true;
            break;
        }
        if (i >= g_warmup) {
            wall.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            const double gpu_us = runtime.last_gpu_time_us();
            if (gpu_us >= 0.0) gpu.push_back(gpu_us);
        }
    }
    runtime.set_gpu_timing_enabled(false);
    if (!failed && !wall.empty()) {
        result.vk_wall = summarize(wall);
        result.vk_gpu = summarize(gpu);
        result.vk_ok = true;
    }

    if (bench.opencl_iter && opencl_available) {
        try {
            std::vector<double> cl_wall;
            cl_wall.reserve(g_runs);
            for (int i = 0; i < g_warmup + g_runs; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                bench.opencl_iter();
                const auto t1 = std::chrono::steady_clock::now();
                if (i >= g_warmup) {
                    cl_wall.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
                }
            }
            result.cl_wall_median_us = summarize(cl_wall).median_us;
            result.cl_ok = result.cl_wall_median_us > 0.0;
        } catch (const std::exception& e) {
            result.error += std::string(result.error.empty() ? "" : "; ") + "opencl baseline failed: " + e.what();
        }
    }
    return result;
}

std::string json_escape(const std::string& text) {
    std::string out;
    for (char c : text) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c);
    }
    return out;
}

void write_reports(const std::vector<CaseResult>& results,
                   const std::filesystem::path& out_dir,
                   const std::string& vk_device,
                   const std::string& cl_device) {
    std::filesystem::create_directories(out_dir);

    // Group rows per op -> <op>.json
    std::vector<std::string> ops;
    for (const auto& row : results) {
        if (std::find(ops.begin(), ops.end(), row.op) == ops.end()) ops.push_back(row.op);
    }
    for (const auto& op : ops) {
        std::ofstream json(out_dir / (op + ".json"));
        json << "{\n  \"op\": \"" << json_escape(op) << "\",\n";
        json << "  \"vulkan_device\": \"" << json_escape(vk_device) << "\",\n";
        json << "  \"opencl_device\": \"" << json_escape(cl_device) << "\",\n";
        json << "  \"methodology\": \"median wall-time of " << g_runs << " runs after " << g_warmup
             << " warmup; gpu_us from VK_QUERY_TYPE_TIMESTAMP; ratio = opencl_wall / vulkan_wall\",\n";
        json << "  \"rows\": [\n";
        bool first = true;
        for (const auto& row : results) {
            if (row.op != op) continue;
            if (!first) json << ",\n";
            first = false;
            json << "    {\"shape\": \"" << json_escape(row.shape) << "\", \"dtype\": \"" << row.dtype << "\"";
            json << ", \"median_us\": " << row.vk_wall.median_us;
            json << ", \"min_us\": " << row.vk_wall.min_us;
            json << ", \"p99_us\": " << row.vk_wall.p99_us;
            json << ", \"gpu_median_us\": " << row.vk_gpu.median_us;
            if (row.work_flops > 0.0) {
                json << ", \"achieved_gflops\": " << row.achieved_throughput();
            } else {
                json << ", \"achieved_gbs\": " << row.achieved_throughput();
            }
            if (row.cl_ok) {
                json << ", \"opencl_baseline_us\": " << row.cl_wall_median_us;
                json << ", \"ratio\": " << row.ratio();
            } else {
                json << ", \"opencl_baseline_us\": \"unavailable\"";
            }
            json << ", \"target_ratio\": " << row.target_ratio;
            json << ", \"result\": \"" << row.verdict() << "\"";
            if (!row.error.empty()) json << ", \"error\": \"" << json_escape(row.error) << "\"";
            json << "}";
        }
        json << "\n  ]\n}\n";
    }

    std::ofstream md(out_dir / "SUMMARY.md");
    md << "# Vulkan port perf record\n\n";
    md << "Device (Vulkan): " << vk_device << "  \n";
    md << "Device (OpenCL baseline): " << (cl_device.empty() ? "unavailable" : cl_device) << "  \n";
    md << "Methodology: median wall-time of " << g_runs << " runs after " << g_warmup
       << " warmup runs; both backends timed as dispatch+wait on the same device. "
       << "`gpu_us` additionally reports the Vulkan GPU-only time from timestamp queries. "
       << "Ratio = OpenCL wall / Vulkan wall (higher is better for Vulkan).\n\n";
    md << "| op | shape | vk wall p50 (us) | vk gpu p50 (us) | opencl p50 (us) | throughput | ratio | target | result |\n";
    md << "|---|---|---|---|---|---|---|---|---|\n";
    for (const auto& row : results) {
        std::ostringstream throughput;
        if (row.work_flops > 0.0) {
            throughput << row.achieved_throughput() << " GFLOP/s";
        } else {
            throughput << row.achieved_throughput() << " GB/s";
        }
        md << "| " << row.op << " | " << row.shape << " | " << row.vk_wall.median_us << " | "
           << row.vk_gpu.median_us << " | ";
        if (row.cl_ok) {
            md << row.cl_wall_median_us;
        } else {
            md << "unavailable";
        }
        md << " | " << throughput.str() << " | ";
        if (row.ratio() >= 0.0) {
            md << row.ratio();
        } else {
            md << "-";
        }
        md << " | " << row.target_ratio << " | " << row.verdict();
        if (!row.error.empty()) md << " (" << row.error << ")";
        md << " |\n";
    }
    md << "\nRegenerate: `build/port-vk/benchmarks/bench_vulkan_perf.exe` from the repo root.\n";
}

float frand(std::uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<float>(static_cast<std::int64_t>(state % 2000) - 1000) / 1000.0f;
}

std::vector<float> random_host(std::size_t count, std::uint64_t seed) {
    std::vector<float> out(count);
    std::uint64_t state = seed | 1u;
    for (auto& value : out) value = frand(state);
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string filter;
    std::filesystem::path out_dir = "reports/vulkan-perf";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if (arg == "--runs") g_runs = std::max(1, std::atoi(next().c_str()));
        else if (arg == "--warmup") g_warmup = std::max(0, std::atoi(next().c_str()));
        else if (arg == "--filter") filter = next();
        else if (arg == "--out") out_dir = next();
    }

    auto runtime = motifcl::VulkanRuntime::create();
    if (!runtime.available()) {
        std::cerr << "Vulkan runtime unavailable: " << runtime.error() << "\n";
        return 77;
    }
    std::cout << "Vulkan device: " << runtime.device_name()
              << " (timestamps: " << (runtime.caps().timestamps ? "yes" : "no")
              << ", period " << runtime.caps().timestamp_period_ns << " ns)\n";

    bool opencl_available = true;
    std::string cl_device;
    std::unique_ptr<motifcl::Backend> cl_backend;
    try {
        cl_backend = std::make_unique<motifcl::Backend>(motifcl::Backend::create_opencl());
        cl_device = cl_backend->device_info().device_name;
    } catch (const std::exception& e) {
        opencl_available = false;
        std::cerr << "OpenCL baseline unavailable: " << e.what() << "\n";
    }

    std::vector<BenchCase> cases;

    // ---- matmul f32 (compute-bound + small) ----
    struct MatmulFixture {
        motifcl::VulkanBuffer a, b, c;
        motifcl::Tensor cl_a, cl_b;
        std::size_t m, k, n;
    };
    std::vector<std::shared_ptr<MatmulFixture>> matmul_fixtures;
    for (const auto dims : {std::array<std::size_t, 3>{64, 64, 64},
                            std::array<std::size_t, 3>{256, 256, 256},
                            std::array<std::size_t, 3>{512, 512, 512}}) {
        auto fixture = std::make_shared<MatmulFixture>();
        fixture->m = dims[0];
        fixture->k = dims[1];
        fixture->n = dims[2];
        const auto a_host = random_host(dims[0] * dims[1], 0x1234);
        const auto b_host = random_host(dims[1] * dims[2], 0x5678);
        fixture->a = runtime.create_buffer(a_host.size() * sizeof(float), a_host.data());
        fixture->b = runtime.create_buffer(b_host.size() * sizeof(float), b_host.data());
        fixture->c = runtime.create_buffer(dims[0] * dims[2] * sizeof(float));
        if (opencl_available) {
            fixture->cl_a = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(dims[0]), static_cast<int64_t>(dims[1])},
                                                      motifcl::DType::F32, a_host.data());
            fixture->cl_b = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(dims[1]), static_cast<int64_t>(dims[2])},
                                                      motifcl::DType::F32, b_host.data());
        }
        matmul_fixtures.push_back(fixture);
        std::ostringstream shape;
        shape << dims[0] << "x" << dims[1] << "x" << dims[2];
        BenchCase bench;
        bench.op = "matmul_f32";
        bench.shape = shape.str();
        bench.target_ratio = 0.33;
        bench.work_flops = 2.0 * dims[0] * dims[1] * dims[2];
        bench.work_bytes = 0.0;
        bench.vulkan_iter = [fixture, &runtime]() -> std::string {
            const auto result = motifcl::run_vulkan_f32_matmul(runtime, fixture->a, fixture->b, fixture->c,
                                                               fixture->m, fixture->k, fixture->n);
            return result.success ? std::string() : result.error;
        };
        if (opencl_available) {
            auto backend_ptr = cl_backend.get();
            bench.opencl_iter = [fixture, backend_ptr]() {
                motifcl::autograd::NoGradGuard guard;
                auto out = motifcl::matmul(fixture->cl_a, fixture->cl_b);
                backend_ptr->finish();
            };
        }
        cases.push_back(std::move(bench));
    }

    // ---- matmul f32 transpose forms (backward shapes) + M=1 decode ----
    struct TransFixture {
        motifcl::VulkanBuffer a, b, c;
        motifcl::Tensor cl_a, cl_b;
        std::size_t m, k, n;
    };
    std::vector<std::shared_ptr<TransFixture>> trans_fixtures;
    enum class MatmulForm { NT, TN, M1NN, M1NT };
    struct TransCase {
        MatmulForm form;
        const char* op;
        std::size_t m, k, n;
        double target;
    };
    const TransCase trans_cases[] = {
        {MatmulForm::NT, "matmul_f32_nt", 256, 256, 256, 0.33},
        {MatmulForm::NT, "matmul_f32_nt", 512, 512, 512, 0.33},
        {MatmulForm::TN, "matmul_f32_tn", 256, 256, 256, 0.33},
        {MatmulForm::TN, "matmul_f32_tn", 512, 512, 512, 0.33},
        {MatmulForm::M1NN, "matmul_f32_m1", 1, 1024, 1024, 0.33},
        {MatmulForm::M1NN, "matmul_f32_m1", 1, 2048, 2048, 0.33},
        {MatmulForm::M1NT, "matmul_f32_m1nt", 1, 1024, 1024, 0.33},
        {MatmulForm::M1NT, "matmul_f32_m1nt", 1, 2048, 2048, 0.33},
    };
    for (const auto& tc : trans_cases) {
        auto fixture = std::make_shared<TransFixture>();
        fixture->m = tc.m;
        fixture->k = tc.k;
        fixture->n = tc.n;
        // A logical shape: NT/M1 forms use [m,k]; TN uses [k,m].
        const std::size_t a_elems = tc.m * tc.k;
        // B logical shape: NT/M1NT use [n,k]; TN/M1NN use [k,n].
        const std::size_t b_elems = tc.k * tc.n;
        const auto a_host = random_host(a_elems, 0xabcd);
        const auto b_host = random_host(b_elems, 0xef01);
        fixture->a = runtime.create_buffer(a_elems * sizeof(float), a_host.data());
        fixture->b = runtime.create_buffer(b_elems * sizeof(float), b_host.data());
        fixture->c = runtime.create_buffer(tc.m * tc.n * sizeof(float));
        if (opencl_available) {
            const auto a_rows = tc.form == MatmulForm::TN ? tc.k : tc.m;
            const auto a_cols = tc.form == MatmulForm::TN ? tc.m : tc.k;
            const auto b_rows = (tc.form == MatmulForm::NT || tc.form == MatmulForm::M1NT) ? tc.n : tc.k;
            const auto b_cols = (tc.form == MatmulForm::NT || tc.form == MatmulForm::M1NT) ? tc.k : tc.n;
            fixture->cl_a = motifcl::Tensor::from_cpu(
                *cl_backend, {static_cast<int64_t>(a_rows), static_cast<int64_t>(a_cols)}, motifcl::DType::F32,
                a_host.data());
            fixture->cl_b = motifcl::Tensor::from_cpu(
                *cl_backend, {static_cast<int64_t>(b_rows), static_cast<int64_t>(b_cols)}, motifcl::DType::F32,
                b_host.data());
        }
        trans_fixtures.push_back(fixture);
        std::ostringstream shape;
        shape << tc.m << "x" << tc.k << "x" << tc.n;
        BenchCase bench;
        bench.op = tc.op;
        bench.shape = shape.str();
        bench.target_ratio = tc.target;
        bench.work_flops = 2.0 * tc.m * tc.k * tc.n;
        bench.work_bytes = 0.0;
        const auto form = tc.form;
        bench.vulkan_iter = [fixture, &runtime, form]() -> std::string {
            motifcl::VulkanOpResult result;
            switch (form) {
                case MatmulForm::NT:
                case MatmulForm::M1NT:
                    result = motifcl::run_vulkan_f32_matmul_transpose_b(runtime, fixture->a, fixture->b, fixture->c,
                                                                        fixture->m, fixture->k, fixture->n);
                    break;
                case MatmulForm::TN:
                    result = motifcl::run_vulkan_f32_matmul_transpose_a(runtime, fixture->a, fixture->b, fixture->c,
                                                                        fixture->m, fixture->k, fixture->n);
                    break;
                case MatmulForm::M1NN:
                    result = motifcl::run_vulkan_f32_matmul(runtime, fixture->a, fixture->b, fixture->c, fixture->m,
                                                            fixture->k, fixture->n);
                    break;
            }
            return result.success ? std::string() : result.error;
        };
        if (opencl_available) {
            auto backend_ptr = cl_backend.get();
            bench.opencl_iter = [fixture, backend_ptr, form]() {
                motifcl::autograd::NoGradGuard guard;
                motifcl::Tensor out;
                switch (form) {
                    case MatmulForm::NT:
                    case MatmulForm::M1NT:
                        out = motifcl::matmul_transpose_b(fixture->cl_a, fixture->cl_b);
                        break;
                    case MatmulForm::TN:
                        out = motifcl::matmul_transpose_a(fixture->cl_a, fixture->cl_b);
                        break;
                    case MatmulForm::M1NN:
                        out = motifcl::matmul(fixture->cl_a, fixture->cl_b);
                        break;
                }
                backend_ptr->finish();
            };
        }
        cases.push_back(std::move(bench));
    }

    // ---- row reductions + elementwise + optimizer (fwd & bwd) ----
    struct RowOpFixture {
        motifcl::VulkanBuffer b0, b1, b2, b3, b4;
        motifcl::Tensor t0, t1, t2;
        std::size_t rows = 0, cols = 0;
    };
    {
        const std::size_t rows = 512, cols = 1024;
        const float eps = 1e-5f;
        const auto x_host = random_host(rows * cols, 0x1111);
        const auto w_host = random_host(cols, 0x2222);
        const auto g_host = random_host(rows * cols, 0x3333);
        // b0=x, b1=w, b2=grad_out, b3=out/dx, b4=row_inv scratch
        auto fx = std::make_shared<RowOpFixture>();
        fx->rows = rows;
        fx->cols = cols;
        fx->b0 = runtime.create_buffer(x_host.size() * sizeof(float), x_host.data());
        fx->b1 = runtime.create_buffer(w_host.size() * sizeof(float), w_host.data());
        fx->b2 = runtime.create_buffer(g_host.size() * sizeof(float), g_host.data());
        fx->b3 = runtime.create_buffer(rows * cols * sizeof(float));
        fx->b4 = runtime.create_buffer(rows * sizeof(float));
        if (opencl_available) {
            fx->t0 = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                               motifcl::DType::F32, x_host.data());
            fx->t1 = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(cols)}, motifcl::DType::F32,
                                               w_host.data());
            fx->t2 = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                               motifcl::DType::F32, g_host.data());
        }
        auto backend_ptr = cl_backend.get();
        auto add_case = [&](const char* op, double target, double bytes,
                            std::function<std::string()> vk, std::function<void()> cl) {
            BenchCase bench;
            bench.op = op;
            std::ostringstream shape;
            shape << rows << "x" << cols;
            bench.shape = shape.str();
            bench.target_ratio = target;
            bench.work_flops = 0.0;
            bench.work_bytes = bytes;
            bench.vulkan_iter = std::move(vk);
            if (opencl_available) bench.opencl_iter = std::move(cl);
            cases.push_back(std::move(bench));
        };
        const double rw2 = 2.0 * rows * cols * sizeof(float);
        const double rw3 = 3.0 * rows * cols * sizeof(float);
        const double rw4 = 4.0 * rows * cols * sizeof(float);
        add_case("softmax_rows_f32", 0.40, rw2,
                 [fx, &runtime]() -> std::string {
                     auto r = motifcl::run_vulkan_softmax_rows(runtime, fx->b0, fx->b3, fx->rows, fx->cols);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr]() {
                     motifcl::autograd::NoGradGuard guard;
                     auto out = motifcl::softmax_rows(fx->t0);
                     backend_ptr->finish();
                 });
        add_case("softmax_rows_bwd_f32", 0.40, rw3,
                 [fx, &runtime]() -> std::string {
                     auto r = motifcl::run_vulkan_softmax_rows_backward(runtime, fx->b0, fx->b2, fx->b3, fx->rows,
                                                                        fx->cols);
                     return r.success ? std::string() : r.error;
                 },
                 std::function<void()>());
        add_case("rmsnorm_f32", 0.40, rw2,
                 [fx, &runtime, eps]() -> std::string {
                     auto r = motifcl::run_vulkan_rmsnorm(runtime, fx->b0, fx->b1, fx->b3, fx->rows, fx->cols, eps);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr, eps]() {
                     motifcl::autograd::NoGradGuard guard;
                     auto out = motifcl::rmsnorm(fx->t0, fx->t1, eps);
                     backend_ptr->finish();
                 });
        add_case("rmsnorm_bwd_x_f32", 0.40, rw3,
                 [fx, &runtime, eps]() -> std::string {
                     auto r = motifcl::run_vulkan_rmsnorm_backward_x(runtime, fx->b0, fx->b1, fx->b2, fx->b3,
                                                                     fx->rows, fx->cols, eps);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr, eps]() {
                     motifcl::autograd::NoGradGuard guard;
                     auto out = motifcl::rmsnorm_backward_x(fx->t0, fx->t1, fx->t2, eps);
                     backend_ptr->finish();
                 });
        add_case("rmsnorm_bwd_w_f32", 0.40, rw2,
                 [fx, &runtime, eps]() -> std::string {
                     auto r = motifcl::run_vulkan_rmsnorm_backward_weight(runtime, fx->b0, fx->b2, fx->b4, fx->b3,
                                                                          fx->rows, fx->cols, eps);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr, eps]() {
                     motifcl::autograd::NoGradGuard guard;
                     auto out = motifcl::rmsnorm_backward_weight(fx->t0, fx->t1, fx->t2, eps);
                     backend_ptr->finish();
                 });
        add_case("gelu_f32", 0.60, rw2,
                 [fx, &runtime]() -> std::string {
                     auto r = motifcl::run_vulkan_gelu(runtime, fx->b0, fx->b3, fx->rows * fx->cols);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr]() {
                     motifcl::autograd::NoGradGuard guard;
                     auto out = motifcl::gelu(fx->t0);
                     backend_ptr->finish();
                 });
        add_case("gelu_bwd_f32", 0.60, rw3,
                 [fx, &runtime]() -> std::string {
                     auto r = motifcl::run_vulkan_gelu_backward(runtime, fx->b0, fx->b2, fx->b3, fx->rows * fx->cols);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr]() {
                     motifcl::autograd::NoGradGuard guard;
                     auto out = motifcl::gelu_backward_op(fx->t0, fx->t2);
                     backend_ptr->finish();
                 });
        add_case("add_f32", 0.60, rw3,
                 [fx, &runtime]() -> std::string {
                     auto r = motifcl::run_vulkan_add(runtime, fx->b0, fx->b2, fx->b3, fx->rows * fx->cols);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr]() {
                     motifcl::autograd::NoGradGuard guard;
                     auto out = motifcl::add(fx->t0, fx->t2);
                     backend_ptr->finish();
                 });
        add_case("sub_f32", 0.60, rw3,
                 [fx, &runtime]() -> std::string {
                     auto r = motifcl::run_vulkan_sub(runtime, fx->b0, fx->b2, fx->b3, fx->rows * fx->cols);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr]() {
                     motifcl::autograd::NoGradGuard guard;
                     auto out = motifcl::sub(fx->t0, fx->t2);
                     backend_ptr->finish();
                 });
        add_case("sgd_update_f32", 0.40, rw3,
                 [fx, &runtime]() -> std::string {
                     auto r = motifcl::run_vulkan_sgd_update(runtime, fx->b0, fx->b2, fx->b3,
                                                             fx->rows * fx->cols, 1e-3f);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr]() {
                     motifcl::autograd::NoGradGuard guard;
                     motifcl::sgd_update(fx->t0, fx->t2, 0.0f);
                     backend_ptr->finish();
                 });
        // mul_scalar / scale: closes the SubBackward/MulBackward/ScalarBackward
        // chain on Vulkan (Slice C2). Same bandwidth profile as add/sub (rw2:
        // read x + write out).
        add_case("mul_scalar_f32", 0.60, rw2,
                 [fx, &runtime]() -> std::string {
                     auto r = motifcl::run_vulkan_mul_scalar(runtime, fx->b0, fx->b3,
                                                              fx->rows * fx->cols, -1.0f);
                     return r.success ? std::string() : r.error;
                 },
                 [fx, backend_ptr]() {
                     motifcl::autograd::NoGradGuard guard;
                     auto out = motifcl::scale(fx->t0, -1.0f);
                     backend_ptr->finish();
                 });
    }

    // SwiGLU fwd + bwd (packed 512 x 2*1024)
    {
        const std::size_t rows = 512, hidden = 1024;
        const auto packed_host = random_host(rows * hidden * 2, 0x4444);
        const auto g_host = random_host(rows * hidden, 0x5555);
        auto fx = std::make_shared<RowOpFixture>();
        fx->rows = rows;
        fx->cols = hidden;
        fx->b0 = runtime.create_buffer(packed_host.size() * sizeof(float), packed_host.data());
        fx->b2 = runtime.create_buffer(g_host.size() * sizeof(float), g_host.data());
        fx->b3 = runtime.create_buffer(rows * hidden * 2 * sizeof(float));
        if (opencl_available) {
            fx->t0 = motifcl::Tensor::from_cpu(*cl_backend,
                                               {static_cast<int64_t>(rows), static_cast<int64_t>(hidden * 2)},
                                               motifcl::DType::F32, packed_host.data());
            fx->t2 = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(rows), static_cast<int64_t>(hidden)},
                                               motifcl::DType::F32, g_host.data());
        }
        auto backend_ptr = cl_backend.get();
        BenchCase fwd;
        fwd.op = "swiglu_f32";
        fwd.shape = "512x2048";
        fwd.target_ratio = 0.60;
        fwd.work_bytes = 3.0 * rows * hidden * sizeof(float);
        fwd.vulkan_iter = [fx, &runtime]() -> std::string {
            auto r = motifcl::run_vulkan_swiglu(runtime, fx->b0, fx->b3, fx->rows, fx->cols);
            return r.success ? std::string() : r.error;
        };
        if (opencl_available) {
            fwd.opencl_iter = [fx, backend_ptr]() {
                motifcl::autograd::NoGradGuard guard;
                auto out = motifcl::swiglu(fx->t0);
                backend_ptr->finish();
            };
        }
        cases.push_back(std::move(fwd));
        BenchCase bwd;
        bwd.op = "swiglu_bwd_f32";
        bwd.shape = "512x2048";
        bwd.target_ratio = 0.60;
        bwd.work_bytes = 5.0 * rows * hidden * sizeof(float);
        bwd.vulkan_iter = [fx, &runtime]() -> std::string {
            auto r = motifcl::run_vulkan_swiglu_backward(runtime, fx->b0, fx->b2, fx->b3, fx->rows, fx->cols);
            return r.success ? std::string() : r.error;
        };
        if (opencl_available) {
            bwd.opencl_iter = [fx, backend_ptr]() {
                motifcl::autograd::NoGradGuard guard;
                auto out = motifcl::swiglu_backward_op(fx->t0, fx->t2);
                backend_ptr->finish();
            };
        }
        cases.push_back(std::move(bwd));
    }

    // ---- Embedding gather + RoPE (Slice E + F perf records) ----
    // Shapes mirror typical transformer prefill: V=4096 vocab, D=64 embed/head_dim,
    // B*T=512 tokens, n_head=8 → channels=64 for rope (single seq, batch folded).
    // Buffers must outlive the bench iteration loops, so hold them in a shared
    // fixture (mirrors RowOpFixture pattern above) — local VulkanBuffer decls
    // would dangle once the setup block exits before iteration starts.
    struct EmbRopeFixture {
        motifcl::VulkanBuffer weight;   // [V, D] f32
        motifcl::VulkanBuffer indices;  // [T] i32
        motifcl::VulkanBuffer emb_out;  // [T, D] f32
        motifcl::VulkanBuffer rope_x;   // [rope_rows, channels] f32
        motifcl::VulkanBuffer rope_out; // [rope_rows, channels] f32
        std::shared_ptr<motifcl::Tensor> cl_w, cl_idx, cl_rope_x;
    };
    {
        const std::size_t V = 4096, D = 64, T = 512;
        const std::size_t n_head = 8, channels = n_head * D;  // 512
        const std::size_t rope_rows = T;                       // batch folded into tokens
        const auto w_host = random_host(V * D, 0x6a6a);
        std::vector<std::int32_t> idx_host(T);
        for (std::size_t t = 0; t < T; ++t) idx_host[t] = static_cast<std::int32_t>((t * 7) % V);
        const auto x_host = random_host(rope_rows * channels, 0x7b7b);

        auto fx = std::make_shared<EmbRopeFixture>();
        fx->weight = runtime.create_buffer(w_host.size() * sizeof(float), w_host.data());
        fx->indices = runtime.create_buffer(idx_host.size() * sizeof(std::int32_t), idx_host.data());
        fx->emb_out = runtime.create_buffer(T * D * sizeof(float));
        fx->rope_x = runtime.create_buffer(x_host.size() * sizeof(float), x_host.data());
        fx->rope_out = runtime.create_buffer(rope_rows * channels * sizeof(float));
        if (opencl_available) {
            fx->cl_w = std::make_shared<motifcl::Tensor>(
                motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(V), static_cast<int64_t>(D)},
                                          motifcl::DType::F32, w_host.data()));
            fx->cl_idx = std::make_shared<motifcl::Tensor>(
                motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(T)}, motifcl::DType::I32,
                                          idx_host.data()));
            fx->cl_rope_x = std::make_shared<motifcl::Tensor>(
                motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(rope_rows),
                                                        static_cast<int64_t>(channels)},
                                          motifcl::DType::F32, x_host.data()));
        }
        auto backend_ptr = cl_backend.get();

        // embedding_gather: read T rows from a [V,D] table + write T*D out.
        // Bandwidth ≈ (T*D + T*D) bytes (index vector negligible).
        {
            BenchCase bc;
            bc.op = "embedding_gather_f32_i32";
            std::ostringstream shape;
            shape << "V" << V << "D" << D << "T" << T;
            bc.shape = shape.str();
            bc.target_ratio = 0.60;
            bc.work_bytes = 2.0 * T * D * sizeof(float);
            bc.vulkan_iter = [fx, &runtime, V, D, T]() -> std::string {
                auto r = motifcl::run_vulkan_embedding_gather(runtime, fx->weight, fx->indices,
                                                               fx->emb_out, V, D, T);
                return r.success ? std::string() : r.error;
            };
            if (opencl_available) {
                bc.opencl_iter = [fx, backend_ptr, V, D]() {
                    motifcl::autograd::NoGradGuard guard;
                    // The Embedding ctor takes Backend&; use the Tensor's own backend
                    // (the fixture-held cl_w keeps the backend alive via shared_ptr).
                    motifcl::Backend& be = fx->cl_w->backend();
                    motifcl::nn::Embedding emb(be, static_cast<int>(V), static_cast<int>(D),
                                                /*skip_weight_init=*/true);
                    emb.weight.data = *fx->cl_w;
                    emb.weight.trainable = false;
                    auto out = emb.forward(*fx->cl_idx);
                    backend_ptr->finish();
                };
            }
            cases.push_back(std::move(bc));
        }

        // rope (interleaved): read x + write out, channels=T*channels total elements.
        {
            BenchCase bc;
            bc.op = "rope_f32";
            std::ostringstream shape;
            shape << "T" << rope_rows << "h" << n_head << "d" << D;
            bc.shape = shape.str();
            bc.target_ratio = 0.40;  // trig (cos/sin/pow) per pair — heavier than add
            bc.work_bytes = 2.0 * rope_rows * channels * sizeof(float);
            bc.vulkan_iter = [fx, &runtime, rope_rows, channels, n_head, D]() -> std::string {
                auto r = motifcl::run_vulkan_rope(runtime, fx->rope_x, fx->rope_out,
                                                   /*batch=*/1, rope_rows, channels, n_head, D,
                                                   /*rotary_dim=*/0, /*token_offset=*/0, 10000.0f,
                                                   /*inverse=*/false);
                return r.success ? std::string() : r.error;
            };
            if (opencl_available) {
                bc.opencl_iter = [fx, n_head, rope_rows, backend_ptr]() {
                    motifcl::autograd::NoGradGuard guard;
                    auto out = motifcl::rope(*fx->cl_rope_x, static_cast<int>(n_head),
                                              /*batch_size=*/1, static_cast<int64_t>(rope_rows), 10000.0f,
                                              /*rotary_dim=*/0, /*token_offset=*/0);
                    backend_ptr->finish();
                };
            }
            cases.push_back(std::move(bc));
        }
    }

    // ---- GQA forward + forward+backward (non-causal, batch=1) ----
    {
        const std::size_t qt = 64, kt = 64, nh = 8, nkh = 2, hd = 64;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        struct GqaFixture {
            motifcl::VulkanBuffer q, k, v, out, g, dq, dk, dv, probs, ds;
            motifcl::Tensor tq, tk, tv, tg;
        };
        auto fx = std::make_shared<GqaFixture>();
        const auto q_host = random_host(qt * nh * hd, 0x6161);
        const auto k_host = random_host(kt * nkh * hd, 0x7272);
        const auto v_host = random_host(kt * nkh * hd, 0x8383);
        const auto g_host = random_host(qt * nh * hd, 0x9494);
        fx->q = runtime.create_buffer(q_host.size() * sizeof(float), q_host.data());
        fx->k = runtime.create_buffer(k_host.size() * sizeof(float), k_host.data());
        fx->v = runtime.create_buffer(v_host.size() * sizeof(float), v_host.data());
        fx->g = runtime.create_buffer(g_host.size() * sizeof(float), g_host.data());
        fx->out = runtime.create_buffer(q_host.size() * sizeof(float));
        fx->dq = runtime.create_buffer(q_host.size() * sizeof(float));
        fx->dk = runtime.create_buffer(k_host.size() * sizeof(float));
        fx->dv = runtime.create_buffer(v_host.size() * sizeof(float));
        fx->probs = runtime.create_buffer(nh * qt * kt * sizeof(float));
        fx->ds = runtime.create_buffer(nh * qt * kt * sizeof(float));
        if (opencl_available) {
            fx->tq = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(qt), static_cast<int64_t>(nh * hd)},
                                               motifcl::DType::F32, q_host.data());
            fx->tk = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(kt), static_cast<int64_t>(nkh * hd)},
                                               motifcl::DType::F32, k_host.data());
            fx->tv = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(kt), static_cast<int64_t>(nkh * hd)},
                                               motifcl::DType::F32, v_host.data());
            fx->tg = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(qt), static_cast<int64_t>(nh * hd)},
                                               motifcl::DType::F32, g_host.data());
        }
        auto backend_ptr = cl_backend.get();
        const double attn_flops = 4.0 * nh * qt * kt * hd;  // qk dots + weighted V
        BenchCase fwd;
        fwd.op = "gqa_fwd_f32";
        fwd.shape = "q64k64h8kv2d64";
        fwd.target_ratio = 0.25;
        fwd.work_flops = attn_flops;
        fwd.vulkan_iter = [fx, &runtime, qt, kt, nh, nkh, hd, scale]() -> std::string {
            auto r = motifcl::run_vulkan_grouped_query_attention(runtime, fx->q, fx->k, fx->v, fx->out, qt, kt,
                                                                 nh, nkh, hd, scale);
            return r.success ? std::string() : r.error;
        };
        if (opencl_available) {
            fwd.opencl_iter = [fx, backend_ptr, qt, kt, nh, nkh]() {
                motifcl::autograd::NoGradGuard guard;
                auto out = motifcl::grouped_query_attention(fx->tq, fx->tk, fx->tv, static_cast<int>(nh),
                                                            static_cast<int>(nkh), false, 1,
                                                            static_cast<int64_t>(qt), static_cast<int64_t>(kt),
                                                            0, 0.0f);
                backend_ptr->finish();
            };
        }
        cases.push_back(std::move(fwd));

        BenchCase fb;
        fb.op = "gqa_fwd_bwd_f32";
        fb.shape = "q64k64h8kv2d64";
        fb.target_ratio = 0.25;
        fb.work_flops = 3.0 * attn_flops;
        fb.vulkan_iter = [fx, &runtime, qt, kt, nh, nkh, hd, scale]() -> std::string {
            auto r = motifcl::run_vulkan_grouped_query_attention(runtime, fx->q, fx->k, fx->v, fx->out, qt, kt,
                                                                 nh, nkh, hd, scale);
            if (!r.success) return r.error;
            r = motifcl::run_vulkan_grouped_query_attention_backward(runtime, fx->q, fx->k, fx->v, fx->g,
                                                                     fx->probs, fx->ds, fx->dq, fx->dk, fx->dv,
                                                                     qt, kt, nh, nkh, hd, scale);
            return r.success ? std::string() : r.error;
        };
        if (opencl_available) {
            fb.opencl_iter = [fx, backend_ptr, qt, kt, nh, nkh]() {
                fx->tq.set_requires_grad(true);
                fx->tk.set_requires_grad(true);
                fx->tv.set_requires_grad(true);
                auto out = motifcl::grouped_query_attention(fx->tq, fx->tk, fx->tv, static_cast<int>(nh),
                                                            static_cast<int>(nkh), false, 1,
                                                            static_cast<int64_t>(qt), static_cast<int64_t>(kt),
                                                            0, 0.0f);
                out.backward(fx->tg);
                backend_ptr->finish();
            };
        }
        cases.push_back(std::move(fb));
    }

    // ---- compact-counter fused state update (memory-native optimizer) ----
    {
        const std::size_t cin = 1024, cout = 256, cC = 3, cN = 8;
        const std::size_t n_groups = cout * (cin / 4);
        struct CounterFixture {
            motifcl::VulkanBuffer state, scale, v, grad_out, x, scale_new, denom;
            std::unique_ptr<motifcl::nn::CounterStateLinear> cl_layer;
            motifcl::Tensor cl_x, cl_go;
        };
        auto fx = std::make_shared<CounterFixture>();
        std::vector<std::uint8_t> state_host(n_groups * 3, 0x25);  // arbitrary mid-range codes
        std::vector<float> ones_host(cout, 1.0f);
        const auto go_host = random_host(cN * cout, 0xaa55);
        const auto x_host = random_host(cN * cin, 0x55aa);
        fx->state = runtime.create_buffer(state_host.size(), state_host.data());
        fx->scale = runtime.create_buffer(cout * sizeof(float), ones_host.data());
        fx->v = runtime.create_buffer(cout * sizeof(float), ones_host.data());
        fx->grad_out = runtime.create_buffer(go_host.size() * sizeof(float), go_host.data());
        fx->x = runtime.create_buffer(x_host.size() * sizeof(float), x_host.data());
        fx->scale_new = runtime.create_buffer(cout * sizeof(float));
        fx->denom = runtime.create_buffer(cout * sizeof(float));
        if (opencl_available) {
            fx->cl_layer = std::make_unique<motifcl::nn::CounterStateLinear>(
                *cl_backend, static_cast<int>(cin), static_cast<int>(cout), static_cast<int>(cC), 0.05f, 0.01f,
                1.0f, 0.9f, 1.0e-8f, 42u);
            fx->cl_x = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(cN), static_cast<int64_t>(cin)},
                                                 motifcl::DType::F32, x_host.data());
            fx->cl_go = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(cN), static_cast<int64_t>(cout)},
                                                  motifcl::DType::F32, go_host.data());
        }
        auto backend_ptr = cl_backend.get();
        BenchCase bench;
        bench.op = "counter_update_f32";
        bench.shape = "in1024_out256_N8";
        bench.target_ratio = 0.40;
        bench.work_bytes = static_cast<double>(n_groups * 3 * 2 + cN * (cin + cout) * sizeof(float));
        bench.vulkan_iter = [fx, &runtime, cC, cin, cout, cN]() -> std::string {
            auto r = motifcl::run_vulkan_compact_counter_apply_update_fused(
                runtime, fx->state, fx->scale, fx->v, fx->grad_out, fx->x, fx->scale_new, fx->denom, cC, cin,
                cout, cN, 0.05f, 0.01f, 0.9f, 1.0e-8f, 1234u);
            return r.success ? std::string() : r.error;
        };
        if (opencl_available) {
            bench.opencl_iter = [fx, backend_ptr]() {
                fx->cl_layer->apply_update_backward(fx->cl_go, fx->cl_x, 1234u);
                backend_ptr->finish();
            };
        }
        cases.push_back(std::move(bench));
    }

    // ---- compact-counter decode + backward-input (retrofit to cached path) ----
    {
        const std::size_t cin = 1024, cout = 256, cC = 3, cN = 8;
        const std::size_t n_groups = cout * (cin / 4);
        struct DecodeFixture {
            motifcl::VulkanBuffer state, scale, weight, grad_out, grad_x;
            std::unique_ptr<motifcl::nn::CounterStateLinear> cl_layer;
            motifcl::Tensor cl_go;
        };
        auto fx = std::make_shared<DecodeFixture>();
        std::vector<std::uint8_t> state_host(n_groups * 3, 0x25);
        std::vector<float> ones_host(cout, 1.0f);
        const auto go_host = random_host(cN * cout, 0x1357);
        fx->state = runtime.create_buffer(state_host.size(), state_host.data());
        fx->scale = runtime.create_buffer(cout * sizeof(float), ones_host.data());
        fx->weight = runtime.create_buffer(cout * cin * sizeof(float));
        fx->grad_out = runtime.create_buffer(go_host.size() * sizeof(float), go_host.data());
        fx->grad_x = runtime.create_buffer(cN * cin * sizeof(float));
        if (opencl_available) {
            fx->cl_layer = std::make_unique<motifcl::nn::CounterStateLinear>(
                *cl_backend, static_cast<int>(cin), static_cast<int>(cout), static_cast<int>(cC), 0.05f, 0.01f,
                1.0f, 0.9f, 1.0e-8f, 42u);
            fx->cl_go = motifcl::Tensor::from_cpu(*cl_backend, {static_cast<int64_t>(cN), static_cast<int64_t>(cout)},
                                                  motifcl::DType::F32, go_host.data());
        }
        auto backend_ptr = cl_backend.get();
        BenchCase decode;
        decode.op = "counter_decode_weight_f32";
        decode.shape = "in1024_out256";
        decode.target_ratio = 0.60;
        decode.work_bytes = static_cast<double>(n_groups * 3 + cout * cin * sizeof(float));
        decode.vulkan_iter = [fx, &runtime, cC, cin, cout]() -> std::string {
            auto r = motifcl::run_vulkan_compact_counter_decode_weight(runtime, fx->state, fx->scale, fx->weight,
                                                                       cin, cout, cC);
            return r.success ? std::string() : r.error;
        };
        if (opencl_available) {
            decode.opencl_iter = [fx, backend_ptr]() {
                motifcl::autograd::NoGradGuard guard;
                auto w = fx->cl_layer->decode_weight();
                backend_ptr->finish();
            };
        }
        cases.push_back(std::move(decode));

        BenchCase bwd_in;
        bwd_in.op = "counter_backward_input_f32";
        bwd_in.shape = "in1024_out256_N8";
        bwd_in.target_ratio = 0.33;  // matmul-like: grad_x = grad_out @ (scale*t)
        bwd_in.work_flops = 2.0 * cN * cin * cout;
        bwd_in.vulkan_iter = [fx, &runtime, cC, cin, cout, cN]() -> std::string {
            auto r = motifcl::run_vulkan_compact_counter_backward_input_u8(runtime, fx->state, fx->scale,
                                                                           fx->grad_out, fx->grad_x, cN, cin,
                                                                           cout, cC);
            return r.success ? std::string() : r.error;
        };
        if (opencl_available) {
            bwd_in.opencl_iter = [fx, backend_ptr]() {
                motifcl::autograd::NoGradGuard guard;
                auto gx = fx->cl_layer->backward_input_from_state(fx->cl_go);
                backend_ptr->finish();
            };
        }
        cases.push_back(std::move(bwd_in));
    }

    // ---- end-to-end SGD training step (same block as test_vulkan_train_step) ----
    {
        struct TrainModel {
            motifcl::Tensor Wq, Wk, Wv, Wo, Wup, Wdown, Whead, Norm1, Norm2, X, Targets;
            const std::int64_t T = 16, n_embd = 64, kv_dim = 32, hidden = 128, vocab = 32;
            const int n_head = 4, n_kv_head = 2;

            void init(motifcl::Backend& backend) {
                auto param = [&](std::int64_t r, std::int64_t c, std::uint32_t seed) {
                    auto host = random_host(static_cast<std::size_t>(r * c), seed);
                    for (auto& v : host) v *= 0.15f;
                    auto t = motifcl::Tensor::from_cpu(backend, {r, c}, motifcl::DType::F32, host.data());
                    t.set_requires_grad(true);
                    return t;
                };
                Wq = param(n_embd, n_embd, 0x11);
                Wk = param(n_embd, kv_dim, 0x22);
                Wv = param(n_embd, kv_dim, 0x33);
                Wo = param(n_embd, n_embd, 0x44);
                Wup = param(n_embd, hidden * 2, 0x55);
                Wdown = param(hidden, n_embd, 0x66);
                Whead = param(n_embd, vocab, 0x77);
                std::vector<float> ones_host(static_cast<std::size_t>(n_embd), 1.0f);
                Norm1 = motifcl::Tensor::from_cpu(backend, {n_embd}, motifcl::DType::F32, ones_host.data());
                Norm2 = motifcl::Tensor::from_cpu(backend, {n_embd}, motifcl::DType::F32, ones_host.data());
                Norm1.set_requires_grad(true);
                Norm2.set_requires_grad(true);
                auto x_host = random_host(static_cast<std::size_t>(T * n_embd), 0x88);
                X = motifcl::Tensor::from_cpu(backend, {T, n_embd}, motifcl::DType::F32, x_host.data());
                std::vector<std::int32_t> targets(static_cast<std::size_t>(T));
                for (std::int64_t t = 0; t < T; ++t) targets[static_cast<std::size_t>(t)] = static_cast<std::int32_t>((t * 7 + 3) % vocab);
                Targets = motifcl::Tensor::from_cpu(backend, {T}, motifcl::DType::I32, targets.data());
            }

            float step(motifcl::VulkanRuntime* batch_runtime) {
                std::vector<motifcl::Tensor*> params = {&Wq, &Wk, &Wv, &Wo, &Wup, &Wdown, &Whead, &Norm1, &Norm2};
                for (auto* p : params) p->zero_grad();
                // Slice J #4: fold forward + backward + optimizer into one
                // batch. Previously this was two batches with a synchronous
                // loss.item() host stall between them; now the scalar loss is
                // read back once after the whole step is submitted.
                if (batch_runtime) batch_runtime->batch_begin();
                auto a = motifcl::rmsnorm(X, Norm1, 1e-5f);
                auto q = motifcl::matmul(a, Wq);
                auto k = motifcl::matmul(a, Wk);
                auto v = motifcl::matmul(a, Wv);
                auto attn = motifcl::grouped_query_attention(q, k, v, n_head, n_kv_head, false, 1, T, T, 0, 0.0f);
                auto o = motifcl::matmul(attn, Wo);
                auto h1 = motifcl::add(X, o);
                auto b = motifcl::rmsnorm(h1, Norm2, 1e-5f);
                auto packed = motifcl::matmul(b, Wup);
                auto m = motifcl::swiglu(packed);
                auto mo = motifcl::matmul(m, Wdown);
                auto h2 = motifcl::add(h1, mo);
                auto logits = motifcl::matmul(h2, Whead);
                auto loss = motifcl::softmax_cross_entropy(logits, Targets);
                loss.backward();
                for (auto* p : params) motifcl::sgd_update(*p, *p->grad(), 1e-3f);
                if (batch_runtime) batch_runtime->batch_end();
                const float value = loss.item();
                return value;
            }
        };
        auto vk_model = std::make_shared<TrainModel>();
        auto cl_model = std::make_shared<TrainModel>();
        std::unique_ptr<motifcl::Backend> vk_backend;
        bool vk_model_ready = false;
        try {
            vk_backend = std::make_unique<motifcl::Backend>(motifcl::Backend::create_vulkan());
            vk_model->init(*vk_backend);
            vk_model_ready = true;
        } catch (const std::exception& e) {
            std::cerr << "train_step vulkan model unavailable: " << e.what() << "\n";
        }
        if (opencl_available) cl_model->init(*cl_backend);
        if (vk_model_ready) {
            BenchCase bench;
            bench.op = "train_step_f32";
            bench.shape = "block_T16_E64";
            bench.target_ratio = 0.40;
            bench.work_bytes = 1.0;  // wall-clock comparison; throughput not meaningful
            auto vk_backend_ptr = vk_backend.get();
            bench.vulkan_iter = [vk_model, vk_backend_ptr]() -> std::string {
                try {
                    (void)vk_model->step(&vk_backend_ptr->vulkan_runtime());
                    return std::string();
                } catch (const std::exception& e) {
                    return e.what();
                }
            };
            if (opencl_available) {
                auto backend_ptr = cl_backend.get();
                bench.opencl_iter = [cl_model, backend_ptr]() {
                    (void)cl_model->step(nullptr);
                    backend_ptr->finish();
                };
            }
            cases.push_back(std::move(bench));
            // Keep the Vulkan backend alive for the duration of the runs.
            static std::unique_ptr<motifcl::Backend> vk_backend_keepalive;
            vk_backend_keepalive = std::move(vk_backend);
        }
    }

    std::vector<CaseResult> results;
    for (const auto& bench : cases) {
        if (!filter.empty() && bench.op.find(filter) == std::string::npos) continue;
        std::cout << "running " << bench.op << " " << bench.shape << " ..." << std::flush;
        auto row = run_case(bench, runtime, opencl_available);
        std::cout << " vk p50 " << row.vk_wall.median_us << " us";
        if (row.cl_ok) std::cout << ", cl p50 " << row.cl_wall_median_us << " us, ratio " << row.ratio();
        std::cout << " " << row.verdict() << "\n";
        if (!row.error.empty()) std::cout << "    error: " << row.error << "\n";
        results.push_back(std::move(row));
    }

    write_reports(results, out_dir, runtime.device_name(), cl_device);
    std::cout << "wrote " << out_dir.string() << "\n";
    return 0;
}
