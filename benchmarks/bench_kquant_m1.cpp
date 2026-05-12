#include <motifcl/motifcl.hpp>
#include <motifcl/gguf.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string model;
    int iters = 8;
    int warmup = 2;
    int limit = 8;
    std::uint64_t max_elements = 64000000ull;
    std::string type = "all";
    bool no_repack = false;
};

struct Candidate {
    motifcl::gguf::TensorInfo info;
    std::uint64_t elements = 0;
};

double elapsed_ms(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

void usage() {
    std::cerr
        << "Usage: bench_kquant_m1 --model MODEL.gguf [options]\n"
        << "\n"
        << "Benchmarks M=1 x [D,N] matmul for GGUF Q4_K/Q6_K tensors and compares\n"
        << "direct K-quant kernels with dequantize+Q4_0_COL repack when tensor size is safe.\n"
        << "\nOptions:\n"
        << "  --model PATH             GGUF file\n"
        << "  --iters N                measured matmul iterations (default: 8)\n"
        << "  --warmup N               warmup iterations (default: 2)\n"
        << "  --limit N                max tensors to benchmark (default: 8)\n"
        << "  --max-elements N         skip repack for larger tensors (default: 64000000)\n"
        << "  --type all|q4|q6         filter tensor type (default: all)\n"
        << "  --no-repack              direct K-quant only\n";
}

Options parse_args(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << "\n";
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--model") opts.model = require_value("--model");
        else if (arg == "--iters") opts.iters = std::stoi(require_value("--iters"));
        else if (arg == "--warmup") opts.warmup = std::stoi(require_value("--warmup"));
        else if (arg == "--limit") opts.limit = std::stoi(require_value("--limit"));
        else if (arg == "--max-elements") opts.max_elements = static_cast<std::uint64_t>(std::stoull(require_value("--max-elements")));
        else if (arg == "--type") opts.type = require_value("--type");
        else if (arg == "--no-repack") opts.no_repack = true;
        else if (arg == "--help" || arg == "-h") {
            usage();
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            usage();
            std::exit(2);
        }
    }
    if (opts.model.empty() || opts.iters <= 0 || opts.warmup < 0 || opts.limit <= 0 ||
        !(opts.type == "all" || opts.type == "q4" || opts.type == "q6")) {
        usage();
        std::exit(2);
    }
    return opts;
}

bool interesting_weight_name(const std::string& name) {
    return name.find(".weight") != std::string::npos &&
           name.find("token_embd") == std::string::npos &&
           name.find("output") == std::string::npos;
}

std::vector<Candidate> select_candidates(const motifcl::gguf::File& file, const Options& opts) {
    std::vector<Candidate> out;
    for (const auto& info : file.tensors()) {
        if (info.dimensions.size() != 2) continue;
        if (info.type != motifcl::gguf::TensorType::Q4_K &&
            info.type != motifcl::gguf::TensorType::Q6_K) {
            continue;
        }
        if (opts.type == "q4" && info.type != motifcl::gguf::TensorType::Q4_K) continue;
        if (opts.type == "q6" && info.type != motifcl::gguf::TensorType::Q6_K) continue;
        const auto elements = motifcl::gguf::tensor_element_count(info);
        if (!interesting_weight_name(info.name) && elements > opts.max_elements) continue;
        out.push_back({info, elements});
    }
    std::stable_sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
        const auto at = static_cast<int>(a.info.type);
        const auto bt = static_cast<int>(b.info.type);
        if (at != bt) return at < bt;
        if (interesting_weight_name(a.info.name) != interesting_weight_name(b.info.name)) {
            return interesting_weight_name(a.info.name);
        }
        return a.elements > b.elements;
    });
    if (static_cast<int>(out.size()) > opts.limit) out.resize(static_cast<std::size_t>(opts.limit));
    return out;
}

template <typename Fn>
double bench_ms(motifcl::Backend& backend,
                int warmup,
                int iters,
                Fn&& fn) {
    motifcl::Tensor y;
    for (int i = 0; i < warmup; ++i) y = fn();
    backend.finish();
    const auto start = Clock::now();
    for (int i = 0; i < iters; ++i) y = fn();
    backend.finish();
    const auto end = Clock::now();
    (void)y;
    return elapsed_ms(start, end) / static_cast<double>(iters);
}

std::vector<int64_t> tensor_shape_i64(const motifcl::gguf::TensorInfo& info) {
    std::vector<int64_t> shape;
    shape.reserve(info.dimensions.size());
    for (auto dim : info.dimensions) shape.push_back(static_cast<int64_t>(dim));
    return shape;
}

void print_profile(motifcl::Backend& backend, const std::string& indent = "    ") {
    int shown = 0;
    for (const auto& row : backend.profiler.summary()) {
        if (shown++ >= 4) break;
        std::cout << indent << row.name
                  << " count=" << row.count
                  << " total_ms=" << std::fixed << std::setprecision(3) << row.total_ms
                  << " avg_ms=" << row.avg_ms << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    const auto opts = parse_args(argc, argv);
    auto file = motifcl::gguf::File::open(opts.model);
    auto candidates = select_candidates(file, opts);
    if (candidates.empty()) {
        std::cerr << "no rank-2 Q4_K/Q6_K tensors selected from " << opts.model << "\n";
        return 2;
    }

    auto backend = motifcl::Backend::create_opencl();
    std::cout << "model=" << opts.model << "\n"
              << "iters=" << opts.iters << " warmup=" << opts.warmup
              << " selected=" << candidates.size() << "\n";

    for (const auto& cand : candidates) {
        const auto& info = cand.info;
        const auto shape = tensor_shape_i64(info);
        const int64_t k = shape[0];
        const int64_t n = shape[1];
        std::cout << "\n"
                  << "tensor=" << info.name << "\n"
                  << "type=" << motifcl::gguf::tensor_type_name(info.type)
                  << " shape=[" << k << "," << n << "]"
                  << " elements=" << cand.elements << "\n";

        auto weight = file.read_tensor_quantized(backend, info.name);
        auto x = motifcl::Tensor::randn(backend, {1, k}, 0.02f);
        backend.profiler.clear();
        backend.profiler.set_enabled(true);
        const double direct_ms = bench_ms(backend, opts.warmup, opts.iters, [&]() {
            return motifcl::matmul(x, weight);
        });
        backend.profiler.set_enabled(false);
        std::cout << "direct_" << motifcl::gguf::tensor_type_name(info.type)
                  << "_ms=" << std::fixed << std::setprecision(3) << direct_ms << "\n";
        print_profile(backend);

        if (opts.no_repack || cand.elements > opts.max_elements) {
            std::cout << "repack_q4_0_col_ms=skipped";
            if (cand.elements > opts.max_elements) std::cout << " reason=max_elements";
            std::cout << "\n";
            continue;
        }

        {
            const auto repack_start = Clock::now();
            auto q4_col = motifcl::nn::repack_gguf_k_quant_to_q4_0_col(backend, file, info.name, false, true);
            backend.finish();
            const double repack_ms = elapsed_ms(repack_start, Clock::now());
            backend.profiler.clear();
            backend.profiler.set_enabled(true);
            const double q4_ms = bench_ms(backend, opts.warmup, opts.iters, [&]() {
                return motifcl::matmul(x, q4_col);
            });
            backend.profiler.set_enabled(false);
            std::cout << "repack_q4_0_col_load_ms=" << std::fixed << std::setprecision(3) << repack_ms << "\n"
                      << "repack_q4_0_col_ms=" << q4_ms << "\n"
                      << "direct_over_repack_q4_0_col_ms_ratio="
                      << (q4_ms > 0.0 ? direct_ms / q4_ms : 0.0) << "\n";
            print_profile(backend);
        }

        {
            const auto repack_start = Clock::now();
            auto q4_tile8 = motifcl::nn::repack_gguf_k_quant_to_q4_0_col(backend, file, info.name, true, true);
            backend.finish();
            const double repack_ms = elapsed_ms(repack_start, Clock::now());
            backend.profiler.clear();
            backend.profiler.set_enabled(true);
            const double q4_ms = bench_ms(backend, opts.warmup, opts.iters, [&]() {
                return motifcl::matmul(x, q4_tile8);
            });
            backend.profiler.set_enabled(false);
            std::cout << "repack_q4_0_col_tile8_load_ms=" << std::fixed << std::setprecision(3) << repack_ms << "\n"
                      << "repack_q4_0_col_tile8_ms=" << q4_ms << "\n"
                      << "direct_over_repack_q4_0_col_tile8_ms_ratio="
                      << (q4_ms > 0.0 ? direct_ms / q4_ms : 0.0) << "\n";
            print_profile(backend);
        }
    }

    return 0;
}
