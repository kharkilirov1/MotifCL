#include <motifcl/runtime/native_matmul.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::int64_t k = 512;
    std::int64_t n = 1024;
    int iters = 500;
    int warmup = 10;
    std::string json_path;
};

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) throw std::runtime_error(std::string(name) + " requires a value");
            return argv[++i];
        };
        if (arg == "--k") {
            options.k = std::stoll(require_value("--k"));
        } else if (arg == "--n") {
            options.n = std::stoll(require_value("--n"));
        } else if (arg == "--iters") {
            options.iters = std::stoi(require_value("--iters"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(require_value("--warmup"));
        } else if (arg == "--json") {
            options.json_path = require_value("--json");
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.k < 0 || options.n < 0 || options.iters <= 0 || options.warmup < 0) {
        throw std::runtime_error("invalid benchmark dimensions or iteration counts");
    }
    return options;
}

template <typename Fn>
double measure_ms(Fn&& fn, int warmup, int iters, float& checksum) {
    for (int i = 0; i < warmup; ++i) fn();
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = Clock::now();
    checksum = fn();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(iters);
}

float checksum(const std::vector<float>& values) {
    float sum = 0.0f;
    for (float value : values) sum += value;
    return sum;
}

std::string json_escape(const std::string& value) {
    std::string out;
    for (char ch : value) {
        if (ch == '"' || ch == '\\') out.push_back('\\');
        out.push_back(ch);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto total = static_cast<std::size_t>(options.k * options.n);
        std::vector<float> a(static_cast<std::size_t>(options.k));
        std::vector<float> b(total);
        std::vector<float> out_scalar(static_cast<std::size_t>(options.n), 0.0f);
        std::vector<float> out_dispatch(static_cast<std::size_t>(options.n), 0.0f);

        for (std::int64_t k = 0; k < options.k; ++k) {
            a[static_cast<std::size_t>(k)] = static_cast<float>((k % 29) - 14) * 0.00390625f;
        }
        for (std::size_t i = 0; i < b.size(); ++i) {
            b[i] = static_cast<float>((static_cast<int>(i % 31) - 15)) * 0.0029296875f;
        }

        float scalar_checksum = 0.0f;
        const double scalar_ms = measure_ms([&] {
            motifcl::native::matmul_f32_m1_scalar(a.data(), b.data(), out_scalar.data(), options.k, options.n);
            return checksum(out_scalar);
        }, options.warmup, options.iters, scalar_checksum);

        float dispatch_checksum = 0.0f;
        const double dispatch_ms = measure_ms([&] {
            motifcl::native::matmul_f32_m1(a.data(), b.data(), out_dispatch.data(), options.k, options.n);
            return checksum(out_dispatch);
        }, options.warmup, options.iters, dispatch_checksum);

        double max_abs_diff = 0.0;
        for (std::size_t i = 0; i < out_scalar.size(); ++i) {
            max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::abs(out_scalar[i] - out_dispatch[i])));
        }
        const double speedup = dispatch_ms > 0.0 ? scalar_ms / dispatch_ms : 0.0;

        std::ostringstream json;
        json << std::fixed << std::setprecision(6)
             << "{\n"
             << "  \"native_matmul_f32_m1_scalar_ms\": " << scalar_ms << ",\n"
             << "  \"native_matmul_f32_m1_dispatch_ms\": " << dispatch_ms << ",\n"
             << "  \"native_matmul_f32_m1_max_abs_diff\": " << max_abs_diff << ",\n"
             << "  \"metadata\": {\n"
             << "    \"kernel\": \"" << json_escape(motifcl::native::matmul_f32_m1_kernel_name()) << "\",\n"
             << "    \"k\": " << options.k << ",\n"
             << "    \"n\": " << options.n << ",\n"
             << "    \"iters\": " << options.iters << ",\n"
             << "    \"warmup\": " << options.warmup << ",\n"
             << "    \"speedup\": " << speedup << ",\n"
             << "    \"scalar_checksum\": " << scalar_checksum << ",\n"
             << "    \"dispatch_checksum\": " << dispatch_checksum << "\n"
             << "  }\n"
             << "}\n";

        if (!options.json_path.empty()) {
            std::ofstream out(options.json_path);
            out << json.str();
        }
        std::cout << json.str();
        return max_abs_diff <= 1e-4 ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "bench_native_matmul failed: " << e.what() << "\n";
        return 1;
    }
}
