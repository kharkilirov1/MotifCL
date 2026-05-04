#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <motifcl/motifcl.hpp>

namespace {

template <typename Fn>
double measure_ms(motifcl::Backend& backend, Fn&& fn, int warmup = 2, int iters = 10) {
    for (int i = 0; i < warmup; ++i) fn();
    backend.finish();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    backend.finish();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(iters);
}

void write_metric(std::ofstream& out, const std::string& name, double value, bool last = false) {
    out << "  \"" << name << "\": " << value << (last ? "\n" : ",\n");
}

void write_string(std::ofstream& out, const std::string& name, const std::string& value, bool last = false) {
    out << "  \"" << name << "\": \"" << value << "\"" << (last ? "\n" : ",\n");
}

} // namespace

int main() {
    try {
        auto backend = motifcl::Backend::create_opencl();
        const auto info = backend.device_info();
        constexpr int n = 256;
        auto A = motifcl::Tensor::randn(backend, {n, n}, 0.02f);
        auto B = motifcl::Tensor::randn(backend, {n, n}, 0.02f);

        auto Aq8 = motifcl::quantize_q8_symmetric(A);
        auto Bq8 = motifcl::quantize_q8_symmetric(B);
        auto Aq4 = motifcl::quantize_q4_symmetric(A);
        auto Bq4 = motifcl::quantize_q4_symmetric(B);
        auto Aq8Rows = motifcl::quantize_q8_symmetric_rows(A);
        auto Bq8Cols = motifcl::quantize_q8_symmetric_cols(B);
        auto Aq4Blocks = motifcl::quantize_q4_symmetric_blocks(A, 64);

        double f32_ms = measure_ms(backend, [&] { motifcl::matmul(A, B); });
        double f32_tile4_ms = measure_ms(backend, [&] { motifcl::matmul_tiled_variant(A, B, 4); });
        double f32_tile8_ms = measure_ms(backend, [&] { motifcl::matmul_tiled_variant(A, B, 8); });
        double f32_tile16_ms = measure_ms(backend, [&] { motifcl::matmul_tiled_variant(A, B, 16); });
        double q8_ms = measure_ms(backend, [&] { motifcl::matmul(Aq8, Bq8); });
        double q4_ms = measure_ms(backend, [&] { motifcl::matmul(Aq4, Bq4); });
        double q8_q4_ms = measure_ms(backend, [&] { motifcl::matmul(Aq8, Bq4); });
        double q8_axis_ms = measure_ms(backend, [&] { motifcl::matmul(Aq8Rows, Bq8Cols); });
        double q4_block_ms = measure_ms(backend, [&] { motifcl::matmul(Aq4Blocks, Bq4); });

        std::ofstream out("motifcl_tuning.json");
        out << "{\n";
        out << "  \"device\": \"" << info.device_name << "\",\n";
        out << "  \"driver\": \"" << info.driver_version << "\",\n";
        write_string(out, "int_dot_mode", backend.int_dot_mode());
        out << "  \"matmul_tile\": \"16x16\",\n";
        out << "  \"softmax_local_size\": 256,\n";
        out << "  \"rmsnorm_local_size\": 256,\n";
        write_metric(out, "matmul_f32_256_ms", f32_ms);
        write_metric(out, "matmul_f32_generated_tile4_256_ms", f32_tile4_ms);
        write_metric(out, "matmul_f32_generated_tile8_256_ms", f32_tile8_ms);
        write_metric(out, "matmul_f32_generated_tile16_256_ms", f32_tile16_ms);
        write_metric(out, "matmul_q8_256_ms", q8_ms);
        write_metric(out, "matmul_q4_256_ms", q4_ms);
        write_metric(out, "matmul_q8_q4_256_ms", q8_q4_ms);
        write_metric(out, "matmul_q8_axis_256_ms", q8_axis_ms);
        write_metric(out, "matmul_q4_block64_256_ms", q4_block_ms, true);
        out << "}\n";
        std::cout << "Wrote motifcl_tuning.json for " << info.device_name << "\n";
    } catch (const std::exception& e) {
        std::cerr << "OpenCL unavailable or tuning failed: " << e.what() << "\n";
        return 0;
    }
}
