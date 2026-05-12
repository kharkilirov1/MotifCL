#include <motifcl/runtime/backend.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/runtime/command_buffer.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace motifcl {

namespace fs = std::filesystem;

namespace {

bool contains(const std::string& s, const std::string& needle) {
    return s.find(needle) != std::string::npos;
}

int env_int(const char* name, int fallback) {
    if (const char* env = std::getenv(name)) {
        if (*env) {
            char* end = nullptr;
            long value = std::strtol(env, &end, 10);
            if (end && *end == '\0') return static_cast<int>(value);
        }
    }
    return fallback;
}

int attention_tile_from_env() {
    const int value = env_int("MOTIFCL_FA_TILE", 16);
    return (value == 8 || value == 16 || value == 32) ? value : 16;
}

int attention_workgroup_from_env() {
    const int value = env_int("MOTIFCL_FA_WG", 128);
    return (value == 64 || value == 128 || value == 256) ? value : 128;
}

std::string build_options_for_source(const std::string& file) {
    std::string options = "-cl-std=CL1.2";
    if (file == "attention.cl") {
        options += " -DFA_TILE=" + std::to_string(attention_tile_from_env());
        options += " -DFA_WG=" + std::to_string(attention_workgroup_from_env());
    }
    return options;
}

std::string default_kernel_dir() {
    if (const char* env = std::getenv("MOTIFCL_KERNEL_DIR")) {
        if (*env) return env;
    }
#ifdef MOTIFCL_BINARY_KERNEL_DIR
    if (fs::exists(MOTIFCL_BINARY_KERNEL_DIR)) return MOTIFCL_BINARY_KERNEL_DIR;
#endif
#ifdef MOTIFCL_SOURCE_DIR
    fs::path source_dir = fs::path(MOTIFCL_SOURCE_DIR) / "kernels";
    if (fs::exists(source_dir)) return source_dir.string();
#endif
#ifdef MOTIFCL_INSTALL_KERNEL_DIR
    if (fs::exists(MOTIFCL_INSTALL_KERNEL_DIR)) return MOTIFCL_INSTALL_KERNEL_DIR;
#endif
    return "kernels";
}

std::string matmul_tiled_variant_name(int tile) {
    return "matmul_tiled_f32_t" + std::to_string(tile);
}

std::string q8_int_dot_variant_name(const std::string& mode) {
    if (mode == "cl_khr_integer_dot_product") return "matmul_q8_0_khr_dot_f32";
    return "matmul_q8_0_dot4_f32";
}

std::string generated_matmul_tiled_source(int tile, const std::string& kernel_name) {
    std::ostringstream src;
    src << "#define TILE " << tile << "\n";
    src << "__kernel void " << kernel_name << "(__global const float* A,\n"
        << "                                      __global const float* B,\n"
        << "                                      __global float* C,\n"
        << "                                      int M,\n"
        << "                                      int N,\n"
        << "                                      int K) {\n"
        << "    int col = get_global_id(0);\n"
        << "    int row = get_global_id(1);\n"
        << "    int local_col = get_local_id(0);\n"
        << "    int local_row = get_local_id(1);\n"
        << "    __local float As[TILE][TILE];\n"
        << "    __local float Bs[TILE][TILE];\n"
        << "    float acc = 0.0f;\n"
        << "    int tiles = (K + TILE - 1) / TILE;\n"
        << "    for (int t = 0; t < tiles; ++t) {\n"
        << "        int a_col = t * TILE + local_col;\n"
        << "        int b_row = t * TILE + local_row;\n"
        << "        As[local_row][local_col] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;\n"
        << "        Bs[local_row][local_col] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;\n"
        << "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        << "        for (int kk = 0; kk < TILE; ++kk) acc += As[local_row][kk] * Bs[kk][local_col];\n"
        << "        barrier(CLK_LOCAL_MEM_FENCE);\n"
        << "    }\n"
        << "    if (row < M && col < N) C[row * N + col] = acc;\n"
        << "}\n";
    return src.str();
}

std::string generated_q8_khr_dot_source(const std::string& kernel_name) {
    std::ostringstream src;
    src << "#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable\n"
        << "__kernel void " << kernel_name << "(__global const char* A,\n"
        << "                                      __global const char* B,\n"
        << "                                      __global float* C,\n"
        << "                                      int M,\n"
        << "                                      int N,\n"
        << "                                      int K,\n"
        << "                                      float scale_a,\n"
        << "                                      float scale_b) {\n"
        << "    int col = get_global_id(0);\n"
        << "    int row = get_global_id(1);\n"
        << "    if (row >= M || col >= N) return;\n"
        << "    int acc = 0;\n"
        << "    int k = 0;\n"
        << "    for (; k + 3 < K; k += 4) {\n"
        << "#if defined(__opencl_c_integer_dot_product_input_4x8bit)\n"
        << "        char4 av = (char4)(A[row * K + k + 0], A[row * K + k + 1], A[row * K + k + 2], A[row * K + k + 3]);\n"
        << "        char4 bv = (char4)(B[(k + 0) * N + col], B[(k + 1) * N + col], B[(k + 2) * N + col], B[(k + 3) * N + col]);\n"
        << "        acc += dot(av, bv);\n"
        << "#else\n"
        << "        acc += ((int)A[row * K + k + 0]) * ((int)B[(k + 0) * N + col]);\n"
        << "        acc += ((int)A[row * K + k + 1]) * ((int)B[(k + 1) * N + col]);\n"
        << "        acc += ((int)A[row * K + k + 2]) * ((int)B[(k + 2) * N + col]);\n"
        << "        acc += ((int)A[row * K + k + 3]) * ((int)B[(k + 3) * N + col]);\n"
        << "#endif\n"
        << "    }\n"
        << "    for (; k < K; ++k) acc += ((int)A[row * K + k]) * ((int)B[k * N + col]);\n"
        << "    C[row * N + col] = ((float)acc) * scale_a * scale_b;\n"
        << "}\n";
    return src.str();
}

} // namespace

KernelCache::KernelCache(OpenCLContext& ctx, std::string kernel_dir, Profiler* profiler)
    : ctx_(&ctx), profiler_(profiler), kernel_dir_(std::move(kernel_dir)) {}

std::string KernelCache::source_file_for_kernel(const std::string& kernel_name) const {
    if (contains(kernel_name, "f16")) return "fp16.cl";
    if (contains(kernel_name, "matmul")) return "matmul.cl";
    if (contains(kernel_name, "quantize") || contains(kernel_name, "dequantize")) return "quant.cl";
    if (contains(kernel_name, "mse") || contains(kernel_name, "cross_entropy") || contains(kernel_name, "mean_reduce")) return "loss.cl";
    if (contains(kernel_name, "softmax") || contains(kernel_name, "causal") || contains(kernel_name, "attention") ||
        contains(kernel_name, "rope") || contains(kernel_name, "qkv") || contains(kernel_name, "gqa") || contains(kernel_name, "grouped_query") ||
        contains(kernel_name, "kv_cache")) return "attention.cl";
    if (contains(kernel_name, "rms") || contains(kernel_name, "norm")) return "norm.cl";
    if (contains(kernel_name, "motif") || contains(kernel_name, "sarc") || contains(kernel_name, "router") ||
        contains(kernel_name, "moe") || contains(kernel_name, "gated_delta") || contains(kernel_name, "sigmoid_gate")) return "motif.cl";
    if (contains(kernel_name, "fused_packed_ple")) return "activation.cl";
    if (contains(kernel_name, "relu") || contains(kernel_name, "gelu") || contains(kernel_name, "silu") ||
        contains(kernel_name, "swiglu") || contains(kernel_name, "exp_") || contains(kernel_name, "sqrt") ||
        contains(kernel_name, "rsqrt")) return "activation.cl";
    if (contains(kernel_name, "embedding") || contains(kernel_name, "token_position")) return "embedding.cl";
    if (contains(kernel_name, "adam") || contains(kernel_name, "sgd")) return "optim.cl";
    if (contains(kernel_name, "sum") || contains(kernel_name, "max") || contains(kernel_name, "reduce")) return "reduce.cl";
    return "basic.cl";
}

std::string KernelCache::load_source(const std::string& filename) const {
    fs::path path = fs::path(kernel_dir_) / filename;
    std::ifstream in(path);
    if (!in.good()) {
        throw Error("failed to open OpenCL kernel source: " + path.string());
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    return oss.str();
}

Kernel KernelCache::get(const std::string& kernel_name) {
    MCL_CHECK(ctx_ != nullptr, "kernel cache has no context");
    auto file = source_file_for_kernel(kernel_name);
    auto options = build_options_for_source(file);
    const auto key = file + "|" + options;
    auto it = programs_.find(key);
    if (it == programs_.end()) {
        auto source = load_source(file);
        auto program = std::make_shared<Program>(*ctx_, source, options, profiler_);
        it = programs_.emplace(key, std::move(program)).first;
    }
    return it->second->get_kernel(kernel_name);
}

Kernel KernelCache::get_matmul_tiled_variant(int tile) {
    MCL_CHECK(ctx_ != nullptr, "kernel cache has no context");
    MCL_CHECK(tile == 4 || tile == 8 || tile == 16, "generated matmul tile must be one of 4, 8, or 16");
    const auto kernel_name = matmul_tiled_variant_name(tile);
    const auto key = std::string("generated:") + kernel_name;
    auto it = programs_.find(key);
    if (it == programs_.end()) {
        auto source = generated_matmul_tiled_source(tile, kernel_name);
        auto program = std::make_shared<Program>(*ctx_, source, "-cl-std=CL1.2", profiler_);
        it = programs_.emplace(key, std::move(program)).first;
    }
    return it->second->get_kernel(kernel_name);
}

Kernel KernelCache::get_q8_int_dot_variant(const std::string& mode) {
    MCL_CHECK(ctx_ != nullptr, "kernel cache has no context");
    if (mode != "cl_khr_integer_dot_product") {
        return get("matmul_q8_0_dot4_f32");
    }
    const auto kernel_name = q8_int_dot_variant_name(mode);
    const auto key = std::string("generated:") + kernel_name;
    auto it = programs_.find(key);
    if (it == programs_.end()) {
        auto source = generated_q8_khr_dot_source(kernel_name);
        std::shared_ptr<Program> program;
        try {
            program = std::make_shared<Program>(*ctx_, source, "-cl-std=CL3.0", profiler_);
        } catch (const Error&) {
            try {
            program = std::make_shared<Program>(*ctx_, source, "-cl-std=CL2.0", profiler_);
            } catch (const Error&) {
                program = std::make_shared<Program>(*ctx_, source, "-cl-std=CL1.2", profiler_);
            }
        }
        it = programs_.emplace(key, std::move(program)).first;
    }
    return it->second->get_kernel(kernel_name);
}

Backend::Backend(OpenCLContext&& context, std::string kernel_dir)
    : ctx(std::move(context)), kernels(ctx, kernel_dir.empty() ? default_kernel_dir() : std::move(kernel_dir), &profiler) {}

Backend::~Backend() {
    if (lifetime_) lifetime_->alive = false;
}

Backend::Backend(Backend&& other) noexcept
    : ctx(std::move(other.ctx)), kernels(ctx, other.kernels.kernel_dir(), &profiler), lifetime_(std::make_shared<BackendLifetime>()) {
    if (other.lifetime_) other.lifetime_->alive = false;
}

Backend& Backend::operator=(Backend&& other) noexcept {
    if (this != &other) {
        if (lifetime_) lifetime_->alive = false;
        ctx = std::move(other.ctx);
        kernels = KernelCache(ctx, other.kernels.kernel_dir(), &profiler);
        if (other.lifetime_) other.lifetime_->alive = false;
        lifetime_ = std::make_shared<BackendLifetime>();
    }
    return *this;
}

Backend Backend::create_opencl(const std::string& kernel_dir) {
    return Backend(OpenCLContext::create_default_gpu(), kernel_dir.empty() ? default_kernel_dir() : kernel_dir);
}

bool Backend::supports_integer_dot() const {
    const auto info = device_info();
    return contains(info.extensions, "cl_khr_integer_dot_product") ||
           contains(info.extensions, "cl_arm_integer_dot_product") ||
           contains(info.extensions, "cl_intel_subgroups") ||
           contains(info.device_vendor, "Advanced Micro Devices") ||
           contains(info.device_vendor, "AMD") ||
           contains(info.device_vendor, "NVIDIA") ||
           contains(info.device_vendor, "Intel");
}

std::string Backend::int_dot_mode() const {
    const auto info = device_info();
    if (contains(info.extensions, "cl_khr_integer_dot_product")) return "cl_khr_integer_dot_product";
    if (contains(info.extensions, "cl_arm_integer_dot_product")) return "cl_arm_integer_dot_product";
    if (contains(info.extensions, "cl_intel_subgroups")) return "cl_intel_subgroups";
    if (supports_integer_dot()) return "vendor_dot4_unrolled";
    return "scalar_fallback";
}

bool Backend::supports_command_buffer() const {
    return query_command_buffer_support(ctx.platform, ctx.device).supported;
}

bool Backend::supports_command_buffer_mutable_dispatch() const {
    return query_command_buffer_support(ctx.platform, ctx.device).mutable_dispatch_supported;
}

std::string Backend::command_buffer_mode() const {
    return query_command_buffer_support(ctx.platform, ctx.device).mode;
}

} // namespace motifcl
