#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <motifcl/runtime/opencl_context.hpp>
#include <motifcl/runtime/program.hpp>
#include <motifcl/runtime/profiler.hpp>

namespace motifcl {

struct BackendLifetime {
    bool alive = true;
};

class KernelCache {
public:
    KernelCache() = default;
    KernelCache(OpenCLContext& ctx, std::string kernel_dir);

    Kernel get(const std::string& kernel_name);
    Kernel get_matmul_tiled_variant(int tile);
    Kernel get_q8_int_dot_variant(const std::string& mode);
    const std::string& kernel_dir() const { return kernel_dir_; }

private:
    OpenCLContext* ctx_ = nullptr;
    std::string kernel_dir_;
    std::unordered_map<std::string, std::shared_ptr<Program>> programs_;

    std::string source_file_for_kernel(const std::string& kernel_name) const;
    std::string load_source(const std::string& filename) const;
};

class Backend {
public:
    OpenCLContext ctx;
    KernelCache kernels;
    Profiler profiler;

    Backend() = default;
    Backend(OpenCLContext&& context, std::string kernel_dir);
    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&& other) noexcept;
    Backend& operator=(Backend&& other) noexcept;

    static Backend create_opencl(const std::string& kernel_dir = "");
    static Backend create() { return create_opencl(); }

    void finish() const { ctx.finish(); }
    DeviceInfo device_info() const { return ctx.info(); }
    bool supports_integer_dot() const;
    std::string int_dot_mode() const;
    std::shared_ptr<BackendLifetime> lifetime_handle() const { return lifetime_; }

private:
    std::shared_ptr<BackendLifetime> lifetime_ = std::make_shared<BackendLifetime>();
};

} // namespace motifcl
