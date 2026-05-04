#pragma once

#include <CL/cl.h>
#include <memory>
#include <string>
#include <vector>

namespace motifcl {

struct OpenCLContextState {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;

    ~OpenCLContextState();
    OpenCLContextState() = default;
    OpenCLContextState(const OpenCLContextState&) = delete;
    OpenCLContextState& operator=(const OpenCLContextState&) = delete;

    bool valid() const { return context != nullptr && queue != nullptr && device != nullptr; }
    void finish() const;
    void release();
};

struct DeviceInfo {
    std::string platform_name;
    std::string platform_vendor;
    std::string device_name;
    std::string device_vendor;
    std::string driver_version;
    std::string device_version;
    std::string extensions;
    cl_ulong global_mem_size = 0;
    cl_ulong local_mem_size = 0;
    cl_uint compute_units = 0;
    std::size_t max_work_group_size = 0;
    cl_uint max_clock_mhz = 0;
    cl_uint preferred_vector_width_float = 0;
};

class OpenCLContext {
public:
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;

    OpenCLContext() = default;
    ~OpenCLContext();

    OpenCLContext(const OpenCLContext&) = delete;
    OpenCLContext& operator=(const OpenCLContext&) = delete;
    OpenCLContext(OpenCLContext&& other) noexcept;
    OpenCLContext& operator=(OpenCLContext&& other) noexcept;

    static OpenCLContext create_default_gpu();
    static std::vector<DeviceInfo> enumerate_devices();

    DeviceInfo info() const;
    void finish() const;
    bool valid() const { return context != nullptr && queue != nullptr && device != nullptr; }
    std::shared_ptr<OpenCLContextState> shared_state() const { return state_; }

private:
    std::shared_ptr<OpenCLContextState> state_;

    void sync_from_state();
    void release();
};

} // namespace motifcl
