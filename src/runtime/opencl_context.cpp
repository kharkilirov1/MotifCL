#include <motifcl/runtime/opencl_context.hpp>

#include <motifcl/core/error.hpp>

#include <algorithm>
#include <cstring>
#include <sstream>
#include <vector>

namespace motifcl {

namespace {

std::string get_platform_string(cl_platform_id platform, cl_platform_info name) {
    size_t size = 0;
    MCL_CHECK_CL(clGetPlatformInfo(platform, name, 0, nullptr, &size));
    std::string out(size ? size - 1 : 0, '\0');
    if (size > 0) MCL_CHECK_CL(clGetPlatformInfo(platform, name, size, out.data(), nullptr));
    return out;
}

std::string get_device_string(cl_device_id device, cl_device_info name) {
    size_t size = 0;
    MCL_CHECK_CL(clGetDeviceInfo(device, name, 0, nullptr, &size));
    std::string out(size ? size - 1 : 0, '\0');
    if (size > 0) MCL_CHECK_CL(clGetDeviceInfo(device, name, size, out.data(), nullptr));
    return out;
}

template <typename T>
T get_device_value(cl_device_id device, cl_device_info name) {
    T value{};
    MCL_CHECK_CL(clGetDeviceInfo(device, name, sizeof(T), &value, nullptr));
    return value;
}

DeviceInfo make_info(cl_platform_id platform, cl_device_id device) {
    DeviceInfo info;
    info.platform_name = get_platform_string(platform, CL_PLATFORM_NAME);
    info.platform_vendor = get_platform_string(platform, CL_PLATFORM_VENDOR);
    info.device_name = get_device_string(device, CL_DEVICE_NAME);
    info.device_vendor = get_device_string(device, CL_DEVICE_VENDOR);
    info.driver_version = get_device_string(device, CL_DRIVER_VERSION);
    info.device_version = get_device_string(device, CL_DEVICE_VERSION);
    info.extensions = get_device_string(device, CL_DEVICE_EXTENSIONS);
    info.global_mem_size = get_device_value<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
    info.local_mem_size = get_device_value<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
    info.compute_units = get_device_value<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
    info.max_work_group_size = get_device_value<std::size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    info.max_clock_mhz = get_device_value<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    info.preferred_vector_width_float = get_device_value<cl_uint>(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
    return info;
}

std::vector<cl_platform_id> platforms() {
    cl_uint count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &count);
    if (err == CL_PLATFORM_NOT_FOUND_KHR || count == 0) {
        throw Error("no OpenCL platforms found; install an OpenCL driver/ICD for your GPU or CPU");
    }
    MCL_CHECK_CL(err);
    std::vector<cl_platform_id> result(count);
    MCL_CHECK_CL(clGetPlatformIDs(count, result.data(), nullptr));
    return result;
}

std::vector<cl_device_id> devices_for(cl_platform_id platform, cl_device_type type) {
    cl_uint count = 0;
    cl_int err = clGetDeviceIDs(platform, type, 0, nullptr, &count);
    if (err == CL_DEVICE_NOT_FOUND || count == 0) return {};
    MCL_CHECK_CL(err);
    std::vector<cl_device_id> result(count);
    MCL_CHECK_CL(clGetDeviceIDs(platform, type, count, result.data(), nullptr));
    return result;
}

} // namespace

OpenCLContextState::~OpenCLContextState() { release(); }

void OpenCLContextState::finish() const {
    MCL_CHECK(valid(), "OpenCL context is not initialized");
    MCL_CHECK_CL(clFinish(queue));
}

void OpenCLContextState::release() {
    if (queue) {
        clReleaseCommandQueue(queue);
        queue = nullptr;
    }
    if (context) {
        clReleaseContext(context);
        context = nullptr;
    }
    platform = nullptr;
    device = nullptr;
}

OpenCLContext::~OpenCLContext() { release(); }

OpenCLContext::OpenCLContext(OpenCLContext&& other) noexcept {
    *this = std::move(other);
}

OpenCLContext& OpenCLContext::operator=(OpenCLContext&& other) noexcept {
    if (this != &other) {
        release();
        state_ = std::move(other.state_);
        sync_from_state();
        other.sync_from_state();
    }
    return *this;
}

void OpenCLContext::sync_from_state() {
    if (state_) {
        platform = state_->platform;
        device = state_->device;
        context = state_->context;
        queue = state_->queue;
    } else {
        platform = nullptr;
        device = nullptr;
        context = nullptr;
        queue = nullptr;
    }
}

void OpenCLContext::release() {
    state_.reset();
    sync_from_state();
}

OpenCLContext OpenCLContext::create_default_gpu() {
    auto plats = platforms();
    cl_platform_id chosen_platform = nullptr;
    cl_device_id chosen_device = nullptr;

    for (auto p : plats) {
        auto gpus = devices_for(p, CL_DEVICE_TYPE_GPU);
        if (!gpus.empty()) {
            chosen_platform = p;
            chosen_device = gpus.front();
            break;
        }
    }
    if (!chosen_device) {
        for (auto p : plats) {
            auto cpus = devices_for(p, CL_DEVICE_TYPE_CPU);
            if (!cpus.empty()) {
                chosen_platform = p;
                chosen_device = cpus.front();
                break;
            }
        }
    }
    if (!chosen_device) {
        throw Error("no OpenCL GPU or CPU device found");
    }

    OpenCLContext out;
    out.state_ = std::make_shared<OpenCLContextState>();
    out.state_->platform = chosen_platform;
    out.state_->device = chosen_device;
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(chosen_platform), 0};
    cl_int err = CL_SUCCESS;
    out.state_->context = clCreateContext(props, 1, &chosen_device, nullptr, nullptr, &err);
    MCL_CHECK_CL(err);
    out.state_->queue = clCreateCommandQueue(out.state_->context, chosen_device, CL_QUEUE_PROFILING_ENABLE, &err);
    MCL_CHECK_CL(err);
    out.sync_from_state();
    return out;
}

std::vector<DeviceInfo> OpenCLContext::enumerate_devices() {
    std::vector<DeviceInfo> infos;
    for (auto p : platforms()) {
        auto devs = devices_for(p, CL_DEVICE_TYPE_ALL);
        for (auto d : devs) infos.push_back(make_info(p, d));
    }
    return infos;
}

DeviceInfo OpenCLContext::info() const {
    MCL_CHECK(valid(), "OpenCL context is not initialized");
    return make_info(platform, device);
}

void OpenCLContext::finish() const {
    MCL_CHECK(state_ && state_->valid(), "OpenCL context is not initialized");
    state_->finish();
}

} // namespace motifcl
