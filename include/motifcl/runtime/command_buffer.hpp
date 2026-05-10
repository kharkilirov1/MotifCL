#pragma once

#include <CL/cl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace motifcl::autograd {
struct GraphKernelLaunchInfo;
}

namespace motifcl {

// Minimal cl_khr_command_buffer / cl_khr_command_buffer_mutable_dispatch declarations.
// Kept local so MotifCL can build with the vendored OpenCL 1.2 header and still
// use driver-level command buffers when a newer ICD exposes the extension entrypoints.
using cl_device_command_buffer_capabilities_khr_mcl = cl_bitfield;
using cl_command_buffer_khr_mcl = struct _cl_command_buffer_khr*;
using cl_sync_point_khr_mcl = cl_uint;
using cl_command_buffer_properties_khr_mcl = intptr_t;
using cl_command_buffer_flags_khr_mcl = cl_bitfield;
using cl_ndrange_kernel_command_properties_khr_mcl = intptr_t;
using cl_mutable_command_khr_mcl = struct _cl_mutable_command_khr*;
using cl_command_buffer_structure_type_khr_mcl = cl_uint;
using cl_mutable_dispatch_fields_khr_mcl = cl_bitfield;

constexpr cl_device_info CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR_MCL = 0x12A9;
constexpr cl_device_info CL_DEVICE_MUTABLE_DISPATCH_CAPABILITIES_KHR_MCL = 0x12B0;
constexpr cl_command_buffer_properties_khr_mcl CL_COMMAND_BUFFER_FLAGS_KHR_MCL = 0x1293;
constexpr cl_command_buffer_flags_khr_mcl CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR_MCL = (1 << 0);
constexpr cl_command_buffer_flags_khr_mcl CL_COMMAND_BUFFER_MUTABLE_KHR_MCL = (1 << 1);
constexpr cl_ndrange_kernel_command_properties_khr_mcl CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR_MCL = 0x12B1;
constexpr cl_mutable_dispatch_fields_khr_mcl CL_MUTABLE_DISPATCH_ARGUMENTS_KHR_MCL = (1 << 3);
constexpr cl_command_buffer_structure_type_khr_mcl CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR_MCL = 0;
constexpr cl_command_buffer_structure_type_khr_mcl CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR_MCL = 1;

struct MutableDispatchArgKHR {
    cl_uint arg_index = 0;
    std::size_t arg_size = 0;
    const void* arg_value = nullptr;
};

struct MutableDispatchConfigKHR {
    cl_command_buffer_structure_type_khr_mcl type = CL_STRUCTURE_TYPE_MUTABLE_DISPATCH_CONFIG_KHR_MCL;
    const void* next = nullptr;
    cl_mutable_command_khr_mcl command = nullptr;
    cl_uint num_args = 0;
    cl_uint num_svm_args = 0;
    cl_uint num_exec_infos = 0;
    cl_uint work_dim = 0;
    const MutableDispatchArgKHR* arg_list = nullptr;
    const MutableDispatchArgKHR* arg_svm_list = nullptr;
    const void* exec_info_list = nullptr;
    const std::size_t* global_work_offset = nullptr;
    const std::size_t* global_work_size = nullptr;
    const std::size_t* local_work_size = nullptr;
};

struct MutableBaseConfigKHR {
    cl_command_buffer_structure_type_khr_mcl type = CL_STRUCTURE_TYPE_MUTABLE_BASE_CONFIG_KHR_MCL;
    const void* next = nullptr;
    cl_uint num_mutable_dispatch = 0;
    const MutableDispatchConfigKHR* mutable_dispatch_list = nullptr;
};

struct CommandBufferSupport {
    bool extension_advertised = false;
    bool mutable_extension_advertised = false;
    bool entrypoints_available = false;
    bool mutable_entrypoints_available = false;
    bool supported = false;
    bool mutable_dispatch_supported = false;
    std::string mode = "host_replay_fallback";
};

CommandBufferSupport query_command_buffer_support(cl_platform_id platform, cl_device_id device);

class DriverCommandBuffer {
public:
    static std::unique_ptr<DriverCommandBuffer> try_create(
        const std::vector<autograd::GraphKernelLaunchInfo>& launches);

    DriverCommandBuffer();
    ~DriverCommandBuffer();
    DriverCommandBuffer(const DriverCommandBuffer&) = delete;
    DriverCommandBuffer& operator=(const DriverCommandBuffer&) = delete;
    DriverCommandBuffer(DriverCommandBuffer&& other) noexcept;
    DriverCommandBuffer& operator=(DriverCommandBuffer&& other) noexcept;

    bool valid() const;
    bool mutable_rebinding() const { return mutable_rebinding_; }
    const std::string& mode() const { return mode_; }
    bool can_execute_with_bindings(const std::unordered_map<int, cl_mem>& tensor_bindings) const;
    bool execute(const std::unordered_map<int, cl_mem>& tensor_bindings);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool mutable_rebinding_ = false;
    std::string mode_ = "host_replay_fallback";
};

} // namespace motifcl
