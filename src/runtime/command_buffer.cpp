#include <motifcl/runtime/command_buffer.hpp>

#include <motifcl/autograd/graph.hpp>

#include <algorithm>
#include <cstring>
#include <utility>

namespace motifcl {

namespace {

constexpr const char* kCommandBufferExt = "cl_khr_command_buffer";
constexpr const char* kMutableDispatchExt = "cl_khr_command_buffer_mutable_dispatch";

bool contains_extension(const std::string& extensions, const char* needle) {
    return extensions.find(needle) != std::string::npos;
}

std::string get_device_string_noexcept(cl_device_id device, cl_device_info name) {
    if (!device) return {};
    std::size_t size = 0;
    if (clGetDeviceInfo(device, name, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::string out(size ? size - 1 : 0, '\0');
    if (clGetDeviceInfo(device, name, size, out.data(), nullptr) != CL_SUCCESS) return {};
    return out;
}

template <typename Fn>
Fn load_extension_fn(cl_platform_id platform, const char* name) {
    void* ptr = nullptr;
    if (platform) ptr = clGetExtensionFunctionAddressForPlatform(platform, name);
    if (!ptr) ptr = clGetExtensionFunctionAddress(name);
    return reinterpret_cast<Fn>(ptr);
}

using clCreateCommandBufferKHR_fn_mcl =
    cl_command_buffer_khr_mcl (*)(cl_uint,
                                  const cl_command_queue*,
                                  const cl_command_buffer_properties_khr_mcl*,
                                  cl_int*);
using clFinalizeCommandBufferKHR_fn_mcl = cl_int (*)(cl_command_buffer_khr_mcl);
using clReleaseCommandBufferKHR_fn_mcl = cl_int (*)(cl_command_buffer_khr_mcl);
using clEnqueueCommandBufferKHR_fn_mcl =
    cl_int (*)(cl_uint, cl_command_queue*, cl_command_buffer_khr_mcl, cl_uint, const cl_event*, cl_event*);
using clCommandNDRangeKernelKHR_fn_mcl =
    cl_int (*)(cl_command_buffer_khr_mcl,
               cl_command_queue,
               const cl_ndrange_kernel_command_properties_khr_mcl*,
               cl_kernel,
               cl_uint,
               const std::size_t*,
               const std::size_t*,
               const std::size_t*,
               cl_uint,
               const cl_sync_point_khr_mcl*,
               cl_sync_point_khr_mcl*,
               cl_mutable_command_khr_mcl*);
using clUpdateMutableCommandsKHR_fn_mcl =
    cl_int (*)(cl_command_buffer_khr_mcl, const MutableBaseConfigKHR*);

struct CommandBufferDispatch {
    clCreateCommandBufferKHR_fn_mcl create = nullptr;
    clFinalizeCommandBufferKHR_fn_mcl finalize = nullptr;
    clReleaseCommandBufferKHR_fn_mcl release = nullptr;
    clEnqueueCommandBufferKHR_fn_mcl enqueue = nullptr;
    clCommandNDRangeKernelKHR_fn_mcl command_ndrange = nullptr;
    clUpdateMutableCommandsKHR_fn_mcl update_mutable = nullptr;

    bool basic_available() const {
        return create && finalize && release && enqueue && command_ndrange;
    }

    bool mutable_available() const {
        return basic_available() && update_mutable;
    }
};

CommandBufferDispatch load_dispatch(cl_platform_id platform) {
    CommandBufferDispatch out;
    out.create = load_extension_fn<clCreateCommandBufferKHR_fn_mcl>(platform, "clCreateCommandBufferKHR");
    out.finalize = load_extension_fn<clFinalizeCommandBufferKHR_fn_mcl>(platform, "clFinalizeCommandBufferKHR");
    out.release = load_extension_fn<clReleaseCommandBufferKHR_fn_mcl>(platform, "clReleaseCommandBufferKHR");
    out.enqueue = load_extension_fn<clEnqueueCommandBufferKHR_fn_mcl>(platform, "clEnqueueCommandBufferKHR");
    out.command_ndrange = load_extension_fn<clCommandNDRangeKernelKHR_fn_mcl>(platform, "clCommandNDRangeKernelKHR");
    out.update_mutable = load_extension_fn<clUpdateMutableCommandsKHR_fn_mcl>(platform, "clUpdateMutableCommandsKHR");
    return out;
}

CommandBufferSupport make_support(cl_platform_id platform, cl_device_id device, const CommandBufferDispatch& dispatch) {
    (void)platform;
    CommandBufferSupport support;
    const auto extensions = get_device_string_noexcept(device, CL_DEVICE_EXTENSIONS);
    support.extension_advertised = contains_extension(extensions, kCommandBufferExt);
    support.mutable_extension_advertised = contains_extension(extensions, kMutableDispatchExt);
    support.entrypoints_available = dispatch.basic_available();
    support.mutable_entrypoints_available = dispatch.mutable_available();
    support.supported = support.extension_advertised && support.entrypoints_available;
    support.mutable_dispatch_supported = support.supported &&
                                         support.mutable_extension_advertised &&
                                         support.mutable_entrypoints_available;

    if (support.mutable_dispatch_supported) {
        support.mode = "cl_khr_command_buffer_mutable_dispatch";
    } else if (support.supported) {
        support.mode = "cl_khr_command_buffer_replay";
    } else if (support.extension_advertised) {
        support.mode = "cl_khr_command_buffer_missing_entrypoints";
    } else {
        support.mode = "host_replay_fallback";
    }
    return support;
}

cl_mem rebound_mem_for_arg(const autograd::GraphKernelLaunchInfo& launch,
                           int arg_index,
                           cl_mem captured,
                           const std::unordered_map<int, cl_mem>& tensor_bindings) {
    for (const auto& binding : launch.tensor_arg_bindings) {
        if (binding.first != arg_index) continue;
        auto it = tensor_bindings.find(binding.second);
        if (it != tensor_bindings.end()) return it->second;
    }
    return captured;
}

bool same_driver_target(const std::vector<autograd::GraphKernelLaunchInfo>& launches) {
    if (launches.empty()) return false;
    const auto platform = launches.front().platform;
    const auto device = launches.front().device;
    const auto context = launches.front().context;
    const auto queue = launches.front().queue;
    return std::all_of(launches.begin(), launches.end(), [&](const auto& launch) {
        return launch.platform == platform &&
               launch.device == device &&
               launch.context == context &&
               launch.queue == queue &&
               launch.kernel != nullptr &&
               launch.work_dim > 0 &&
               !launch.global_work_size.empty();
    });
}

} // namespace

CommandBufferSupport query_command_buffer_support(cl_platform_id platform, cl_device_id device) {
    return make_support(platform, device, load_dispatch(platform));
}

struct DriverCommandBuffer::Impl {
    CommandBufferDispatch dispatch;
    cl_command_buffer_khr_mcl command_buffer = nullptr;
    cl_command_queue queue = nullptr;
    std::vector<autograd::GraphKernelLaunchInfo> launches;
    std::vector<cl_mutable_command_khr_mcl> mutable_commands;

    ~Impl() {
        if (command_buffer && dispatch.release) {
            dispatch.release(command_buffer);
            command_buffer = nullptr;
        }
    }
};

DriverCommandBuffer::DriverCommandBuffer() = default;
DriverCommandBuffer::~DriverCommandBuffer() = default;

DriverCommandBuffer::DriverCommandBuffer(DriverCommandBuffer&& other) noexcept {
    *this = std::move(other);
}

DriverCommandBuffer& DriverCommandBuffer::operator=(DriverCommandBuffer&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        mutable_rebinding_ = other.mutable_rebinding_;
        mode_ = std::move(other.mode_);
        other.mutable_rebinding_ = false;
        other.mode_ = "host_replay_fallback";
    }
    return *this;
}

bool DriverCommandBuffer::valid() const {
    return impl_ && impl_->command_buffer != nullptr;
}

bool DriverCommandBuffer::can_execute_with_bindings(const std::unordered_map<int, cl_mem>& tensor_bindings) const {
    return valid() && (tensor_bindings.empty() || mutable_rebinding_);
}

std::unique_ptr<DriverCommandBuffer> DriverCommandBuffer::try_create(
    const std::vector<autograd::GraphKernelLaunchInfo>& launches) {
    if (!same_driver_target(launches)) return nullptr;

    const auto platform = launches.front().platform;
    const auto device = launches.front().device;
    auto dispatch = load_dispatch(platform);
    auto support = make_support(platform, device, dispatch);
    if (!support.supported) return nullptr;

    auto build = [&](bool mutable_dispatch) -> std::unique_ptr<DriverCommandBuffer> {
        auto out = std::unique_ptr<DriverCommandBuffer>(new DriverCommandBuffer());
        out->impl_ = std::make_unique<Impl>();
        out->impl_->dispatch = dispatch;
        out->impl_->queue = launches.front().queue;
        out->impl_->launches = launches;
        out->impl_->mutable_commands.resize(launches.size(), nullptr);

        cl_command_buffer_properties_khr_mcl mutable_props[] = {
            CL_COMMAND_BUFFER_FLAGS_KHR_MCL,
            CL_COMMAND_BUFFER_MUTABLE_KHR_MCL,
            0
        };
        const cl_command_buffer_properties_khr_mcl* props = mutable_dispatch ? mutable_props : nullptr;

        cl_int err = CL_SUCCESS;
        cl_command_queue queue = out->impl_->queue;
        out->impl_->command_buffer = dispatch.create(1, &queue, props, &err);
        if (err != CL_SUCCESS || !out->impl_->command_buffer) return nullptr;

        cl_sync_point_khr_mcl previous_sync = 0;
        bool has_previous_sync = false;
        cl_ndrange_kernel_command_properties_khr_mcl mutable_ndrange_props[] = {
            CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR_MCL,
            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR_MCL,
            0
        };

        for (std::size_t i = 0; i < launches.size(); ++i) {
            const auto& launch = launches[i];
            cl_sync_point_khr_mcl sync_point = 0;
            cl_mutable_command_khr_mcl mutable_handle = nullptr;
            const auto* local = launch.has_local_work_size ? launch.local_work_size.data() : nullptr;
            const auto* ndrange_props = mutable_dispatch ? mutable_ndrange_props : nullptr;
            err = dispatch.command_ndrange(out->impl_->command_buffer,
                                           launch.queue,
                                           ndrange_props,
                                           launch.kernel,
                                           launch.work_dim,
                                           nullptr,
                                           launch.global_work_size.data(),
                                           local,
                                           has_previous_sync ? 1u : 0u,
                                           has_previous_sync ? &previous_sync : nullptr,
                                           &sync_point,
                                           mutable_dispatch ? &mutable_handle : nullptr);
            if (err != CL_SUCCESS) return nullptr;
            previous_sync = sync_point;
            has_previous_sync = true;
            out->impl_->mutable_commands[i] = mutable_handle;
        }

        err = dispatch.finalize(out->impl_->command_buffer);
        if (err != CL_SUCCESS) return nullptr;

        out->mutable_rebinding_ = mutable_dispatch;
        out->mode_ = mutable_dispatch ? "cl_khr_command_buffer_mutable_dispatch"
                                      : "cl_khr_command_buffer_replay";
        return out;
    };

    if (support.mutable_dispatch_supported) {
        if (auto mutable_cb = build(true)) return mutable_cb;
    }
    return build(false);
}

bool DriverCommandBuffer::execute(const std::unordered_map<int, cl_mem>& tensor_bindings) {
    if (!valid()) return false;
    if (!tensor_bindings.empty() && !mutable_rebinding_) return false;

    if (mutable_rebinding_) {
        const std::size_t count = impl_->launches.size();
        std::vector<std::vector<cl_mem>> arg_values(count);
        std::vector<std::vector<MutableDispatchArgKHR>> arg_lists(count);
        std::vector<MutableDispatchConfigKHR> configs;
        configs.reserve(count);

        for (std::size_t i = 0; i < count; ++i) {
            const auto& launch = impl_->launches[i];
            const auto mutable_command = impl_->mutable_commands[i];
            if (!mutable_command || launch.buffer_args.empty()) continue;

            arg_values[i].reserve(launch.buffer_args.size());
            arg_lists[i].reserve(launch.buffer_args.size());
            for (const auto& arg : launch.buffer_args) {
                arg_values[i].push_back(rebound_mem_for_arg(launch, arg.first, arg.second, tensor_bindings));
                arg_lists[i].push_back(MutableDispatchArgKHR{
                    static_cast<cl_uint>(arg.first),
                    sizeof(cl_mem),
                    &arg_values[i].back()
                });
            }

            MutableDispatchConfigKHR config;
            config.command = mutable_command;
            config.num_args = static_cast<cl_uint>(arg_lists[i].size());
            config.work_dim = launch.work_dim;
            config.arg_list = arg_lists[i].data();
            configs.push_back(config);
        }

        if (!configs.empty()) {
            MutableBaseConfigKHR base;
            base.num_mutable_dispatch = static_cast<cl_uint>(configs.size());
            base.mutable_dispatch_list = configs.data();
            const cl_int err = impl_->dispatch.update_mutable(impl_->command_buffer, &base);
            if (err != CL_SUCCESS) return false;
        }
    }

    cl_command_queue queue = impl_->queue;
    const cl_int err = impl_->dispatch.enqueue(1, &queue, impl_->command_buffer, 0, nullptr, nullptr);
    return err == CL_SUCCESS;
}

} // namespace motifcl
