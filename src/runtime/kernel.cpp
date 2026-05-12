#include <motifcl/runtime/kernel.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/autograd/graph.hpp>
#include <motifcl/runtime/buffer.hpp>
#include <motifcl/runtime/opencl_context.hpp>
#include <motifcl/runtime/profiler.hpp>

#include <type_traits>
#include <utility>

namespace motifcl {

namespace {

double event_elapsed_ms(cl_event event) {
    MCL_CHECK(event != nullptr, "cannot query profiling on null event");
    MCL_CHECK_CL(clWaitForEvents(1, &event));
    cl_ulong start = 0, end = 0;
    MCL_CHECK_CL(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr));
    MCL_CHECK_CL(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
    return static_cast<double>(end - start) * 1e-6;
}

std::vector<std::pair<int, cl_mem>> buffer_arg_pairs(const std::vector<KernelBufferArgSnapshot>& args) {
    std::vector<std::pair<int, cl_mem>> out;
    out.reserve(args.size());
    for (const auto& arg : args) {
        out.emplace_back(arg.index, arg.mem);
    }
    return out;
}

}

Kernel::Kernel(OpenCLContext& ctx, cl_kernel kernel, std::string name, Profiler* profiler)
    : ctx_(&ctx), state_(ctx.shared_state()), kernel_(kernel), name_(std::move(name)), profiler_(profiler) {}

Kernel::~Kernel() { release(); }

Kernel::Kernel(Kernel&& other) noexcept { *this = std::move(other); }

Kernel& Kernel::operator=(Kernel&& other) noexcept {
    if (this != &other) {
        release();
        ctx_ = other.ctx_;
        state_ = std::move(other.state_);
        kernel_ = other.kernel_;
        name_ = std::move(other.name_);
        profiler_ = other.profiler_;
        buffer_args_ = std::move(other.buffer_args_);
        other.ctx_ = nullptr;
        other.kernel_ = nullptr;
        other.profiler_ = nullptr;
    }
    return *this;
}

void Kernel::release() {
    if (kernel_) {
        clReleaseKernel(kernel_);
        kernel_ = nullptr;
    }
    ctx_ = nullptr;
    state_.reset();
    profiler_ = nullptr;
}

void Kernel::set_arg_raw(int index, std::size_t size, const void* ptr) {
    MCL_CHECK(kernel_ != nullptr, "kernel is not initialized");
    MCL_CHECK_CL(clSetKernelArg(kernel_, static_cast<cl_uint>(index), size, ptr));
}

void Kernel::remember_buffer_arg(int index, cl_mem mem) {
    for (auto& arg : buffer_args_) {
        if (arg.index == index) {
            arg.mem = mem;
            return;
        }
    }
    buffer_args_.push_back({index, mem});
}

void Kernel::set_arg(int index, const Buffer& buffer) {
    cl_mem mem = buffer.raw();
    set_arg_raw(index, sizeof(cl_mem), &mem);
    remember_buffer_arg(index, mem);
}

void Kernel::set_arg_mem(int index, cl_mem mem) {
    set_arg_raw(index, sizeof(cl_mem), &mem);
    remember_buffer_arg(index, mem);
}

void Kernel::set_arg_local(int index, std::size_t bytes) {
    MCL_CHECK(kernel_ != nullptr, "kernel is not initialized");
    MCL_CHECK_CL(clSetKernelArg(kernel_, static_cast<cl_uint>(index), bytes, nullptr));
}

Event Kernel::launch1d(std::size_t global, std::size_t local) {
    MCL_CHECK(state_ && state_->valid(), "kernel launch without valid context");
    MCL_CHECK(global > 0, "global work size must be positive");
    size_t g[1] = {global};
    size_t l[1] = {local};
    const bool profile = profiler_ && profiler_->enabled();
    cl_event event = nullptr;
    MCL_CHECK_CL(clEnqueueNDRangeKernel(state_->queue, kernel_, 1, nullptr, g, local ? l : nullptr, 0, nullptr,
                                        profile ? &event : nullptr));
    if (profile) profiler_->add(name_, event_elapsed_ms(event));
    if (autograd::is_graph_capturing()) {
        MCL_CHECK_CL(clRetainKernel(kernel_));
        using KernelHandle = std::remove_pointer_t<cl_kernel>;
        std::shared_ptr<KernelHandle> retained(kernel_, [](KernelHandle* k) {
            if (k) clReleaseKernel(k);
        });
        auto state = state_;
        auto buffer_args = buffer_arg_pairs(buffer_args_);
        autograd::GraphKernelLaunchInfo launch;
        launch.kernel_name = name_;
        launch.platform = state->platform;
        launch.device = state->device;
        launch.context = state->context;
        launch.queue = state->queue;
        launch.kernel = retained.get();
        launch.work_dim = 1;
        launch.global_work_size = {global};
        if (local) launch.local_work_size = {local};
        launch.has_local_work_size = local != 0;
        launch.buffer_args = buffer_args;
        launch.retained_state = state;
        launch.retained_kernel = retained;
        autograd::record_kernel_launch(std::move(launch), [state, retained, global, local, buffer_args](const std::unordered_map<int, cl_mem>& bindings,
                                                                                                        const std::vector<std::pair<int, int>>& tensor_arg_bindings) {
            MCL_CHECK(state && state->valid(), "replay without live OpenCL context");
            for (const auto& arg : buffer_args) {
                cl_mem mem = arg.second;
                for (const auto& binding : tensor_arg_bindings) {
                    if (binding.first != arg.first) continue;
                    auto it = bindings.find(binding.second);
                    if (it != bindings.end()) {
                        mem = it->second;
                        break;
                    }
                }
                MCL_CHECK_CL(clSetKernelArg(retained.get(), static_cast<cl_uint>(arg.first), sizeof(cl_mem), &mem));
            }
            size_t rg[1] = {global};
            size_t rl[1] = {local};
            MCL_CHECK_CL(clEnqueueNDRangeKernel(state->queue, retained.get(), 1, nullptr, rg, local ? rl : nullptr, 0, nullptr, nullptr));
        });
    }
    return Event(event);
}

Event Kernel::launch2d(std::size_t gx, std::size_t gy, std::size_t lx, std::size_t ly) {
    MCL_CHECK(state_ && state_->valid(), "kernel launch without valid context");
    MCL_CHECK(gx > 0 && gy > 0, "global work size must be positive");
    size_t g[2] = {gx, gy};
    size_t l[2] = {lx, ly};
    const bool profile = profiler_ && profiler_->enabled();
    cl_event event = nullptr;
    MCL_CHECK_CL(clEnqueueNDRangeKernel(state_->queue, kernel_, 2, nullptr, g, (lx && ly) ? l : nullptr, 0, nullptr,
                                        profile ? &event : nullptr));
    if (profile) profiler_->add(name_, event_elapsed_ms(event));
    if (autograd::is_graph_capturing()) {
        MCL_CHECK_CL(clRetainKernel(kernel_));
        using KernelHandle = std::remove_pointer_t<cl_kernel>;
        std::shared_ptr<KernelHandle> retained(kernel_, [](KernelHandle* k) {
            if (k) clReleaseKernel(k);
        });
        auto state = state_;
        auto buffer_args = buffer_arg_pairs(buffer_args_);
        autograd::GraphKernelLaunchInfo launch;
        launch.kernel_name = name_;
        launch.platform = state->platform;
        launch.device = state->device;
        launch.context = state->context;
        launch.queue = state->queue;
        launch.kernel = retained.get();
        launch.work_dim = 2;
        launch.global_work_size = {gx, gy};
        if (lx && ly) launch.local_work_size = {lx, ly};
        launch.has_local_work_size = lx != 0 && ly != 0;
        launch.buffer_args = buffer_args;
        launch.retained_state = state;
        launch.retained_kernel = retained;
        autograd::record_kernel_launch(std::move(launch), [state, retained, gx, gy, lx, ly, buffer_args](const std::unordered_map<int, cl_mem>& bindings,
                                                                                                         const std::vector<std::pair<int, int>>& tensor_arg_bindings) {
            MCL_CHECK(state && state->valid(), "replay without live OpenCL context");
            for (const auto& arg : buffer_args) {
                cl_mem mem = arg.second;
                for (const auto& binding : tensor_arg_bindings) {
                    if (binding.first != arg.first) continue;
                    auto it = bindings.find(binding.second);
                    if (it != bindings.end()) {
                        mem = it->second;
                        break;
                    }
                }
                MCL_CHECK_CL(clSetKernelArg(retained.get(), static_cast<cl_uint>(arg.first), sizeof(cl_mem), &mem));
            }
            size_t rg[2] = {gx, gy};
            size_t rl[2] = {lx, ly};
            MCL_CHECK_CL(clEnqueueNDRangeKernel(state->queue, retained.get(), 2, nullptr, rg, (lx && ly) ? rl : nullptr, 0, nullptr, nullptr));
        });
    }
    return Event(event);
}

} // namespace motifcl
