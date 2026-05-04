#include <motifcl/runtime/program.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/runtime/opencl_context.hpp>

#include <vector>

namespace motifcl {

Program::Program(OpenCLContext& ctx, const std::string& source, const std::string& options) : ctx_(&ctx), state_(ctx.shared_state()) {
    MCL_CHECK(ctx.valid(), "OpenCL context is not initialized");
    MCL_CHECK(state_ && state_->valid(), "OpenCL context state is not initialized");
    const char* src = source.c_str();
    size_t len = source.size();
    cl_int err = CL_SUCCESS;
    program_ = clCreateProgramWithSource(state_->context, 1, &src, &len, &err);
    MCL_CHECK_CL(err);
    err = clBuildProgram(program_, 1, &state_->device, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program_, state_->device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size ? log_size - 1 : 0, '\0');
        if (log_size) clGetProgramBuildInfo(program_, state_->device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        throw Error(std::string("OpenCL program build failed: ") + cl_error_name(err) + "\n" + log);
    }
}

Program::~Program() { release(); }

Program::Program(Program&& other) noexcept { *this = std::move(other); }

Program& Program::operator=(Program&& other) noexcept {
    if (this != &other) {
        release();
        ctx_ = other.ctx_;
        state_ = std::move(other.state_);
        program_ = other.program_;
        other.ctx_ = nullptr;
        other.program_ = nullptr;
    }
    return *this;
}

void Program::release() {
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
    ctx_ = nullptr;
    state_.reset();
}

Kernel Program::get_kernel(const std::string& name) {
    MCL_CHECK(program_ != nullptr, "program is not initialized");
    MCL_CHECK(state_ && state_->valid(), "program has no live OpenCL context");
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program_, name.c_str(), &err);
    MCL_CHECK_CL(err);
    return Kernel(*ctx_, kernel, name);
}

} // namespace motifcl
