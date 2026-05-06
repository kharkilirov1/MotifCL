#include <motifcl/runtime/buffer.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/runtime/backend.hpp>
#include <motifcl/runtime/event.hpp>
#include <motifcl/runtime/opencl_context.hpp>
#include <motifcl/runtime/profiler.hpp>

#include <utility>

namespace motifcl {

Buffer::Buffer(OpenCLContext& ctx,
               std::size_t nbytes,
               cl_mem_flags flags,
               Profiler* profiler,
               std::shared_ptr<BackendLifetime> profiler_lifetime)
    : ctx_(&ctx),
      state_(ctx.shared_state()),
      nbytes_(nbytes),
      profiler_(profiler),
      profiler_lifetime_(std::move(profiler_lifetime)) {
    MCL_CHECK(ctx.valid(), "OpenCL context is not initialized");
    MCL_CHECK(state_ && state_->valid(), "OpenCL context state is not initialized");
    MCL_CHECK(nbytes > 0, "cannot allocate zero-byte buffer");
    cl_int err = CL_SUCCESS;
    mem_ = clCreateBuffer(state_->context, flags, nbytes, nullptr, &err);
    MCL_CHECK_CL(err);
}

Buffer::~Buffer() { release(); }

Buffer::Buffer(Buffer&& other) noexcept { *this = std::move(other); }

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        release();
        ctx_ = other.ctx_;
        state_ = std::move(other.state_);
        mem_ = other.mem_;
        nbytes_ = other.nbytes_;
        profiler_ = other.profiler_;
        profiler_lifetime_ = std::move(other.profiler_lifetime_);
        other.ctx_ = nullptr;
        other.mem_ = nullptr;
        other.nbytes_ = 0;
        other.profiler_ = nullptr;
        other.profiler_lifetime_.reset();
    }
    return *this;
}

void Buffer::release() {
    if (mem_) {
        clReleaseMemObject(mem_);
        mem_ = nullptr;
    }
    ctx_ = nullptr;
    state_.reset();
    nbytes_ = 0;
    profiler_ = nullptr;
    profiler_lifetime_.reset();
}

void Buffer::set_profiler(Profiler* profiler, std::shared_ptr<BackendLifetime> profiler_lifetime) {
    profiler_ = profiler;
    profiler_lifetime_ = std::move(profiler_lifetime);
}

void Buffer::upload(const void* data, std::size_t bytes, std::size_t offset) {
    MCL_CHECK(valid(), "upload to invalid buffer");
    MCL_CHECK(offset + bytes <= nbytes_, "upload range exceeds buffer size");
    MCL_CHECK(state_ && state_->valid(), "upload without live OpenCL context");
    const bool profile = profiler_ && profiler_lifetime_ && profiler_lifetime_->alive && profiler_->enabled();
    cl_event event = nullptr;
    MCL_CHECK_CL(clEnqueueWriteBuffer(state_->queue, mem_, CL_TRUE, offset, bytes, data, 0, nullptr, profile ? &event : nullptr));
    if (event) {
        Event profiled(event);
        profiler_->add("transfer.upload", profiled.elapsed_ms());
    }
}

void Buffer::download(void* data, std::size_t bytes, std::size_t offset) const {
    MCL_CHECK(valid(), "download from invalid buffer");
    MCL_CHECK(offset + bytes <= nbytes_, "download range exceeds buffer size");
    MCL_CHECK(state_ && state_->valid(), "download without live OpenCL context");
    const bool profile = profiler_ && profiler_lifetime_ && profiler_lifetime_->alive && profiler_->enabled();
    cl_event event = nullptr;
    MCL_CHECK_CL(clEnqueueReadBuffer(state_->queue, mem_, CL_TRUE, offset, bytes, data, 0, nullptr, profile ? &event : nullptr));
    if (event) {
        Event profiled(event);
        profiler_->add("transfer.download", profiled.elapsed_ms());
    }
}

bool Buffer::same_context(const OpenCLContext& ctx) const {
    return state_ && state_ == ctx.shared_state();
}

} // namespace motifcl
