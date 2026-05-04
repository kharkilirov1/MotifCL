#include <motifcl/runtime/buffer.hpp>

#include <motifcl/core/error.hpp>
#include <motifcl/runtime/opencl_context.hpp>

namespace motifcl {

Buffer::Buffer(OpenCLContext& ctx, std::size_t nbytes, cl_mem_flags flags) : ctx_(&ctx), state_(ctx.shared_state()), nbytes_(nbytes) {
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
        other.ctx_ = nullptr;
        other.mem_ = nullptr;
        other.nbytes_ = 0;
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
}

void Buffer::upload(const void* data, std::size_t bytes, std::size_t offset) {
    MCL_CHECK(valid(), "upload to invalid buffer");
    MCL_CHECK(offset + bytes <= nbytes_, "upload range exceeds buffer size");
    MCL_CHECK(state_ && state_->valid(), "upload without live OpenCL context");
    MCL_CHECK_CL(clEnqueueWriteBuffer(state_->queue, mem_, CL_TRUE, offset, bytes, data, 0, nullptr, nullptr));
}

void Buffer::download(void* data, std::size_t bytes, std::size_t offset) const {
    MCL_CHECK(valid(), "download from invalid buffer");
    MCL_CHECK(offset + bytes <= nbytes_, "download range exceeds buffer size");
    MCL_CHECK(state_ && state_->valid(), "download without live OpenCL context");
    MCL_CHECK_CL(clEnqueueReadBuffer(state_->queue, mem_, CL_TRUE, offset, bytes, data, 0, nullptr, nullptr));
}

} // namespace motifcl
