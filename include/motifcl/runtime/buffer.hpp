#pragma once

#include <CL/cl.h>
#include <cstddef>
#include <memory>

namespace motifcl {

class OpenCLContext;
struct OpenCLContextState;

class Buffer {
public:
    Buffer() = default;
    Buffer(OpenCLContext& ctx, std::size_t nbytes, cl_mem_flags flags = CL_MEM_READ_WRITE);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    void upload(const void* data, std::size_t bytes, std::size_t offset = 0);
    void download(void* data, std::size_t bytes, std::size_t offset = 0) const;

    cl_mem raw() const { return mem_; }
    std::size_t nbytes() const { return nbytes_; }
    bool valid() const { return mem_ != nullptr; }
    OpenCLContext& context() const { return *ctx_; }

private:
    OpenCLContext* ctx_ = nullptr;
    std::shared_ptr<OpenCLContextState> state_;
    cl_mem mem_ = nullptr;
    std::size_t nbytes_ = 0;

    void release();
};

} // namespace motifcl
