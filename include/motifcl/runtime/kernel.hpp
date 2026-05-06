#pragma once

#include <CL/cl.h>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

#include <motifcl/runtime/event.hpp>

namespace motifcl {

class Buffer;
class OpenCLContext;
class Profiler;
struct OpenCLContextState;

class Kernel {
public:
    Kernel() = default;
    Kernel(OpenCLContext& ctx, cl_kernel kernel, std::string name, Profiler* profiler = nullptr);
    ~Kernel();

    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    Kernel(Kernel&& other) noexcept;
    Kernel& operator=(Kernel&& other) noexcept;

    void set_arg(int index, const Buffer& buffer);
    void set_arg_mem(int index, cl_mem mem);
    void set_arg_local(int index, std::size_t bytes);

    template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value || std::is_enum<T>::value>>
    void set_arg(int index, const T& value) {
        set_arg_raw(index, sizeof(T), &value);
    }

    Event launch1d(std::size_t global, std::size_t local = 0);
    Event launch2d(std::size_t gx, std::size_t gy, std::size_t lx = 0, std::size_t ly = 0);
    const std::string& name() const { return name_; }

private:
    OpenCLContext* ctx_ = nullptr;
    std::shared_ptr<OpenCLContextState> state_;
    cl_kernel kernel_ = nullptr;
    std::string name_;
    Profiler* profiler_ = nullptr;

    void set_arg_raw(int index, std::size_t size, const void* ptr);
    void release();
};

} // namespace motifcl
