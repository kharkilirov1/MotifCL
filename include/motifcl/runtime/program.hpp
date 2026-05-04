#pragma once

#include <CL/cl.h>
#include <memory>
#include <string>

#include <motifcl/runtime/kernel.hpp>

namespace motifcl {

class OpenCLContext;
struct OpenCLContextState;

class Program {
public:
    Program() = default;
    Program(OpenCLContext& ctx, const std::string& source, const std::string& options = "");
    ~Program();

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    Kernel get_kernel(const std::string& name);
    cl_program raw() const { return program_; }

private:
    OpenCLContext* ctx_ = nullptr;
    std::shared_ptr<OpenCLContextState> state_;
    cl_program program_ = nullptr;

    void release();
};

} // namespace motifcl
