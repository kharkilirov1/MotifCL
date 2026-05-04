#pragma once

#include <CL/cl.h>
#include <stdexcept>
#include <string>

namespace motifcl {

class Error : public std::runtime_error {
public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};

const char* cl_error_name(cl_int err);
void check_cl(cl_int err, const char* expr, const char* file, int line);

} // namespace motifcl

#define MCL_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            throw ::motifcl::Error(std::string("MotifCL check failed: ") + (msg) + \
                                   " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        } \
    } while (0)

#define MCL_CHECK_CL(expr) ::motifcl::check_cl((expr), #expr, __FILE__, __LINE__)
