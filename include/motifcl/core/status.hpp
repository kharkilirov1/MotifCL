#pragma once

#include <string>

namespace motifcl {

enum class StatusCode {
    Ok,
    RuntimeError,
    InvalidArgument,
    OpenCLError
};

struct Status {
    StatusCode code = StatusCode::Ok;
    std::string message;

    bool ok() const { return code == StatusCode::Ok; }
    static Status ok_status() { return {}; }
    static Status error(StatusCode c, std::string msg) { return {c, std::move(msg)}; }
};

} // namespace motifcl
