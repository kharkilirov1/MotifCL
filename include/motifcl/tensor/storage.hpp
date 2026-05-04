#pragma once

#include <cstddef>
#include <motifcl/runtime/buffer.hpp>

namespace motifcl {

class Backend;

class Storage {
public:
    Storage() = default;
    Storage(Backend& backend, std::size_t nbytes);

    Buffer buffer;
    std::size_t nbytes = 0;
};

} // namespace motifcl
