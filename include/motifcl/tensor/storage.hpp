#pragma once

#include <cstddef>
#include <motifcl/runtime/buffer.hpp>

namespace motifcl {

class Backend;

class Storage {
public:
    Storage() = default;
    Storage(Backend& backend, std::size_t nbytes);
    ~Storage();

    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
    Storage(Storage&&) noexcept = default;
    Storage& operator=(Storage&&) noexcept = default;

    Buffer buffer;
    std::size_t nbytes = 0;
    bool pooled = false;
};

void clear_memory_pool();
std::size_t memory_pool_cached_blocks();
std::size_t memory_pool_cached_bytes();

} // namespace motifcl
