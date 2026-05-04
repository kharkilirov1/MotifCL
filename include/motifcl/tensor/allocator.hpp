#pragma once

#include <cstddef>
#include <vector>

#include <motifcl/runtime/buffer.hpp>

namespace motifcl {

class Backend;

class Allocator {
public:
    explicit Allocator(Backend& backend) : backend_(&backend) {}
    Buffer allocate(std::size_t nbytes);
    void release(Buffer&& buffer);
    std::size_t cached_blocks() const { return free_blocks_.size(); }

private:
    Backend* backend_ = nullptr;
    std::vector<Buffer> free_blocks_;
};

} // namespace motifcl
