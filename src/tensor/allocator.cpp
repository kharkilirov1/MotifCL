#include <motifcl/tensor/allocator.hpp>

#include <motifcl/runtime/backend.hpp>

namespace motifcl {

Buffer Allocator::allocate(std::size_t nbytes) {
    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
        if (it->nbytes() >= nbytes) {
            Buffer out = std::move(*it);
            free_blocks_.erase(it);
            return out;
        }
    }
    return Buffer(backend_->ctx, nbytes);
}

void Allocator::release(Buffer&& buffer) {
    if (buffer.valid()) free_blocks_.push_back(std::move(buffer));
}

} // namespace motifcl
