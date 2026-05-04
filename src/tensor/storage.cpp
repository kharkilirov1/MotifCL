#include <motifcl/tensor/storage.hpp>

#include <motifcl/runtime/backend.hpp>

namespace motifcl {

Storage::Storage(Backend& backend, std::size_t bytes) : buffer(backend.ctx, bytes), nbytes(bytes) {}

} // namespace motifcl
