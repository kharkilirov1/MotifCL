#include <motifcl/tensor/storage.hpp>

#include <motifcl/runtime/backend.hpp>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace motifcl {

namespace {

struct PooledBlock {
    Buffer buffer;
};

std::mutex g_pool_mutex;
std::vector<PooledBlock> g_pool;

bool memory_pool_enabled() {
    const char* disabled = std::getenv("MOTIFCL_DISABLE_MEMORY_POOL");
    return !(disabled && *disabled && std::string(disabled) != "0");
}

std::size_t max_cached_blocks() {
    if (const char* env = std::getenv("MOTIFCL_MEMORY_POOL_MAX_BLOCKS")) {
        const auto value = std::strtoull(env, nullptr, 10);
        if (value > 0) return static_cast<std::size_t>(value);
    }
    return 512;
}

Buffer acquire_buffer(Backend& backend, std::size_t bytes, bool& pooled) {
    pooled = memory_pool_enabled();
    if (!pooled) return Buffer(backend.ctx, bytes, CL_MEM_READ_WRITE, &backend.profiler, backend.lifetime_handle());
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    auto best = g_pool.end();
    for (auto it = g_pool.begin(); it != g_pool.end(); ++it) {
        if (it->buffer.valid() &&
            it->buffer.same_context(backend.ctx) &&
            it->buffer.nbytes() >= bytes &&
            (best == g_pool.end() || it->buffer.nbytes() < best->buffer.nbytes())) {
            best = it;
        }
    }
    if (best != g_pool.end()) {
        Buffer out = std::move(best->buffer);
        out.set_profiler(&backend.profiler, backend.lifetime_handle());
        g_pool.erase(best);
        return out;
    }
    return Buffer(backend.ctx, bytes, CL_MEM_READ_WRITE, &backend.profiler, backend.lifetime_handle());
}

void release_buffer(Buffer&& buffer) {
    if (!buffer.valid() || !memory_pool_enabled()) return;
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (g_pool.size() >= max_cached_blocks()) return;
    g_pool.push_back(PooledBlock{std::move(buffer)});
}

} // namespace

Storage::Storage(Backend& backend, std::size_t bytes) : buffer(acquire_buffer(backend, bytes, pooled)), nbytes(bytes) {}

Storage::~Storage() {
    if (pooled && buffer.valid()) release_buffer(std::move(buffer));
}

void clear_memory_pool() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_pool.clear();
}

std::size_t memory_pool_cached_blocks() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    return g_pool.size();
}

std::size_t memory_pool_cached_bytes() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    std::size_t total = 0;
    for (const auto& block : g_pool) total += block.buffer.nbytes();
    return total;
}

} // namespace motifcl
