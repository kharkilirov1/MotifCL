#include <motifcl/runtime/profiler.hpp>

namespace motifcl {

void Profiler::add(std::string name, double ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.push_back({std::move(name), ms});
}

std::vector<ProfileRecord> Profiler::records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_;
}

void Profiler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.clear();
}

} // namespace motifcl
