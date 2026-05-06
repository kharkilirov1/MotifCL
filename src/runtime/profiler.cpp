#include <motifcl/runtime/profiler.hpp>

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace motifcl {

void Profiler::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

bool Profiler::enabled() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return enabled_;
}

void Profiler::add(std::string name, double ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_) return;
    records_.push_back({std::move(name), ms});
}

std::vector<ProfileRecord> Profiler::records() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_;
}

std::vector<ProfileSummary> Profiler::summary() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<std::string, ProfileSummary> grouped;
    for (const auto& record : records_) {
        auto& item = grouped[record.name];
        item.name = record.name;
        item.count += 1;
        item.total_ms += record.ms;
        item.max_ms = std::max(item.max_ms, record.ms);
    }
    std::vector<ProfileSummary> out;
    out.reserve(grouped.size());
    for (auto& kv : grouped) {
        if (kv.second.count > 0) kv.second.avg_ms = kv.second.total_ms / static_cast<double>(kv.second.count);
        out.push_back(std::move(kv.second));
    }
    std::sort(out.begin(), out.end(), [](const ProfileSummary& a, const ProfileSummary& b) {
        if (a.total_ms != b.total_ms) return a.total_ms > b.total_ms;
        return a.name < b.name;
    });
    return out;
}

void Profiler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    records_.clear();
}

} // namespace motifcl
