#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace motifcl {

struct ProfileRecord {
    std::string name;
    double ms = 0.0;
};

struct ProfileSummary {
    std::string name;
    std::size_t count = 0;
    double total_ms = 0.0;
    double avg_ms = 0.0;
    double max_ms = 0.0;
};

class Profiler {
public:
    void set_enabled(bool enabled);
    bool enabled() const;
    void add(std::string name, double ms);
    std::vector<ProfileRecord> records() const;
    std::vector<ProfileSummary> summary() const;
    void clear();

private:
    mutable std::mutex mutex_;
    std::vector<ProfileRecord> records_;
    bool enabled_ = false;
};

} // namespace motifcl
