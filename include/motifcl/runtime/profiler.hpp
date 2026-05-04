#pragma once

#include <mutex>
#include <string>
#include <vector>

namespace motifcl {

struct ProfileRecord {
    std::string name;
    double ms = 0.0;
};

class Profiler {
public:
    void add(std::string name, double ms);
    std::vector<ProfileRecord> records() const;
    void clear();

private:
    mutable std::mutex mutex_;
    std::vector<ProfileRecord> records_;
};

} // namespace motifcl
