#include <motifcl/core/logging.hpp>

#include <iostream>
#include <mutex>

namespace motifcl {
namespace {
std::mutex& log_mutex() {
    static std::mutex m;
    return m;
}
LogLevel& global_level() {
    static LogLevel level = LogLevel::Info;
    return level;
}
const char* prefix(LogLevel level) {
    switch (level) {
        case LogLevel::Debug: return "[MotifCL:debug] ";
        case LogLevel::Info: return "[MotifCL] ";
        case LogLevel::Warning: return "[MotifCL:warn] ";
        case LogLevel::Error: return "[MotifCL:error] ";
        case LogLevel::Off: return "";
    }
    return "[MotifCL] ";
}
} // namespace

void Logger::set_level(LogLevel level) { global_level() = level; }
LogLevel Logger::level() { return global_level(); }

void Logger::log(LogLevel level, const std::string& message) {
    if (level < global_level() || global_level() == LogLevel::Off) return;
    std::lock_guard<std::mutex> lock(log_mutex());
    std::cerr << prefix(level) << message << '\n';
}

} // namespace motifcl
