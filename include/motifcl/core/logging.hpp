#pragma once

#include <string>

namespace motifcl {

enum class LogLevel {
    Debug = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Off = 4
};

class Logger {
public:
    static void set_level(LogLevel level);
    static LogLevel level();
    static void log(LogLevel level, const std::string& message);
    static void debug(const std::string& message) { log(LogLevel::Debug, message); }
    static void info(const std::string& message) { log(LogLevel::Info, message); }
    static void warn(const std::string& message) { log(LogLevel::Warning, message); }
    static void error(const std::string& message) { log(LogLevel::Error, message); }
};

} // namespace motifcl

#define MCL_LOG_INFO(msg) ::motifcl::Logger::info(msg)
#define MCL_LOG_WARN(msg) ::motifcl::Logger::warn(msg)
#define MCL_LOG_ERROR(msg) ::motifcl::Logger::error(msg)
