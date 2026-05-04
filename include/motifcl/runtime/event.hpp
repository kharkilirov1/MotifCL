#pragma once

#include <CL/cl.h>

namespace motifcl {

class Event {
public:
    Event() = default;
    explicit Event(cl_event event);
    ~Event();

    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;

    bool valid() const { return event_ != nullptr; }
    cl_event raw() const { return event_; }
    void wait() const;
    double elapsed_ms() const;

private:
    cl_event event_ = nullptr;
};

} // namespace motifcl
