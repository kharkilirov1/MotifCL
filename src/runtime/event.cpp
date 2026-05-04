#include <motifcl/runtime/event.hpp>

#include <motifcl/core/error.hpp>

namespace motifcl {

Event::Event(cl_event event) : event_(event) {}

Event::~Event() {
    if (event_) clReleaseEvent(event_);
}

Event::Event(Event&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        if (event_) clReleaseEvent(event_);
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void Event::wait() const {
    if (event_) MCL_CHECK_CL(clWaitForEvents(1, &event_));
}

double Event::elapsed_ms() const {
    MCL_CHECK(event_ != nullptr, "cannot query profiling on null event");
    wait();
    cl_ulong start = 0, end = 0;
    MCL_CHECK_CL(clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr));
    MCL_CHECK_CL(clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
    return static_cast<double>(end - start) * 1e-6;
}

} // namespace motifcl
