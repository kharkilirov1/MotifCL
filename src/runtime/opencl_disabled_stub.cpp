// Build-time replacement for the OpenCL ICD loader when
// -DMOTIFCL_ENABLE_OPENCL=OFF.
//
// The full MotifCL library (Tensor, ops, autograd, nn) still compiles against
// the vendored CL type declarations, but nothing links against a real OpenCL
// loader: every entry point below reports "no platforms" / invalid operation,
// so Backend::create_opencl() fails with the normal "no OpenCL platforms
// found" error and every OpenCL-gated code path stays cold. The Vulkan
// backend is unaffected. This file must match the prototypes in
// third_party/opencl/CL/cl.h.
#include <CL/cl.h>

extern "C" {

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
    (void)num_entries;
    (void)platforms;
    if (num_platforms) *num_platforms = 0;
    return CL_SUCCESS;
}

cl_int clGetPlatformInfo(cl_platform_id, cl_platform_info, size_t, void*, size_t* size_ret) {
    if (size_ret) *size_ret = 0;
    return CL_INVALID_OPERATION;
}

cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id* devices, cl_uint* num_devices) {
    (void)devices;
    if (num_devices) *num_devices = 0;
    return CL_DEVICE_NOT_FOUND;
}

cl_int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void*, size_t* size_ret) {
    if (size_ret) *size_ret = 0;
    return CL_INVALID_OPERATION;
}

cl_context clCreateContext(const cl_context_properties*, cl_uint, const cl_device_id*,
                           void (*)(const char*, const void*, size_t, void*), void*, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

cl_int clReleaseContext(cl_context) { return CL_INVALID_OPERATION; }

cl_command_queue clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties,
                                      cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

cl_int clReleaseCommandQueue(cl_command_queue) { return CL_INVALID_OPERATION; }

cl_mem clCreateBuffer(cl_context, cl_mem_flags, size_t, void*, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

cl_mem clCreateSubBuffer(cl_mem, cl_mem_flags, cl_buffer_create_type, const void*, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

cl_int clRetainMemObject(cl_mem) { return CL_INVALID_OPERATION; }
cl_int clReleaseMemObject(cl_mem) { return CL_INVALID_OPERATION; }

cl_int clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint,
                            const cl_event*, cl_event*) {
    return CL_INVALID_OPERATION;
}

cl_int clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint,
                           const cl_event*, cl_event*) {
    return CL_INVALID_OPERATION;
}

cl_program clCreateProgramWithSource(cl_context, cl_uint, const char**, const size_t*, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

cl_int clBuildProgram(cl_program, cl_uint, const cl_device_id*, const char*, void (*)(cl_program, void*),
                      void*) {
    return CL_INVALID_OPERATION;
}

cl_int clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void*,
                             size_t* size_ret) {
    if (size_ret) *size_ret = 0;
    return CL_INVALID_OPERATION;
}

cl_int clReleaseProgram(cl_program) { return CL_INVALID_OPERATION; }

cl_kernel clCreateKernel(cl_program, const char*, cl_int* errcode_ret) {
    if (errcode_ret) *errcode_ret = CL_INVALID_OPERATION;
    return nullptr;
}

cl_int clRetainKernel(cl_kernel) { return CL_INVALID_OPERATION; }
cl_int clSetKernelArg(cl_kernel, cl_uint, size_t, const void*) { return CL_INVALID_OPERATION; }

cl_int clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*,
                              const size_t*, cl_uint, const cl_event*, cl_event*) {
    return CL_INVALID_OPERATION;
}

cl_int clReleaseKernel(cl_kernel) { return CL_INVALID_OPERATION; }
cl_int clFinish(cl_command_queue) { return CL_INVALID_OPERATION; }
cl_int clWaitForEvents(cl_uint, const cl_event*) { return CL_INVALID_OPERATION; }
cl_int clReleaseEvent(cl_event) { return CL_INVALID_OPERATION; }

cl_int clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void*, size_t* size_ret) {
    if (size_ret) *size_ret = 0;
    return CL_INVALID_OPERATION;
}

void* clGetExtensionFunctionAddress(const char*) { return nullptr; }
void* clGetExtensionFunctionAddressForPlatform(cl_platform_id, const char*) { return nullptr; }

} // extern "C"
