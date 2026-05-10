#ifndef MOTIFCL_MINIMAL_CL_H
#define MOTIFCL_MINIMAL_CL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint16_t cl_ushort;
typedef uint8_t cl_uchar;
typedef cl_uint cl_bool;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef intptr_t cl_context_properties;
typedef intptr_t cl_properties;
typedef uint32_t cl_platform_info;
typedef uint32_t cl_device_info;
typedef uint32_t cl_program_build_info;
typedef uint32_t cl_event_info;
typedef uint32_t cl_profiling_info;
typedef uint32_t cl_buffer_create_type;

typedef struct _cl_platform_id* cl_platform_id;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_event* cl_event;

#define CL_SUCCESS                                  0
#define CL_DEVICE_NOT_FOUND                       -1
#define CL_BUILD_PROGRAM_FAILURE                  -11
#define CL_INVALID_VALUE                          -30
#define CL_INVALID_DEVICE                         -33
#define CL_INVALID_CONTEXT                        -34
#define CL_INVALID_MEM_OBJECT                     -38
#define CL_INVALID_BINARY                         -42
#define CL_INVALID_BUILD_OPTIONS                  -43
#define CL_INVALID_PROGRAM                        -44
#define CL_INVALID_PROGRAM_EXECUTABLE             -45
#define CL_INVALID_KERNEL_NAME                    -46
#define CL_INVALID_KERNEL_DEFINITION              -47
#define CL_INVALID_KERNEL                         -48
#define CL_INVALID_ARG_INDEX                      -49
#define CL_INVALID_ARG_VALUE                      -50
#define CL_INVALID_ARG_SIZE                       -51
#define CL_INVALID_KERNEL_ARGS                    -52
#define CL_INVALID_WORK_DIMENSION                 -53
#define CL_INVALID_WORK_GROUP_SIZE                -54
#define CL_INVALID_WORK_ITEM_SIZE                 -55
#define CL_INVALID_GLOBAL_OFFSET                  -56
#define CL_INVALID_EVENT_WAIT_LIST                -57
#define CL_INVALID_OPERATION                      -59
#define CL_OUT_OF_RESOURCES                       -5
#define CL_OUT_OF_HOST_MEMORY                     -6
#define CL_PLATFORM_NOT_FOUND_KHR                 -1001

#define CL_FALSE                                   0
#define CL_TRUE                                    1

#define CL_DEVICE_TYPE_DEFAULT                     (1ULL << 0)
#define CL_DEVICE_TYPE_CPU                         (1ULL << 1)
#define CL_DEVICE_TYPE_GPU                         (1ULL << 2)
#define CL_DEVICE_TYPE_ACCELERATOR                 (1ULL << 3)
#define CL_DEVICE_TYPE_ALL                         0xFFFFFFFFULL

#define CL_MEM_READ_WRITE                          (1ULL << 0)
#define CL_MEM_WRITE_ONLY                          (1ULL << 1)
#define CL_MEM_READ_ONLY                           (1ULL << 2)
#define CL_MEM_USE_HOST_PTR                        (1ULL << 3)
#define CL_MEM_ALLOC_HOST_PTR                      (1ULL << 4)
#define CL_MEM_COPY_HOST_PTR                       (1ULL << 5)

#define CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE     (1ULL << 0)
#define CL_QUEUE_PROFILING_ENABLE                  (1ULL << 1)

#define CL_PLATFORM_PROFILE                        0x0900
#define CL_PLATFORM_VERSION                        0x0901
#define CL_PLATFORM_NAME                           0x0902
#define CL_PLATFORM_VENDOR                         0x0903
#define CL_PLATFORM_EXTENSIONS                     0x0904

#define CL_DEVICE_TYPE                             0x1000
#define CL_DEVICE_VENDOR_ID                        0x1001
#define CL_DEVICE_MAX_COMPUTE_UNITS                0x1002
#define CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS         0x1003
#define CL_DEVICE_MAX_WORK_GROUP_SIZE              0x1004
#define CL_DEVICE_MAX_CLOCK_FREQUENCY              0x100C
#define CL_DEVICE_ADDRESS_BITS                     0x100D
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE               0x1010
#define CL_DEVICE_GLOBAL_MEM_SIZE                  0x101F
#define CL_DEVICE_LOCAL_MEM_SIZE                   0x1023
#define CL_DEVICE_NAME                             0x102B
#define CL_DEVICE_VENDOR                           0x102C
#define CL_DRIVER_VERSION                          0x102D
#define CL_DEVICE_PROFILE                          0x102E
#define CL_DEVICE_VERSION                          0x102F
#define CL_DEVICE_EXTENSIONS                       0x1030
#define CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT     0x100B

#define CL_CONTEXT_PLATFORM                        0x1084

#define CL_PROGRAM_BUILD_STATUS                    0x1181
#define CL_PROGRAM_BUILD_OPTIONS                   0x1182
#define CL_PROGRAM_BUILD_LOG                       0x1183

#define CL_PROFILING_COMMAND_QUEUED                0x1280
#define CL_PROFILING_COMMAND_SUBMIT                0x1281
#define CL_PROFILING_COMMAND_START                 0x1282
#define CL_PROFILING_COMMAND_END                   0x1283

#define CL_BUFFER_CREATE_TYPE_REGION               0x1220

typedef struct _cl_buffer_region {
    size_t origin;
    size_t size;
} cl_buffer_region;

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms);
cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices);
cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
cl_context clCreateContext(const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices, void (*pfn_notify)(const char*, const void*, size_t, void*), void* user_data, cl_int* errcode_ret);
cl_int clReleaseContext(cl_context context);
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret);
cl_int clReleaseCommandQueue(cl_command_queue command_queue);
cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret);
cl_mem clCreateSubBuffer(cl_mem buffer, cl_mem_flags flags, cl_buffer_create_type buffer_create_type, const void* buffer_create_info, cl_int* errcode_ret);
cl_int clRetainMemObject(cl_mem memobj);
cl_int clReleaseMemObject(cl_mem memobj);
cl_int clEnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write, size_t offset, size_t cb, const void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t cb, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char** strings, const size_t* lengths, cl_int* errcode_ret);
cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void (*pfn_notify)(cl_program, void*), void* user_data);
cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
cl_int clReleaseProgram(cl_program program);
cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret);
cl_int clRetainKernel(cl_kernel kernel);
cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value);
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event);
cl_int clReleaseKernel(cl_kernel kernel);
cl_int clFinish(cl_command_queue command_queue);
cl_int clWaitForEvents(cl_uint num_events, const cl_event* event_list);
cl_int clReleaseEvent(cl_event event);
cl_int clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret);
void* clGetExtensionFunctionAddress(const char* func_name);
void* clGetExtensionFunctionAddressForPlatform(cl_platform_id platform, const char* func_name);

#ifdef __cplusplus
}
#endif

#endif
