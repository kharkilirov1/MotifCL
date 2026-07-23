cmake_minimum_required(VERSION 3.20)
project(MotifCL VERSION 1.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MOTIFCL_BUILD_TESTS "Build tests" ON)
option(MOTIFCL_BUILD_PYTHON "Build Python bindings when pybind11 is available" ON)
option(MOTIFCL_BUILD_EXAMPLES "Build examples" ON)
option(MOTIFCL_BUILD_TOOLS "Build command-line tools" ON)
option(MOTIFCL_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(MOTIFCL_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(MOTIFCL_INSTALL "Enable install/export rules" ON)
option(MOTIFCL_ENABLE_OPENCL "Build OpenCL-backed MotifCL runtime/library" ON)
option(MOTIFCL_ENABLE_SANITIZERS "Enable ASan/UBSan on supported compilers" OFF)
option(MOTIFCL_ENABLE_FAST_MATH "Enable non-IEEE fast-math compiler flags for performance builds" OFF)
option(MOTIFCL_ENABLE_NATIVE_ARCH "Enable host-native CPU codegen flags for local optimized builds" OFF)
option(MOTIFCL_ENABLE_LTO "Enable interprocedural optimization/link-time optimization for release builds" OFF)
option(MOTIFCL_OPENCL_FAST_RELAXED_MATH "Build OpenCL kernels with fast relaxed math options" OFF)
option(MOTIFCL_ANDROID_MI9_TUNING "Enable Android/MI 9 arm64 host-side tuning flags" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerOptions)
include(GNUInstallDirs)

if(MOTIFCL_ENABLE_OPENCL)
find_package(OpenCL QUIET)
if(NOT OpenCL_FOUND)
    if(WIN32 AND MSVC)
        find_library(MOTIFCL_OPENCL_LIBRARY
            NAMES OpenCL OpenCL.lib
        )
        if(NOT MOTIFCL_OPENCL_LIBRARY)
            if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|AARCH64|aarch64")
                    set(_motifcl_opencl_machine ARM64)
                else()
                    set(_motifcl_opencl_machine x64)
                endif()
            else()
                set(_motifcl_opencl_machine x86)
            endif()
            set(_motifcl_opencl_implib "${CMAKE_BINARY_DIR}/OpenCL.lib")
            set(_motifcl_msvc_lib_tool "${CMAKE_AR}")
            if(NOT _motifcl_msvc_lib_tool OR _motifcl_msvc_lib_tool MATCHES "NOTFOUND")
                get_filename_component(_motifcl_msvc_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
                find_program(_motifcl_msvc_lib_tool NAMES lib.exe HINTS "${_motifcl_msvc_bin}")
            endif()
            if(NOT _motifcl_msvc_lib_tool OR _motifcl_msvc_lib_tool MATCHES "NOTFOUND")
                message(FATAL_ERROR "MSVC lib.exe not found; cannot generate OpenCL import library.")
            endif()
            execute_process(
                COMMAND "${_motifcl_msvc_lib_tool}" /nologo
                        "/def:${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenCL.def"
                        "/machine:${_motifcl_opencl_machine}"
                        "/out:${_motifcl_opencl_implib}"
                RESULT_VARIABLE _motifcl_opencl_implib_result
            )
            if(NOT _motifcl_opencl_implib_result EQUAL 0)
                message(FATAL_ERROR "Failed to generate MSVC OpenCL import library from cmake/OpenCL.def")
            endif()
            set(MOTIFCL_OPENCL_LIBRARY "${_motifcl_opencl_implib}" CACHE FILEPATH "OpenCL ICD loader import library" FORCE)
        endif()
    elseif(WIN32 AND NOT MOTIFCL_OPENCL_LIBRARY)
        foreach(_motifcl_opencl_candidate
                "$ENV{SystemRoot}/System32/OpenCL.dll"
                "$ENV{WINDIR}/System32/OpenCL.dll"
                "C:/Windows/System32/OpenCL.dll")
            if(EXISTS "${_motifcl_opencl_candidate}")
                set(MOTIFCL_OPENCL_LIBRARY "${_motifcl_opencl_candidate}" CACHE FILEPATH "OpenCL ICD loader library" FORCE)
                break()
            endif()
        endforeach()
    else()
        find_library(MOTIFCL_OPENCL_LIBRARY
            NAMES OpenCL libOpenCL.so.1 OpenCL.dll
            HINTS
                /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
                "$ENV{SystemRoot}/System32"
                "$ENV{WINDIR}/System32"
                "C:/Windows/System32"
        )
    endif()
    if(NOT MOTIFCL_OPENCL_LIBRARY)
        message(FATAL_ERROR "OpenCL library not found. Install an OpenCL ICD loader, e.g. ocl-icd-opencl-dev, or set MOTIFCL_OPENCL_LIBRARY.")
    endif()
    add_library(OpenCL::OpenCL UNKNOWN IMPORTED)
    set_target_properties(OpenCL::OpenCL PROPERTIES
        IMPORTED_LOCATION "${MOTIFCL_OPENCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencl"
    )
    message(STATUS "Using vendored minimal OpenCL headers and library: ${MOTIFCL_OPENCL_LIBRARY}")
endif()
endif()

set(MOTIFCL_SOURCES
    src/core/error.cpp
    src/core/logging.cpp
    src/core/shape.cpp
    src/core/types.cpp
    src/runtime/opencl_context.cpp
    src/runtime/event.cpp
    src/runtime/buffer.cpp
    src/runtime/program.cpp
    src/runtime/kernel.cpp
    src/runtime/command_buffer.cpp
    src/runtime/backend.cpp
    src/runtime/microkernel.cpp
    src/runtime/native_matmul.cpp
    src/runtime/profiler.cpp
    src/tensor/storage.cpp
    src/tensor/allocator.cpp
    src/tensor/tensor.cpp
    src/ops/basic_ops.cpp
    src/ops/matmul.cpp
    src/ops/reduce.cpp
    src/ops/indexing.cpp
    src/ops/fp16.cpp
    src/ops/norm.cpp
    src/ops/activation.cpp
    src/ops/fused_transformer.cpp
    src/ops/attention.cpp
    src/ops/loss.cpp
    src/ops/optim.cpp
    src/ops/quant.cpp
    src/autograd/graph.cpp
    src/autograd/tape.cpp
    src/autograd/backward.cpp
    src/nn/module.cpp
    src/nn/linear.cpp
    src/nn/compact_counter.cpp
    src/nn/group_counter_import.cpp
    src/nn/reversible.cpp
    src/nn/embedding.cpp
    src/nn/rmsnorm.cpp
    src/nn/attention.cpp
    src/nn/transformer.cpp
    src/nn/gemma.cpp
    src/nn/hf_compat.cpp
    src/motif/motif_linear.cpp
    src/motif/motif_lora.cpp
    src/motif/sarc_residual.cpp
    src/motif/router.cpp
    src/motif/motif_block.cpp
    src/train/adam.cpp
    src/train/mixed_precision.cpp
    src/train/static_train_step.cpp
    src/train/sgd.cpp
    src/train/trainer.cpp
    src/train/training_utils.cpp
    src/serialization.cpp
    src/gguf.cpp
)

add_library(motifcl_vulkan_runtime src/runtime/vulkan_backend.cpp)
add_library(MotifCL::vulkan_runtime ALIAS motifcl_vulkan_runtime)
set_target_properties(motifcl_vulkan_runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(motifcl_vulkan_runtime
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(motifcl_vulkan_runtime PRIVATE ${CMAKE_DL_LIBS})
motifcl_apply_compiler_options(motifcl_vulkan_runtime)

if(NOT MOTIFCL_ENABLE_OPENCL)
    # OpenCL-off build: the full library still compiles against the vendored
    # CL type declarations, but links a local no-op stub instead of any
    # OpenCL loader. Backend::create_opencl() then fails at runtime with the
    # normal "no OpenCL platforms found" error; the Vulkan backend is the
    # only live GPU path.
    list(APPEND MOTIFCL_SOURCES src/runtime/opencl_disabled_stub.cpp)
endif()

add_library(motifcl ${MOTIFCL_SOURCES})
add_library(MotifCL::motifcl ALIAS motifcl)
set_target_properties(motifcl PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(motifcl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(motifcl
    PUBLIC CL_TARGET_OPENCL_VERSION=120
    PRIVATE
        MOTIFCL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        MOTIFCL_BINARY_KERNEL_DIR="${CMAKE_BINARY_DIR}/kernels"
        MOTIFCL_INSTALL_KERNEL_DIR="${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_DATADIR}/motifcl/kernels"
        $<$<BOOL:${MOTIFCL_OPENCL_FAST_RELAXED_MATH}>:MOTIFCL_OPENCL_FAST_RELAXED_MATH=1>
)

if(MOTIFCL_ENABLE_OPENCL)
    target_link_libraries(motifcl
        PUBLIC OpenCL::OpenCL
        PRIVATE motifcl_vulkan_runtime ${CMAKE_DL_LIBS}
    )
else()
    target_include_directories(motifcl PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencl>)
    target_link_libraries(motifcl PRIVATE motifcl_vulkan_runtime ${CMAKE_DL_LIBS})
endif()
motifcl_apply_compiler_options(motifcl)

if(MOTIFCL_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _motifcl_ipo_supported OUTPUT _motifcl_ipo_error)
    if(_motifcl_ipo_supported)
        set_property(TARGET motifcl PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set_property(TARGET motifcl PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    else()
        message(WARNING "MOTIFCL_ENABLE_LTO requested but IPO/LTO is not supported: ${_motifcl_ipo_error}")
    endif()
endif()

add_custom_target(motifcl_copy_kernels ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        "${CMAKE_BINARY_DIR}/kernels"
    COMMENT "Copying MotifCL OpenCL kernels to the build tree"
)
add_dependencies(motifcl motifcl_copy_kernels)

if(NOT MOTIFCL_ENABLE_OPENCL)
    message(STATUS "MOTIFCL_ENABLE_OPENCL=OFF: motifcl builds against the local OpenCL stub; "
                   "Backend::create_opencl() fails at runtime and Vulkan is the only GPU backend.")
endif()

if(MOTIFCL_BUILD_EXAMPLES)
    if(TARGET motifcl)
        add_subdirectory(examples/cpp)
    else()
        message(STATUS "Skipping examples because motifcl core target is disabled.")
    endif()
endif()

if(MOTIFCL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MOTIFCL_BUILD_TOOLS)
    if(TARGET motifcl)
        add_subdirectory(tools)
    else()
        message(STATUS "Skipping tools because motifcl core target is disabled.")
    endif()
endif()

if(MOTIFCL_BUILD_BENCHMARKS)
    if(TARGET motifcl)
        add_subdirectory(benchmarks)
    else()
        message(STATUS "Skipping benchmarks because motifcl core target is disabled.")
    endif()
endif()

if(MOTIFCL_BUILD_PYTHON AND TARGET motifcl)
    set(PYBIND11_FINDPYTHON ON)
    if(WIN32 AND NOT DEFINED Python_FIND_REGISTRY)
        set(Python_FIND_REGISTRY NEVER)
    endif()
    if(NOT DEFINED Python_FIND_STRATEGY)
        set(Python_FIND_STRATEGY LOCATION)
    endif()
    if(WIN32 AND Python_EXECUTABLE MATCHES "WindowsApps")
        unset(Python_EXECUTABLE CACHE)
        unset(MOTIFCL_PYTHON_EXECUTABLE CACHE)
    endif()
    if(NOT Python_EXECUTABLE)
        find_program(MOTIFCL_PYTHON_EXECUTABLE NAMES python python3)
        if(MOTIFCL_PYTHON_EXECUTABLE)
            set(Python_EXECUTABLE "${MOTIFCL_PYTHON_EXECUTABLE}" CACHE FILEPATH "Python executable used for MotifCL bindings")
        endif()
    endif()
    find_package(Python COMPONENTS Interpreter Development.Module QUIET)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND AND Python_EXECUTABLE)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            RESULT_VARIABLE _motifcl_pybind11_cmakedir_result
            OUTPUT_VARIABLE _motifcl_pybind11_cmakedir
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_motifcl_pybind11_cmakedir_result EQUAL 0 AND EXISTS "${_motifcl_pybind11_cmakedir}/pybind11Config.cmake")
            find_package(pybind11 CONFIG QUIET PATHS "${_motifcl_pybind11_cmakedir}" NO_DEFAULT_PATH)
        endif()
    endif()
    if(pybind11_FOUND AND Python_FOUND)
        add_subdirectory(python)
    else()
        message(WARNING "pybind11 or Python development module not found; Python bindings are present but will not be built. Install pybind11/Python dev files or set -DMOTIFCL_BUILD_PYTHON=OFF.")
    endif()
elseif(MOTIFCL_BUILD_PYTHON)
    message(STATUS "Skipping Python bindings because MOTIFCL_ENABLE_OPENCL=OFF disables the current motifcl core target.")
endif()

if(MOTIFCL_INSTALL)
    include(CMakePackageConfigHelpers)
    if(TARGET motifcl)
        set(_motifcl_install_targets motifcl motifcl_vulkan_runtime)
    else()
        set(_motifcl_install_targets motifcl_vulkan_runtime)
    endif()
    install(TARGETS ${_motifcl_install_targets}
        EXPORT MotifCLTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    if(MOTIFCL_ENABLE_OPENCL)
        install(DIRECTORY third_party/opencl/CL DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        install(DIRECTORY kernels DESTINATION ${CMAKE_INSTALL_DATADIR}/motifcl)
    endif()
    install(EXPORT MotifCLTargets
        NAMESPACE MotifCL::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MotifCL
    )
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MotifCLConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/MotifCLConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MotifCL
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/MotifCLConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/MotifCLConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/MotifCLConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MotifCL
    )
endif()
