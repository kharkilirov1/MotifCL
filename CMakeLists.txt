cmake_minimum_required(VERSION 3.20)
project(MotifCL VERSION 1.0.0 LANGUAGES CXX)

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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerOptions)
include(GNUInstallDirs)

find_package(OpenCL QUIET)
if(NOT OpenCL_FOUND)
    if(WIN32 AND NOT MOTIFCL_OPENCL_LIBRARY)
        foreach(_motifcl_opencl_candidate
                "$ENV{SystemRoot}/System32/OpenCL.dll"
                "$ENV{WINDIR}/System32/OpenCL.dll"
                "C:/Windows/System32/OpenCL.dll")
            if(EXISTS "${_motifcl_opencl_candidate}")
                set(MOTIFCL_OPENCL_LIBRARY "${_motifcl_opencl_candidate}" CACHE FILEPATH "OpenCL ICD loader library" FORCE)
                break()
            endif()
        endforeach()
    endif()
    find_library(MOTIFCL_OPENCL_LIBRARY
        NAMES OpenCL libOpenCL.so.1 OpenCL.dll
        HINTS
            /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
            "$ENV{SystemRoot}/System32"
            "$ENV{WINDIR}/System32"
            "C:/Windows/System32"
    )
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
    src/runtime/backend.cpp
    src/runtime/profiler.cpp
    src/tensor/storage.cpp
    src/tensor/allocator.cpp
    src/tensor/tensor.cpp
    src/ops/basic_ops.cpp
    src/ops/matmul.cpp
    src/ops/reduce.cpp
    src/ops/norm.cpp
    src/ops/activation.cpp
    src/ops/attention.cpp
    src/ops/loss.cpp
    src/ops/optim.cpp
    src/ops/quant.cpp
    src/autograd/graph.cpp
    src/autograd/tape.cpp
    src/autograd/backward.cpp
    src/nn/module.cpp
    src/nn/linear.cpp
    src/nn/embedding.cpp
    src/nn/rmsnorm.cpp
    src/nn/attention.cpp
    src/nn/transformer.cpp
    src/motif/motif_linear.cpp
    src/motif/motif_lora.cpp
    src/motif/sarc_residual.cpp
    src/motif/router.cpp
    src/motif/motif_block.cpp
    src/train/adam.cpp
    src/train/sgd.cpp
    src/train/trainer.cpp
)

add_library(motifcl ${MOTIFCL_SOURCES})
add_library(MotifCL::motifcl ALIAS motifcl)

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
)

target_link_libraries(motifcl PUBLIC OpenCL::OpenCL)
motifcl_apply_compiler_options(motifcl)

add_custom_target(motifcl_copy_kernels ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        "${CMAKE_BINARY_DIR}/kernels"
    COMMENT "Copying MotifCL OpenCL kernels to the build tree"
)
add_dependencies(motifcl motifcl_copy_kernels)

if(MOTIFCL_BUILD_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()

if(MOTIFCL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(MOTIFCL_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if(MOTIFCL_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(MOTIFCL_BUILD_PYTHON)
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
endif()

if(MOTIFCL_INSTALL)
    include(CMakePackageConfigHelpers)
    install(TARGETS motifcl
        EXPORT MotifCLTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY third_party/opencl/CL DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY kernels DESTINATION ${CMAKE_INSTALL_DATADIR}/motifcl)
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
