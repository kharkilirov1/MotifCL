set(MOTIFCL_TOOLS
    dump_opencl_info
    compare_cpu_gpu
    kernel_tuner
)

foreach(tool ${MOTIFCL_TOOLS})
    add_executable(motifcl_${tool} ${tool}.cpp)
    target_link_libraries(motifcl_${tool} PRIVATE motifcl)
    motifcl_apply_compiler_options(motifcl_${tool})
endforeach()
