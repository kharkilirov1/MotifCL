set(MOTIFCL_TOOLS
    dump_opencl_info
    compare_cpu_gpu
    kernel_tuner
    generate_transformer
    warm_decode_bench
)

foreach(tool ${MOTIFCL_TOOLS})
    add_executable(motifcl_${tool} ${tool}.cpp)
    target_link_libraries(motifcl_${tool} PRIVATE motifcl)
    motifcl_apply_compiler_options(motifcl_${tool})
    if(MOTIFCL_INSTALL)
        install(TARGETS motifcl_${tool} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endforeach()
