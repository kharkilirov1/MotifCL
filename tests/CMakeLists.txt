set(MOTIFCL_TESTS
    test_shape
    test_buffer
    test_backend_lifetime
    test_tensor
    test_quant
    test_matmul
    test_softmax
    test_attention
    test_rmsnorm
    test_layernorm
    test_graph_capture
    test_backward
    test_adam
    test_transformer
    test_fused_transformer
    test_embedding
    test_cross_entropy
    test_gpt
    test_production_stack
    test_modern_transformer
    test_gemma_compat
    test_hf_modern_compat
    test_hf_dense_families
    test_gguf
    test_microkernel
    test_native_matmul
    test_counter_state
    test_group_counter_import
    test_ternary_solver
    test_memory_truth_gate
    test_f16_matmul_autograd
    test_reversible_attn
    test_vulkan_backend
    test_vulkan_counter_fused
    test_vulkan_gqa_p0
    test_vulkan_gqa_training
    test_vulkan_kv_decode_p0
    test_vulkan_q4_fused_p0
    test_vulkan_row_bias_p0
    test_artifact_fuzz
    test_split_qkv_train_repro
    test_vulkan_train_step
    test_vulkan_reversible
    test_vulkan_memory_native
)

# MinGW test executables need the toolchain runtime DLLs (libstdc++-6.dll,
# libgcc_s_seh-1.dll, libwinpthread-1.dll) resolved from the *same* toolchain
# that built them. When ctest runs from an MSYS/Git-Bash shell, /mingw64/bin
# shadows them with incompatible copies -> 0xc0000139 at launch. Prepend the
# compiler's own bin dir for every test process.
set(_motifcl_test_env "")
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    get_filename_component(_motifcl_mingw_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    file(TO_NATIVE_PATH "${_motifcl_mingw_bin}" _motifcl_mingw_bin_native)
    set(_motifcl_test_env "PATH=path_list_prepend:${_motifcl_mingw_bin_native}")
endif()

function(motifcl_register_test test_name)
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
    if(_motifcl_test_env)
        set_tests_properties(${test_name} PROPERTIES
            ENVIRONMENT_MODIFICATION "${_motifcl_test_env}")
    endif()
endfunction()

if(TARGET motifcl)
    foreach(test_name ${MOTIFCL_TESTS})
        add_executable(${test_name} ${test_name}.cpp)
        target_link_libraries(${test_name} PRIVATE motifcl)
        motifcl_apply_compiler_options(${test_name})
        motifcl_register_test(${test_name})
        if(test_name STREQUAL "test_group_counter_import" OR
           test_name STREQUAL "test_ternary_solver")
            set_tests_properties(${test_name} PROPERTIES
                WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
        endif()
    endforeach()
endif()

if(TARGET motifcl_vulkan_runtime)
    add_executable(test_vulkan_runtime_standalone test_vulkan_runtime_standalone.cpp)
    target_link_libraries(test_vulkan_runtime_standalone PRIVATE motifcl_vulkan_runtime)
    motifcl_apply_compiler_options(test_vulkan_runtime_standalone)
    motifcl_register_test(test_vulkan_runtime_standalone)
endif()

# Throughput bench (not a ctest): stories15M-shaped counter train step.
if(TARGET motifcl)
    add_executable(bench_ternary15m bench_ternary15m.cpp)
    target_link_libraries(bench_ternary15m PRIVATE motifcl)
    motifcl_apply_compiler_options(bench_ternary15m)
endif()
