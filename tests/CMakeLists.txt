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
    test_gguf
)

foreach(test_name ${MOTIFCL_TESTS})
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE motifcl)
    motifcl_apply_compiler_options(${test_name})
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()
