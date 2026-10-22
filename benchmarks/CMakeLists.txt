set(MOTIFCL_BENCHMARKS
    bench_matmul
    bench_native_matmul
    bench_matmul_gpt_shapes
    bench_modern_transformer
    bench_kquant_m1
    bench_softmax
    bench_attention
    bench_lora
    bench_vulkan_perf
    bench_vulkan_gemm_shapes
    bench_vulkan_counter_gemm
)

foreach(bench ${MOTIFCL_BENCHMARKS})
    add_executable(${bench} ${bench}.cpp)
    target_link_libraries(${bench} PRIVATE motifcl)
    motifcl_apply_compiler_options(${bench})
endforeach()
