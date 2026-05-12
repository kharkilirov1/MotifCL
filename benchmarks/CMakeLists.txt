set(MOTIFCL_BENCHMARKS
    bench_matmul
    bench_matmul_gpt_shapes
    bench_modern_transformer
    bench_kquant_m1
    bench_softmax
    bench_attention
    bench_lora
)

foreach(bench ${MOTIFCL_BENCHMARKS})
    add_executable(${bench} ${bench}.cpp)
    target_link_libraries(${bench} PRIVATE motifcl)
    motifcl_apply_compiler_options(${bench})
endforeach()
