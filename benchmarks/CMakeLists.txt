set(MOTIFCL_BENCHMARKS
    bench_matmul
    bench_softmax
    bench_attention
    bench_lora
)

foreach(bench ${MOTIFCL_BENCHMARKS})
    add_executable(${bench} ${bench}.cpp)
    target_link_libraries(${bench} PRIVATE motifcl)
    motifcl_apply_compiler_options(${bench})
endforeach()
