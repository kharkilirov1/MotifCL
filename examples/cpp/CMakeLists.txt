set(MOTIFCL_EXAMPLES
    01_device_info
    02_matmul
    03_mlp_train
    04_bigram_train
    05_tiny_transformer
    06_motif_lora
    07_long_train_checkpoint
    08_shakespeare_10m_train
)

foreach(example ${MOTIFCL_EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE motifcl)
    motifcl_apply_compiler_options(${example})
endforeach()
