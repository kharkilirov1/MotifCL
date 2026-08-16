set(MOTIFCL_EXAMPLES
    01_device_info
    02_matmul
    03_mlp_train
    04_bigram_train
    05_tiny_transformer
    06_motif_lora
    07_long_train_checkpoint
    08_shakespeare_10m_train
    09_fog_v3_rx580_pretrain
    10_fog_v3_operator_gate
    11_fog_v3_structured_machine_gate
    12_fog_v3_reasoning_sft
    13_fog_v3_inference
    14_fog_v3_logic_demo
    15_fog_v3_e2e_prompt
    16_fog_v3_ablation_test
    17_fog_v3_ghost_room_test
    18_fog_v3_fairy_tale_logic
    19_fog_v3_advanced_logic_suite
    20_fog_v3_ultimate_benchmark
)

foreach(example ${MOTIFCL_EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE motifcl)
    motifcl_apply_compiler_options(${example})
endforeach()
