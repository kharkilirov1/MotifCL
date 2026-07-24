# Project role and evidence boundary

## Position in the project family

MotifCL is a Vulkan-first native neural runtime for legacy AMD GPUs. It is maintained as a
supporting systems project for
[`memory-native`](https://github.com/kharkilirov1/memory-native), which is the canonical research
implementation of the finite-state optimizer-in-weight method.

The repositories answer different questions:

| Repository | Primary question |
|---|---|
| `memory-native` | Does the training method learn, scale, and reproduce in a familiar PyTorch/MLX environment? |
| MotifCL | Can the relevant operators and compact state execute efficiently in a native Vulkan runtime on constrained hardware? |

## Evidence policy

Claims must identify the repository, commit, backend, device, command, and witness that produced
them.

- A PyTorch/T4 result is not a MotifCL Vulkan result.
- A MotifCL kernel parity test is not convergence evidence for the training method.
- A skip-capable GPU test is not backend evidence unless strict mode proves that the intended
  device path executed.
- Historical reports are not substitutes for a fresh release gate.
- Modeled memory and throughput are labeled separately from measured values.

For Vulkan tests, use `MOTIFCL_REQUIRE_VULKAN_COMPUTE=1` where supported so an unavailable device
cannot turn a skipped path into a false green result.

## Precision policy

The RX 580/Polaris target drives the default training design:

- Vulkan FP32 is the supported primary training path.
- BF16 training is not a target capability of Polaris.
- Full FP16 backward is not a project milestone for the RX 580 profile because the hardware
  lacks modern mixed-precision matrix acceleration and previous FP16 paths did not establish a
  useful end-to-end advantage.
- Q4/Q8/K-quant paths are intended for inference.
- Compact counter state and reversible execution are the memory-oriented training research paths.

Mixed-precision infrastructure may remain useful for portability experiments on other devices,
but it must not displace correctness and measured performance on the named target.

## Current scope

MotifCL is suitable for:

- Vulkan runtime and kernel research;
- compact Transformer inference;
- small-model FP32 training experiments;
- operator parity and backend studies;
- memory-native state/update experiments;
- legacy-GPU portability work.

It is not presented as:

- a full replacement for PyTorch;
- a production distributed-training platform;
- proof of large-model convergence;
- a universal loader for every HF architecture;
- a hardened service for untrusted model artifacts.

## Maintenance priorities

1. preserve strict Vulkan training witnesses;
2. keep HF/GGUF compatibility claims tied to executable tests;
3. separate inference support from backward/training support;
4. keep generated SPIR-V synchronized with shader sources;
5. improve install/export and consumer-build reliability;
6. document unsupported architectures and formats explicitly.
