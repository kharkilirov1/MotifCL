# MotifCL Build Plan

## Overview
Build MotifCL — a C++17/OpenCL neural compute framework for legacy AMD GPUs (RX 580/gfx803).
Layered architecture: Runtime → Tensor Engine → Ops → Autograd → NN API → Motif modules → Python bindings.

## Execution Strategy
Build in staged phases, each creating a working, testable increment.

## Stage 1 — Foundation Runtime (Phase 1)
**Goal**: OpenCL context, buffer, program, kernel cache, backend, error handling.
**Output**: Working C++ runtime that can create context, allocate buffers, compile kernels, run fill/add kernels, download results.
**Files**: `include/motifcl/runtime/*`, `src/runtime/*`, `kernels/basic.cl`, `tools/dump_opencl_info.cpp`, CMakeLists.txt root.

## Stage 2 — Tensor Engine (Phase 2)
**Goal**: DType, Shape, Storage, Tensor, Allocator, CPU/GPU transfer.
**Output**: Create tensors, views, copy CPU↔GPU, basic tensor properties.
**Files**: `include/motifcl/tensor/*`, `src/tensor/*`

## Stage 3 — Core Ops + Kernels (Phase 3)
**Goal**: Basic element-wise ops, matmul (naive + tiled), softmax, reduce, norm, loss, optim kernels.
**Output**: Functional API: add, mul, matmul, relu, gelu, softmax, mse_loss, cross_entropy, adam_update.
**Files**: `include/motifcl/ops/*`, `src/ops/*`, `kernels/*.cl`
**Tests**: CPU reference comparison for each op.

## Stage 4 — Autograd MVP (Phase 4)
**Goal**: Node, Tape, backward pass for core ops.
**Output**: Can build MLP, compute loss, call `.backward()`, gradients flow correctly.
**Files**: `include/motifcl/autograd/*`, `src/autograd/*`
**Tests**: Gradient numerical check.

## Stage 5 — NN Modules (Phase 5)
**Goal**: Module base, Parameter, Linear, GELU, RMSNorm, Sequential, MLP.
**Output**: PyTorch-like C++ API for building models.
**Files**: `include/motifcl/nn/*`, `src/nn/*`
**Tests**: MLP trains on toy data (XOR or random regression).

## Stage 6 — Python Bindings (Phase 6)
**Goal**: pybind11 bindings for Backend, Tensor, ops, nn, optim.
**Output**: `import motifcl` works in Python, can create tensors, build model, train.
**Files**: `python/*`
**Tests**: Python test script trains MLP.

## Stage 7 — Transformer Stack (Phases 7-8)
**Goal**: Embedding, RoPE, CausalSelfAttention, MLP, TransformerBlock, GPTModel, training loop.
**Output**: Character-level transformer trains and generates text.
**Files**: `include/motifcl/nn/{embedding,attention,transformer}.hpp`

## Stage 8 — LoRA + Motif Research Layer (Phases 9-10)
**Goal**: LoRALinear, MotifLinear, MotifLoRA, MotifRouter, SARCResidual, MotifTransformerBlock.
**Output**: Specialized research modules for adapter learning experiments.
**Files**: `include/motifcl/motif/*`, `src/motif/*`

## Deliverables
- Complete `motifcl/` source tree
- `CMakeLists.txt` with options for tests, Python, examples
- Working C++ and Python APIs
- Tests and benchmarks
- Examples in `examples/cpp/` and `examples/python/`

## Skill Loading
- Stage 1-8: `vibecoding-general-swarm` for C++ framework implementation
