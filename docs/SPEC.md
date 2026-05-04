# MotifCL Specification

## Overview
MotifCL is a C++17/OpenCL neural compute framework for legacy AMD GPUs (RX 580 / gfx803).
It provides a layered architecture: Runtime → Tensor Engine → Ops → Autograd → NN API → Motif Research modules → Python bindings.

## Architecture Layers
```
┌─────────────────────────────────────────────┐
│ Python API                                  │
├─────────────────────────────────────────────┤
│ High-level NN API (Linear, Attention, etc.) │
├─────────────────────────────────────────────┤
│ Autograd / Graph Engine                     │
├─────────────────────────────────────────────┤
│ Tensor Engine                               │
├─────────────────────────────────────────────┤
│ Ops Layer (matmul, gelu, softmax, adam)     │
├─────────────────────────────────────────────┤
│ Kernel Runtime (cache, events, queue)       │
├─────────────────────────────────────────────┤
│ OpenCL Runtime (context, device, queue)     │
├─────────────────────────────────────────────┤
│ Hardware (RX 580 / AMD OpenCL driver)       │
└─────────────────────────────────────────────┘
```

## Interface Contracts

### Core Types (include/motifcl/core/)
- `DType`: enum class { F32, F16, I32, U8, Q8_0, Q4_0 }
- `Shape`: vector<int64_t> with numel(), ndim(), operator[]
- `DeviceType`: enum class { CPU, OpenCL }
- `Status`: error code system with StatusCode enum
- `Error`: exception type with message + code
- `Logging`: simple severity-based logger

### Runtime (include/motifcl/runtime/)
- `OpenCLContext`: platform, device, context, queue. Static factory `create_default_gpu()`. Methods: finish(), flush().
- `Buffer`: cl_mem wrapper. Upload/download CPU<->GPU. RAII cleanup.
- `Program`: compiles OpenCL C source. `get_kernel(name)` returns Kernel.
- `Kernel`: set_arg(int, Buffer|int|float), launch1d(global, local), launch2d(gx, gy, lx, ly).
- `KernelCache`: string->Kernel LRU cache tied to Backend.
- `Backend`: owns OpenCLContext, KernelCache, Profiler. Static `create()`. Non-copyable.
- `Profiler`: CL event timing via profiling info.

### Tensor (include/motifcl/tensor/)
- `Storage`: owns Buffer + nbytes. Shared by views.
- `Tensor`: backend ptr, shared Storage, Shape, strides, DType, offset, requires_grad, grad, grad_fn.
  Static factories: `from_cpu()`, `zeros()`, `randn()`, `empty()`.
  Methods: `view()`, `contiguous()`, `to_cpu()`, `numel()`, `nbytes()`, `dtype()`, `shape()`, `stride()`.
  Data is always on GPU; `to_cpu()` downloads.

### Ops (include/motifcl/ops/)
Functional API (free functions taking Tensor, returning Tensor):
- Basic: `add`, `sub`, `mul`, `div`, `scale`, `neg`, `fill`, `copy`
- Activations: `relu`, `gelu`, `silu`, `exp`, `sqrt`, `rsqrt`
- Reduce: `sum`, `mean`, `max_reduce`
- Matmul: `matmul`, `matmul_t`, `linear`, `linear_bias`
- Norm: `rmsnorm`, `layernorm`
- Attention: `softmax_rows`, `causal_mask`, `attention_scores`, `attention_apply`, `rope`
- Loss: `mse_loss`, `cross_entropy`
- Optim: `sgd_update`, `adam_update`
All ops create output Tensor with proper shape; some register backward nodes if requires_grad.

### Autograd (include/motifcl/autograd/)
- `Node`: abstract base. `inputs`, `outputs`, virtual `backward(grad_output)`.
- `Tape`: vector of shared_ptr<Node>. `add()`, `backward(loss)`, `clear()`.
- Backward implementations: `MatMulBackward`, `AddBackward`, `ReluBackward`, etc.
- `backward(Tensor loss)` traverses tape in reverse, calling each node's backward.

### NN (include/motifcl/nn/)
- `Parameter`: Tensor data + Tensor grad + bool trainable.
- `Module`: abstract. `forward(x)`, `parameters()`, `zero_grad()`, `to(backend)`.
- `Linear`: weight + bias Parameters. `forward(x) = matmul(x, W) + bias`.
- `RMSNorm`: weight Parameter + eps. `forward(x) = x / RMS(x) * weight`.
- `GELU`, `ReLU`, `SiLU`: activation modules.
- `Sequential`: vector<Module*>. Runs in order.
- `Embedding`: lookup table. `forward(indices)`.
- `SelfAttention`: Q/K/V/O Linear projections. n_head, head_dim. causal mask support.
- `MLP`: fc1 + fc2 Linear with GELU.
- `TransformerBlock`: RMSNorm → SelfAttention → residual → RMSNorm → MLP → residual.
- `GPTModel`: Embedding → TransformerBlocks → final Linear.

### Motif (include/motifcl/motif/)
- `MotifLinear`: multiple motif slices with router weights.
- `MotifLoRA`: frozen base + LoRA adapters per motif.
- `SARCResidual`: `y = x + gamma * F(x) / (RMS(x) + eps)`.
- `MotifRouter`: computes routing weights.
- `MotifTransformerBlock`: uses MotifAttention + MotifMLP + SARC.

### Train (include/motifcl/train/)
- `Optimizer`: abstract. `step()`, `zero_grad()`.
- `Adam`: lr, beta1, beta2, eps, step_count. Maintains m/v tensors per parameter.
- `SGD`: lr, momentum. Optional momentum buffer.
- `Trainer`: owns model + optimizer + dataloader. `fit(epochs)`.

## Build System
- CMake 3.20+, C++17 standard.
- `find_package(OpenCL REQUIRED)` for main library.
- `find_package(pybind11)` for Python module.
- Options: MOTIFCL_BUILD_TESTS, MOTIFCL_BUILD_PYTHON, MOTIFCL_BUILD_EXAMPLES.
- Library target: `motifcl` (shared/static).
- Python target: `motifcl_py` module renamed to `motifcl`.

## Kernel Files
All in `kernels/` directory. Loaded at runtime from build directory via `file(COPY kernels ...)`.
- `basic.cl`: fill, add, mul, scale, copy
- `matmul.cl`: naive and tiled variants
- `reduce.cl`: sum, max reductions
- `softmax.cl`: row-wise softmax
- `norm.cl`: rmsnorm, layernorm
- `activation.cl`: relu, gelu, silu
- `attention.cl`: rope, causal_mask, attention scores/apply
- `loss.cl`: mse, cross-entropy
- `optim.cl`: sgd, adam updates
- `motif.cl`: motif_linear, motif_lora, sarc_residual

## Testing Strategy
- CPU reference implementation for every op.
- Random input → CPU vs GPU → compare max abs error.
- Gradient check: analytical vs numerical for autograd.
- MLP trains on XOR or synthetic regression.

## Execution Mode
Phase 1: Immediate mode (eager). Each op launches kernel immediately.
Future: Lazy graph + captured training step.
