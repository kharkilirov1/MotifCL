# MotifCL

MotifCL is a C++17/OpenCL neural compute framework for legacy AMD GPUs, especially Polaris/RX 580-class hardware. It is designed as a compact research-first alternative to heavy ML runtimes: OpenCL runtime, Tensor API, eager autograd for core training loops, neural-network modules, Transformer forward stack, LoRA/Motif/SARC research modules, Python bindings, tests, tools, benchmarks, and CMake install/export support.

This release is a productionized source tree rather than a single-file kernel demo. The C++ core builds as a reusable library target `MotifCL::motifcl`; examples, tests, tools and benchmarks are separate targets; kernels are installed and discoverable through `MOTIFCL_KERNEL_DIR` or the installed data directory.

## Implemented stack

- Runtime: OpenCL context/device selection, command queue with profiling, shared OpenCL runtime-state ownership for buffers/programs/kernels/replay, RAII buffers, programs, kernels, events, profiler, kernel cache.
- Tensor engine: `Tensor`, `Storage`, `Shape`, `DType`, views, CPU upload/download, `empty`, `zeros`, `ones`, `full`, `randn`, `uniform`.
- Ops: elementwise math, scalar ops, row bias, workgroup row reductions, register-blocked F32 matmul, generated F32 matmul tile variants, Q8_0/Q4_0 symmetric quantize/dequantize, per-tensor/per-axis/blockwise quantization metadata, mixed Q8/Q4 quantized matmul, register-blocked unscaled quantized matmul paths, extension-gated Q8 integer-dot path with portable fallback, activations, softmax rows, causal mask, FlashAttention-style tiled multi-head attention forward/backward, workgroup-reduced RMSNorm/LayerNorm, MSE loss, softmax cross entropy for I32 targets, Adam and SGD kernels.
- Autograd: eager backward path for add/sub/mul/div/scale, matmul, ReLU, GELU, MSE, softmax-cross-entropy logits, RMSNorm, token/position embedding, and correctness-first fused multi-head attention backward.
- Graph capture: thread-local static op-sequence tracing through `CapturedGraph`, `GraphCaptureGuard`, tensor IDs, replayable captured OpenCL kernel launches, host-reduction replay for scalar MSE / softmax-cross-entropy reductions, linear scheduler/executor, tensor-spec metadata, buffer-planning liveness estimates, shape-polymorphic signatures, and C++/Python APIs for inspecting/executing captured dependencies.
- NN: `Parameter`, `Module`, `Linear`, `ReLU`, `GELU`, `Sequential`, `Embedding`, `RMSNorm`, `SelfAttention`, `MLP`, `TransformerBlock`, `GPTModel`.
- Motif/research: `MotifLinear`, `MotifLoRA`, `Router`, `MotifTransformerBlock`, `SARCResidual`.
- Training: `Adam`, `SGD`, generic `Trainer` with dataloader and loss callback.
- Python: pybind11 extension source plus thin package wrappers.
- QA: CTest suite, CPU/GPU compare tool, OpenCL info dumper, kernel tuner, microbenchmarks.

## Build

```bash
cmake -S . -B build \
  -DMOTIFCL_BUILD_EXAMPLES=ON \
  -DMOTIFCL_BUILD_TESTS=ON \
  -DMOTIFCL_BUILD_TOOLS=ON \
  -DMOTIFCL_BUILD_BENCHMARKS=ON \
  -DMOTIFCL_BUILD_PYTHON=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Python wheel builds use the isolated PEP 517 dependencies from `pyproject.toml`:

```bash
python -m pip install build
python -m build --wheel
python -m pip install dist/motifcl-*.whl
python -c "import motifcl as mcl; print(mcl.Backend)"
```

Direct CMake Python builds are also supported when `pybind11` and Python development metadata are visible to CMake:

```bash
cmake -S . -B build -DMOTIFCL_BUILD_PYTHON=ON
cmake --build build -j
PYTHONPATH=$PWD/build/python python -c "import motifcl as mcl; print(mcl.Backend)"
```

The source build can work even when OpenCL development headers are missing, because a minimal vendored `CL/cl.h` is included. You still need an OpenCL ICD loader/library and an actual OpenCL driver at runtime.

## Install and consume from another CMake project

```bash
cmake --install build --prefix /opt/motifcl
```

Consumer:

```cmake
find_package(MotifCL REQUIRED CONFIG)
add_executable(app main.cpp)
target_link_libraries(app PRIVATE MotifCL::motifcl)
```

Kernel discovery order:

1. `MOTIFCL_KERNEL_DIR`, if set;
2. build-tree `build/kernels`;
3. source-tree `kernels`;
4. installed `share/motifcl/kernels`;
5. relative `kernels` directory.

## C++ quick start

```cpp
#include <motifcl/motifcl.hpp>

int main() {
    auto backend = motifcl::Backend::create_opencl();
    auto x = motifcl::Tensor::randn(backend, {32, 128});
    auto w = motifcl::Tensor::randn(backend, {128, 64});
    auto y = motifcl::gelu(motifcl::matmul(x, w));
    auto host = y.to_vector<float>();
}
```

## Tiny GPT forward and cross entropy

```cpp
#include <cstdint>
#include <motifcl/motifcl.hpp>

int main() {
    auto backend = motifcl::Backend::create_opencl();
    motifcl::nn::GPTModel model(backend, 128, 64, 128, 4, 2, 256);

    std::int32_t ids_host[] = {1, 2, 3, 4};
    std::int32_t tgt_host[] = {2, 3, 4, 5};
    auto ids = motifcl::Tensor::from_cpu(backend, {1, 4}, motifcl::DType::I32, ids_host);
    auto tgt = motifcl::Tensor::from_cpu(backend, {4}, motifcl::DType::I32, tgt_host);

    auto logits = model.forward(ids).view({4, 128});
    auto loss = motifcl::softmax_cross_entropy(logits, tgt);
    loss.backward();
}
```

## Tools

```bash
./build/tools/motifcl_dump_opencl_info
./build/tools/motifcl_compare_cpu_gpu
./build/tools/motifcl_kernel_tuner
```

`motifcl_kernel_tuner` writes `motifcl_tuning.json` with local F32, generated F32 tile variants, Q8_0, Q4_0, mixed Q8/Q4, per-axis Q8, blockwise Q4 matmul timings, and the selected integer-dot mode for the active OpenCL device.

## Benchmarks

```bash
cmake -S . -B build -DMOTIFCL_BUILD_BENCHMARKS=ON
cmake --build build -j
./build/benchmarks/bench_matmul
./build/benchmarks/bench_softmax
./build/benchmarks/bench_attention
./build/benchmarks/bench_lora
```

## Current engineering boundaries

MotifCL is now structured and buildable as a real library, but it is still intentionally small. It does not attempt to be a full PyTorch replacement. Transformer forward and a small GPT-style backward/training path work through token/position embeddings, RMSNorm, FlashAttention-style tiled multi-head attention, MLP, and cross entropy. Q8_0 and Q4_0 quantize/dequantize and matmul are implemented as correctness-first symmetric paths with per-tensor, per-row/per-column, flat blockwise, mixed Q8/Q4 matmul support, register-blocked unscaled quantized kernels, a `cl_khr_integer_dot_product` generated-kernel path when the driver exposes it, and portable fallback kernels. F32 matmul now uses a 4x4 per-thread register-blocked 32x32 workgroup tile for normal non-transposed matmul, while transpose and scaled-quant metadata paths keep conservative fallback kernels. Rowwise sum/max, RMSNorm, RMS-per-row, LayerNorm, and RMSNorm backward-X use cooperative workgroup row reductions instead of one-thread-per-row reductions when the device supports 256-workitem groups. Static graph capture can replay captured GPU kernel launches and scalar host reductions over the same tensors, exposes a linear schedule, records tensor specs, estimates buffer reuse/liveness, and can emit shape-polymorphic signatures. Runtime OpenCL handles are shared across buffers/programs/kernels/replay so tensor storage can be downloaded/released safely after a Backend scope ends; creating new ops still intentionally requires a live Backend/KernelCache. Remaining limits are mainly true dynamic-shape rebinding/recompilation, planner-materialized buffer allocation for arbitrary captured graphs, hand-written vendor ISA assembly kernels, deeper backend-handle ergonomics for post-Backend new ops, and broader production performance tuning.
