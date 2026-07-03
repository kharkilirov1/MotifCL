# MotifCL Apple ports — macOS and iPhone

## macOS

MotifCL can use the existing OpenCL backend on macOS. The `macos-release`
CMake preset is guarded to Darwin hosts and enables:

- `Release`
- universal `arm64;x86_64` build
- tools, examples, tests, and benchmarks
- fast OpenCL kernel math
- IPO/LTO when supported by the Apple toolchain

Run on a Mac:

```bash
cmake --preset macos-release
cmake --build --preset macos-release
ctest --test-dir build/macos-release --output-on-failure
```

## iPhone / iOS

iOS does not provide OpenCL. A real native iPhone compute backend must be Metal
or a Vulkan-over-Metal layer such as MoltenVK. This folder therefore contains a
native Metal device probe as the first non-OpenCL iPhone step; it validates the
Apple GPU path without pretending the full MotifCL tensor/autograd stack already
runs on Metal.

On a Mac with Xcode:

```bash
# macOS Metal probe
./ports/apple/build-metal-probe.sh macos

# iPhone arm64 probe binary; deploy/sign with your normal Xcode/iOS flow.
./ports/apple/build-metal-probe.sh ios
```

The remaining iPhone work is to map MotifCL kernels to MSL compute pipelines and
then make OpenCL an optional backend in the core library.
