# MotifCL portability matrix

## RX 580 / Polaris

Use `rx580-release` for the local optimized OpenCL path. It enables native tools
and benchmarks plus OpenCL fast-math switches. Runtime defaults select the
checked-in Polaris tuning baseline (`MOTIFCL_MATMUL_F32_TILE=16`,
`MOTIFCL_FA_TILE=16`, `MOTIFCL_FA_WG=128`).

## Android / Xiaomi Mi 9

The native Android path is in `ports/android`. The connected Mi 9 was probed as
`arm64-v8a`, Android SDK 35, with `/vendor/lib64/libOpenCL.so` present. The host
currently has `adb` but no Android SDK/NDK environment, so use
`ports/android/build-mi9-native.ps1 -ProbeOnly` for read-only device checks and
install/set `ANDROID_HOME` or `ANDROID_NDK_HOME` before compiling.

## macOS

Use `macos-release` on a Darwin host. This keeps the current OpenCL backend,
builds universal `arm64;x86_64`, and enables the same optimized release switches
where supported by the Apple toolchain.

## iPhone / iOS

iOS has no OpenCL runtime, so the MotifCL tensor/autograd stack cannot be
truthfully marked iPhone-native until a Metal backend is implemented or the
OpenCL backend becomes optional. `ports/apple/metal_probe.mm` is the first
native Metal validation step; build it on a Mac with Xcode using
`ports/apple/build-metal-probe.sh ios`.
