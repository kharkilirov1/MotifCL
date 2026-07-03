# MotifCL native Android port — Xiaomi Mi 9 / arm64-v8a

This port builds the native C++ MotifCL core plus `motifcl_dump_opencl_info` and
`motifcl_compare_cpu_gpu` with the Android NDK, links against the device OpenCL
loader, then runs the binaries directly through `adb shell`.

Verified local device probe:

- model: `MI 9`
- device: `cepheus`
- ABI: `arm64-v8a`
- Android SDK level: `35`
- OpenCL loader present: `/vendor/lib64/libOpenCL.so`
- Vulkan loader present: `/system/lib64/libvulkan.so`

Host blocker observed on this machine: `ANDROID_HOME`, `ANDROID_SDK_ROOT`, and
`JAVA_HOME` are unset, so the device can be probed through `adb`, but the native
Android build needs an Android SDK + NDK install before it can compile.

## Build and run

```powershell
# Optional read-only probe; works with only adb.
powershell -ExecutionPolicy Bypass -File ports/android/build-mi9-native.ps1 -ProbeOnly

# Full native build; requires ANDROID_HOME or ANDROID_NDK_HOME.
powershell -ExecutionPolicy Bypass -File ports/android/build-mi9-native.ps1
```

The script:

1. detects the connected Mi 9 through `adb`;
2. pulls `/vendor/lib64/libOpenCL.so` into `build/android-mi9-arm64/opencl/arm64-v8a/`;
3. configures CMake with `ANDROID_ABI=arm64-v8a`;
4. enables MI 9 host-side tuning and fast OpenCL math;
5. builds `motifcl_dump_opencl_info` and `motifcl_compare_cpu_gpu`;
6. pushes and runs them in `/data/local/tmp/motifcl`.

Use the raw CMake preset only when these environment variables are already set:

```powershell
$env:ANDROID_NDK_HOME="C:\path\to\Android\Sdk\ndk\<version>"
$env:MOTIFCL_ANDROID_OPENCL_ARM64_LIB="C:\path\to\libOpenCL.so"
cmake --preset android-mi9-arm64
cmake --build --preset android-mi9-arm64
```
