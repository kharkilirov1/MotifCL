# MotifCL RX 580 / Polaris optimized profile

The `rx580-release` CMake preset is a performance-oriented build profile for
the existing Polaris/OpenCL path:

- builds tools, examples, tests, and benchmarks;
- enables IPO/LTO where supported;
- enables OpenCL fast relaxed math;
- keeps Python bindings off for a tighter native binary build.

Runtime tuning defaults are applied by the helper script after the normal
correctness tests complete:

- `MOTIFCL_MATMUL_F32_TILE=16`
- `MOTIFCL_FA_TILE=16`
- `MOTIFCL_FA_WG=128`
- `MOTIFCL_OPENCL_BUILD_OPTIONS="-cl-mad-enable -cl-no-signed-zeros"`

```powershell
powershell -ExecutionPolicy Bypass -File ports/rx580/run-rx580-release.ps1
```

The profile matches the checked-in `motifcl_tuning.json` baseline for
Ellesmere/Polaris where generated tile-16 F32 matmul is the fastest recorded
256x256 path.
