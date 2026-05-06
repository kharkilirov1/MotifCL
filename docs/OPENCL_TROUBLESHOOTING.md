# OpenCL troubleshooting

## No platform/device found

Install an OpenCL ICD loader and at least one vendor/CPU device driver.

Linux CI uses:

```bash
sudo apt-get install -y ocl-icd-opencl-dev pocl-opencl-icd clinfo
```

Windows builds can compile against the vendored minimal OpenCL headers plus `C:\Windows\System32\OpenCL.dll`, but runtime tests still require a usable ICD/device.

## Kernel source discovery

Kernel lookup order:

1. `MOTIFCL_KERNEL_DIR`;
2. build-tree `build/kernels`;
3. source-tree `kernels`;
4. installed `share/motifcl/kernels`;
5. relative `kernels`.

## Performance knobs

- Run `motifcl_kernel_tuner` to write `motifcl_tuning.json`.
- Use `MOTIFCL_MATMUL_F32_TILE=4`, `8`, or `16` to force generated tiled matmul for regression comparison.
- Leave the variable unset for the default register-blocked matmul path.
