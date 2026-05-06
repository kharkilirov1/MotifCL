# Python API notes

The Python package is a thin pybind11 wrapper over the C++ runtime.

Smoke-test after building a wheel:

```bash
python -m build --wheel
python -m pip install dist/motifcl-*.whl
python - <<'PY'
import motifcl as mcl
b = mcl.Backend.create()
x = mcl.ones(b, [2, 3])
bias = mcl.tensor_f32(b, [3], [1, 2, 3])
print(mcl.add(x, bias).cpu())
PY
```

Typed editors can consume `python/motifcl/__init__.pyi`.

Exposed production APIs include:

- deterministic RNG: `manual_seed`;
- tensor factories: `randn`, `uniform`, `zeros`, `ones`, `tensor_f32`, `tensor_i32`;
- extended ops: `add_broadcast`, `mul_broadcast`, `sum_all`, `mean_all`, `slice_rows`, `dropout`, `masked_fill`;
- checkpointing: `save_tensor`, `load_tensor`, `save_parameters`, `load_parameters`;
- optimizers: `Adam`, `SGD`.
