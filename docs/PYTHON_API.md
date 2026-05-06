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
- extended ops: `add_broadcast`, `mul_broadcast`, `sum_all`, `mean_all`, `slice_rows`, GPU `dropout`, GPU `masked_fill`;
- modern transformer ops: `swiglu`, `qkv_split`, `rope`, `grouped_query_attention`, `grouped_query_attention_masked`, `kv_cache_append`;
- modern transformer modules: `TransformerConfig`, `KVCache`, `ModernSelfAttention`, `ModernMLP`, `ModernTransformerBlock`, `ModernGPTModel`;
- checkpointing: `save_tensor`, `load_tensor`, `save_parameters`, `load_parameters`;
- optimizers: `Adam`, `SGD`.

Minimal modern GPT smoke:

```python
import motifcl as mcl

b = mcl.Backend.create()
cfg = mcl.TransformerConfig()
cfg.vocab_size = 32000
cfg.block_size = 256
cfg.n_embd = 512
cfg.n_head = 8
cfg.n_kv_head = 2     # GQA; set to 1 for MQA or n_head for normal MHA
cfg.n_layer = 6
cfg.use_rope = True
cfg.use_swiglu = True

model = mcl.ModernGPTModel(b, cfg)
tokens = mcl.tensor_i32(b, [1, 4], [1, 2, 3, 4])
logits = model.forward(tokens)
print(logits.shape)

caches = model.create_kv_cache(b, batch_size=1)
step_logits = model.forward_with_cache(mcl.tensor_i32(b, [1, 1], [1]), caches)
print(step_logits.shape, caches[0].length)
```
