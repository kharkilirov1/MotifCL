# Python API notes

The Python package exposes the C++ runtime plus a small PyTorch-like
`motifcl.nn` facade for ergonomic model composition.

Smoke-test after building a wheel:

```bash
python -m build --wheel
python -m pip install dist/motifcl-*.whl
python - <<'PY'
import motifcl as mcl
b = mcl.Backend.create()
x = mcl.ones(b, [2, 3])
bias = mcl.tensor_f32(b, [3], [1, 2, 3])
print((x + bias).cpu())
PY
```

Typed editors can consume `python/motifcl/__init__.pyi`.

Exposed production APIs include:

- deterministic RNG: `manual_seed`;
- tensor factories: `randn`, `uniform`, `zeros`, `ones`, `tensor_f32`, `tensor_i32`;
- extended ops: `add_broadcast`, `mul_broadcast`, `sum_all`, `mean_all`, `slice_rows`, GPU `dropout`, GPU `masked_fill`;
- modern transformer ops: `swiglu`, `qkv_split`, `rope`, `rope_positions`,
  `grouped_query_attention`, `grouped_query_attention_masked`,
  `kv_cache_append`, `kv_cache_append_positions`, and GPU `rowwise_sample`
  with greedy/temperature/top-k/top-p modes;
- modern transformer modules: `TransformerConfig`, `KVCache`, `ModernSelfAttention`, `ModernMLP`, `ModernTransformerBlock`, `ModernGPTModel`;
- PyTorch-like facade: `motifcl.nn.Module`, `__call__`, recursive `parameters()`,
  `named_parameters()`, `state_dict()`, `load_state_dict()`, `train()/eval()`,
  and `Tensor` operator overloads for tensor/tensor `+`, `-`, `*`, `@` plus scalar multiply;
- HF-style modern transformer compatibility: `HFArchitecture`, `HFTransformerConfig`, `QuantizationPolicy`, `load_hf_transformer_config_json`, `make_hf_transformer_model`, `load_hf_transformer_weights`, `load_hf_tokenizer`, `generate_hf_text`, `generate_hf_batch_text`;
- graph/runtime inspection: `Backend.command_buffer_mode()`, `Backend.supports_command_buffer_mutable_dispatch()`, and `GraphExecutor.execution_mode()`;
- checkpointing: `save_tensor`, `load_tensor`, `save_parameters`,
  `load_parameters`, `save_quantized_transformer_checkpoint`,
  `save_quantized_transformer_checkpoint_policy`,
  `load_quantized_transformer_checkpoint`;
- optimizers: `Adam`, `SGD`.

PyTorch-like model smoke:

```python
import motifcl as mcl
import motifcl.nn as nn

b = mcl.Backend.create()

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(b, 3, 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(b, 4, 2)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

model = TinyMLP()
x = mcl.randn(b, [8, 3])
y = model(x)
loss = mcl.mse_loss(y, mcl.zeros(b, [8, 2]))
opt = mcl.Adam(model.parameters(), lr=1e-3)
opt.zero_grad()
loss.backward()
opt.step()

with mcl.no_grad():
    print(model(x).shape)
```

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

HF-style modern decoder smoke:

```python
import motifcl as mcl

b = mcl.Backend.create()
cfg = mcl.load_hf_transformer_config_json("model/config.json")
model = mcl.make_hf_transformer_model(b, cfg)
report = mcl.load_hf_transformer_weights(b, model, ["model/model.safetensors"], cfg, strict=False)
policy = mcl.QuantizationPolicy()
policy.default_dtype = mcl.DType.Q4_0
policy.lm_head_dtype = mcl.DType.Q8_0
policy.q8_layers = [0]
model.enable_quantized_inference_policy(policy)
tokenizer = mcl.load_hf_tokenizer("model", cfg)
opts = mcl.GenerateOptions()
opts.top_p = 0.9
print(mcl.generate_hf_text(b, model, tokenizer, "Hello", opts))
print(mcl.generate_hf_batch_text(b, model, tokenizer, ["Hello", "Hi"]))
mcl.save_quantized_transformer_checkpoint_policy(model, "model-q4q8.mclq", policy)
loaded = mcl.make_hf_transformer_model(b, cfg)
mcl.load_quantized_transformer_checkpoint(loaded, b, "model-q4q8.mclq")
```
