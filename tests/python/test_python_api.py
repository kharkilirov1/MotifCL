import os
import json
import math
import struct
import tempfile

import motifcl as mcl
import motifcl.nn as nn


def _seq(count, scale=0.01):
    return [scale * (i + 1) for i in range(count)]




def _append_varint(buf, value):
    while value >= 0x80:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.append(value)


def _append_len_field(buf, field, payload):
    _append_varint(buf, (field << 3) | 2)
    _append_varint(buf, len(payload))
    buf.extend(payload)


def _write_sentencepiece_model(path, pieces):
    data = bytearray()
    for piece in pieces:
        msg = bytearray()
        _append_len_field(msg, 1, piece.encode("utf-8"))
        _append_varint(msg, (3 << 3) | 0)
        _append_varint(msg, 1)
        _append_len_field(data, 1, msg)
    trainer = bytearray()
    _append_varint(trainer, (3 << 3) | 0)
    _append_varint(trainer, 1)
    _append_len_field(data, 2, trainer)
    with open(path, "wb") as f:
        f.write(data)


def _write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    offset = 0
    for name, shape, values in tensors:
        raw = struct.pack("<" + "f" * len(values), *values)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


def test_python_extended_api_smoke():
    backend = mcl.Backend.create()
    mcl.manual_seed(123)
    x = mcl.tensor_f32(backend, [2, 3], [1, 2, 3, 4, 5, 6])
    bias = mcl.tensor_f32(backend, [3], [10, 20, 30])
    assert mcl.add(x, bias).cpu() == [11, 22, 33, 14, 25, 36]
    assert mcl.slice_rows(x, 1, 2).cpu() == [4, 5, 6]
    assert abs(mcl.mean_all(x).item() - 3.5) < 1e-6
    fused = mcl.add_bias_gelu_rows(x, mcl.tensor_f32(backend, [3], [0.5, -0.5, 1.0])).cpu()
    expected = []
    for v in [1.5, 1.5, 4.0, 4.5, 4.5, 7.0]:
        t = 0.7978845608028654 * (v + 0.044715 * v * v * v)
        expected.append(0.5 * v * (1.0 + math.tanh(t)))
    assert all(abs(a - b) < 1e-4 for a, b in zip(fused, expected))

    if mcl.backend_supports_fp16(backend):
        a16 = mcl.cast_f32_to_f16(x)
        assert all(abs(a - b) < 5e-3 for a, b in zip(mcl.cast_f16_to_f32(a16).cpu(), x.cpu()))
        w16 = mcl.cast_f32_to_f16(mcl.tensor_f32(backend, [3, 2], [7, 8, 9, 10, 11, 12]))
        assert all(abs(a - b) < 5e-2 for a, b in zip(mcl.matmul_f16_accum_f32(a16, w16).cpu(), [58, 64, 139, 154]))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tensor.mclt")
        mcl.save_tensor(x, path)
        assert mcl.load_tensor(backend, path).cpu() == x.cpu()

    logits = mcl.tensor_f32(backend, [2, 3], [1, 3, 2, 5, 4, 0])
    sampled = mcl.rowwise_sample(logits, temperature=0.0)
    assert sampled.cpu_i32() == [1, 0]
    sampled_top_p = mcl.rowwise_sample(logits, temperature=1.0, top_p=0.01)
    assert sampled_top_p.cpu_i32() == [1, 0]

    with tempfile.TemporaryDirectory() as tmp:
        sp_path = os.path.join(tmp, "tokenizer.model")
        _write_sentencepiece_model(sp_path, ["<unk>", "▁hello", "▁world"])
        tok = mcl.GemmaTokenizer.load_vocab(sp_path)
        assert tok.tokenizer_model_type() == "Unigram"
        assert tok.encode("hello world", False, False) == [1, 2]
        assert tok.decode([1, 2], False) == "hello world"


def test_python_pytorch_like_api_smoke():
    backend = mcl.Backend.create()
    mcl.manual_seed(777)

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(backend, 3, 4)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(backend, 4, 2)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    model = TinyMLP()
    x = mcl.tensor_f32(backend, [2, 3], [1, 2, 3, 4, 5, 6])
    y = model(x)
    assert y.shape == [2, 2]
    assert len(model.parameters()) == 4
    assert list(model.state_dict().keys()) == ["fc1.param_0", "fc1.param_1", "fc2.param_0", "fc2.param_1"]
    model.load_state_dict(model.state_dict())
    assert model.eval().training is False
    assert model.train().training is True

    assert (x + x).shape == [2, 3]
    assert (x @ mcl.tensor_f32(backend, [3, 2], [1, 0, 0, 1, 1, 1])).shape == [2, 2]
    assert (2.0 * x).shape == [2, 3]

    previous = mcl.is_grad_enabled()
    with mcl.no_grad():
        assert not mcl.is_grad_enabled()
        assert model(x).shape == [2, 2]
    assert mcl.is_grad_enabled() == previous

    target = mcl.tensor_f32(backend, [2, 2], [0, 0, 0, 0])
    loss = mcl.mse_loss(y, target)
    opt = mcl.SGD(model.parameters(), lr=0.0)
    opt.zero_grad()
    loss.backward()
    opt.step()

    model.fc1.enable_quantized_inference(mcl.DType.Q4_0)
    assert model.fc1.quantized_inference_enabled()
    with tempfile.TemporaryDirectory() as tmp:
        qpath = os.path.join(tmp, "fc1.q4.mclt")
        mcl.save_tensor(model.fc1.quantized_weight(), qpath)
        loaded = mcl.load_tensor(backend, qpath)
        assert loaded.dtype == mcl.DType.Q4_0
        assert loaded.has_quant_scales()


def test_python_modern_transformer_api_smoke():
    backend = mcl.Backend.create()
    mcl.manual_seed(321)

    packed = mcl.tensor_f32(
        backend,
        [2, 8],
        [1, 2, 3, 4, 10, 11, 20, 21, 5, 6, 7, 8, 12, 13, 22, 23],
    )
    split = mcl.qkv_split(packed, 4, 2)
    assert split.q.cpu() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert split.k.cpu() == [10, 11, 12, 13]
    assert split.v.cpu() == [20, 21, 22, 23]

    sw = mcl.swiglu(mcl.tensor_f32(backend, [1, 4], [1.0, 2.0, 3.0, 4.0]))
    assert sw.shape == [1, 2]

    q = mcl.randn(backend, [4, 16])
    k = mcl.randn(backend, [4, 8])
    v = mcl.randn(backend, [4, 8])
    gqa = mcl.grouped_query_attention(q, k, v, 4, 2, True, 1, 4, 4, 0)
    assert gqa.shape == [4, 16]
    mask = mcl.tensor_i32(
        backend,
        [4, 4],
        [
            0, 1, 1, 1,
            0, 0, 1, 1,
            0, 0, 0, 1,
            0, 0, 0, 0,
        ],
    )
    gqa_masked = mcl.grouped_query_attention_masked(q, k, v, mask, 4, 2, False, 1, 4, 4, 0)
    assert gqa_masked.shape == [4, 16]

    cfg = mcl.TransformerConfig()
    cfg.vocab_size = 32
    cfg.block_size = 8
    cfg.n_embd = 16
    cfg.n_head = 4
    cfg.n_kv_head = 2
    cfg.n_layer = 1
    cfg.mlp_hidden = 32
    cfg.use_rope = True
    cfg.use_swiglu = True
    cfg.use_qkv_bias = True

    attn = mcl.ModernSelfAttention(backend, cfg)
    cache = mcl.KVCache(backend, 1, cfg.block_size, cfg.n_kv_head, cfg.n_embd // cfg.n_head)
    y_cache = attn.forward_with_cache(mcl.randn(backend, [1, cfg.n_embd]), cache, 1, 1)
    assert y_cache.shape == [1, cfg.n_embd]
    assert cache.length == 1

    model = mcl.ModernGPTModel(backend, cfg)
    tokens = mcl.tensor_i32(backend, [1, 4], [1, 2, 3, 4])
    logits = model.forward(tokens)
    assert logits.shape == [1, 4, cfg.vocab_size]
    masked_logits = model.forward_masked(tokens, mask)
    assert masked_logits.shape == [1, 4, cfg.vocab_size]
    caches = model.create_kv_cache(backend, 1)
    step_logits = model.forward_with_cache(mcl.tensor_i32(backend, [1, 1], [1]), caches)
    assert step_logits.shape == [1, 1, cfg.vocab_size]
    assert caches[0].length == 1
    masked_caches = model.create_kv_cache(backend, 1)
    cache_mask = mcl.tensor_i32(backend, [1, 1, cfg.block_size], [0] * cfg.block_size)
    masked_step = model.forward_with_cache_masked(mcl.tensor_i32(backend, [1, 1], [1]), cache_mask, masked_caches)
    assert masked_step.shape == [1, 1, cfg.vocab_size]
    assert masked_caches[0].length == 1
    assert len(model.parameters()) > 0


def test_python_hf_modern_transformer_compat_smoke():
    backend = mcl.Backend.create()
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(
                """{
  "model_type": "llama",
  "vocab_size": 256,
  "max_position_embeddings": 8,
  "hidden_size": 16,
  "intermediate_size": 32,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "num_key_value_heads": 2,
  "rms_norm_eps": 0.00001,
  "rope_theta": 10000.0,
  "attention_bias": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
}"""
            )
        tokenizer_path = os.path.join(tmp, "tokenizer.json")
        with open(tokenizer_path, "w", encoding="utf-8") as f:
            json.dump({"model": {"type": "BPE", "vocab": {"<pad>": 0, "<bos>": 1, "<eos>": 2, "A": 65, "B": 66, "AB": 200}, "merges": ["A B"]}}, f)

        cfg = mcl.load_hf_transformer_config_json(config_path)
        assert cfg.architecture == mcl.HFArchitecture.Llama
        assert cfg.architecture_name == "llama"
        assert cfg.transformer.n_embd == 16
        assert cfg.transformer.n_kv_head == 2
        assert "model.layers.0.self_attn.q_proj.weight" in mcl.expected_hf_transformer_weight_names(cfg)
        mapped = mcl.map_hf_transformer_weight_name(cfg.architecture, "model.layers.0.mlp.down_proj.weight")
        assert mapped.kind == "down_proj"

        weights_path = os.path.join(tmp, "model.safetensors")
        _write_safetensors(weights_path, [
            ("model.embed_tokens.weight", [256, 16], _seq(256 * 16, 0.0001)),
            ("model.norm.weight", [16], [1.0] * 16),
            ("lm_head.weight", [256, 16], _seq(256 * 16, 0.0002)),
            ("model.layers.0.input_layernorm.weight", [16], [1.0] * 16),
            ("model.layers.0.post_attention_layernorm.weight", [16], [1.0] * 16),
            ("model.layers.0.self_attn.q_proj.weight", [16, 16], _seq(16 * 16, 0.0003)),
            ("model.layers.0.self_attn.k_proj.weight", [8, 16], _seq(8 * 16, 0.0004)),
            ("model.layers.0.self_attn.v_proj.weight", [8, 16], _seq(8 * 16, 0.0005)),
            ("model.layers.0.self_attn.o_proj.weight", [16, 16], _seq(16 * 16, 0.0006)),
            ("model.layers.0.mlp.gate_proj.weight", [32, 16], _seq(32 * 16, 0.0007)),
            ("model.layers.0.mlp.up_proj.weight", [32, 16], _seq(32 * 16, 0.0008)),
            ("model.layers.0.mlp.down_proj.weight", [16, 32], _seq(16 * 32, 0.0009)),
        ])

        model = mcl.make_hf_transformer_model(backend, cfg)
        report = mcl.load_hf_transformer_weights(backend, model, [weights_path], cfg, strict=True)
        assert report.missing == []
        assert report.loaded_tensors >= 12
        tokens = mcl.tensor_i32(backend, [1, 4], [65, 66, 67, 68])
        logits = model.forward(tokens)
        assert logits.shape == [1, 4, 256]

        tokenizer = mcl.load_hf_tokenizer(tmp, cfg)
        assert tokenizer.encode("AB", False, False) == [200]
        assert tokenizer.decode([200], True) == "AB"
        opts = mcl.GenerateOptions()
        opts.max_new_tokens = 0
        assert mcl.generate_hf_text(backend, model, tokenizer, "AB", opts) == "AB"
        assert mcl.generate_batch_text(backend, model, tokenizer, ["AB", "A"], opts) == ["AB", "A"]
        assert mcl.generate_hf_batch_text(backend, model, tokenizer, ["AB", "A"], opts) == ["AB", "A"]
        opts.prefill_prompt = False
        assert mcl.generate_hf_text(backend, model, tokenizer, "AB", opts) == "AB"
        opts.prefill_prompt = True
        opts.gpu_greedy_sampling = True
        mcl.enable_hf_transformer_quantized_inference(model, mcl.DType.Q8_0)
        assert model.quantized_inference_enabled()
        assert model.quantized_weight_dtype() == mcl.DType.Q8_0
        opts.prefill_prompt = True
        assert mcl.generate_hf_text(backend, model, tokenizer, "AB", opts) == "AB"
        mcl.enable_hf_transformer_quantized_inference(model, mcl.DType.Q4_0)
        assert model.quantized_weight_dtype() == mcl.DType.Q4_0
        assert mcl.generate_hf_text(backend, model, tokenizer, "AB", opts) == "AB"
        opts.max_new_tokens = 1
        opts.temperature = 1.0
        opts.top_k = 1
        batch_ids = mcl.generate_batch(backend, model, [[65], [65, 66]], opts)
        assert len(batch_ids) == 2 and len(batch_ids[0]) == 2 and len(batch_ids[1]) == 3
        opts.top_k = 0
        opts.top_p = 0.01
        batch_ids_top_p = mcl.generate_batch(backend, model, [[65], [65, 66]], opts)
        assert len(batch_ids_top_p) == 2 and len(batch_ids_top_p[0]) == 2 and len(batch_ids_top_p[1]) == 3
        with tempfile.TemporaryDirectory() as qtmp:
            mcl.save_quantized_transformer_checkpoint(model, qtmp, mcl.DType.Q4_0)
            loaded = mcl.make_hf_transformer_model(backend, cfg)
            mcl.load_quantized_transformer_checkpoint(loaded, backend, qtmp)
            assert loaded.quantized_inference_enabled()
            assert loaded.quantized_weight_dtype() == mcl.DType.Q4_0
        policy = mcl.QuantizationPolicy()
        policy.default_dtype = mcl.DType.Q4_0
        policy.lm_head_dtype = mcl.DType.Q8_0
        policy.q8_layers = [0]
        model.enable_quantized_inference_policy(policy)
        with tempfile.TemporaryDirectory() as qtmp:
            mcl.save_quantized_transformer_checkpoint_policy(model, qtmp, policy)
            loaded = mcl.make_hf_transformer_model(backend, cfg)
            mcl.load_quantized_transformer_checkpoint(loaded, backend, qtmp)
            assert loaded.quantized_lm_head().dtype == mcl.DType.Q8_0
        mcl.disable_hf_transformer_quantized_inference(model)
        assert not model.quantized_inference_enabled()
