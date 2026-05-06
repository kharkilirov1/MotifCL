import os
import math
import tempfile

import motifcl as mcl


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
