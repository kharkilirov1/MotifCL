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
