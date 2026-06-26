"""Isolation for the deep nonlinear regression plateau seen on the GPU MLP test.
Same task (random ReLU teacher), three students: dense FP32+Adam, ternary-QAT+Adam,
counter+RMS (self-update). If ternary-QAT also plateaus far above dense, the gap is
the cost of TERNARIZING a continuous FP32 target (not the counter optimizer)."""
import torch, torch.nn as nn, torch.nn.functional as F
from counter_state_rms import RMSCounterLinear
from parity_gate import TernaryQATLinear

torch.manual_seed(0)
N, din, dh, dout, steps = 512, 8, 32, 4, 1500
W1 = torch.randn(din, dh) * 0.5
W2 = torch.randn(dh, dout) * 0.5
x = torch.randn(N, din, requires_grad=True)
y = (F.relu(x.detach() @ W1) @ W2).detach()


def run(make1, make2, opt_fn):
    torch.manual_seed(1)
    l1, l2 = make1(), make2()
    params = list(l1.parameters()) + list(l2.parameters())
    opt = opt_fn(params) if (opt_fn and params) else None
    curve = []
    for s in range(steps):
        pred = l2(F.relu(l1(x)))
        loss = ((pred - y) ** 2).mean()
        if opt:
            opt.zero_grad()
        loss.backward()
        if opt:
            opt.step()
        if s % 300 == 0 or s == steps - 1:
            curve.append((s, loss.item()))
    with torch.no_grad():
        l1.eval() if hasattr(l1, "eval") else None
        # disable counter self-update for clean eval
        for m in (l1, l2):
            if hasattr(m, "update_enabled"):
                m.update_enabled = False
        pred = l2(F.relu(l1(x)))
        fmse = ((pred - y) ** 2).mean().item()
    return fmse, curve


dense, dc = run(lambda: nn.Linear(din, dh, bias=False),
                lambda: nn.Linear(dh, dout, bias=False),
                lambda p: torch.optim.Adam(p, 1e-2))
qat, qc = run(lambda: TernaryQATLinear(din, dh),
              lambda: TernaryQATLinear(dh, dout),
              lambda p: torch.optim.Adam(p, 1e-2))
cnt, cc = run(lambda: RMSCounterLinear(din, dh, C=11, lr=0.01, lr_scale=2e-3),
              lambda: RMSCounterLinear(dh, dout, C=11, lr=0.01, lr_scale=2e-3),
              None)

print("curves (step: mse):")
print("  dense      ", [f"{s}:{v:.3f}" for s, v in dc])
print("  ternaryQAT ", [f"{s}:{v:.3f}" for s, v in qc])
print("  counter+RMS", [f"{s}:{v:.3f}" for s, v in cc])
print(f"\nFINAL  dense={dense:.5f}  ternary-QAT={qat:.5f}  counter+RMS={cnt:.5f}")
print(f"  QAT/dense    = {qat/dense:.1f}x   (cost of ternarization alone)")
print(f"  counter/dense= {cnt/dense:.1f}x")
print(f"  counter/QAT  = {cnt/qat:.2f}x   (pure counter-optimizer cost on this task)")
