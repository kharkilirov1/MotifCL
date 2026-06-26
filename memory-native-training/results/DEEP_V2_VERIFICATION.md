# Deep-v2 verification (reproduced on CPU)

The deep-research-v2 pass shipped four new gates/ablations with result tables. I re-ran them
on CPU (torch 2.12) rather than trusting the quoted numbers. Verdict: **everything reproduces.**

| claim (deep-v2 doc) | doc value | my CPU re-run | verdict |
|---|---|---|---|
| Memory budget, L=24/d=2048/seq=1024, 6-bit + 4-bit reversible acts | 21.10 / 2.06 / 1.44 GiB | 21.096 / 2.063 / 1.438 | ✅ exact |
| Bit-budget MSE, C=3 / 5 / 8 (direct, 801 steps, 5 seeds) | 0.01787 / 0.00785 / 0.00209 | 0.01787 / 0.00785 / 0.00209 | ✅ exact |
| Bit-budget, C=8/11 reach exact recovery | acc→1.0 | acc 1.0, hit ~767–783 | ✅ |
| Activation int8 / int4 / int3 ≈ fp (unbiased Q(X)) | all ≈ fp | int4 0.0414 ≈ fp 0.0411 | ✅ confirmed |
| Activation-memory gate: node still stores `Tensor x` | WARN | WARN | ✅ |

Reproduce:
```bash
python tools/memory_budget_calculator.py --layers 24 --d-model 2048 --seq 1024 --batch 1 \
  --state-bits 6 --counter-act-bits 4
python benchmarks/python/bitbudget_counter_ablation.py --modes direct --seeds 5 --steps 801
python benchmarks/python/activation_quantization_witness.py --seeds 3 --steps 1200
python tools/activation_memory_gate.py
```

## One reproducibility caveat
The bit-budget MSE numbers are **step-count sensitive**: at 600 steps the same ordering holds
but the absolute floor is ~10× higher (nothing has converged yet). The doc's specific numbers
require its 801 steps (and the activation witness's "MSE 0" needs the 1200-step long run). The
*scientific conclusions* (more bits → lower residual; 6-bit best; low-bit unbiased activations
≈ fp) are robust to step count; the headline decimals are not. Pin steps/seeds when quoting.

## Note on engine lineage
These tools were authored against a **more advanced engine** than this branch: in the deep-v2
`compact_counter.cpp` the backward computes `grad_x` directly from packed state
(`backward_input_from_state`) and updates from `(x, grad_out)`, so it keeps **no transient
dense weight at all** (this branch still decodes a short-lived `w` for `grad_x`). The deep-v2
`tools/memory_truth_gate.py` is paired with that stricter engine and is therefore **not
vendored here** — adopting it requires adopting the deep-v2 engine source (an open decision;
see chat). The verified tools above are engine-agnostic (or only WARN) and run clean on this
branch. The accessible PyTorch port realizes the same ideas independently: `act_save_bits`
(unbiased low-bit saved activation) and `memory_native.training_budget` (the budget model).
