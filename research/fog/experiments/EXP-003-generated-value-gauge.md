# EXP-003 — Generated-value gauge identifiability

Status: **PASSED (controlled mechanistic gate)**  
Date: 2026-08-14

## Question

If a recurrent latent machine is supervised only at selected terminal depths,
does a correct final answer force every intermediate latent register to live in
the same canonical coordinate system?

## Construction

Values live in `Z_31` and are represented by a frozen 30-dimensional Fourier
codebook.  An exact latent addition operator creates `E(x+b)` directly; the
result need not have appeared in the input.  After every transition a shared
trainable gauge rotation `G_phi` is applied to every Fourier plane.

No intermediate hop is decoded or supervised.  The loss only aligns the final
latent register with the canonical terminal target.

Because semantic addition commutes with `G_phi`, supervision at depth `d`
constrains only `G_phi^d`.

## Arms

1. terminal depth `{2}` only;
2. terminal depth `{3}` only;
3. coprime terminal depths `{2,3}`, initialized inside the canonical basin;
4. the same `{2,3}` constraints from a deliberately far/noncanonical init.

Three seeds are run for every arm.

## Falsifiable expectations

- `{2}` admits a noncanonical exact order-2 gauge;
- `{3}` admits a noncanonical exact order-3 gauge;
- exact simultaneous satisfaction of `{2,3}` admits only identity in this
  scalar gauge family because `gcd(2,3)=1`;
- `gcd=1` is an identifiability statement, not a guarantee that gradient
  optimization has no bad local minima.

## Reproduction

```bash
python generated_value_gauge_experiment.py \
  --output artifacts/research/exp_003/metrics.json \
  --steps 300 --batch-size 128 --eval-examples 1024 \
  --eval-max-depth 16 --seeds 0 1 2
```
