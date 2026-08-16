# Memory-native training, deep pass v2

This pass asks a stricter question than the first memory-truth patch:

> after eliminating dense `w` and dense `grad_w`, where is the next memory wall and what is the mathematical lower bound?

The answer is that the project has already removed the classic optimizer/gradient peak for counter
linear weights.  Further progress comes from three fronts:

1. **state bits**: can the synapse be smaller than 6 bits?
2. **activation contract**: can a layer update without storing the full input activation `x`?
3. **structure**: can we avoid storing an independent state for every dense matrix element?

---

## 1. Accounting model

For a GPT-style stack with linear parameter count

\[
N_{lin}\approx \alpha Ld^2,
\]

and vocabulary embedding count

\[
N_{emb}=Vd,
\]

a normal BF16 AdamW-style training state is roughly

\[
2N \;\text{params}+2N\;\text{grads}+4N\;\text{master}+4N\;m+4N\;v=16N\;\text{bytes}.
\]

The finite-state counter path is instead approximately

\[
M_{counter}=\frac{b_s}{8}N_{lin}+b_{emb}N_{emb}/8 + O(Ld),
\]

where `O(Ld)` is row scale/RMS state.  With fused update, there is no global gradient buffer and no
per-weight optimizer state:

\[
M_{grad}=M_{optim}=0.
\]

But the input activation needed for the update remains a separate question:

\[
\nabla W = \Delta^T X.
\]

A layer cannot form this correlation from `Delta` alone.  It must either store `X`, reconstruct `X`,
or use a lossy/unbiased representation `Q(X)`.

A helper was added:

```bash
python tools/memory_budget_calculator.py --layers 24 --d-model 2048 --seq 1024 --batch 1 \
  --state-bits 6 --counter-act-bits 4 --activation-policy-counter reversible
```

Example output for `L=24,d=2048,V=50257,seq=1024`:

| regime | estimated total |
|---|---:|
| BF16 AdamW-style | ~21.10 GiB |
| finite-state 6-bit + 4-bit/reversible activations | ~2.06 GiB |
| visible ternary entropy lower-bound | ~1.44 GiB |

This is not an allocator-level profiler; it is a transparent symbolic budget.  It is useful because it
shows what is still worth attacking.  For the above shape, after weight/gradient/optimizer elimination,
activation policy dominates the remaining *trainability* question more than another half-bit on the
weight state.

---

## 2. Can we go below 6 bits per synapse?

The current strong state family is

\[
t\in\{-1,0,+1\},\qquad c\in\{-(C-1),\ldots,C-1\}.
\]

The number of states is

\[
S=3(2C-1).
\]

So:

| C | states | logical bits |
|---:|---:|---:|
| 3 | 15 | 4 |
| 5 | 27 | 5 |
| 8 | 45 | 6 |
| 11 | 63 | 6 |

The information-theoretic minimum for a visible ternary weight alone is

\[
\log_2 3 \approx 1.585 \text{ bits/weight}.
\]

But training is not just storing a visible ternary matrix.  It also needs a residual/error-feedback
channel.  A counter synapse tracks the hidden real update

\[
\theta = s\left(t+\frac{c}{C}\right),
\]

with visible forward weight

\[
w=st.
\]

The residual error is bounded by

\[
|\theta-w| < s.
\]

Smaller C means coarser hidden residual and more update loss.  A new CPU ablation was added:

```bash
python benchmarks/python/bitbudget_counter_ablation.py --seeds 5 --steps 801 \
  --json-out results/bitbudget_counter_ablation.json
```

Observed on the small 16x16 ternary teacher recovery witness:

| C | states | bits | mode | mean final MSE |
|---:|---:|---:|---|---:|
| 3 | 15 | 4 | direct | 0.01787 |
| 5 | 27 | 5 | direct | 0.00785 |
| 8 | 45 | 6 | direct | 0.00209 |
| 11 | 63 | 6 | ternary | 0.00081 |

Interpretation:

* 4-bit and 5-bit states can learn, but they leave a larger residual floor.
* 6-bit remains the best fixed-state training point in this family.
* `C=11` is the cleanest 6-bit state: it uses 63 of 64 codes.
* Empirical entropy after training is lower than the logical bit width; this is useful for checkpoint
  compression, but not automatically for random-access GPU training.

### Practical conclusion

Do **not** replace 6-bit training state globally with 5-bit yet.  Better plan:

1. train early/mid phase with 6-bit `C=11`,
2. measure per-layer counter entropy and flip rate,
3. freeze or demote stable layers to 5-bit or visible ternary,
4. keep unstable layers at 6-bit.

This is a layerwise/statewise bit annealing schedule, not a single global precision.

---

## 3. Activation memory: the next real wall

The memory-native native layer still stores `Tensor x` in the backward node.  This is correct for ordinary
autograd because update needs

\[
\Delta^T X.
\]

But it means the next strict gate should be about activation storage, not dense gradients.

There are three mathematically valid contracts:

### A. Reconstruct X exactly or approximately

Use reversible blocks:

\[
y_1=x_1+F(x_2),\qquad y_2=x_2+G(y_1),
\]

then backward reconstructs

\[
x_2=y_2-G(y_1),\qquad x_1=y_1-F(x_2).
\]

The counter update is committed only after old-state recomputation is finished.

### B. Store a low-bit unbiased activation Q(X)

If stochastic activation quantization satisfies

\[
\mathbb E[Q(X)\mid X]=X,
\]

then

\[
\mathbb E[\Delta^T Q(X)\mid X,\Delta]=\Delta^T X.
\]

The update remains unbiased, with extra variance.  For uniform quantization step `h`, per-element error
variance is bounded by

\[
\operatorname{Var}(Q(x)-x)\le h^2/4.
\]

A new witness was added:

```bash
python benchmarks/python/activation_quantization_witness.py --seeds 3 --steps 1200 \
  --json-out results/activation_quantization_witness_long.json
```

Result on a 32x32 teacher recovery task:

| saved activation | effective bits/elem | median hit | mean final MSE |
|---|---:|---:|---:|
| fp | 16.00 | 945 | 0 |
| int8 | 8.50 | 909 | 0 |
| int4 | 4.50 | 932 | 0 |
| int3 | 3.50 | 932 | 0 |

This is only a witness, not an LLM result.  It says that **4-bit saved activations are a plausible
fallback when full reversibility is not yet available**.

### C. Store only a sketch of X

Use random projection/sketching:

\[
S\in\mathbb R^{r\times B},\qquad \widehat{\nabla W}=\Delta^T S^T S X.
\]

If

\[
\mathbb E[S^TS]=I,
\]

then the gradient estimate is unbiased.  This is a GRASS-like direction: optimize in a smaller subspace,
reduce gradient/optimizer memory, and never materialize full gradients.  It is probably better for
fine-tuning or adapter channels than for all from-scratch dense weights.

---

## 4. Row-scale/RMS memory

Current native state has:

```text
state: packed 6-bit per weight
scale: fp32 per output row
v: fp32 per output row
```

The `scale/v` cost is `O(rows)`, not `O(weights)`.  For large dense layers it is tiny:

\[
\frac{8\cdot out}{0.75\cdot out\cdot in}=\frac{10.67}{in}.
\]

At `in=2048`, this is about 0.52%.  Therefore changing row state from FP32+FP32 to FP16/log16+FP16/log16
is good hygiene, but not the main memory breakthrough.  It is worth doing for small layers and embeddings,
not because it changes the asymptotics.

The safer design is log-domain row stats:

\[
\ell_s=\operatorname{round}_{int16}(a\log s+b),\qquad
\ell_v=\operatorname{round}_{uint16}(a\log(v+\epsilon)+b).
\]

Why log? Because scale and RMS are positive and multiplicative error is more important than additive error.
A 16-bit log scale gives small relative error across a wide dynamic range.

---

## 5. Entropy coding versus training random access

After training, the state distribution is not uniform.  If the empirical entropy is

\[
H(Q)<6,
\]

then checkpoint/inference files can be entropy-compressed below 6 bits/state.

But during training the GPU needs fast random/block access.  Variable-length Huffman-like coding would
hurt kernels and atomics.  The practical split is:

* training: fixed 6-bit block packing;
* checkpoint: entropy compressed;
* inference: visible ternary pack at ~1.585 bits/weight, plus row scales.

---

## 6. The deeper way to save memory: stop using independent dense matrices

Even 6-bit state is still `O(d^2)`.  To go below that, we need structure:

\[
W x = \sum_{r=1}^R D_{r,0} H P_{r,1} D_{r,1} H P_{r,2} \cdots x + UV^Tx.
\]

Here:

* `H` is fixed Hadamard/FFT-like mixing,
* `P` is fixed or learned permutation,
* `D` is a trainable finite-state diagonal/block-diagonal operator,
* `UV^T` is a small low-rank escape channel.

This changes memory from

\[
O(d^2)
\]

to roughly

\[
O(Rd + rd),
\]

at the cost of expressivity.  This is not just compression; it is a different architecture.  It should be
tested as an optional block, not forced into all layers.

A practical hybrid block:

```text
local attention / SSM recurrence
+ structured finite-state mixer
+ small dense finite-state escape matrix
+ low-rank adapter channel
```

The dense escape channel is important.  Pure structured matrices often fail because they impose the wrong
inductive bias.  A small dense escape preserves universality while still making most memory subquadratic.

---

## 7. New gates added in this pass

### `tools/memory_budget_calculator.py`

Symbolic budget for BF16 AdamW versus finite-state/reversible/low-bit activation training.

### `benchmarks/python/bitbudget_counter_ablation.py`

Tests 4/5/6-bit finite-state families on discrete teacher recovery.

### `benchmarks/python/activation_quantization_witness.py`

Tests whether low-bit stored activations can still support the counter update.

### `tools/activation_memory_gate.py`

Static warning gate: reports that the current native layer still stores `Tensor x` in `CounterBackwardNode`.
It is not a failure for ordinary non-reversible training, but it marks the next frontier.

---

## 8. Recommended next engineering order

1. Keep `C=11`, packed 6-bit, as the default training state.
2. Add layerwise bit annealing: 6-bit active layers, 5-bit stable layers, visible ternary frozen layers.
3. Implement optional 4-bit stochastic activation save for non-reversible blocks.
4. Build a real reversible counter block where `CounterBackwardNode` does not own `Tensor x`.
5. Quantize row scale/RMS to log16 after correctness is stable.
6. Add entropy-compressed checkpoint export.
7. Start structured-mixer experiments only after the dense finite-state baseline has a real corpus parity test.

The deepest conclusion is:

> after gradient/optimizer elimination, memory is no longer primarily an optimizer problem.  It becomes an information-flow problem: what minimal information must survive from forward to backward to compute useful correlations?
