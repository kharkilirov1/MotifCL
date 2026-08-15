# RESULT-018 — A shared invariant chart eliminates CAST for ADD/MUL

One-step:

- ADD motif: **100%**;
- MUL frequency-permutation motif: **100%**;
- wrong control that applies the ADD-local motif to MUL: 3.12% (chance-like).

Random mixed programs:

- 3 seeds;
- depth 1,2,4,8,16,32,64,128,256;
- correct operator grammar: **100% at every depth** with target cosine 1;
- no chart switches and no intermediate hard snap.

The wrong-local-MUL control falls toward chance by depth 8–32.

Interpretation:

> EXP-008 demonstrates operator-specific **locality**, not a theorem that
> non-commuting operations require separate charts.  A shared representation
> closed under the full operator action can trade additional width for simpler
> chart management.

Evidence: `artifacts/research/exp_019/metrics.json`.
