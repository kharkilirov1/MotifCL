# RESULT-012 — Geometric canonicalization does not identify semantic closure

EXP-013 failed on all three new seeds.

Representative metrics:

- seed 10: successor 100%, binary 18.83%, depth-32 2.93%;
- seed 11: successor 83.87%, binary 12.70%, depth-32 3.32%;
- seed 12: successor 67.74%, binary 13.01%, depth-32 2.73%.

The canonicalizer **did** move states closer to the learned codebook.  For
example seed 12 raw nearest-code cosine ~0.787 becomes ~0.979 after projection.
That geometric improvement does not recover the correct identity.

Conclusion:

> being near *some* canonical state is not enough; the transition law must make
> the correct semantic branch identifiable.

This rejects a generic "just add an attractor/codebook projection" fix for the
EXP-012 off-manifold failure.

Evidence: `artifacts/research/exp_013/metrics.json`.
