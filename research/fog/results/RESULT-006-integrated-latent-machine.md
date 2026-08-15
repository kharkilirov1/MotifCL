# RESULT-006 — Data and control registers compose into an OOD latent machine

Across three independent ALU pair splits, OOD programs of length 5–10 gave:

| arm | final value | HALT step |
|---|---:|---:|
| Fourier/local ALU + Fourier/local PC | **100%** | **100%** |
| Fourier/full ALU + good PC | 0.13% | **100%** |
| good ALU + Fourier/full PC | 3.29% | **0%** |
| random/full ALU + good PC | 1.74% | **100%** |
| good machine, value shuffled after hop 2 | 3.68% | **100%** |
| good machine, PC shifted after hop 2 | 2.95% | **0%** |

Chance final value is `1/31 = 3.23%`.

This cleanly factorizes failure modes:

- corrupt/wrong **data path** destroys the answer but leaves control/HALT intact;
- corrupt/wrong **control path** destroys HALT and therefore the program trace;
- only the joint structured machine generalizes both dimensions simultaneously.

The integrated positive arm satisfies all of the following:

- generated latent values are reused without intermediate decoding;
- every intended arithmetic transition is OOD relative to one-step ALU fitting;
- PC positions beyond the fitted prefix are OOD;
- program lengths 5–10 are OOD;
- HALT is read from instruction memory, not supplied as external `R`.

Limit: modules are learned/fitted separately in controlled algebraic charts, not
end-to-end from language or terminal-only task loss.

Evidence: `artifacts/research/exp_007/metrics.json`.
