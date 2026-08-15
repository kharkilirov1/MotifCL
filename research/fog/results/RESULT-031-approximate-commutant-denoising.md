# RESULT-031 — repeated operator redundancy acts as structural error correction

EXP-032 passed the controlled noisy sweep.

All 36 repeated-irrep arms passed the fixed approximate-commutant gap gate.
Generic random operator-pair controls passed it **0 times**.

Mean depth-256 accuracy:

| relative action noise | raw noisy | shared-block compiled |
|---:|---:|---:|
| 3% | 50.3% | **100.0%** |
| 5% | 27.4% | **100.0%** |
| 10% | 37.9% | **99.27%** |
| 15% | 24.3% | **96.77%** |

Multiplicity 3 and 4 were especially robust: at 15% noise all six runs returned
to 100% after shared-block averaging.  Multiplicity 2 has less redundancy and
was the main source of the remaining errors.

The optional polar projection changed little in this sweep.  The dominant gain
therefore came from **discovering that several latent subspaces should implement
the same operator and averaging them**, not from an S3-specific law.

## Interpretation

Repeated latent modules can serve as a structural error-correcting code for
recurrent dynamics.  Approximate symmetry is visible as a low-singular-value
cluster in the commutator operator even after raw long-horizon behavior has
already degraded badly.
