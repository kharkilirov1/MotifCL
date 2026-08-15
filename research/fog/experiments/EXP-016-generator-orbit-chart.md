# EXP-016 — Generator-orbit chart and matching closure

Status: **PASSED WITH MATCHED CLOSURE**  
Date: 2026-08-14

## Question

Can finite-table identity parameters be removed by generating every code from
one transition law?

Compare:

1. `free_codebook`: independent phase vector per identity;
2. `generator_orbit`: `E(x)=T^x E(0)`;
3. `generator_orbit_closure`: the same orbit plus the global law `T^n=I` and
   identity consistency.

Only successor terminal depths 1/2/3 are labeled.
