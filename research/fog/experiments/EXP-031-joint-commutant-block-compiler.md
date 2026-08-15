# EXP-031 — joint commutant block compiler

Status: **PASSED controlled**  
Date: 2026-08-14

## Question

Can the structural compiler recover a useful latent grammar when **no operator
and no short composition has a simple spectrum** because the representation
contains repeated copies of the same joint irreducible block?

## Construction

Use the real 2D standard irreducible representation of `S3`, repeated with
multiplicity `m in {2,3,4}` and hidden behind a random orthogonal gauge.

The compiler receives only the two dense action matrices.  It is not given the
hidden gauge, semantic codebook, multiplicity, or block width.

For an absolutely irreducible real block repeated `m` times, the joint
commutant should have dimension `m^2`.  The compiler:

1. solves `XA=AX` and `XB=BX`;
2. infers multiplicity from the commutant dimension;
3. diagonalizes a generic *symmetric commutant element* to split invariant
   copies;
4. aligns equivalent blocks using intertwiners;
5. stores one shared 2x2 operator block instead of one dense action per copy.

## Falsification control

Replace one of three repeated copies with an inequivalent rotation block.  A
clean `m=3` repeated-irrep claim must then be rejected.

## Locked protocol

- seeds: 110, 111, 112;
- multiplicities: 2, 3, 4;
- recurrent program depth: 256;
- 2,048 evaluation trajectories per run;
- short-word spectrum search verifies that a simple spectral anchor is absent.

Machine artifact: `artifacts/research/exp_031/metrics.json`.
