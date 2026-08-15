# Research Decisions

## D-001 — Do not scale before composition / learned geometry

**Decision:** parameter count is not the primary next milestone.

**Reason:** controlled gates already show that small systems can bind, compose
and compute when the state geometry and operator law are right.  Scaling cannot
distinguish a reusable law from a larger transition table.

**Revisit when:** a learned (not hand-designed) chart survives generated-value
recurrence and production transfer.

## D-002 — Defer latent critic

**Decision:** do not implement a self-critic as the next feature.

**Reason:** the central unresolved problem is still learning a stable reusable
transition system.  A critic adds degrees of freedom before the internal machine
state is learned robustly.

**Revisit when:** learned atlas + adaptive execution are stable and branching
trajectories exist.

## D-003 — Separate semantic identity from role

**Decision:** role and identity are distinct architectural axes.

**Reason:** unrelated `KEY_B` and `VALUE_B` embeddings prevent a correct payload
from automatically becoming a valid next address.

## D-004 — Preserve negative evidence

Failed seeds, interventions, wrong-chart controls and over-capacity memorization
arms are first-class results.

## D-005 — Treat intermediate canonicality as an empirical claim

**Decision:** never infer "the model reasoned through canonical states" from
terminal accuracy alone.

**Reason:** EXP-003 constructs exact order-2/order-3 latent gauge cycles that are
perfect at supervised terminal depths and wrong at intermediate hops.

**Consequence:** recurrent experiments must report intermediate probes / geometry
or use constraints that make coordinate reuse identifiable.

## D-006 — Prefer shared laws over finite transition tables

**Decision:** new recurrent modules should expose state-independent transition
structure wherever the task suggests one.

**Reason:** EXP-006 shows that a dense recurrent PC matrix can be perfect on the
observed prefix yet fail immediately on unseen state positions.  Temporal weight
sharing alone is not algorithmic generalization.

## D-007 — Co-design geometry and operator bias

**Decision:** evaluate a representation together with the complexity class of
operators acting on it.

**Reason:** EXP-004 shows both failure modes:

- random geometry + large capacity memorizes;
- good Fourier geometry + over-general full operator still under-generalizes;
- good geometry + correct local operator law generalizes exactly.

## D-008 — Multi-chart is promising but chart switching is a separate module

**Decision:** do not simply concatenate several operator-specific charts with an
unconstrained dense bridge and call the problem solved.

**Reason:** EXP-008/009 show strong operator specificity and poor held-out
identity generalization for linear chart bridges.

**Current design candidates:** canonical identity bus, explicit chart-transition
operators, or jointly updated redundant views with cycle/equivariance losses.

## D-009 — Recurrent correctness requires geometric stability, not top-1 alone

**Decision:** every reusable latent transition should report a closure-defect
metric and a local perturbation-gain diagnostic in addition to one-step accuracy.

**Reason:** EXP-014 flexible seed 22 is 100% on the complete one-step binary law
with target cosine ~0.996, yet falls to 0% by depth 64.  The instability is
visible early as non-zero closure angle and perturbation gain around/above one.

## D-010 — Prefer laws by construction when recurrence depends on them

**Decision:** when a small structural operator family can guarantee closure,
associativity, norm preservation or state sharing, compare it directly against
an unconstrained learned operator rather than relying only on regularization.

**Reason:** EXP-012/013 admit off-manifold algebras; EXP-014 structured-normed
operators are 3/3 seed stable through depth 64.

## D-011 — Match algebraic constraints to the representation parameterization

**Decision:** do not add symbolic/algebraic penalties in isolation.  Ask whether
the model parameterization makes the penalized law global over the states it
controls.

**Reason:** a root/closure penalty did not stabilize a free learned codebook in
EXP-010.  The same kind of closure becomes decisive in EXP-016 after all
identities are parameterized as one generator orbit `E(x)=T^xE(0)`; orbit+closure
is 5/5 seed stable.

## D-012 — Revise the multi-chart decision

**Previous view:** operator-specific local charts suggested multi-chart registers
plus a learned bridge.

**Revised decision:** treat multi-chart as one compression option, not the
default assumption.

**Reason:** EXP-019 shows that a full additive-character representation supports
both ADD and MUL exactly in one register space when the operator grammar uses
separate motifs: local complex product for ADD and frequency permutation for
MUL.  No CAST is needed through mixed depth 256.

**Consequence:** future comparisons must include both:

1. compact specialized charts + explicit CAST;
2. wider operator-invariant shared representations + sparse action motifs.

## D-013 — Think in operator orbits when allocating latent width

**Decision:** measure how a proposed basis expands under the operator family.

**Reason:** EXP-020 shows an exact controlled relation: a multiplier subgroup of
size `q` has a frequency orbit of size `q`, and real width `2q` supports exact
mixed ADD/MUL programs.  A multiplier outside the subgroup immediately leaves
that frequency orbit.

**Consequence:** latent width is partly an invariance/closure budget, not only an
information-storage budget.

## D-014 — Separate operator induction from operator execution

**Decision:** report router and executor accuracy separately.

**Reason:** EXP-015 had a router that was exact on non-ambiguous demonstrations
while one MUL executor seed was poor.  EXP-017 repaired the executor without
changing the induction mechanism.  EXP-018 then combined inference and recurrent
execution successfully.

## D-015 — Distinguish transition-fit capacity from representation capacity

**Decision:** dimension sweeps must evaluate the *generated operator algebra*,
not only the locally supervised generator edges.

**Reason:** EXP-021/022 repeatedly obtain 100% top-1 on both `x->x+1` and
`x->3x` in dimensions below 30 while arbitrary powers/mixed programs remain near
chance.  Local edge fit dramatically underestimates the representation width
needed by the joint operator family.

## D-016 — Use isotropy/rank constraints when learning invariant state spaces

**Decision:** when a target operator family is expected to occupy a full
invariant component, monitor effective rank and test a centered/isotropic
geometry constraint before adding more task labels.

**Reason:** d=30 in EXP-021 collapsed to effective rank ~21.7.  EXP-022 fills the
30D subspace using only unlabeled geometry and immediately recovers exact powers,
semidirect composition and mixed programs.

## D-017 — Treat representation-theoretic dimension as an architectural budget

**Decision:** for shared linear/sparse recurrent actions, derive the smallest
common invariant representation when possible and compare model width against
that value.

**Reason:** NOTE-007 proves a 30D requirement for exact linear equivariance of
the tested affine generators; EXP-022 shows a sharp experimental 29->30 jump on
three fresh boundary seeds.

## D-018 — Treat dense operator weights as gauge-dependent evidence

**Decision:** do not infer computational complexity from matrix density in an
arbitrary learned latent gauge.

**Reason:** EXP-023 converts apparently dense d=30 actions to diagonal/monomial
motifs using only a recovered spectral gauge, with no loss through depth 64.

## D-019 — Add a structural compile step before recurrent deployment

**Decision:** for operator families with identifiable laws, compare raw learned
execution against a compiled/projection path.

**Reason:** EXP-024 shows motif projection can denoise long-horizon dynamics;
EXP-025 can infer the finite-order/conjugacy law from dense matrices before
compilation.

**Boundary:** do not project onto a family unless its law passes a fixed
acceptance residual; d=29 controls are rejected.

## D-020 — Minimize the learned operator library by discovered relations

**Decision:** redundant operators should be expressed as compositions of
primitive learned generators when matrix relations are validated.

**Reason:** EXP-026 removes S5/S7 after discovering S5~=S3^20 and S7~=S3^28,
without changing depth-64 execution.

## D-021 — Budget transition data against residual completion freedom

**Decision:** reason about action sample complexity through the dimension of the
unconstrained complement, not only through example count.

**Reason:** NOTE-009 and EXP-027/028 show that k known directions for an
orthogonal d-dimensional operator leave O(d-k) freedom. Structural motifs can
resolve the final discrete d-1 orientation bit, but not a generic continuous
O(2) ambiguity at d-2.

## D-022 — Canonicalize same-algebra modules before learning a CAST

**Decision:** if two modules realize the same identifiable operator algebra,
first attempt invariant/spectral gauge canonicalization and only learn a bridge
for residual semantic differences.

**Reason:** EXP-029 transfers identities and mid-program recurrent states across
independent seeds at 100% with zero trained bridge parameters.

**Boundary:** a simple-spectrum anchor is no longer required after EXP-030/031.
Canonical interoperability for repeated/more general joint blocks still needs a
shared structural anchor inside the recovered joint algebra.

## D-023 — Search the generated operator algebra for a canonical anchor

**Decision:** a structural compiler should be allowed to synthesize candidate
operators by composition rather than requiring one learned primitive to have a
simple spectrum.

**Reason:** EXP-030 starts with order-10 and order-3 degenerate operators but
finds `T=BC`, an order-30 simple-spectrum action, then recompiles both originals
as powers of T with 100% depth-64 execution.

**Consequence:** spectral degeneracy of individual modules is not by itself a
reason to abandon canonicalization; first inspect the closure of their grammar.

## D-024 — Use the commutant before declaring spectral canonicalization impossible

**Decision:** if every learned operator/composition has repeated spectrum, inspect
the *joint commutant* and its multiplicity structure before introducing an
arbitrary learned CAST.

**Reason:** EXP-031 recovers repeated S3 irreducible copies from commutant
dimensions 4/9/16 and compresses them to one shared block with exact depth-256
execution.

## D-025 — Permit compilation from traces, but require an identification gap

**Decision:** production structural analysis may begin from logged hidden-state
transition pairs rather than direct action matrices, but downstream structural
claims require a fixed singular-gap/residual gate.

**Reason:** EXP-033 compiles noisy trajectory-only estimates to ~100% at `2d`
probes in the tested system, while the `d`-probe regime is not reliably
identifiable and usually triggers abstention.

## D-026 — Treat local Jacobian fields as gauge data and use loop holonomy as evidence

**Decision:** when a recurrent operator is state-conditioned, do not force one
global matrix.  First test a local-gauge factorization and use closed-loop
holonomy to decide whether a stronger recurrent law can be projected safely.

**Reason:** EXP-034 has no good observed global action matrix.  Gauge
synchronization plus accepted cycle closure restores 100% depth-256 tracking up
to 10% Jacobian noise on all tested seeds.  At the one 15% arm whose holonomy
exceeds the fixed threshold, the compiler abstains instead of forcing the law.

## D-027 — Instrument nonlinear recurrence as a black box before hard-coding operators

**Decision:** for production transfer, first attempt active hidden-state probing
(orbit/neighborhood discovery + local Jacobian/JVP estimates) before adding a
new explicit operator module to the architecture.

**Reason:** EXP-035 recovers a stable operator grammar from a nonlinear black-box
map without semantic coordinates, a supplied graph, or precomputed Jacobians.
All 12 main arms through 10% Jacobian noise compile to 100% depth-256 tangent
tracking.

**Boundary:** the current gate is a finite 2D conformal orbit.  Production use
requires continuous-manifold clustering and low-rank derivative sketches.

## D-028 — Use economical JVP evidence rather than materialized production Jacobians

**Decision:** expose a one-step recurrent transition API and use randomized
JVP/VJP sketches as the default high-dimensional structural evidence gate.

**Reason:** EXP-036 recovers the correct finite-order/conjugacy relations through
`d_model=320`, and the real attention/backbone transition is probeable without
constructing an `O(d^2)` Jacobian.

## D-029 — Treat operator choice as a finite grammar, not a soft mixture

**Decision:** `register_machine_v3` uses hard forward operator selection.  During
training the router uses a straight-through one-hot estimator; inference uses
true argmax selection.

**Reason:** EXP-037 showed that a soft convex mixture of individually legal
operators drifts off the recurrent manifold.  Hard routing with a legal
`BLOCK_PRODUCT` primitive keeps generated states canonical and exact through OOD
depth.

## D-030 — Start v3 from the query register; make exact binding a READ candidate

**Decision:** slot 0 initially contains the exact query/value identity.  The
query-conditioned binder produces a separate addressed payload candidate; the
operator grammar may READ it or combine it with the current value.

**Reason:** copying the first payload into the value register *before* the first
machine tick destroys the original operand/state needed for generated-value
computation.  The new semantics preserves v2 lookup through READ while allowing
computation from tick one.

## D-031 — Freeze architecture at the model-readiness gate; move uncertainty to training

**Decision:** after EXP-036/037, do not add another architectural mechanism
without new falsifying evidence.  Build and train the 10.245M v3 preset first.

**Reason:** instantiate/backward/JVP/checkpoint/reload and generated-value gates
now pass.  The largest remaining uncertainty is semantic training, not whether a
recurrent register machine can be constructed.
