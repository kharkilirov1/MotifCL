# Current Research Status

Updated: 2026-08-14

## Evidence ladder

### E2 — exact latent binding (released v2)

`query_bound_v2` established exact protected one-hop binding under controlled
lookup, including the full 10M token/backbone gate and causal interventions.

Interpretation: continuous latent capacity itself is not the main bottleneck.
Addressing, coordinate compatibility, state preservation and readout are.

### E3 — recurrent composition and generated latent computation

Controlled results now establish all of the following:

- **EXP-001:** a protected payload reused as the next address composes unseen
  permutations exactly; validation remains exact through depth 64.
- **EXP-003:** terminal-only supervision can hide exact periodic latent gauges;
  intermediate canonicality cannot be inferred from terminal accuracy.
- **EXP-004/005:** a learned one-step ALU can generalize to held-out operand
  pairs and feed its own continuous generated state through depth 32.
- **EXP-006/007:** separate latent value/PC registers can execute OOD-length
  unique-HALT programs, with causal separation of data and control failures.

### E3 — learned geometry and recurrent stability

- **EXP-010:** a latent chart can be learned from a local successor cycle rather
  than supplied as a fixed Fourier codebook; one-step generalization can still
  coexist with deep recurrent drift.
- **EXP-011:** multi-depth consistency on the *same local law* stabilizes the
  learned chart: 3/3 seeds give 100% unseen binary generalization and 100%
  recurrence through depth 64.
- **EXP-012:** jointly learning both chart and a flexible local bilinear operator
  is optimization-unstable.  A bad seed can be almost associative/commutative
  in continuous space while its outputs lie far off the canonical code manifold.
- **EXP-013:** inserting a soft codebook canonicalizer does **not** repair this.
  It moves states closer to the manifold geometrically without identifying the
  correct semantic branch.
- **EXP-014:** law-by-construction fixes the failure.  A normed structured
  operator with learned coordinates gives 100% binary law and 100% depth-64
  recurrence on 3/3 new seeds.  Matched flexible-penalty arms remain unstable;
  one arm is 100% one-step yet falls to 0% at depth 64.
- **stability diagnostic:** closure defect plus local perturbation gain exposes
  the recurrent failure before top-1 accuracy collapses.

### E3 — finite operator grammar / operator induction

- **EXP-015:** latent consistency can infer whether demonstrations implement ADD
  or MUL without an explicit operation label.  On semantically non-ambiguous
  episodes routing is exact; one free-codebook executor remained seed-unstable.
- **EXP-016:** parameterizing identities as one generator orbit removes the
  finite-table representation, but orbit sharing alone has hard periodic minima.
  Adding the *matching global closure* `T^n=I` makes the same construction 5/5
  seed stable: 100% binary law and 100% depth-64 recurrence.
- **EXP-017:** the robust orbit+closure modules plugged into the finite grammar
  give 100% routing and answering on every non-ambiguous episode across 3 seeds.
- **EXP-018:** after inferring the hidden operator from two demonstrations, the
  selected learned operator executes its own generated continuous state through
  depth 64 at 100% on 3/3 seeds.

### E3 — one invariant chart can support non-commuting operators

EXP-008 originally showed strong pressure toward operator-specific *local*
charts: additive Fourier + local law solves ADD but not MUL, and multiplicative
log-Fourier does the converse.  EXP-009 showed that a naive dense bridge between
those specialized charts does not generalize to held-out identities.

**EXP-019 revises the interpretation.**  In the full additive-character basis:

- `ADD` is a local complex-product motif;
- `MUL` is an operand-conditioned frequency-permutation motif;
- random mixed ADD/MUL programs stay exactly correct through depth 256 with no
  chart switch and no hard intermediate decode.

So multiple charts are an optional compression strategy, not a mathematical
necessity.  A sufficiently operator-invariant shared representation can avoid
CAST entirely if the grammar contains the correct action motifs.

**EXP-020:** if allowed multiplicative operators form a subgroup of size `q`, a
frequency orbit of size `q` is invariant and a real width `2q` supports exact
mixed programs through depth 128.  For the full multiplicative group over
`F_31`, the nonzero frequency orbit has size 30.

Interpretation: latent width can be governed by the **orbit of the representation
under the operator set**, not merely by the number of stored facts.

### E3 — learned invariant affine representation (EXP-021/022)

The invariant representation no longer has to be supplied as Fourier features.

- EXP-021 learns 31 identity codes plus only two shared linear generator actions
  `A:x->x+1` and `M:x->3x`.  Dimensions 8..30 can reach 100% top-1 on both
  trained generators while the generated affine grammar still fails near chance.
- EXP-022 adds only centered/isotropic code geometry.  On seeds 70/71/72, d=30
  learns the complete generated affine action and mixed programs at 100%; d=16
  and d=24 remain near chance despite 100% generator top-1.
- New boundary seeds 73/74/75 show the transition sharply: d=28/29 remain near
  chance, while **d=30 is exact on all three seeds**.
- NOTE-007 proves `d>=30` for an exact linear equivariant latent representation
  of these two affine generators that distinguishes all `F_31` states.

This is the strongest current evidence that operator grammar can determine a
minimum useful latent dimension independently of local classification fit.

### E3 — structural operator compiler and latent interoperability (EXP-023..035)

- **EXP-023:** a semantics-free A-spectral gauge turns learned dense d=30
  actions into diagonal/monomial motifs; hard-pruned depth-64 execution remains
  100% on 3/3 seeds, while d=29 controls fail.
- **EXP-024:** motif projection is a recurrent denoiser.  At 5% action-weight
  noise, dense depth-32 mean falls to 67.7%, support projection recovers 92.2%,
  and family-law projection returns to 100%.
- **EXP-025:** from dense matrices alone the compiler recovers order 31 and
  `M A M^-1 ~= A^3`; all d=30 runs pass a fixed residual threshold and compile
  to 100% depth-64 grammars, while d=29 is rejected.
- **EXP-026:** independently learned S3/S5/S7 actions compress to one primitive
  scaling generator; S5~=S3^20 and S7~=S3^28 are discovered automatically and
  the minimal library remains 100% at depth 64.
- **EXP-027/028:** transition supervision has a complement-dimension law.
  k=29 leaves an orientation bit that the motif compiler can repair; k=28 leaves
  continuous O(2) ambiguity and remains unresolved; k=30 is directly exact.
- **EXP-029:** independently trained models canonicalize to the same latent
  coordinates and exchange mid-program continuous states with 100% accuracy
  through a zero-parameter structural bridge.
- **EXP-030:** individual learned actions may both have degenerate spectra, yet
  the compiler can search their generated algebra and synthesize a new primitive
  simple-spectrum anchor.  Order-10/order-3 actions produce an order-30 anchor
  and recompile to 100% depth-64 execution on 3/3 d=30 seeds.
- **EXP-031:** repeated joint irreducible blocks defeat every simple-spectrum
  composition, but the joint commutant exposes multiplicity `m` through an
  `m^2`-dimensional algebra.  Intertwiner alignment compresses all copies to one
  shared 2x2 grammar and keeps depth-256 execution at 100% on 9/9 runs.
- **EXP-032:** approximate commutant structure survives 3--15% action noise.
  Shared-block averaging raises mean depth-256 accuracy from 27.4% to 100% at
  5% noise and remains 96.8% on average at 15%; generic random pairs never pass
  the structural gap gate.
- **EXP-033:** no action matrix or identity codebook is required.  From noisy
  continuous transition pairs, `~2d` probes per operator were sufficient for the
  current ridge->commutant pipeline on 12/12 tested arms; compiled depth-256
  accuracy averaged 99.99%.
- **EXP-034:** even a single observed global matrix is unnecessary.  A
  state-conditioned local Jacobian field is gauge-synchronized into shared
  operators; when measured loop holonomy passes a fixed gate, cycle-law
  projection restores 100% depth-256 tracking through 10% Jacobian noise on all
  tested seeds.  At insufficient holonomy evidence the compiler abstains.
- **EXP-035:** the compiler starts from a genuinely nonlinear black-box hidden
  map.  From one hidden state it actively discovers a 14-state orbit and edge
  graph, estimates local Jacobians by finite differences, synchronizes their
  gauges, and compiles accepted loop laws.  All 12 main runs through 10%
  Jacobian noise recover **100% depth-256 tangent tracking**.

Interpretation: in this controlled linear regime, representation learning and
sparse motif discovery can be separated into a **neural learner + structural
compiler** pipeline.  Matrix density in the original gauge is not a reliable
measure of computational complexity.

### E4 — production-shaped generated-value register machine (EXP-036/037)

The controlled research line is now implemented in a buildable production-shaped
model contract rather than only stand-alone micro-gates.

- **EXP-036:** randomized JVPs verify finite-order/conjugacy laws without
  materializing full Jacobians at widths 128, 256 and the production
  `d_model=320`.  The real FOG attention/backbone transition is JVP-probeable
  through a probe-only math-SDPA fallback.
- **`register_machine_v3`:** typed value/control/scratch registers, shared
  recurrent transition, exact READ candidate, IDENTITY, parameter-free
  `BLOCK_PRODUCT`, four flexible low-rank bilinear operators, optional HALT,
  and a public `transition_memory()` structural-probe boundary.
- **EXP-037:** on a task where the next state is absent from prompt payloads,
  the full v3 cell must `READ operand -> generate value -> reuse generated
  value as next address`.  With straight-through hard operator routing, 3/3
  seeds are exact at depths 1,2,3,4,6,8 and the generated-state cosine stays
  above 0.9999979.
- **Model-readiness gate:** the 10,245,433-parameter reference preset can be
  instantiated, differentiated through the machine cell, JVP-probed through the
  real recurrent transition, checkpointed, and strict-reloaded.

Interpretation: **the architecture is now build-ready and research-training-ready.**
This is not yet E5 semantic reasoning: exact binding still uses structured
address/payload relations and the natural-language operator-induction problem
remains open.

## What is not yet established

- Decisive recurrent transfer of the released trained 10M `query_bound_v2`
  checkpoint (the binary is absent from the supplied archive and cannot be
  downloaded in the current runtime because outbound DNS is unavailable).
- End-to-end *online compilation* of a trained natural-language v3 backbone.
  EXP-036 closes the first high-dimensional JVP instrumentation step and the v3
  transition itself is probeable, but no semantically trained v3 checkpoint has
  yet been structurally compiled.
- A learned open-ended operator grammar rather than a finite candidate library.
- General mixed-operation reasoning outside controlled finite algebra.
- Learned adaptive halting (current HALT evidence uses an explicit instruction).
- Branching/search/self-correction and verifier learning.
- Natural-language / code / mathematical reasoning advantage over matched
  looping/text-CoT baselines.
- 100M+ scaling evidence.

## Current bottleneck

EXP-036 closes the first high-dimensional derivative-probing bottleneck, and
EXP-037 closes the first production-shaped generated-value register-machine gate.
The question has therefore moved from *can we build the machine?* to:

> **Can a built `register_machine_v3` learn semantic binding and operator
> induction from natural-language/code contexts while preserving the recurrent
> closure/stability properties established in controlled tasks?**

The strongest current design principles are:

1. identity must be reusable as the next register value/address;
2. terminal top-1 accuracy is not a recurrent-stability certificate;
3. measure closure defect and perturbation gain before long-horizon claims;
4. temporal weight sharing is insufficient — representation/action sharing must
   also be structural;
5. geometry and operator class are one coupled design choice;
6. algebraic penalties are weaker than architectural laws when the latter are
   available;
7. a regularizer is most effective when its parameterization makes the law
   global (EXP-016);
8. atlas switching and a wider invariant shared representation are competing
   design points with different capacity costs;
9. operator-family selection can be separated from operator execution;
10. repeated representation copies create a commutant signature and can act as
    structural error-correcting redundancy;
11. compilation should abstain when commutant/holonomy evidence does not cross a
    fixed structural gate.

## Production gap

`register_machine_v3` is now a separate opt-in production contract; legacy and
`query_bound_v2` semantics remain source/checkpoint compatible.  The reference
v3 preset is 10,245,433 parameters and passes the mechanical model-readiness
gate.  The released v2 trained-checkpoint recurrence transfer remains a parallel
historical gate, but it is no longer a blocker for constructing the next model.
The remaining gap is **training semantic capabilities into v3**, not constructing
its recurrent machinery.
