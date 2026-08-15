# Research Log

## 2026-08-14 — Workflow reset and composition hypothesis

- Inspected the released v2 repository and artifacts.
- Baseline core test suite: `python -m pytest -q tests binding_diagnostics_test.py` -> **60 passed, 2 skipped** in the current environment.
- Full bare `pytest` collection is not a valid baseline in this container because `tokenizers` is not installed; classify as `ENVIRONMENT`, not model failure.
- Identified production-v2 recurrence gap: each reasoning iteration uses the original lexical query for protected binding.
- Identified a second interface issue for multi-hop use: separate key/value token namespaces do not provide a canonical identity code that can be reused across roles.
- Froze EXP-001 protocol before locked-test access.

## 2026-08-14 — EXP-001 passed

- Only trainable parameter: address `logit_scale`; final scale 2.4493382.
- Validation recurrent accuracy: 100% at every depth 1–16; stress validation stayed 100% through depth 64.
- Locked test (1,024 examples): 100% recurrent accuracy at every depth 1–16.
- Static depth-2 accuracy: 12.99%; hop-2 corruption: 15.23% for later depths.
- Derived a symmetric one-dimensional fixed-point model predicting the observed soft-state mass/cosine almost exactly.
- Decision: proceed to EXP-002 production 10M recurrent token composition.

## 2026-08-14 — EXP-002 production recurrence implementation

- Added backward-compatible `binding_query_update = static | primary_recurrent` to `FOGReasonerConfig`; default remains `static` so old checkpoints keep old behavior.
- `FOGLatentReasoner.reason()` can now feed the previous protected primary into the next binding address while auxiliary workspace still receives contextual backbone state.
- Added canonical shared-state token serialization with a masked role gap so identical value/source identities do not create duplicate address candidates.
- Fresh 10M geometry smoke: recurrent 100% at R=1–4; static collapses after R=1; hop-2 corruption collapses later outputs.
- The supplied ZIP does not contain `checkpoints/`, so the decisive trained-checkpoint EXP-002 run remains pending. GitHub text tooling can see the historical binary in the repository tree but cannot materialize the binary checkpoint in this runtime.

## 2026-08-14 — EXP-003: periodic latent gauge identified

- Added a controlled generated-value computation over `Z_31` with no
  intermediate supervision.
- Depth-2-only terminal loss converged to `phi≈pi`: canonical only at even hops.
- Depth-3-only converged to `phi≈2pi/3`: canonical every third hop.
- Coprime `{2,3}` from the canonical basin converged to identity and stayed
  canonical through depth 16.
- Far-init `{2,3}` found a non-zero-loss local basin, separating identifiability
  from optimization.
- Added `NOTE-002` gcd gauge law.

## 2026-08-14 — EXP-004/005: operator-compatible generated computation

- Fourier/local addition: 100% held-out pair generalization across three splits.
- Random/full control: 100% train interpolation, 2.85% held-out (chance 3.23%).
- Fourier/full control: 100% train but only 56.67% held-out.
- Recurrent reuse of the learned Fourier/local ALU stayed 100% through depth 32
  while every runtime transition pair was held out.
- Conclusion: geometry and operator bias jointly determine reusable computation.

## 2026-08-14 — EXP-006/007: control plane and integrated latent machine

- Fourier/local PC successor learned from positions 0..3 generalized to all OOD
  program positions and lengths 5..10; unique HALT accuracy 100%.
- Dense/full PC controls memorized the prefix and got 0% OOD HALT.
- Integrated good ALU + good PC reached 100% value / 100% HALT on OOD programs.
- Value shuffle after hop 2 reduced final value to ~3.68% while preserving HALT.
- PC shift after hop 2 reduced value to ~2.95% and HALT to 0%.

## 2026-08-14 — EXP-008/009: multi-chart pressure and bridge failure

- Additive Fourier chart makes ADD local/exact but MUL local transfer is 0%.
- Multiplicative log-Fourier chart makes MUL local/exact but ADD local transfer is 0%.
- Linear chart bridges fit seen identities at 100% but recover only ~12% held-out
  identities; all-identity fit returns to 100%.
- Current priority moved to end-to-end learned latent atlas and chart-transition
  laws rather than critic/scaling.

## 2026-08-14 — EXP-010/011: learned chart and depth consistency

- A chart learned only from a closed successor cycle generalized to all unseen
  binary additions, but one seed drifted badly by depth 32.
- Direct scalar root regularization did not reliably fix the drift.
- Multi-depth consistency on the same successor law (depths 1/2/3) gave 3/3
  seeds 100% binary generalization and 100% recurrence through depth 64.

## 2026-08-14 — EXP-012/013: off-manifold joint algebra

- Joint chart + flexible local operator is not seed stable.
- Bad seed can have near-zero continuous associativity/commutativity while
  arbitrary pair outputs are far from every canonical identity.
- Soft codebook canonicalization moves outputs toward the manifold but does not
  restore the correct semantic branch; EXP-013 rejected this generic fix.

## 2026-08-14 — EXP-014: law by construction + stability gate

- Structured normed operator: 3/3 seeds, 100% binary law, 100% depth 64.
- Flexible matched seed 22: 100% one-step binary but 0% depth 64.
- Measured closure angle + local perturbation gain; these expose instability
  before top-1 accuracy fails.
- Added NOTE-005.

## 2026-08-14 — EXP-015..018: finite operator grammar

- Demonstration consistency identifies ADD vs MUL exactly on every
  non-ambiguous episode without an explicit operation label.
- A free-codebook MUL executor exposed a separate geometry failure despite
  correct routing.
- Generator-orbit parameterization alone introduced hard periodic minima.
- Adding the matching global orbit closure `T^n=I` made the representation 5/5
  seed stable; the same type of closure regularizer had been ineffective when
  identities were independent.
- Robust operator grammar: 3/3 seeds exact on all non-ambiguous episodes.
- Infer-then-recur: 3/3 seeds, 100% routing and answer through depth 64.

## 2026-08-14 — EXP-019: multi-chart hypothesis revised

- Derived `chi_k(x+y)=chi_k(x)chi_k(y)` and
  `chi_k(xy)=chi_{ky}(x)` in the full additive-character basis.
- Same register space supports ADD as a local product and MUL as a frequency
  permutation.
- Random mixed ADD/MUL programs remain exact through depth 256 without CAST.
- Wrong local MUL motif collapses to chance.
- Multi-chart is now treated as a compression option, not a necessity.

## 2026-08-14 — EXP-020: operator-orbit representation scaling

- For multiplier subgroups of sizes 1,2,3,5,6,10,15,30, a matching frequency
  orbit of size q is exactly invariant.
- Real width 2q runs mixed ADD/MUL-by-subgroup programs at 100% through depth128.
- Outside-subgroup scaling leaves the selected orbit immediately.
- Added NOTE-006 and IDEA-008.

## 2026-08-14 — production checkpoint access retry

- The released checkpoint binary is still not present in the supplied archive.
- Direct `curl` from this runtime cannot resolve `raw.githubusercontent.com`.
- EXP-002 remains pending for environmental artifact-access reasons, not because
  of a model failure.

## 2026-08-14 — EXP-021/022: learned invariant representation and 30D threshold

- Removed the hand-designed Fourier codebook entirely.
- Learned 31 identity codes plus only two shared linear actions from
  `x->x+1` and `x->3x` generator transitions.
- EXP-021: d=8..30 can fit both generator edges at 100% top-1 while generated
  affine programs stay near chance; d=31 nearly solves the full grammar.
- Mechanistic audit found d=30 code effective rank only ~21.7.
- EXP-022 added centered/isotropic code geometry, with no new semantic labels.
- Seeds 70/71/72: d16/d24 remain near chance on mixed programs, d30 is exact.
- Boundary seeds 73/74/75: d28/d29 remain near chance, d30 is exact 3/3.
- Added NOTE-007 proving d>=30 for exact linear equivariant codes distinguishing
  all F_31 states under the two affine generators.
- Current next bottleneck: learn sparse/local action structure inside the learned
  invariant representation rather than supplying the action motif.

## 2026-08-14 — EXP-023..025: structural operator compiler

- EXP-023: learned dense d=30 actions become diagonal/monomial in an
  automatically recovered A-spectral gauge; hard-pruned depth-64 execution stays
  100% on seeds 73/74/75. d=29 controls remain mixed and fail.
- EXP-024: spectral motif projection acts as a recurrent denoiser. At 5% weight
  noise, dense depth-32 mean falls to 67.7%, support projection recovers 92.2%,
  and support+family closure returns to 100% on 3/3 seeds.
- EXP-025: from dense matrices alone the compiler infers order 31 and
  `M A M^-1 ~= A^3`, rejects d=29, and compiles accepted d=30 runs to 100%
  depth-64 sparse grammars.
- Added IDEA-009 and NOTE-008.

## 2026-08-14 — EXP-026: automatic grammar compression

- Trained A,S3,S5,S7 as independent dense actions with no relation labels.
- Compiler chose S3 as order-30 primitive and recovered S5~=S3^20,
  S7~=S3^28 on new seeds 80/81/82.
- Removed redundant S5/S7 matrices; minimal {A,S3} grammar retained 100%
  depth-64 execution.
- d=29 controls were rejected by large relation residuals.

## 2026-08-14 — EXP-027/028: transition sample law

- Partial action supervision shows a spanning-set boundary in d=30.
- k=30 observed source states recovers the final unseen transition and full
  grammar on all seeds.
- k=29 leaves a discrete orientation ambiguity: seeds 90/91 choose the correct
  branch, seed 92 chooses det(M)~+1 and mixed execution falls to ~1.6%.
- The spectral motif compiler repairs that seed to 100% without new transition
  labels.
- k=28 leaves an O(2) continuous complement ambiguity and remains unresolved.
- Added NOTE-009.

## 2026-08-14 — EXP-029: canonical gauge interoperability

- Independently trained seed 73/74/75 representations canonicalize to codebooks
  with mean pairwise cosine >0.9999999.
- Structural bridge uses learned spectra plus E(0) phase anchor, with no trained
  bridge parameters or paired non-anchor states.
- All 31 identities transfer at 100% across every seed pair.
- 32 source recurrent steps -> structural bridge -> 32 target recurrent steps
  remains 100% on all three model pairs.
- Added NOTE-010.

## 2026-08-14 — EXP-030: degenerate operators synthesize their own anchor

- Trained only B:x->27x (order10) and C:x->25x (order3); neither has simple
  spectrum in d=30.
- Compiler searched B^a C^b and selected T=BC, discovered order30 with 30
  distinct spectral roots on seeds 100/101/102.
- Recovered B~=T^21 and C~=T^10; compiled depth64 programs remained 100%.
- Added NOTE-011 and D-023: search generated algebra, not only individual
  operators, for a canonical spectral anchor.

## 2026-08-14 — EXP-031: repeated joint irreducible blocks

- Built repeated 2D S3 irreps with multiplicity 2/3/4 behind random gauges.
- No searched operator word had more than two distinct eigenvalues, so a simple
  spectral anchor is impossible.
- Joint commutant dimensions were exactly 4/9/16; symmetric commutant separation
  plus intertwiner alignment recovered one shared block grammar.
- Depth-256 compiled execution was 100% on 9/9 runs.
- Broken-sharing controls produced commutant dimension 5 and were rejected.
- Added NOTE-012 and D-024.

## 2026-08-14 — EXP-032: approximate commutant denoising

- Replaced exact commutant nullspace by a fixed singular-gap criterion.
- All 36 repeated noisy arms (3--15% action noise) were structurally detected;
  generic random operator pairs were rejected.
- Shared-block averaging raised mean depth-256 accuracy to 100% at 3/5% noise,
  99.27% at 10%, and 96.77% at 15%.
- Multiplicity itself behaves as recurrent error-correcting redundancy.

## 2026-08-14 — EXP-033: trajectory-only structural recovery

- Removed privileged action-matrix and identity-codebook input from the compiler.
- Ridge system identification from noisy continuous hidden-state pairs followed
  by commutant compilation gives 99.99% mean depth-256 accuracy at ~2d probes per
  operator over the tested multiplicity/noise sweep.
- ~d probes are not reliably identifiable in the current two-stage pipeline.
- Added D-025.

## 2026-08-14 — EXP-034: state-conditioned local Jacobian field

- Removed the assumption that one global action matrix exists in observed
  coordinates.
- Gauge synchronization factors noisy local Jacobians into local charts plus
  shared O(2) actions.
- Observable cycle holonomy gates finite-order projection.
- All seeds through 10% Jacobian noise return to 100% depth-256 tracking after
  accepted projection.
- At seed142/15% holonomy residual 0.5355 exceeds the fixed 0.40 gate; compiler
  abstains and does not create a false success.
- Added NOTE-013 and D-026.

## 2026-08-14 — EXP-035: nonlinear black-box structural compilation

- Replaced supplied local-Jacobian field with a nonlinear black-box map
  `w=z+alpha*z^2` conjugating a dihedral latent action.
- Compiler receives only one starting hidden state and callable operators.
- Active exploration discovers 14 hidden orbit states and infers cycle graphs
  `[7,7]` and seven 2-cycles without semantic labels.
- Local Jacobians are estimated by finite differences; both black-box operators
  have nontrivial global-linear fit error (~0.161 and ~0.129).
- On seeds 150/151/152, structural compilation returns 100% depth-256 tangent
  tracking at 1/3/5/10% Jacobian noise (12/12 main arms).
- Exploratory seed150 high-noise sweep degrades at 15/20%; at 30% the 7-cycle
  holonomy gate rejects the law, demonstrating abstention rather than forced
  projection.
- Added NOTE-014, IDEA-010 and D-027.

## 2026-08-14 — EXP-036: high-dimensional JVP structural evidence

- Added an explicit one-step production transition API and randomized JVP
  relation/gain probes.
- Recovered `(order A=31, order M=30, conjugacy exponent=3)` at 128/256/320D
  without full Jacobians.
- Scoped structural probes to math-SDPA because CPU flash SDPA lacks forward AD;
  normal inference remains unchanged.

## 2026-08-14 — register_machine_v3 / EXP-037: model-side generated computation

- Added typed value/control/scratch registers and a finite operator bank to a
  new checkpoint-compatible architecture contract.
- Corrected first-tick semantics so value starts from query identity and exact
  binding is a separate READ candidate.
- Flexible bilinear operators alone did not yield stable recurrence; a
  law-compatible `BLOCK_PRODUCT` primitive did.
- Soft operator mixtures still drifted, motivating straight-through hard routing.
- With hard routing, 3/3 seeds achieve 100% at depths 1,2,3,4,6,8 with minimum
  mean canonical-state cosine >0.9999979.
- Added the 10,245,433-parameter reference preset and model builder.  The builder
  validates forward/backward, machine gradients, a real JVP transition probe,
  checkpoint save and exact strict reload.
- Milestone reached: **BUILD READY / RESEARCH TRAINING READY**.
