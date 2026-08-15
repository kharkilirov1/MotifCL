# FOG Research Roadmap

Ordered by dependency and information gain.

## M1 — Exact binding (PASSED)

Released `query_bound_v2` one-hop evidence.

## M2 — Recurrent binding composition (PASSED controlled; production pending)

- EXP-001: passed.
- EXP-002 trained 10M transfer: pending checkpoint binary availability.

## M3 — Gauge-stable recurrent registers (PASSED controlled / diagnostic ongoing)

Established:

- periodic hidden coordinate gauges under terminal-only loss (EXP-003);
- stable soft attractor dynamics (NOTE-001);
- multi-depth consistency can stabilize a learned chart (EXP-011);
- closure-defect / local-gain diagnostic predicts recurrent fragility before
  top-1 failure.

Next: port the stability diagnostic to production latent transitions.

## M4 — Generated-value computation (PASSED controlled)

EXP-004/005 establish learned generated continuous values reused on entirely
held-out transitions.

## M5 — Latent control plane (PASSED controlled micro-gates)

EXP-006/007 establish value register, PC register, instruction memory, explicit
HALT and OOD program-length execution.

Still missing: terminal-only learned control and adaptive/dynamics-based halt.

## M6 — Learned representation × operator law (CURRENT CORE RESEARCH)

### M6.1 Learned chart (PASSED controlled)

EXP-010/011 show that a chart can emerge from a local transition law and become
stable under recurrence without a fixed Fourier codebook.

### M6.2 Joint chart/operator learning (PARTIAL)

EXP-012 finds off-manifold continuous algebras; EXP-013 shows that a generic
soft canonicalizer is not enough.  EXP-014 shows law-by-construction is robust.

Current goal: learn richer law-by-construction families or discover their
parameters/router from data without hard-coding task semantics.

### M6.3 Representation closure under an operator grammar (NEW PRIORITY)

EXP-019/020 introduce a stronger alternative to mandatory multi-chart routing:

- choose a shared representation closed under the operator family;
- implement each operator by its sparse action motif;
- representation width follows operator-orbit size.

EXP-021/022 now close the first item in a linear controlled setting: a 30D
invariant affine representation is learned from only two generator transition
laws, and NOTE-007 predicts the observed 30D threshold.

Controlled follow-up is now passed through EXP-029:

1. EXP-023 discovers diagonal/monomial sparse actions by canonical gauge;
2. EXP-024 projects noisy learned actions back onto recurrently stable families;
3. EXP-025 infers finite order and conjugacy law without semantic metadata;
4. EXP-026 compresses redundant learned actions to primitive generators;
5. EXP-027/028 characterize partial-transition sample complexity and structural
   completion;
6. EXP-029 aligns independently trained modules through canonical gauge rather
   than a learned CAST.

### M6.4 Structural compiler beyond clean linear actions (CURRENT CORE)

Now passed in controlled stages:

1. **EXP-030:** individually degenerate actions can synthesize a simple anchor
   from their generated operator algebra;
2. **EXP-031:** when *no composition* can have simple spectrum because of
   repeated joint irreducible blocks, the commutant recovers multiplicity and a
   shared block grammar;
3. **EXP-032:** approximate commutant singular gaps survive noisy actions and
   repeated-block averaging denoises long recurrent execution;
4. **EXP-033:** the same compiler can start from noisy hidden-state trajectory
   pairs rather than privileged action matrices or a semantic codebook;
5. **EXP-034:** a state-conditioned field of local Jacobians can be factorized
   by gauge synchronization, and observable loop holonomy can gate a stronger
   recurrent law projection;
6. **EXP-035:** from one hidden state, a nonlinear black-box recurrent map can be
   actively explored to infer its finite context graph and finite-difference
   Jacobians before the same gauge/holonomy compiler is applied.

Next experiments:

1. replace the finite orbit by a continuous hidden-state cloud with unknown
   neighborhood count / manifold structure;
2. replace full Jacobians by low-rank JVP/VJP sketches suitable for large hidden
   dimensions;
3. general high-dimensional/non-orthogonal local gauges and repeated joint
   irreps with real/complex/quaternionic commutants;
4. online/differentiable motif compilation during training rather than post-hoc;
5. compare universal invariant basis vs specialized atlas + structural CAST on
   parameter count, sample complexity and recurrent stability;
6. port closure/gain, commutant-gap and holonomy diagnostics to production
   `query_bound_v2`.

## M7 — Finite operator grammar induction (PASSED controlled, expand next)

EXP-017/018 show:

- infer operator family from demonstrations without an explicit label;
- execute the selected learned primitive recurrently through depth 64.

Next:

- more than two candidate operator families;
- noisy/partial demonstrations;
- compositional programs whose primitive changes mid-program;
- learned motif discovery, not only selection from a supplied library.

## M8 — Production-shaped register machine (PASSED model-ready)

- EXP-036: high-dimensional JVP evidence gate works through `d_model=320` and
  through the real FOG recurrent transition.
- `register_machine_v3`: typed registers + finite operator bank + recurrent
  value-as-address + optional HALT are integrated into production code.
- EXP-037: full model-side generated-value/re-addressing gate passes 3/3 seeds
  through OOD depth 8.
- `build_fog_machine.py`: exact parameter, forward/backward, machine-gradient,
  JVP, save and strict-reload gates pass for the 10,245,433-parameter preset.

**Decision:** architecture construction is no longer the blocker.  From here the
main path is a training ladder on the built model.

Next model-training ladder:

1. lexical pretraining / migration;
2. exact READ/binding warmup;
3. variable-depth generated-operator curriculum with hard routing;
4. learned HALT/control after recurrent stability passes;
5. semantic/contextual binding and operator induction;
6. matched baselines and only then scaling.

The released v2 checkpoint recurrence transfer remains a useful historical
comparison, but is not required to initialize or train v3.

## M9 — Adaptive computation (NEXT AFTER STABLE TRAINED V3)

Only after the learned transition is stable:

- learned HALT policy;
- fixed-point / cycle / confidence stopping;
- compute-accuracy tradeoff.

## M10 — Branching and verifier

After reliable heterogeneous multi-step execution:

- parallel latent branches;
- verifier/rejector;
- rollback/retry;
- branch-budget allocation.

## M11 — Semantic + scale gates

- algorithmic/code/logical tasks without hand-designed field representations;
- matched looping Transformer / text-CoT / latent baselines;
- natural language;
- 100M+ scaling;
- throughput, memory and quality curves.

Scaling before M6/M7 risks producing a larger memorizing transition system.
