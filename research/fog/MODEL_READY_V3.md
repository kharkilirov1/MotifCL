# FOG Register Machine v3 — model-ready specification

Date: 2026-08-14

## Status

**BUILD READY / RESEARCH TRAINING READY.**

The repository can now instantiate, differentiate, structurally probe, save and
strictly reload the next FOG model.  This is a model-readiness claim, not a claim
that an untrained checkpoint is a useful language model.

Reference preset:

- architecture: `register_machine_v3`;
- parameters: **10,245,433** at vocab 8192 / max sequence 512;
- `d_model=320`, four lexical backbone layers;
- `K=4` fixed typed registers;
- max recurrent depth default 8;
- one exact query-bound proposal lane;
- value register starts from the exact query code;
- register 1 is control; registers 2--3 are scratch;
- operator candidates: READ, IDENTITY, parameter-free BLOCK_PRODUCT and four
  learned low-rank bilinear operators;
- straight-through hard operator routing;
- optional learned HALT head;
- direct first-token latent readout;
- explicit `transition_memory()` API for JVP/VJP instrumentation.

## Why this architecture now

Every major choice has a controlled reason:

- recurrent primary state: EXP-001;
- canonical generated state must be reusable: EXP-003--005;
- shared transition law, not a per-position table: EXP-006/007;
- operator-compatible geometry/laws matter more than raw capacity: EXP-004,
  EXP-014, EXP-019--022;
- one-step correctness is not enough; closure/gain must be monitored: EXP-014;
- structural compilation can recover/denoise/compress learned dynamics:
  EXP-023--035;
- full Jacobians are unnecessary at d=320: EXP-036;
- discrete operator routing prevents invalid soft-mixture dynamics: EXP-037.

## Build gate

Run:

```bash
python build_fog_machine.py \
  --output checkpoints/fog_machine_v3_10m_init.pt \
  --reasoning-steps 8 --jvp-probes 2 --seed 42
```

The gate requires:

1. exact parameter contract;
2. multi-token forward loss is finite;
3. backward reaches the machine cell;
4. a JVP sketch runs through the real recurrent transition;
5. checkpoint is written;
6. fresh model strict-loads every parameter exactly.

The current generated report is
`checkpoints/fog_machine_v3_10m_init.model_ready.json`.

Machine-readable build manifest: `research/MODEL_BUILD_MANIFEST.json`.

## Current mechanical verification

The final post-EXP-037 rebuild reports:

- `status = MODEL_READY`;
- `parameters = 10,245,433`;
- `machine_hard_routing = true`;
- 24 machine-cell gradient tensors are finite/nonzero in the builder smoke;
- production transition JVP gain: mean ~0.496, p95 ~0.502 (2 random probes);
- checkpoint size ~41.0 MB FP32;
- checkpoint SHA-256: `2b5ecb6bbe08b4ad8a08a87c79e8fb3aee4acc1b546892ea2f99b7e16e80c17f`;
- `strict_reload = true`;
- core regression after the final build: **121 passed / 2 skipped**.

These numbers are mechanical readiness evidence, not quality metrics for an
untrained language model.

## Real training entry points

`train_real.py` now accepts `--architecture register_machine_v3` for
`init-model`, pretraining and SFT.  Causal-LM pretraining uses the lexical
backbone exactly as before; machine parameters become active in latent SFT or a
machine curriculum.

Example initialization with the normal tokenizer environment:

```bash
python train_real.py init-model \
  --architecture register_machine_v3 \
  --reasoning-steps 8 \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --output checkpoints/fog_machine_v3_10m_init.pt
```

## Recommended training ladder

### Stage A — lexical backbone

Pretrain causal LM first.  Do not ask an untrained operator grammar to discover
language and recurrent computation simultaneously.

### Stage B — exact READ/binding warmup

Preserve the binding-v2 capability before increasing generated-operator mass.
Monitor address accuracy and READ routing.

### Stage C — operator curriculum

Train structured generated-value and program tasks with **variable/copime
terminal depths**, not a single fixed depth.  Include tasks where the correct
intermediate value is absent from prompt payloads.  Monitor operator-selection
entropy, canonical-state cosine, closure defect and JVP gain.

### Stage D — HALT/control

Only after a stable recurrent transition exists, train the control register and
HALT head on variable program lengths.  Keep an external maximum-depth safety
cap.

### Stage E — semantic reasoning

Move from offset-structured binding to semantic/contextual operator induction.
This remains the largest open model-side research problem.

## Hard go/no-go gates before scaling

Do **not** scale a checkpoint merely because one-step accuracy is high.  Require:

- generated-value intervention dependence;
- OOD recurrent depth above training depth;
- closure defect below a fixed threshold;
- JVP local-gain distribution not showing uncontrolled amplification;
- stable hard operator identities rather than high-entropy mixture behavior;
- HALT interventions that separate data and control failures;
- structural compiler must abstain when evidence is weak.

## Remaining limitation

The current production-shaped exact binding mechanism still assumes structured
relative address->payload offsets.  v3 is therefore ready to build and train as
a research model, but **natural-language semantic binding/operator induction is
not yet solved**.  That becomes the next model-level research frontier rather
than a blocker for constructing the model itself.
