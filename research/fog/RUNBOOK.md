# FOG Research Runbook

Shortest reproducible commands for the active research workflow.

## Environment

```bash
python -m pip install -e '.[dev]'
```

Use `python -m pytest` so the repository root is reliably on `sys.path`.

## Core regression

```bash
python -m pytest -q tests binding_diagnostics_test.py
```

A bare full collection additionally requires optional `tokenizers` paths.
Environment failures must be recorded separately from model/experiment failures.

## EXP-001 — recurrent binding composition

```bash
python recurrent_binding_composition_experiment.py \
  --steps 160 --train-max-depth 4 \
  --eval-max-depth 16 --eval-examples 1024 \
  --batch-size 128 --eval-batch-size 128 --threads 4
```

Locked test has already been opened; do not tune EXP-001 on it.

## EXP-002 — production 10M recurrence

Smoke only:

```bash
python production_recurrent_token_composition_experiment.py \
  --max-depth 4 --examples 32 --batch-size 16
```

Decisive validation once the released checkpoint is present:

```bash
python production_recurrent_token_composition_experiment.py \
  --checkpoint checkpoints/fog_binding_v2_10m_token_lookup_bf16.pt \
  --output artifacts/research/exp_002/trained_validation.json \
  --split validation --max-depth 4 --examples 1024 --batch-size 64
```

## EXP-003 — generated-value gauge

```bash
python generated_value_gauge_experiment.py \
  --output artifacts/research/exp_003/metrics.json \
  --steps 300 --batch-size 128 --eval-examples 1024 \
  --eval-max-depth 16 --seeds 0 1 2
```

## EXP-004 — geometry × operator bias

```bash
python operator_compatible_geometry_experiment.py \
  --output artifacts/research/exp_004/metrics.json
```

## EXP-005 — learned generated-value recurrence

```bash
python learned_generated_value_recurrence_experiment.py \
  --output artifacts/research/exp_005/metrics.json \
  --examples 2048
```

## EXP-006 — latent PC + unique HALT

```bash
python latent_program_counter_experiment.py \
  --output artifacts/research/exp_006/metrics.json \
  --examples 4096
```

## EXP-007 — integrated latent register machine

```bash
python integrated_latent_machine_experiment.py \
  --output artifacts/research/exp_007/metrics.json \
  --examples 2048
```

## EXP-008 — operator-specific charts

```bash
python multichart_operator_specificity_experiment.py \
  --output artifacts/research/exp_008/metrics.json
```

## EXP-009 — chart bridge

```bash
python chart_bridge_experiment.py \
  --output artifacts/research/exp_009/metrics.json
```

## Adding a new research direction

1. Reserve the next unused IDEA/EXP number.
2. Write the hypothesis and falsifier before locked evaluation.
3. Separate environment, optimizer, representation and task failures.
4. Put machine-readable metrics under `artifacts/research/exp_NNN/`.
5. Write a RESULT doc even for a negative result.
6. Record architectural consequences in `DECISIONS.md`.
7. Only then change production defaults.

## EXP-010 — learned cyclic chart

```bash
python learned_cyclic_chart_experiment.py \
  --output artifacts/research/exp_010/metrics.json
```

## EXP-011 — multi-depth chart consistency

```bash
python learned_chart_depth_consistency_experiment.py \
  --output artifacts/research/exp_011/metrics.json
```

## EXP-012 — jointly learned chart + flexible operator

The formal artifact was assembled from per-seed runs to avoid execution
timeouts:

```bash
python joint_chart_operator_experiment.py --seeds 0 --output artifacts/research/exp_012/seed0.json
python joint_chart_operator_experiment.py --seeds 1 --output artifacts/research/exp_012/seed1.json
python joint_chart_operator_experiment.py --seeds 2 --output artifacts/research/exp_012/seed2.json
```

## EXP-013 — in-transition soft canonicalizer

```bash
python canonicalized_joint_algebra_experiment.py \
  --seeds 10 11 12 --steps 800 \
  --output artifacts/research/exp_013/metrics.json
```

## EXP-014 — law by construction

```bash
python normed_operator_parameterization_experiment.py \
  --seeds 20 21 22 --steps 800 \
  --output artifacts/research/exp_014/metrics.json
```

Stability case study:

```bash
python transition_stability_diagnostic.py \
  --seed 22 --output artifacts/research/stability_diagnostic/seed22.json
```

## EXP-015 — finite operator induction

```bash
python operator_grammar_induction_experiment.py \
  --seeds 30 --steps 600 --episodes 3000 \
  --output artifacts/research/exp_015/seed30.json
```

Use separate seed files when runtime limits make a multi-seed command unsafe.

## EXP-016 — generator orbit + matching closure

```bash
python generator_orbit_chart_experiment.py \
  --seeds 40 41 42 43 44 --steps 600 \
  --output artifacts/research/exp_016/metrics.json
```

The decisive closure arm is also stored separately in
`artifacts/research/exp_016/orbit_closure.json`.

## EXP-017 — robust operator grammar

```bash
python robust_operator_grammar_experiment.py \
  --seeds 50 --steps 600 --episodes 3000 \
  --output artifacts/research/exp_017/seed50.json
```

## EXP-018 — infer primitive then recur

```bash
python inferred_operator_recurrent_execution_experiment.py \
  --seeds 60 --steps 600 --episodes 1000 \
  --output artifacts/research/exp_018/seed60.json
```

## EXP-019 — single invariant chart for mixed ADD/MUL

```bash
python single_chart_affine_machine_experiment.py \
  --depth 256 --examples 4096 \
  --output artifacts/research/exp_019/metrics.json
```

## EXP-020 — operator-orbit scaling

```bash
python operator_orbit_scaling_experiment.py \
  --depth 128 --examples 4096 \
  --output artifacts/research/exp_020/metrics.json
```

## EXP-021 — learned affine representation from two generators

```bash
python learned_affine_representation_experiment.py \
  --seeds 0 --dimensions 2 4 8 16 24 30 31 \
  --steps 1000 --program-depth 8 --examples 512 \
  --output artifacts/research/exp_021/seed0.json
```

## EXP-022 — isotropic learned affine representation

Primary new-seed sweep:

```bash
python isotropic_affine_representation_experiment.py \
  --seeds 70 --dimensions 16 24 30 \
  --steps 1200 --program-depth 16 --examples 1024 \
  --output artifacts/research/exp_022/seed70.json
```

Boundary confirmation uses dimensions `28 29 30` on seeds 73/74/75.  See
`artifacts/research/exp_022/metrics.json`.

## EXP-023 — spectral gauge motif discovery

```bash
python spectral_gauge_motif_discovery_experiment.py \
  --dimensions 29 30 --seeds 73 74 75 \
  --program-depth 64 --examples 512 \
  --output artifacts/research/exp_023/metrics.json
```

## EXP-024 — motif projection denoising

Run per seed if runtime limits are tight:

```bash
python motif_projection_denoising_experiment.py \
  --seeds 73 --noise 0 0.03 0.05 0.10 \
  --program-depth 32 --examples 64 \
  --output artifacts/research/exp_024/seed73.json
```

The combined artifact is `artifacts/research/exp_024/metrics.json`.

## EXP-025 — automatic operator-law discovery

```bash
python automatic_operator_law_discovery_experiment.py \
  --dimensions 29 30 --seeds 73 74 75 \
  --program-depth 64 --examples 128 \
  --output artifacts/research/exp_025/metrics.json
```

## EXP-026 — operator grammar compression

```bash
python operator_grammar_compression_experiment.py \
  --dimensions 30 --seeds 80 81 82 --steps 1600 \
  --program-depth 64 --examples 128 \
  --output artifacts/research/exp_026/d30.json
```

Run the matched d=29 control separately. Combined artifact:
`artifacts/research/exp_026/metrics.json`.

## EXP-027 — partial transition sample complexity

```bash
python partial_transition_sample_complexity_experiment.py \
  --coverage 24 29 30 --seeds 90 91 92 --steps 1500 \
  --program-depth 16 --examples 128 \
  --output artifacts/research/exp_027/metrics.json
```

## EXP-028 — structural completion threshold

```bash
python structural_completion_sample_threshold_experiment.py \
  --coverage 28 29 30 --seeds 90 91 92 --steps 1500 \
  --program-depth 16 --examples 128 \
  --output artifacts/research/exp_028/metrics.json
```

## EXP-029 — canonical gauge interoperability

```bash
python canonical_gauge_interoperability_experiment.py \
  --seeds 73 74 75 --pre-depth 32 --post-depth 32 --examples 256 \
  --output artifacts/research/exp_029/metrics.json
```

## EXP-030 — synthesize a simple-spectrum anchor from degenerate operators

```bash
python degenerate_joint_anchor_synthesis_experiment.py \
  --dimensions 30 --seeds 100 101 102 --steps 1400 \
  --program-depth 64 --examples 128 \
  --output artifacts/research/exp_030/metrics.json
```

## EXP-031 — joint commutant block compiler

```bash
python joint_commutant_block_compiler_experiment.py \
  --multiplicities 2 3 4 --seeds 110 111 112 \
  --program-depth 256 --examples 2048 \
  --output artifacts/research/exp_031/metrics.json
```

## EXP-032 — approximate commutant denoising

```bash
python approximate_commutant_denoising_experiment.py \
  --multiplicities 2 3 4 --seeds 120 121 122 \
  --noise 0.03 0.05 0.10 0.15 \
  --program-depth 256 --examples 2048 \
  --output artifacts/research/exp_032/metrics.json
```

## EXP-033 — trajectory-only commutant recovery

```bash
python trajectory_only_commutant_recovery_experiment.py \
  --multiplicities 3 4 --seeds 130 131 132 \
  --sample-factors 1 2 4 --noise 0.05 0.10 \
  --program-depth 256 --examples 2048 \
  --output artifacts/research/exp_033/metrics.json
```

## EXP-034 — state-conditioned local Jacobian gauge synchronization

The full sweep can exceed a single runtime limit; run per seed and merge the
`rows` arrays without changing protocol fields:

```bash
python state_conditioned_jacobian_gauge_experiment.py \
  --seeds 140 --noise 0.03 0.05 0.10 0.15 \
  --program-depth 256 --examples 1024 \
  --output artifacts/research/exp_034/seed_140.json
```

Repeat for seeds 141/142.  The assembled artifact is
`artifacts/research/exp_034/metrics.json`.

## EXP-035 — nonlinear black-box structural compiler

Run per seed if needed for runtime isolation:

```bash
python nonlinear_blackbox_structural_compiler_experiment.py \
  --seeds 150 --noise 0.01 0.03 0.05 0.10 \
  --program-depth 256 --examples 1024 \
  --output artifacts/research/exp_035/seed_150.json
```

Repeat for seeds 151/152, then assemble the unchanged `rows` arrays into
`artifacts/research/exp_035/metrics.json`.  The separate exploratory high-noise
probe is `artifacts/research/exp_035/high_noise_seed150.json`.

## EXP-036 — high-dimensional JVP relation discovery

```bash
python highdim_jvp_relation_discovery_experiment.py \
  --dimensions 128 256 --perturbations 0 0.01 0.03 0.05 \
  --output artifacts/research/exp_036/metrics.json
```

The production-width 320D probe is stored separately under
`artifacts/research/exp_036/`.

## EXP-037 — register-machine generated value / recurrent re-addressing

```bash
python register_machine_v3_generated_value_experiment.py \
  --seeds 0 1 2 --steps 150 \
  --output artifacts/research/exp_037/metrics.json
```

Pass criterion: exact at every evaluated depth 1,2,3,4,6,8 on every seed, with
hard selected operator traces and canonical-state cosine reported.

## Build the reference FOG register machine v3

This is the model-readiness command and should be rerun after any architectural
change:

```bash
python build_fog_machine.py \
  --output checkpoints/fog_machine_v3_10m_init.pt \
  --reasoning-steps 8 --jvp-probes 2 --seed 42
```

Expected report: `status=MODEL_READY`, architecture `register_machine_v3`,
10,245,433 unique parameters, nonzero machine gradients, a finite JVP gain probe,
a recorded checkpoint SHA-256, and `strict_reload=true`.

With the optional tokenizer dependency installed, the equivalent normal pipeline
entry point is:

```bash
python train_real.py init-model \
  --architecture register_machine_v3 --reasoning-steps 8 \
  --tokenizer tokenizer/tinystories_3k_bpe.json \
  --output checkpoints/fog_machine_v3_10m_init.pt
```
