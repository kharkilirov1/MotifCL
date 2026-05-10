# Production hardening roadmap

This file tracks the remaining engineering boundaries after the modern HF
compatibility pass.

## Highest-impact remaining work

1. **Quantized modern inference follow-up**
   - Existing: Q4_0/Q8_0 no-grad inference path for modern `Linear`
     projections and LM head, CLI `--quant`, common HF safetensors loading,
     persisted quantized tensor metadata in `save_tensor()` / `load_tensor()`,
     mixed per-layer `QuantizationPolicy`, and whole-model quantized
     transformer checkpoint manifests.
   - Missing: calibration/quality tooling, external checkpoint-format export
     polish, and hardware-backed perf/accuracy acceptance criteria per model
     family.

2. **FP16 training coverage**
   - Existing: FP16 cast kernels, FP16 inference matmul, FP32-master
     `MixedPrecisionAdam`, dynamic loss scaling.
   - Missing: broad FP16 backward kernels for training ops, especially matmul
     backward variants and fused transformer training paths.

3. **Graph execution**
   - Existing: capture/replay, tensor specs, same-shape `cl_mem` rebinding,
     command-buffer replay/update when supported by the ICD.
   - Missing: true dynamic-shape recompile/rebind and planner-materialized
     allocation replacement for arbitrary captured outputs.

4. **Modern HF compatibility**
   - Existing: generic config path, common LLaMA/Gemma/Mistral/Qwen2-style
     safetensors layout, tokenizer simple-vocab/byte fallback plus binary
     `tokenizer.model` pieces and HF BPE/Unigram support, cached generation,
     fused padded variable-length batch prefill with per-row positioned decode
     cache writes, GPU greedy/top-k/top-p/temperature sampling, and optional
     real-HF smoke coverage.
   - Missing: full SentencePiece normalizer parity, architecture-specific
     weight mappers for layouts that diverge from `model.layers.*`, fully
     compact packed ragged KV-cache storage, and exact full-vocab GPU top-p
     without the current 128-candidate cap.

5. **Performance gates**
   - Existing: local tuner, checked-in Ellesmere seed baseline, regression
     comparison script with required-key/min-metric and metadata checks, CI
     smoke for the script, and `tools/release_check.py` to orchestrate local
     release-readiness checks.
   - Missing: device/driver matrix baselines collected as CI artifacts and
     hardware-backed perf gates for supported GPUs.

6. **Release hygiene**
   - Keep `CHANGELOG.md` and docs synchronized with public API changes.
   - Before tagging: run `python tools/release_check.py --wheel --hf-run
     --require-clean-git` from a clean branch, and archive local
     `motifcl_tuning.json` next to the matching baseline/device metadata.
