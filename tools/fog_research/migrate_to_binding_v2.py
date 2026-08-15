#!/usr/bin/env python3
"""Create a query-bound-v2 initialization from a legacy lexical checkpoint.

Only tensors with explicitly compatible semantics are copied.  The result is
an initialization checkpoint: its token embedding and causal backbone retain
lexical pretraining, while the new binder/direct reader still require task
training.  Optimizer state is intentionally never migrated across architectures.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from fog_lmw import (
    FOGReasonerConfig,
    FOGLatentReasoner,
    fog_binding_v2_10m_config,
)
from fog_lmw.checkpoint import atomic_torch_save, sha256_file


def migrate(source: Path, output: Path, *, seed: int = 42) -> dict:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    legacy = FOGReasonerConfig(**payload["model_config"])
    if legacy.architecture_version != "legacy_v1":
        raise ValueError("source checkpoint is not legacy_v1")
    if legacy.vocab_size != 8192 or legacy.d_model != 320:
        raise ValueError("automatic migration currently targets the released 10M geometry")

    torch.manual_seed(seed)
    config = fog_binding_v2_10m_config(
        vocab_size=legacy.vocab_size,
        max_seq_len=legacy.max_seq_len,
        reasoning_steps=legacy.reasoning_steps,
        dropout=legacy.dropout,
    )
    model = FOGLatentReasoner(config)
    target = model.state_dict()
    source_state = payload["model_state_dict"]
    copied = []
    initialized = []
    # The v2 reusable-memory gate has different semantics from the legacy
    # append/compress gate even when tensor shapes happen to match.  Copy only
    # lexical/backbone tensors whose function is unchanged.
    allowed_prefixes = ("token.", "backbone.", "lm_head.")
    for name, tensor in target.items():
        candidate = (
            source_state.get("token.weight")
            if name == "direct_head.weight"
            else source_state.get(name)
        )
        if (
            (name.startswith(allowed_prefixes) or name == "direct_head.weight")
            and candidate is not None
            and candidate.shape == tensor.shape
        ):
            target[name] = candidate.to(dtype=tensor.dtype).clone()
            copied.append(name)
        else:
            initialized.append(name)
    # Preserve slot-role geometry from the legacy generic slot queries.
    legacy_query = source_state.get("planner.query")
    if legacy_query is not None and legacy_query.shape == target["planner.slot_role"].shape:
        target["planner.slot_role"] = legacy_query.float().clone()
        copied.append("planner.slot_role <- planner.query")
        initialized.remove("planner.slot_role")
    model.load_state_dict(target, strict=True)
    if model.lm_head.weight is not model.token.weight:
        raise AssertionError("tied lexical weights were lost")

    migration = {
        "kind": "legacy_v1_to_query_bound_v2_initialization",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "source_global_step": int(payload.get("global_step", 0)),
        "copied_tensors": copied,
        "new_or_reinitialized_tensors": initialized,
        "optimizer_state_migrated": False,
        "warning": "binding writer and direct reader are not trained by migration",
    }
    out = {
        "format_version": 2,
        "checkpoint_kind": "inference",
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "global_step": int(payload.get("global_step", 0)),
        "consumed_tokens": int(payload.get("consumed_tokens", 0)),
        "tokenizer": payload.get("tokenizer"),
        "metadata": {"stage": "binding-v2-migrated-init", "migration": migration},
    }
    atomic_torch_save(out, output)
    result = {
        "output": str(output),
        "sha256": sha256_file(output),
        "architecture_version": config.architecture_version,
        "parameters": sum(p.numel() for p in model.parameters()),
        "copied_tensor_count": len(copied),
        "initialized_tensor_count": len(initialized),
        "migration": migration,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
