#!/usr/bin/env python3
"""EXP-002 implementation gate for production query-bound-v2 recurrence.

The decisive protocol is documented in
`research/experiments/EXP-002-production-recurrent-token-composition.md`.

This script uses the real FOGLatentReasoner token embeddings, causal backbone,
query-conditioned planner, reusable K-slot memory and direct tied readout.  It
serializes a permutation table with one canonical STATE token namespace shared
by source, payload and query roles.  A masked gap ensures that only source
positions are valid address candidates for the configured binding offset.

If --checkpoint is omitted, the script is only an implementation smoke test on
a fresh 10M geometry.  Such a run must not be reported as a trained-model
result.  The decisive EXP-002 run requires the released 10M v2 checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import torch
from torch.nn import functional as F

from fog_lmw import FOGReasonerConfig, FOGLatentReasoner
from fog_lmw.presets import fog_binding_v2_10m_config
from matched_structured_lookup_experiment import (
    StructuredBatch,
    StructuredTaskConfig,
    _update_stream_digest,
    make_batch,
    verify_mapping_holdout,
)
from recurrent_binding_composition_experiment import target_for_depth


EXPERIMENT_NAME = "exp_002_production_recurrent_token_composition"
PAD_ID = 0
STATE_BASE = 100
Mode = Literal["recurrent", "static"]
Intervention = Literal["normal", "corrupt_after_2"]


def token_batch_shared_identity(
    structured: StructuredBatch,
    *,
    pad_id: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Serialize sources, aligned payloads, a masked gap, then the query.

    For table size N and binding offset N:

      positions [0,N)   : source STATE ids, valid
      positions [N,2N)  : payload STATE ids, valid
      positions [2N,3N) : masked gap
      position  3N       : query STATE id, valid

    Therefore only source positions have a valid token exactly N positions
    later.  Payload occurrences of the same identity cannot become accidental
    competing addresses.
    """

    n = structured.row_sources.size(1)
    sources = STATE_BASE + structured.row_sources
    payloads = STATE_BASE + structured.row_values
    gap = torch.full(
        (sources.size(0), n), pad_id, dtype=torch.long, device=sources.device
    )
    query = (STATE_BASE + structured.query_keys)[:, None]
    prompt = torch.cat([sources, payloads, gap, query], dim=1)
    mask = torch.cat(
        [
            torch.ones_like(sources, dtype=torch.bool),
            torch.ones_like(payloads, dtype=torch.bool),
            torch.zeros_like(gap, dtype=torch.bool),
            torch.ones_like(query, dtype=torch.bool),
        ],
        dim=1,
    )
    return prompt, mask


def load_model(
    *,
    checkpoint: Path | None,
    task: StructuredTaskConfig,
    max_depth: int,
    device: torch.device,
) -> tuple[FOGLatentReasoner, str]:
    if checkpoint is None:
        cfg = fog_binding_v2_10m_config(
            vocab_size=8192,
            max_seq_len=max(128, 3 * task.table_size + 1 + 16),
            reasoning_steps=max_depth,
            dropout=0.0,
        )
        source_kind = "fresh_10m_geometry_smoke_only"
        state_dict = None
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = FOGReasonerConfig(**payload["model_config"])
        if cfg.architecture_version != "query_bound_v2":
            raise ValueError("checkpoint must use query_bound_v2")
        source_kind = "released_or_supplied_checkpoint"
        state_dict = payload["model_state_dict"]

    # Same number of binding relations as the released (2,) preset, so this
    # changes no parameter shapes.  The long offset + mask supplies role.
    cfg.binding_offsets = (task.table_size,)
    cfg.binding_query_update = "primary_recurrent"
    cfg.reasoning_steps = max_depth
    cfg.dropout = 0.0
    cfg.max_seq_len = max(cfg.max_seq_len, 3 * task.table_size + 1 + cfg.latent_slots + 4)
    cfg.validate()
    model = FOGLatentReasoner(cfg)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    model.float().to(device).eval()
    return model, source_kind


def _planner_corruption_hook():
    counter = {"hop": 0}

    def hook(module, args, kwargs):
        counter["hop"] += 1
        # Third planner call receives the query produced after two completed
        # recurrent hops. Roll it across examples while leaving tables fixed.
        if counter["hop"] == 3:
            state = kwargs.get("binding_query_state")
            if state is None:
                raise AssertionError("query-conditioned planner omitted binding query")
            kwargs["binding_query_state"] = state.roll(1, dims=0)
        return args, kwargs

    return hook


@torch.inference_mode()
def evaluate_depth(
    model: FOGLatentReasoner,
    task: StructuredTaskConfig,
    *,
    data_seed: int,
    split: Literal["validation", "test"],
    depth: int,
    examples: int,
    batch_size: int,
    device: torch.device,
    mode: Mode,
    intervention: Intervention = "normal",
) -> dict:
    original_update = model.cfg.binding_query_update
    model.cfg.binding_query_update = (
        "primary_recurrent" if mode == "recurrent" else "static"
    )
    correct = 0
    nll = 0.0
    digest = hashlib.sha256()
    address_hits = [0 for _ in range(depth)]
    address_mass = [0.0 for _ in range(depth)]
    oracle_cosine = [0.0 for _ in range(depth)]
    total = 0
    try:
        for start in range(0, examples, batch_size):
            count = min(batch_size, examples - start)
            batch = make_batch(
                task,
                data_seed=data_seed,
                split=split,
                start_index=start,
                batch_size=count,
            )
            _update_stream_digest(digest, batch)
            batch = batch.to(device)
            prompt, mask = token_batch_shared_identity(batch)
            target_state = target_for_depth(batch, depth, device)
            target_token = STATE_BASE + target_state

            handle = None
            if intervention == "corrupt_after_2" and depth >= 3:
                handle = model.planner.register_forward_pre_hook(
                    _planner_corruption_hook(), with_kwargs=True
                )
            try:
                _, aux = model.reason(
                    prompt,
                    prompt_attention_mask=mask,
                    reasoning_steps=depth,
                    return_diagnostics=True,
                )
            finally:
                if handle is not None:
                    handle.remove()

            primary = aux["primary_latent"]
            logits = model.direct_vocab_logits(primary)
            correct += int(logits.argmax(-1).eq(target_token).sum())
            nll += float(F.cross_entropy(logits.float(), target_token, reduction="sum"))
            total += count

            mapping = torch.tensor(batch.mappings, dtype=torch.long, device=device)
            state = batch.query_keys
            rows = torch.arange(count, device=device)
            for hop, hist in enumerate(aux["history"]):
                weights = hist["planner"]["binding_attention"][:, 0]
                expected = batch.row_sources.eq(state[:, None])
                expected_index = expected.float().argmax(-1)
                address_hits[hop] += int(weights.argmax(-1).eq(expected_index).sum())
                address_mass[hop] += float(weights[rows, expected_index].sum())
                state = mapping[rows, state]
                oracle = model.token(STATE_BASE + state)
                oracle_cosine[hop] += float(
                    F.cosine_similarity(
                        hist["latent"][:, 0].float(), oracle.float(), dim=-1
                    ).sum()
                )
    finally:
        model.cfg.binding_query_update = original_update

    return {
        "split": split,
        "depth": depth,
        "mode": mode,
        "intervention": intervention,
        "correct": correct,
        "count": total,
        "accuracy": correct / total,
        "nll": nll / total,
        "stream_sha256": digest.hexdigest(),
        "hops": [
            {
                "hop": hop + 1,
                "address_hit_accuracy": address_hits[hop] / total,
                "correct_address_mass": address_mass[hop] / total,
                "oracle_cosine": oracle_cosine[hop] / total,
            }
            for hop in range(depth)
        ],
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=Path("artifacts/research/exp_002/smoke.json"))
    p.add_argument("--data-seed", type=int, default=20260814)
    p.add_argument("--table-size", type=int, default=8)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--examples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    return p


def main() -> None:
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    if args.split == "test" and args.checkpoint is None:
        raise ValueError("fresh-init smoke must not touch the locked test split")
    task = StructuredTaskConfig(table_size=args.table_size)
    protocol = verify_mapping_holdout(
        task,
        data_seed=args.data_seed,
        train_examples=512,
        validation_examples=args.examples,
        test_examples=args.examples,
    )
    device = torch.device(args.device)
    model, source_kind = load_model(
        checkpoint=args.checkpoint,
        task=task,
        max_depth=args.max_depth,
        device=device,
    )
    results = {
        mode: [
            evaluate_depth(
                model,
                task,
                data_seed=args.data_seed,
                split=args.split,
                depth=depth,
                examples=args.examples,
                batch_size=args.batch_size,
                device=device,
                mode=mode,  # type: ignore[arg-type]
            )
            for depth in range(1, args.max_depth + 1)
        ]
        for mode in ("recurrent", "static")
    }
    results["recurrent_corrupt_after_2"] = [
        evaluate_depth(
            model,
            task,
            data_seed=args.data_seed,
            split=args.split,
            depth=depth,
            examples=args.examples,
            batch_size=args.batch_size,
            device=device,
            mode="recurrent",
            intervention="corrupt_after_2",
        )
        for depth in range(3, args.max_depth + 1)
    ]
    payload = {
        "experiment": EXPERIMENT_NAME,
        "source_kind": source_kind,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "decisive_evidence": args.checkpoint is not None,
        "task_config": asdict(task),
        "model_config": asdict(model.cfg),
        "protocol": protocol,
        "split": args.split,
        "results": results,
    }
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_kind": source_kind,
                "recurrent_accuracy": [r["accuracy"] for r in results["recurrent"]],
                "static_accuracy": [r["accuracy"] for r in results["static"]],
                "decisive_evidence": payload["decisive_evidence"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
