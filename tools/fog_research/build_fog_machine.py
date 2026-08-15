#!/usr/bin/env python3
"""Build and mechanically validate the model-ready FOG register machine v3."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
import json
from pathlib import Path

import torch

from fog_lmw import (
    FOG_MACHINE_V3_10M_PARAMETER_COUNT,
    FOGReasonerConfig,
    FOGLatentReasoner,
    fog_machine_v3_10m_config,
)
from fog_lmw.checkpoint import load_training_checkpoint, save_training_checkpoint
from fog_lmw.structural import jvp_gain_stats


def unique_parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


def build(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    config = fog_machine_v3_10m_config(
        max_seq_len=args.max_seq_len,
        reasoning_steps=args.reasoning_steps,
        dropout=0.0,
        adaptive_halting=False,
    )
    model = FOGLatentReasoner(config)
    parameters = unique_parameter_count(model)
    expected = (
        FOG_MACHINE_V3_10M_PARAMETER_COUNT
        if args.max_seq_len == 512
        else parameters
    )
    if parameters != expected:
        raise AssertionError(f"parameter contract failed: {parameters} != {expected}")

    batch = args.smoke_batch
    prompt_len = min(12, args.max_seq_len - config.latent_slots - 4)
    if prompt_len <= max(config.binding_offsets):
        raise ValueError("max_seq_len is too short for the smoke prompt")
    prompt = torch.randint(4, config.vocab_size, (batch, prompt_len))
    answer = torch.randint(4, config.vocab_size, (batch, 3))
    answer[:, 0] = 1  # arbitrary BOS ID for the smoke contract

    model.train()
    loss, aux = model.loss(prompt, answer, reasoning_steps=min(3, args.reasoning_steps))
    if not torch.isfinite(loss):
        raise RuntimeError("smoke loss is not finite")
    loss.backward()
    machine_grads = [
        p.grad for p in model.machine_cell.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not machine_grads or not any(torch.isfinite(g).all() and g.abs().max() > 0 for g in machine_grads):
        raise RuntimeError("machine cell did not receive a finite non-zero gradient")
    model.zero_grad(set_to_none=True)

    # Probe the real shared transition. This does not materialize a Jacobian.
    model.eval()
    first, _ = model.transition_memory(prompt[:1], None)
    def transition(memory: torch.Tensor) -> torch.Tensor:
        return model.transition_memory(prompt[:1], memory)[0]
    gain = jvp_gain_stats(transition, first.detach(), probes=args.jvp_probes, seed=args.seed + 1)

    output = Path(args.output)
    save_training_checkpoint(
        output,
        model=model,
        config=config,
        optimizer=None,
        scheduler=None,
        scaler=None,
        global_step=0,
        consumed_tokens=0,
        tokenizer_path=None,
        training_args={"builder": "build_fog_machine.py", **vars(args)},
        extra={
            "stage": "model_ready_initialized",
            "structural_contract": "typed_registers/shared_transition/jvp_probeable",
        },
    )

    restored_config = FOGReasonerConfig(**torch.load(output, map_location="cpu", weights_only=False)["model_config"])
    restored = FOGLatentReasoner(restored_config)
    load_training_checkpoint(output, model=restored, restore_rng=False)
    for (name_a, p_a), (name_b, p_b) in zip(model.named_parameters(), restored.named_parameters(), strict=True):
        if name_a != name_b or not torch.equal(p_a, p_b):
            raise RuntimeError(f"strict reload mismatch at {name_a}/{name_b}")

    checkpoint_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "status": "MODEL_READY",
        "architecture": config.architecture_version,
        "parameters": parameters,
        "config": asdict(config),
        "smoke": {
            "loss": float(loss.detach()),
            "token_accuracy": float(aux["token_accuracy"].detach()),
            "machine_gradient_tensors": len(machine_grads),
        },
        "transition_probe": gain.to_dict(),
        "checkpoint": str(output),
        "checkpoint_bytes": output.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "strict_reload": True,
    }
    report_path = output.with_suffix(".model_ready.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="checkpoints/fog_machine_v3_10m_init.pt")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--reasoning-steps", type=int, default=8)
    p.add_argument("--smoke-batch", type=int, default=2)
    p.add_argument("--jvp-probes", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    report = build(args)
    print(json.dumps({k: report[k] for k in ("status", "architecture", "parameters", "checkpoint", "strict_reload")}, indent=2))


if __name__ == "__main__":
    main()
