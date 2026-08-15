from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import random
from typing import Any

import torch

from .config import FOGReasonerConfig


CHECKPOINT_FORMAT_VERSION = 2


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    config: FOGReasonerConfig,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    scaler: Any | None,
    global_step: int,
    consumed_tokens: int,
    tokenizer_path: str | Path | None,
    training_args: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            None if optimizer is None else optimizer.state_dict()
        ),
        "scheduler_state_dict": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "global_step": int(global_step),
        "consumed_tokens": int(consumed_tokens),
        "rng_state": capture_rng_state(),
        "training_args": training_args,
        "tokenizer": (
            None
            if tokenizer_path is None
            else {
                "path": str(tokenizer_path),
                "sha256": sha256_file(tokenizer_path),
            }
        ),
        "extra": extra or {},
    }
    return atomic_torch_save(payload, path)


def save_inference_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    config: FOGReasonerConfig,
    global_step: int,
    consumed_tokens: int,
    tokenizer_path: str | Path | None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save portable weights without optimizer state (~model FP32 size)."""

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": "inference",
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "global_step": int(global_step),
        "consumed_tokens": int(consumed_tokens),
        "tokenizer": (
            None
            if tokenizer_path is None
            else {
                "path": str(tokenizer_path),
                "sha256": sha256_file(tokenizer_path),
            }
        ),
        "metadata": metadata or {},
    }
    return atomic_torch_save(payload, path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    tokenizer_path: str | Path | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {payload.get('format_version')}"
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if payload.get("checkpoint_kind") == "inference":
        if optimizer is not None or scheduler is not None or scaler is not None:
            raise ValueError("inference checkpoint has no optimizer/scheduler/scaler state")
        saved_tokenizer = payload.get("tokenizer")
        if saved_tokenizer is not None and tokenizer_path is not None:
            actual = sha256_file(tokenizer_path)
            if actual != saved_tokenizer["sha256"]:
                raise ValueError("tokenizer hash differs from checkpoint")
        return payload
    if optimizer is not None and payload["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload["scaler_state_dict"] is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    saved_tokenizer = payload.get("tokenizer")
    if saved_tokenizer is not None and tokenizer_path is not None:
        actual = sha256_file(tokenizer_path)
        if actual != saved_tokenizer["sha256"]:
            raise ValueError("tokenizer hash differs from checkpoint")
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
