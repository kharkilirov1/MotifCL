#!/usr/bin/env python3
"""Export a portable BF16 inference checkpoint and verify the result.

The exporter intentionally accepts only the project's portable inference
format.  It loads the input with ``weights_only=True``, verifies the embedded
tokenizer hash, casts only after a strict model load, and writes atomically.
No timestamp is stored, so exporting the same input to the same path is
byte-reproducible with a fixed PyTorch version.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from fog_lmw import (
    FOG_10M_PARAMETER_COUNT,
    FOG_BINDING_V2_10M_PARAMETER_COUNT,
    FOGReasonerConfig,
    FOGLatentReasoner,
)
from fog_lmw.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    atomic_torch_save,
    sha256_file,
)


DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _load_portable_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a dictionary")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {payload.get('format_version')}"
        )
    if payload.get("checkpoint_kind") != "inference":
        raise ValueError(
            "source must be a portable inference checkpoint; export a training "
            "checkpoint with save_inference_checkpoint first"
        )
    for key in ("model_config", "model_state_dict", "global_step", "consumed_tokens"):
        if key not in payload:
            raise ValueError(f"checkpoint is missing required field {key!r}")
    return payload


def _resolve_tokenizer(
    input_path: Path,
    tokenizer_record: dict[str, Any] | None,
    explicit_path: str | Path | None,
) -> Path | None:
    if tokenizer_record is None:
        if explicit_path is not None:
            raise ValueError("--tokenizer was given but the checkpoint has no tokenizer")
        return None

    saved_path = Path(str(tokenizer_record["path"]))
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    else:
        candidates.extend(
            [
                saved_path,
                Path.cwd() / saved_path,
                input_path.parent / saved_path,
                input_path.parent.parent / saved_path,
            ]
        )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.is_file():
            continue
        actual_hash = sha256_file(candidate)
        expected_hash = str(tokenizer_record["sha256"])
        if actual_hash != expected_hash:
            raise ValueError(
                f"tokenizer hash mismatch for {candidate}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        return candidate

    searched = ", ".join(str(candidate) for candidate in seen)
    raise FileNotFoundError(f"could not find the checkpoint tokenizer; searched: {searched}")


def _probe(model: FOGLatentReasoner) -> tuple[torch.Tensor, float]:
    """Return deterministic strict-memory logits and their finite CE loss."""

    if model.cfg.vocab_size < 8:
        raise ValueError("verification probe requires vocab_size >= 8")
    prompt = torch.tensor([[1, 3, 4, 5, 2]], dtype=torch.long)
    answer = torch.tensor([[1, 6, 7, 2]], dtype=torch.long)
    lexical_prompt = torch.tensor([[1]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        memory, _ = model.reason(prompt)
        if model.cfg.readout_mode == "direct_latent":
            _, aux = model.reason(prompt)
            primary = aux["primary_latent"]
            logits = model.direct_vocab_logits(primary)[:, None, :]
            loss = F.cross_entropy(logits[:, 0].float(), answer[:, 1])
        else:
            logits = model.decode(lexical_prompt, memory, answer[:, :-1])
            loss = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), answer[:, 1:].reshape(-1)
            )
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise ValueError("verification probe produced non-finite logits or loss")
    return logits.float().cpu(), float(loss)


def export_checkpoint(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dtype_name: str = "bf16",
    tokenizer_path: str | Path | None = None,
    expected_parameters: int | None = None,
    max_logit_error: float = 0.05,
    force: bool = False,
) -> dict[str, Any]:
    """Export and verify one deterministic, portable inference checkpoint."""

    source = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("input and output paths must differ")
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing file: {output}")
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported dtype {dtype_name!r}; choose from {sorted(DTYPES)}")
    if max_logit_error < 0.0:
        raise ValueError("max_logit_error must be non-negative")

    payload = _load_portable_payload(source)
    resolved_tokenizer = _resolve_tokenizer(
        source, payload.get("tokenizer"), tokenizer_path
    )
    config = FOGReasonerConfig(**payload["model_config"])
    if expected_parameters is None:
        expected_parameters = (
            FOG_BINDING_V2_10M_PARAMETER_COUNT
            if config.architecture_version == "query_bound_v2"
            else FOG_10M_PARAMETER_COUNT
        )
    model = FOGLatentReasoner(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if expected_parameters is not None and parameters != expected_parameters:
        raise ValueError(
            f"parameter count mismatch: expected {expected_parameters}, got {parameters}"
        )
    if model.token.weight is not model.lm_head.weight:
        raise ValueError("token embedding and LM head are not tied")
    if config.readout_mode == "direct_latent" and model.token.weight is not model.direct_head.weight:
        raise ValueError("token embedding and direct head are not tied")

    reference_logits, reference_loss = _probe(model)
    target_dtype = DTYPES[dtype_name]
    model.to(dtype=target_dtype)
    if model.token.weight is not model.lm_head.weight:
        raise ValueError("dtype conversion broke tied token/LM-head weights")
    if config.readout_mode == "direct_latent" and model.token.weight is not model.direct_head.weight:
        raise ValueError("dtype conversion broke tied token/direct-head weights")

    original_metadata = payload.get("metadata")
    metadata = dict(original_metadata) if isinstance(original_metadata, dict) else {}
    metadata["export"] = {
        "source_checkpoint": source.name,
        "source_sha256": sha256_file(source),
        "state_dict_dtype": str(target_dtype).removeprefix("torch."),
        "exporter": "export_checkpoint.py",
        "lossy": target_dtype != torch.float32,
    }
    output_payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": "inference",
        "model_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "global_step": int(payload["global_step"]),
        "consumed_tokens": int(payload["consumed_tokens"]),
        # Preserve the portable path and verified hash exactly as recorded.
        "tokenizer": payload.get("tokenizer"),
        "metadata": metadata,
    }
    atomic_torch_save(output_payload, output)

    exported = _load_portable_payload(output)
    reloaded = FOGLatentReasoner(FOGReasonerConfig(**exported["model_config"]))
    reloaded.load_state_dict(exported["model_state_dict"], strict=True)
    reloaded_parameters = sum(parameter.numel() for parameter in reloaded.parameters())
    if reloaded_parameters != parameters:
        raise ValueError("reloaded checkpoint parameter count changed")
    if reloaded.token.weight is not reloaded.lm_head.weight:
        raise ValueError("reloaded token embedding and LM head are not tied")
    if config.readout_mode == "direct_latent" and reloaded.token.weight is not reloaded.direct_head.weight:
        raise ValueError("reloaded token embedding and direct head are not tied")
    actual_logits, actual_loss = _probe(reloaded)
    absolute_error = (actual_logits - reference_logits).abs()
    max_error = float(absolute_error.max())
    mean_error = float(absolute_error.mean())
    argmax_agreement = float(
        (actual_logits.argmax(dim=-1) == reference_logits.argmax(dim=-1))
        .float()
        .mean()
    )
    if max_error > max_logit_error:
        raise ValueError(
            f"exported logits differ by {max_error:.6g}, exceeding "
            f"--max-logit-error {max_logit_error:.6g}"
        )

    state_dtypes = sorted(
        {str(value.dtype).removeprefix("torch.") for value in exported["model_state_dict"].values() if value.is_floating_point()}
    )
    expected_state_dtype = str(target_dtype).removeprefix("torch.")
    if state_dtypes != [expected_state_dtype]:
        raise ValueError(
            f"unexpected floating state dtypes {state_dtypes}; "
            f"expected only {expected_state_dtype}"
        )

    return {
        "ok": True,
        "input": str(source),
        "output": str(output),
        "parameters": parameters,
        "global_step": int(exported["global_step"]),
        "consumed_tokens": int(exported["consumed_tokens"]),
        "state_dict_dtype": expected_state_dtype,
        "source_bytes": source.stat().st_size,
        "output_bytes": output.stat().st_size,
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(output),
        "tokenizer": None if resolved_tokenizer is None else str(resolved_tokenizer),
        "reference_loss": reference_loss,
        "reloaded_loss": actual_loss,
        "max_logit_error": max_error,
        "mean_logit_error": mean_error,
        "argmax_agreement": argmax_agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="portable FP32 inference checkpoint")
    parser.add_argument("output", help="new portable checkpoint path")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--tokenizer", help="tokenizer path (auto-detected by default)")
    parser.add_argument(
        "--expected-parameters",
        type=int,
        default=None,
        help="strict parameter-count gate (default: auto by architecture)",
    )
    parser.add_argument("--max-logit-error", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = export_checkpoint(
        args.input,
        args.output,
        dtype_name=args.dtype,
        tokenizer_path=args.tokenizer,
        expected_parameters=args.expected_parameters,
        max_logit_error=args.max_logit_error,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
