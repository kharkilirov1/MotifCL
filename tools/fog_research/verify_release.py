#!/usr/bin/env python3
"""Verify the primary trained tokenizer/checkpoint and optional gradients."""

from __future__ import annotations

import argparse
import json

import torch

from fog_lmw import (
    FOG_10M_PARAMETER_COUNT,
    FOG_BINDING_V2_10M_PARAMETER_COUNT,
    FOGReasonerConfig,
    FOGLatentReasoner,
)
from fog_lmw.checkpoint import load_training_checkpoint, sha256_file
from fog_lmw.data import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer", default="tokenizer/tinystories_3k_bpe.json"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/fog_binding_v2_10m_token_lookup_bf16.pt",
    )
    parser.add_argument("--forward-backward", action="store_true")
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = FOGReasonerConfig(**payload["model_config"])
    model = FOGLatentReasoner(config)
    load_training_checkpoint(
        args.checkpoint,
        model=model,
        tokenizer_path=args.tokenizer,
        restore_rng=False,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = (
        FOG_BINDING_V2_10M_PARAMETER_COUNT
        if config.architecture_version == "query_bound_v2"
        else FOG_10M_PARAMETER_COUNT
    )
    if parameters != expected_parameters:
        raise AssertionError(
            f"parameter mismatch for {config.architecture_version}: "
            f"expected {expected_parameters}, got {parameters}"
        )
    if model.token.weight is not model.lm_head.weight:
        raise AssertionError("token embedding and LM head are not tied")
    if config.readout_mode == "direct_latent" and model.token.weight is not model.direct_head.weight:
        raise AssertionError("token embedding and direct head are not tied")
    probe = "FOG release check: мир, 你好"
    if tokenizer.decode(tokenizer.encode(probe)) != probe:
        raise AssertionError("tokenizer UTF-8 round trip failed")

    result: dict[str, object] = {
        "ok": True,
        "parameters": parameters,
        "architecture_version": config.architecture_version,
        "vocab_size": tokenizer.vocab_size,
        "tied_weights": True,
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": payload["global_step"],
        "consumed_tokens": payload.get("consumed_tokens", 0),
        "checkpoint_kind": payload.get("checkpoint_kind", "training"),
        "checkpoint_metadata": payload.get("metadata", payload.get("extra", {})),
    }
    if args.forward_backward:
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model.train()
        prompt_tokens = tokenizer.encode("Question: What is 19 + 23?", add_bos=True, add_eos=True)
        answer_tokens = tokenizer.encode("42", add_bos=True, add_eos=True)
        prompt = torch.tensor([prompt_tokens], dtype=torch.long)
        answer = torch.tensor([answer_tokens], dtype=torch.long)
        loss, _ = model(
            prompt,
            answer,
            decoder_prompt_ids=torch.tensor([[tokenizer.bos_token_id]]),
        )
        loss.backward()
        if config.architecture_version == "query_bound_v2":
            checked_grads = [
                model.planner.bind.q_proj.weight.grad,
                model.planner.bind.k_proj.weight.grad,
            ]
            gradient_label = "binding_gradient_norm"
        else:
            checked_grads = [
                parameter.grad
                for parameter in model.memory.compress.parameters()
                if parameter.grad is not None
            ]
            gradient_label = "compressor_gradient_norm"
        if not checked_grads or any(grad is None for grad in checked_grads):
            raise AssertionError("required architecture gradients are missing")
        if not all(torch.isfinite(grad).all() for grad in checked_grads):
            raise AssertionError("required architecture gradients are non-finite")
        result["latent_loss"] = float(loss.detach())
        result[gradient_label] = float(
            torch.sqrt(sum(grad.float().pow(2).sum() for grad in checked_grads))
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
