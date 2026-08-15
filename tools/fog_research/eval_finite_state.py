"""Load a saved finite-state checkpoint and reproduce its exact metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fog_lmw import FOGReasonerConfig, FOGLatentReasoner
from train_finite_state import evaluate_lengths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        default=Path("artifacts/finite_state_bottleneck_seed0/recurrent.pt"),
    )
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FOGLatentReasoner(FOGReasonerConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    variant = payload["training"]["variant"]
    decoder_bottleneck = payload["training"].get("decoder_bottleneck", False)
    result = {
        "checkpoint": str(args.checkpoint),
        "variant": variant,
        "decoder_bottleneck": decoder_bottleneck,
        "depths_1_4": evaluate_lengths(
            model,
            (1, 2, 3, 4),
            variant=variant,
            decoder_bottleneck=decoder_bottleneck,
        ),
        "depths_5_8": evaluate_lengths(
            model,
            (5, 6, 7, 8),
            variant=variant,
            decoder_bottleneck=decoder_bottleneck,
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
