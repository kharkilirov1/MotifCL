#!/usr/bin/env python3
"""Generate deterministic disjoint addition pairs for a latent-memory gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATES = (
    "What is {a} plus {b}?",
    "Add {a} and {b}.",
    "Calculate the sum of {a} and {b}.",
    "A box has {a} red balls and {b} blue balls. How many balls are there?",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-number", type=int, default=19)
    parser.add_argument("--train-output", default="data_cache/addition_train.jsonl")
    parser.add_argument("--validation-output", default="data_cache/addition_validation.jsonl")
    args = parser.parse_args()
    if args.max_number < 2:
        raise ValueError("max-number must be >= 2")
    train, validation = [], []
    for a in range(args.max_number + 1):
        for b in range(args.max_number + 1):
            template_index = (a * 7 + b * 11) % len(TEMPLATES)
            row = {
                "question": TEMPLATES[template_index].format(a=a, b=b),
                "answer": f"{a} + {b} = {a + b}.\n#### {a + b}",
                "a": a,
                "b": b,
                "template": template_index,
            }
            # Stable pair-disjoint 80/20 split. Because 53 and 97 are both 2
            # modulo 5, the rule is symmetric in a and b: a validation pair's
            # swapped pair is also validation, so neither orientation leaks
            # into train.
            target = validation if (a * 53 + b * 97) % 5 == 0 else train
            target.append(row)
    for path_string, rows in (
        (args.train_output, train),
        (args.validation_output, validation),
    ):
        path = Path(path_string)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps({"train": len(train), "validation": len(validation)}))


if __name__ == "__main__":
    main()
