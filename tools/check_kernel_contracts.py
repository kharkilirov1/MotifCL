#!/usr/bin/env python3
"""Verify that every concrete OpenCL kernel is covered by a validation contract family."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACTS = ROOT / "docs" / "kernel_validation_contracts.json"
KERNEL_RE = re.compile(r"^\s*__kernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def kernel_names() -> list[str]:
    names: set[str] = set()
    for path in sorted((ROOT / "kernels").glob("*.cl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = KERNEL_RE.match(line)
            if match:
                names.add(match.group(1))
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--list", action="store_true", help="print covered kernel count and unmatched names")
    args = parser.parse_args()

    data = json.loads(args.contracts.read_text(encoding="utf-8"))
    ignored = set(data.get("ignored_kernel_names", []))
    families = data.get("families", [])
    patterns = [(item.get("name", "<unnamed>"), re.compile(item["pattern"])) for item in families]
    names = [name for name in kernel_names() if name not in ignored]
    unmatched = [name for name in names if not any(pattern.match(name) for _, pattern in patterns)]

    if args.list or unmatched:
        print(f"kernel_contracts: kernels={len(names)} families={len(families)} unmatched={len(unmatched)}")
        for name in unmatched:
            print(f"unmatched: {name}")
    if unmatched:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
