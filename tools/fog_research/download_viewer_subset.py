#!/usr/bin/env python3
"""Download a deterministic JSONL slice through HF Dataset Viewer API.

This helper is for small reproducible experiments when `datasets` streaming is
unavailable. It rejects truncated cells and records source row indices and a
content hash. It is not intended for mirroring full datasets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode


BASE_URL = "https://datasets-server.huggingface.co/rows"


def fetch_page(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> dict:
    query = urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    completed = subprocess.run(
        [
            "curl",
            "--retry",
            "6",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "20",
            "--max-time",
            "90",
            "-fsSL",
            f"{BASE_URL}?{query}",
        ],
        check=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.offset < 0 or args.limit < 1:
        raise ValueError("offset must be >= 0 and limit must be >= 1")
    if not 1 <= args.page_size <= 100:
        raise ValueError("page-size must be in [1, 100]")
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    requests = []
    stop = args.offset + args.limit
    for start in range(args.offset, stop, args.page_size):
        requests.append((start, min(args.page_size, stop - start)))
    pages: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                fetch_page,
                args.dataset,
                args.config,
                args.split,
                start,
                length,
            ): start
            for start, length in requests
        }
        for future in as_completed(futures):
            start = futures[future]
            pages[start] = future.result()
            print(f"downloaded offset {start}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    total_rows = None
    for start, requested_length in requests:
        page = pages[start]
        total_rows = page.get("num_rows_total", total_rows)
        if page.get("partial"):
            raise RuntimeError(f"Dataset Viewer returned a partial page at {start}")
        page_rows = page.get("rows", [])
        if len(page_rows) != requested_length:
            raise RuntimeError(
                f"expected {requested_length} rows at {start}, got {len(page_rows)}"
            )
        for item in page_rows:
            if item.get("truncated_cells"):
                raise RuntimeError(f"truncated cell at source row {item['row_idx']}")
            expected_index = args.offset + len(rows)
            if item["row_idx"] != expected_index:
                raise RuntimeError(
                    f"row index discontinuity: {item['row_idx']} != {expected_index}"
                )
            row = {"_source_row_idx": item["row_idx"], **item["row"]}
            rows.append(row)

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "offset": args.offset,
        "limit": len(rows),
        "source_total_rows": total_rows,
        "sha256": sha256,
        "viewer_endpoint": BASE_URL,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
