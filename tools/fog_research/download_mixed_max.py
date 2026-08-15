import os
import json
import sys
import time
from datasets import load_dataset


def count_lines(path):
    n = 0
    with open(path, encoding="utf-8") as fh:
        for _ in fh:
            n += 1
    return n


def main():
    fw_cap = int(os.environ.get("FW_CAP", "1500000"))
    os.makedirs("data", exist_ok=True)
    train_file = "data/train_mixed_max.jsonl"
    eval_file = "data/eval_mixed_max.jsonl"

    count = count_lines(train_file) + count_lines(eval_file)
    print(f"Resume: {count} rows already written", flush=True)

    print("Loading TinyStories (full train split) ...", flush=True)
    ts_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    try:
        ts_total = ts_dataset.info.splits["train"].num_examples
    except Exception:
        ts_total = 2_119_719
    print(f"TinyStories total rows: {ts_total}", flush=True)

    ts_done = count >= ts_total
    if ts_done:
        print("TinyStories complete, moving to FineWeb-Edu.", flush=True)
    else:
        print(f"Continuing TinyStories from row {count}", flush=True)

    print("Loading FineWeb-Edu sample-10BT ...", flush=True)
    try:
        fw_dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
        )
    except Exception as e:
        print(f"Error loading FineWeb-Edu: {e}. Falling back to more TinyStories.", flush=True)
        fw_dataset = None

    start = time.perf_counter()

    def log_progress(src, n):
        el = time.perf_counter() - start
        rate = n / max(el, 1e-6)
        print(
            f"[{src}] {n} rows this run | total {count} | {rate:.0f} rows/s | "
            f"elapsed {el/60:.1f} min",
            flush=True,
        )

    with open(train_file, "a", encoding="utf-8") as f_train, open(eval_file, "a", encoding="utf-8") as f_eval:
        def write_row(record):
            nonlocal count
            if count % 20 == 0:
                f_eval.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                f_train.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

        if not ts_done:
            n = 0
            for item in ts_dataset.skip(count):
                text = item.get("text", "")
                if not text or not text.strip():
                    continue
                write_row({"text": text})
                n += 1
                if n % 100_000 == 0:
                    log_progress("TinyStories", n)
            log_progress("TinyStories", n)
            print(f"TinyStories finished, total {count}", flush=True)
        else:
            print(f"TinyStories already complete (total {count})", flush=True)

        if fw_dataset is None:
            print("FineWeb-Edu unavailable; falling back to repeated TinyStories.", flush=True)
            fw_dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

        fw_offset = max(0, count - ts_total)
        print(f"FineWeb-Edu: skipping {fw_offset}, cap {fw_cap}", flush=True)
        n = 0
        for item in fw_dataset.skip(fw_offset):
            text = item.get("text", "")
            if not text or not text.strip():
                continue
            write_row({"text": text})
            n += 1
            if n % 100_000 == 0:
                log_progress("FineWeb-Edu", n)
            if n >= fw_cap:
                break
        log_progress("FineWeb-Edu", n)

    print(f"Done! total rows now: {count}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
