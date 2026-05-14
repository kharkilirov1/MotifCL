#!/usr/bin/env python3
"""Small Ollama-compatible local HTTP front-end for MotifCL.

The server keeps `motifcl_generate_transformer --jsonl-repl` alive, so model
weights are loaded once and subsequent `/api/generate` calls reuse the hot
runtime.  It intentionally uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import queue
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_runner() -> Path:
    root = _repo_root()
    candidates = [
        root / "build_kquant_prefill" / "tools" / "motifcl_generate_transformer.exe",
        root / "build" / "tools" / "motifcl_generate_transformer.exe",
        root / "build" / "release" / "tools" / "motifcl_generate_transformer.exe",
        root / "build_kquant_prefill" / "tools" / "motifcl_generate_transformer",
        root / "build" / "tools" / "motifcl_generate_transformer",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _ns_since(start: float) -> int:
    return int((time.perf_counter() - start) * 1_000_000_000)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _ollama_options(body: dict[str, Any]) -> dict[str, Any]:
    opts = body.get("options") if isinstance(body.get("options"), dict) else {}
    out: dict[str, Any] = {}
    for src, dst in [
        ("num_predict", "num_predict"),
        ("max_new_tokens", "max_new_tokens"),
        ("temperature", "temperature"),
        ("top_k", "top_k"),
        ("top_p", "top_p"),
        ("seed", "seed"),
        ("ignore_eos", "ignore_eos"),
    ]:
        if src in body:
            out[dst] = body[src]
        if src in opts:
            out[dst] = opts[src]
    return out


def _chat_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        if content:
            lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def _stream_chunks(text: str, target_chars: int = 96) -> list[str]:
    if not text:
        return [""]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for part in text.split(" "):
        piece = part if not current else " " + part
        current.append(piece)
        size += len(piece)
        if size >= target_chars:
            chunks.append("".join(current))
            current = []
            size = 0
    if current:
        chunks.append("".join(current))
    return chunks


class MotifCLWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.Lock()
        self.proc: subprocess.Popen[str] | None = None
        self.stderr_tail: "queue.Queue[str]" = queue.Queue(maxsize=256)
        self.ready = threading.Event()
        self.startup_ms = 0.0
        self.start()

    def command(self) -> list[str]:
        cmd = [
            str(self.args.runner),
            "--model",
            str(self.args.model),
            "--jsonl-repl",
            "--completion-only",
            "--max-new-tokens",
            str(self.args.max_new_tokens),
            "--temperature",
            str(self.args.temperature),
            "--top-k",
            str(self.args.top_k),
            "--top-p",
            str(self.args.top_p),
        ]
        if self.args.ctx_size:
            cmd += ["--ctx-size", str(self.args.ctx_size)]
        if self.args.arch:
            cmd += ["--arch", self.args.arch]
        if self.args.tokenizer:
            cmd += ["--tokenizer", str(self.args.tokenizer)]
        if self.args.random_init:
            cmd.append("--random-init")
        if self.args.no_prefill:
            cmd.append("--no-prefill")
        if self.args.force_prefill:
            cmd.append("--force-prefill")
        if self.args.disable_adaptive_prefill:
            cmd.append("--disable-adaptive-prefill")
        if self.args.cpu_sampling:
            cmd.append("--cpu-sampling")
        for extra in self.args.extra_arg or []:
            cmd.append(extra)
        return cmd

    def start(self) -> None:
        if self.proc and self.proc.poll() is None:
            return
        self.ready.clear()
        started = time.perf_counter()
        self.proc = subprocess.Popen(
            self.command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_repo_root()),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        if not self.ready.wait(timeout=self.args.startup_timeout):
            if self.proc.poll() is not None:
                raise RuntimeError(f"MotifCL worker exited during startup; logs={self.recent_logs()}")
            print(
                f"warning: MotifCL worker did not report ready within {self.args.startup_timeout:.1f}s",
                file=sys.stderr,
            )
        self.startup_ms = (time.perf_counter() - started) * 1000.0

    def _drain_stderr(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            if "REPL ready" in line:
                self.ready.set()
            try:
                self.stderr_tail.put_nowait(line.rstrip())
            except queue.Full:
                try:
                    self.stderr_tail.get_nowait()
                except queue.Empty:
                    pass
                self.stderr_tail.put_nowait(line.rstrip())

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def recent_logs(self) -> list[str]:
        items = list(self.stderr_tail.queue)
        return items[-32:]

    def generate(self, prompt: str, options: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if not self.alive():
                self.start()
            assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
            request = {"prompt": prompt, **options}
            self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
            if not line:
                code = self.proc.poll()
                raise RuntimeError(f"MotifCL worker exited before response; code={code}; logs={self.recent_logs()}")
            data = json.loads(line)
            if not data.get("ok", False):
                raise RuntimeError(str(data.get("error", "MotifCL generation failed")))
            return data

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


class MotifCLHandler(BaseHTTPRequestHandler):
    server_version = "MotifCLOllamaCompat/0.1"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    @property
    def worker(self) -> MotifCLWorker:
        return self.server.worker  # type: ignore[attr-defined]

    @property
    def model_name(self) -> str:
        return self.server.model_name  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/health"):
            self._send_json({"ok": True, "model": self.model_name, "worker_alive": self.worker.alive()})
            return
        if self.path == "/api/version":
            self._send_json({"version": "motifcl-ollama-compat-0.1"})
            return
        if self.path in ("/api/tags", "/v1/models"):
            model = {
                "name": self.model_name,
                "model": self.model_name,
                "modified_at": _now_iso(),
                "size": 0,
                "digest": "motifcl-local",
                "details": {"format": "motifcl", "family": "local"},
            }
            if self.path == "/v1/models":
                self._send_json({"object": "list", "data": [{"id": self.model_name, "object": "model"}]})
            else:
                self._send_json({"models": [model]})
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/generate":
                self._handle_generate()
            elif self.path == "/api/chat":
                self._handle_chat()
            elif self.path == "/api/show":
                self._send_json({"modelfile": f"FROM {self.model_name}\n", "details": {"family": "motifcl"}})
            elif self.path in ("/v1/completions", "/v1/chat/completions"):
                self._handle_openai_compat()
            else:
                self._send_json({"error": "not found"}, 404)
        except Exception as exc:  # keep daemon alive and return JSON errors
            self._send_json({"error": str(exc), "worker_logs": self.worker.recent_logs()}, 500)

    def _handle_generate(self, body: dict[str, Any] | None = None) -> None:
        if body is None:
            body = self._read_json()
        prompt = str(body.get("prompt", ""))
        stream = bool(body.get("stream", True))
        start = time.perf_counter()
        result = self.worker.generate(prompt, _ollama_options(body))
        response = str(result.get("response", ""))
        total_duration = _ns_since(start)
        created = _now_iso()
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.end_headers()
            for chunk in _stream_chunks(response):
                payload = {"model": self.model_name, "created_at": created, "response": chunk, "done": False}
                self.wfile.write(_json_bytes(payload) + b"\n")
                self.wfile.flush()
            final = {
                "model": self.model_name,
                "created_at": created,
                "response": "",
                "done": True,
                "total_duration": total_duration,
                "load_duration": 0,
                "worker_total_ms": result.get("total_ms"),
            }
            self.wfile.write(_json_bytes(final) + b"\n")
            return
        self._send_json(
            {
                "model": self.model_name,
                "created_at": created,
                "response": response,
                "done": True,
                "total_duration": total_duration,
                "load_duration": 0,
                "worker_total_ms": result.get("total_ms"),
            }
        )

    def _handle_chat(self) -> None:
        body = self._read_json()
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        prompt = _chat_prompt(messages)
        stream = bool(body.get("stream", True))
        start = time.perf_counter()
        result = self.worker.generate(prompt, _ollama_options(body))
        response = str(result.get("response", ""))
        total_duration = _ns_since(start)
        created = _now_iso()
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.end_headers()
            for chunk in _stream_chunks(response):
                payload = {
                    "model": self.model_name,
                    "created_at": created,
                    "message": {"role": "assistant", "content": chunk},
                    "done": False,
                }
                self.wfile.write(_json_bytes(payload) + b"\n")
                self.wfile.flush()
            final = {
                "model": self.model_name,
                "created_at": created,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": total_duration,
                "load_duration": 0,
                "worker_total_ms": result.get("total_ms"),
            }
            self.wfile.write(_json_bytes(final) + b"\n")
            return
        self._send_json(
            {
                "model": self.model_name,
                "created_at": created,
                "message": {"role": "assistant", "content": response},
                "done": True,
                "total_duration": total_duration,
                "load_duration": 0,
                "worker_total_ms": result.get("total_ms"),
            }
        )

    def _handle_openai_compat(self) -> None:
        body = self._read_json()
        if self.path == "/v1/chat/completions":
            messages = body.get("messages") if isinstance(body.get("messages"), list) else []
            prompt = _chat_prompt(messages)
        else:
            prompt = str(body.get("prompt", ""))
        options: dict[str, Any] = {}
        if "max_tokens" in body:
            options["num_predict"] = body["max_tokens"]
        if "temperature" in body:
            options["temperature"] = body["temperature"]
        result = self.worker.generate(prompt, options)
        text = str(result.get("response", ""))
        created = int(time.time())
        if self.path == "/v1/chat/completions":
            payload = {
                "id": "motifcl-chatcmpl",
                "object": "chat.completion",
                "created": created,
                "model": self.model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            }
        else:
            payload = {
                "id": "motifcl-cmpl",
                "object": "text_completion",
                "created": created,
                "model": self.model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
            }
        self._send_json(payload)

    def log_message(self, fmt: str, *args: Any) -> None:
        if not getattr(self.server, "quiet", False):  # type: ignore[attr-defined]
            super().log_message(fmt, *args)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MotifCL Ollama-compatible local server")
    parser.add_argument("--model", required=True, help="HF model directory or .gguf file")
    parser.add_argument("--name", default=None, help="API model name (default: model path name)")
    parser.add_argument("--runner", type=Path, default=_default_runner())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--no-prefill", action="store_true")
    parser.add_argument("--force-prefill", action="store_true")
    parser.add_argument("--disable-adaptive-prefill", action="store_true")
    parser.add_argument("--cpu-sampling", action="store_true")
    parser.add_argument("--extra-arg", action="append", help="Raw extra argument passed to motifcl_generate_transformer")
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--warmup-prompt", default="Hello")
    parser.add_argument("--warmup-tokens", type=int, default=1)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.runner = Path(args.runner).resolve()
    if not args.runner.exists():
        raise SystemExit(f"runner not found: {args.runner}")
    model_name = args.name or Path(args.model).stem or "motifcl-local"
    worker = MotifCLWorker(args)
    if not args.no_warmup and args.warmup_tokens > 0:
        warm_started = time.perf_counter()
        worker.generate(args.warmup_prompt, {"num_predict": args.warmup_tokens, "ignore_eos": True})
        print(f"warmup_ms={(time.perf_counter() - warm_started) * 1000.0:.3f}", file=sys.stderr)
    server = ThreadingHTTPServer((args.host, args.port), MotifCLHandler)
    server.worker = worker  # type: ignore[attr-defined]
    server.model_name = model_name  # type: ignore[attr-defined]
    server.quiet = args.quiet  # type: ignore[attr-defined]
    print(f"MotifCL Ollama-compatible server listening on http://{args.host}:{args.port}", file=sys.stderr)
    print(f"model={model_name} runner={args.runner}", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
