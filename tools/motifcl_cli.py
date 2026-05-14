#!/usr/bin/env python3
r"""One-command MotifCL launcher.

This is the Ollama-like front door for local development:

    .\motifcl.cmd             # build-if-needed, auto-pick local model, start hot server
    .\motifcl.cmd run "Hi"    # auto-start server and stream an answer
    .\motifcl.cmd down        # stop the launcher-owned server

The implementation intentionally stays stdlib-only and delegates serving to
`tools/motifcl_ollama_server.py`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools" / "motifcl_ollama_server.py"
STATE_DIR = ROOT / "build" / "motifcl_launcher"
STATE_FILE = STATE_DIR / "server.json"
STDOUT_LOG = STATE_DIR / "server.out.log"
STDERR_LOG = STATE_DIR / "server.err.log"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11435


def log(message: str) -> None:
    print(f"[motifcl] {message}", flush=True)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def exe_suffix() -> str:
    return ".exe" if os.name == "nt" else ""


def default_build_dir() -> Path:
    preferred = ROOT / "build_kquant_prefill"
    return preferred if preferred.exists() else ROOT / "build"


def runner_in_build(build_dir: Path) -> Path:
    return build_dir / "tools" / f"motifcl_generate_transformer{exe_suffix()}"


def existing_runner_candidates() -> list[Path]:
    names = [
        ROOT / "build_kquant_prefill" / "tools" / f"motifcl_generate_transformer{exe_suffix()}",
        ROOT / "build" / "tools" / f"motifcl_generate_transformer{exe_suffix()}",
        ROOT / "build" / "release" / "tools" / f"motifcl_generate_transformer{exe_suffix()}",
    ]
    if os.name == "nt":
        names.extend([
            ROOT / "build_kquant_prefill" / "tools" / "motifcl_generate_transformer",
            ROOT / "build" / "tools" / "motifcl_generate_transformer",
        ])
    return names


def run_checked(cmd: list[str], cwd: Path = ROOT) -> None:
    log("running: " + " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_runner(args: argparse.Namespace) -> Path:
    if getattr(args, "runner", None):
        runner = Path(args.runner).resolve()
        if not runner.exists():
            raise SystemExit(f"runner not found: {runner}")
        return runner

    for candidate in existing_runner_candidates():
        if candidate.exists() and not getattr(args, "rebuild", False):
            return candidate.resolve()

    build_dir = Path(getattr(args, "build_dir", default_build_dir())).resolve()
    runner = runner_in_build(build_dir)
    if getattr(args, "no_build", False):
        raise SystemExit(f"runner not found and --no-build is set: {runner}")

    if not (build_dir / "CMakeCache.txt").exists():
        run_checked(["cmake", "-S", str(ROOT), "-B", str(build_dir), "-DMOTIFCL_BUILD_TOOLS=ON"])
    jobs = str(getattr(args, "jobs", os.cpu_count() or 4))
    run_checked(["cmake", "--build", str(build_dir), "--target", "motifcl_generate_transformer", "-j", jobs])
    if not runner.exists():
        raise SystemExit(f"build completed but runner is missing: {runner}")
    return runner.resolve()


def model_name_from_path(path: Path) -> str:
    if path.is_file():
        parent = path.parent.name
        stem = path.stem
        name = parent if parent and parent.lower() not in {"models", "build"} else stem
    else:
        name = path.name
    name = re.sub(r"-?gguf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^A-Za-z0-9_.:-]+", "-", name).strip("-_.:")
    return name or "motifcl-local"


def model_aliases(path: Path) -> set[str]:
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    raw = {path.name, path.stem, path.parent.name, model_name_from_path(path), rel}
    if path.is_file():
        raw.add(path.parent.stem)
    return {normalize(x) for x in raw if x}


def discover_models() -> list[Path]:
    roots = [
        ROOT / "build" / "models",
        ROOT / "build_kquant_prefill" / "models",
        ROOT / "models",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.gguf"):
            resolved = path.resolve()
            if resolved not in seen:
                found.append(resolved)
                seen.add(resolved)
    found.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return found


def resolve_model(value: str | None) -> Path:
    if value:
        candidate = Path(value)
        if candidate.exists():
            return candidate.resolve()
        candidate = ROOT / value
        if candidate.exists():
            return candidate.resolve()

    models = discover_models()
    if not value:
        if not models:
            raise SystemExit(
                "no local GGUF model found. Put a .gguf under build/models or pass a model path, e.g.\n"
                "  .\\motifcl.cmd up .\\path\\model.gguf"
            )
        return models[0]

    wanted = normalize(value)
    exact = [p for p in models if wanted in model_aliases(p)]
    if exact:
        return exact[0]
    fuzzy = [p for p in models if any(wanted in alias or alias in wanted for alias in model_aliases(p))]
    if fuzzy:
        return fuzzy[0]
    raise SystemExit(f"model not found: {value}. Run `.\\motifcl.cmd list` to see discovered models.")


def read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_state(data: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def http_json(host: str, port: int, path: str, body: dict[str, Any] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    url = f"http://{host}:{port}{path}"
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 local-only URL
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def health(host: str, port: int, timeout: float = 2.0) -> dict[str, Any] | None:
    try:
        return http_json(host, port, "/health", timeout=timeout)
    except Exception:
        return None


def wait_ready(host: str, port: int, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        last = health(host, port, timeout=2.0)
        if last and last.get("ok") and last.get("worker_alive"):
            return last
        time.sleep(0.5)
    raise SystemExit(f"server did not become ready within {timeout:.1f}s; last_health={last}; stderr={STDERR_LOG}")


def kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def stop_owned_server() -> bool:
    state = read_state()
    pid = int(state.get("pid") or 0)
    if not pid:
        return False
    kill_pid(pid)
    for _ in range(20):
        time.sleep(0.1)
        if not health(str(state.get("host", DEFAULT_HOST)), int(state.get("port", DEFAULT_PORT)), timeout=0.5):
            break
    try:
        STATE_FILE.unlink()
    except FileNotFoundError:
        pass
    return True


def server_command(args: argparse.Namespace, model: Path, runner: Path, name: str) -> list[str]:
    cmd = [
        sys.executable,
        str(SERVER),
        "--model", str(model),
        "--name", name,
        "--runner", str(runner),
        "--host", str(args.host),
        "--port", str(args.port),
        "--ctx-size", str(args.ctx_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top-k", str(args.top_k),
        "--top-p", str(args.top_p),
        "--startup-timeout", str(args.startup_timeout),
        "--warmup-tokens", str(args.warmup_tokens),
    ]
    if args.arch:
        cmd += ["--arch", args.arch]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.random_init:
        cmd.append("--random-init")
    if args.no_prefill:
        cmd.append("--no-prefill")
    if args.force_prefill:
        cmd.append("--force-prefill")
    if args.disable_adaptive_prefill:
        cmd.append("--disable-adaptive-prefill")
    if args.cpu_sampling:
        cmd.append("--cpu-sampling")
    if args.no_warmup:
        cmd.append("--no-warmup")
    if args.quiet:
        cmd.append("--quiet")
    for extra in args.extra_arg or []:
        cmd += ["--extra-arg", extra]
    return cmd


def ensure_up(args: argparse.Namespace, model_value: str | None = None) -> dict[str, Any]:
    model = resolve_model(model_value or getattr(args, "model", None))
    name = args.name or model_name_from_path(model)
    current = health(args.host, args.port)
    if current and current.get("ok"):
        if current.get("model") == name:
            log(f"already ready: http://{args.host}:{args.port} model={name}")
            return {"model": name, "model_path": str(model), "already_running": True}
        state = read_state()
        if int(state.get("port") or 0) == args.port and state.get("pid"):
            log(f"restarting launcher-owned server on port {args.port}: {current.get('model')} -> {name}")
            stop_owned_server()
        else:
            raise SystemExit(
                f"port {args.port} already serves model={current.get('model')}; use --port or stop that server first"
            )

    runner = ensure_runner(args)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    out = STDOUT_LOG.open("ab")
    err = STDERR_LOG.open("ab")
    cmd = server_command(args, model, runner, name)
    log(f"starting hot server: model={name} port={args.port}")
    creationflags = 0
    start_new_session = False
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        start_new_session = True
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=out,
        stderr=err,
        creationflags=creationflags,
        start_new_session=start_new_session,
    )
    write_state({
        "pid": proc.pid,
        "host": args.host,
        "port": args.port,
        "model": name,
        "model_path": str(model),
        "runner": str(runner),
        "command": cmd,
        "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "stdout_log": str(STDOUT_LOG),
        "stderr_log": str(STDERR_LOG),
    })
    ready = wait_ready(args.host, args.port, args.startup_timeout + max(5.0, args.warmup_tokens * 5.0))
    log(f"ready: http://{args.host}:{args.port} model={ready.get('model')}")
    log(f"logs: {STDERR_LOG}")
    return {"model": name, "model_path": str(model), "pid": proc.pid}


def command_up(args: argparse.Namespace) -> int:
    ensure_up(args, args.model)
    return 0


def command_serve(args: argparse.Namespace) -> int:
    model = resolve_model(args.model)
    name = args.name or model_name_from_path(model)
    runner = ensure_runner(args)
    cmd = server_command(args, model, runner, name)
    log(f"serving foreground: http://{args.host}:{args.port} model={name}")
    return subprocess.call(cmd, cwd=str(ROOT))


def command_down(args: argparse.Namespace) -> int:
    stopped = stop_owned_server()
    log("stopped" if stopped else "no launcher-owned server pidfile found")
    return 0


def command_status(args: argparse.Namespace) -> int:
    current = health(args.host, args.port)
    state = read_state()
    if current and current.get("ok"):
        log(f"ready: http://{args.host}:{args.port} model={current.get('model')} worker_alive={current.get('worker_alive')}")
    else:
        log(f"not responding on http://{args.host}:{args.port}")
    if state:
        log(f"pid={state.get('pid')} model_path={state.get('model_path')} stderr={state.get('stderr_log')}")
    return 0


def command_list(_: argparse.Namespace) -> int:
    models = discover_models()
    if not models:
        log("no .gguf models discovered under build/models, build_kquant_prefill/models, or models")
        return 1
    for idx, path in enumerate(models, start=1):
        name = model_name_from_path(path)
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"{idx}. {name}\t{rel}")
    return 0


def split_run_args(args: argparse.Namespace) -> tuple[str | None, str]:
    if args.model:
        return args.model, args.prompt or " ".join(args.text)
    if args.prompt:
        return None, args.prompt
    if not args.text:
        return None, ""
    first = args.text[0]
    try:
        resolve_model(first)
        return first, " ".join(args.text[1:])
    except SystemExit:
        return None, " ".join(args.text)


def stream_generate(args: argparse.Namespace, prompt: str) -> None:
    body = {
        "model": args.name or "motifcl-local",
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": args.num_predict or args.max_new_tokens,
            "temperature": args.request_temperature,
            "top_k": args.request_top_k,
            "top_p": args.request_top_p,
        },
    }
    if args.ignore_eos:
        body["options"]["ignore_eos"] = True
    if args.seed is not None:
        body["options"]["seed"] = args.seed
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}/api/generate",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:  # noqa: S310 local-only URL
        for raw in resp:
            if not raw.strip():
                continue
            event = json.loads(raw.decode("utf-8"))
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("done"):
                if args.show_timing and event.get("worker_total_ms") is not None:
                    print(f"\n[motifcl] worker_total_ms={event.get('worker_total_ms')}", file=sys.stderr)
                break
            chunk = str(event.get("response", ""))
            if chunk:
                print(chunk, end="", flush=True)
    print()


def command_run(args: argparse.Namespace) -> int:
    model_value, prompt = split_run_args(args)
    up = ensure_up(args, model_value)
    if not args.name:
        args.name = str(up.get("model") or "motifcl-local")
    if prompt:
        stream_generate(args, prompt)
        return 0
    log("interactive mode; Ctrl+C or empty line exits")
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        stream_generate(args, line)
    return 0


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--name", default=None, help="API model name")
    p.add_argument("--runner", default=None, help="explicit motifcl_generate_transformer binary")
    p.add_argument("--build-dir", type=Path, default=default_build_dir())
    p.add_argument("--jobs", type=int, default=min(os.cpu_count() or 4, 8))
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--ctx-size", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0, help="server default temperature")
    p.add_argument("--top-k", type=int, default=0, help="server default top_k")
    p.add_argument("--top-p", type=float, default=1.0, help="server default top_p")
    p.add_argument("--arch", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--no-prefill", action="store_true")
    p.add_argument("--force-prefill", action="store_true")
    p.add_argument("--disable-adaptive-prefill", action="store_true")
    p.add_argument("--cpu-sampling", action="store_true")
    p.add_argument("--extra-arg", action="append")
    p.add_argument("--startup-timeout", type=float, default=300.0)
    p.add_argument("--warmup-tokens", type=int, default=1)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--quiet", action="store_true", default=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ollama-like one-command launcher for MotifCL")
    sub = parser.add_subparsers(dest="command")

    up = sub.add_parser("up", help="build-if-needed and start a background hot server")
    add_common(up)
    up.add_argument("model", nargs="?", help="model path or discovered alias; default: newest local GGUF")
    up.set_defaults(func=command_up)

    serve = sub.add_parser("serve", help="foreground server, like ollama serve")
    add_common(serve)
    serve.add_argument("model", nargs="?", help="model path or discovered alias; default: newest local GGUF")
    serve.set_defaults(func=command_serve)

    run = sub.add_parser("run", help="auto-start server and stream a prompt, like ollama run")
    add_common(run)
    run.add_argument("--model", default=None, help="model path or discovered alias")
    run.add_argument("--prompt", default=None)
    run.add_argument("--num-predict", type=int, default=None)
    run.add_argument("--request-temperature", type=float, default=0.0)
    run.add_argument("--request-top-k", type=int, default=0)
    run.add_argument("--request-top-p", type=float, default=1.0)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--ignore-eos", action="store_true")
    run.add_argument("--request-timeout", type=float, default=3600.0)
    run.add_argument("--show-timing", action="store_true")
    run.add_argument("text", nargs="*", help="optional [model] prompt...; if first arg is a model alias, it is used")
    run.set_defaults(func=command_run)

    down = sub.add_parser("down", help="stop the launcher-owned background server")
    down.add_argument("--host", default=DEFAULT_HOST)
    down.add_argument("--port", type=int, default=DEFAULT_PORT)
    down.set_defaults(func=command_down)

    status = sub.add_parser("status", help="show server status")
    status.add_argument("--host", default=DEFAULT_HOST)
    status.add_argument("--port", type=int, default=DEFAULT_PORT)
    status.set_defaults(func=command_status)

    list_cmd = sub.add_parser("list", help="list discovered local GGUF models")
    list_cmd.set_defaults(func=command_list)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    if not argv:
        argv = ["up"]
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print()
        return 130
    except urllib.error.URLError as exc:
        raise SystemExit(f"HTTP error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
