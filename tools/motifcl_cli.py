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
DEFAULT_AUTO_CTX_SIZE = 8192


def configure_stdio() -> None:
    """Keep streaming output usable on Windows when the model emits emoji/CJK."""
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


configure_stdio()


def log(message: str) -> None:
    print(f"[motifcl] {message}", flush=True)


def experimental_runtime_notes(args: argparse.Namespace) -> list[str]:
    notes: list[str] = []
    if getattr(args, "paged_kv", False):
        notes.append(f"experimental paged KV cache enabled explicitly (--paged-kv, page_size={args.kv_page_size})")
    kv_dtype = str(getattr(args, "kv_cache_dtype", "f32")).lower()
    if kv_dtype not in {"", "f32"}:
        notes.append(f"experimental compressed KV cache enabled explicitly (--kv-cache-dtype={kv_dtype})")
    return notes


def warn_experimental_runtime(args: argparse.Namespace) -> None:
    for note in experimental_runtime_notes(args):
        log(f"warning: {note}; keep it behind measured runs and stable fallback")


def path_entries(value: str | None = None) -> list[str]:
    raw = os.environ.get("PATH", "") if value is None else value
    return [x.strip('"') for x in raw.split(os.pathsep) if x]


def path_contains(directory: Path, value: str | None = None) -> bool:
    target = str(directory.resolve()).lower()
    return any(str(Path(entry).resolve()).lower() == target for entry in path_entries(value))


def default_shim_dir() -> Path:
    return Path.home() / ".local" / "bin"


def add_user_path_windows(directory: Path) -> bool:
    if os.name != "nt":
        return False
    import winreg

    key_path = "Environment"
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
        try:
            current, reg_type = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current, reg_type = "", winreg.REG_EXPAND_SZ
        if path_contains(directory, current):
            return False
        updated = current + (os.pathsep if current else "") + str(directory)
        winreg.SetValueEx(key, "Path", 0, reg_type, updated)
    os.environ["PATH"] = os.environ.get("PATH", "") + (os.pathsep if os.environ.get("PATH") else "") + str(directory)
    return True


def write_shim(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        shim = directory / "motifcl.cmd"
        shim.write_text(
            "@echo off\r\n"
            f'call "{ROOT / "motifcl.cmd"}" %*\r\n',
            encoding="ascii",
        )
    else:
        shim = directory / "motifcl"
        shim.write_text(
            "#!/usr/bin/env sh\n"
            f'exec "{ROOT / "motifcl.cmd"}" "$@"\n',
            encoding="utf-8",
        )
        shim.chmod(0o755)
    return shim


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


def infer_chat_template(model: Path, name: str, requested: str | None) -> str | None:
    value = (requested or "auto").strip().lower()
    if value in {"none", "raw", "off", "false", "no"}:
        return None
    if value and value != "auto":
        return value
    haystack = f"{name} {model.name} {model.parent.name} {model.stem}".lower()
    if "gemma" in haystack:
        return "gemma"
    if "qwen" in haystack:
        return "chatml"
    if "llama" in haystack:
        return "llama3"
    if "mistral" in haystack:
        return "mistral"
    return None


def auto_ctx_limit() -> int:
    raw = os.environ.get("MOTIFCL_CTX_AUTO_MAX", "").strip()
    if not raw:
        return DEFAULT_AUTO_CTX_SIZE
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_AUTO_CTX_SIZE
    return value if value > 0 else DEFAULT_AUTO_CTX_SIZE


def model_context_length(model: Path, runner: Path, arch: str | None = None) -> int | None:
    cmd = [str(runner), "--model", str(model), "--inspect"]
    if arch:
        cmd += ["--arch", arch]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30.0,
            check=False,
        )
    except Exception:
        return None
    match = re.search(r"^\s*context_length:\s*(\d+)\s*$", proc.stdout or "", flags=re.MULTILINE)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def resolve_ctx_size(value: str | int | None, model: Path, runner: Path, arch: str | None = None) -> int:
    native = model_context_length(model, runner, arch) or DEFAULT_AUTO_CTX_SIZE
    raw = str(value if value is not None else "auto").strip().lower()
    if raw in {"", "auto"}:
        return max(1, min(native, auto_ctx_limit()))
    if raw in {"max", "native", "model"}:
        return native
    try:
        requested = int(raw)
    except ValueError:
        raise SystemExit("--ctx-size must be an integer, auto, or max/native/model")
    if requested <= 0:
        return native
    return max(1, min(requested, native))


def discover_models() -> list[Path]:
    roots = [
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
                "no local GGUF model found. Put a .gguf under models/ or pass a model path, e.g.\n"
                "  .\\motifcl.cmd up .\\path\\model.gguf"
            )
        return models[0]

    wanted = normalize(value)
    exact = [p for p in models if wanted in model_aliases(p)]
    if exact:
        return exact[0]
    if not wanted:
        raise SystemExit(f"model not found: {value}. Run `.\\motifcl.cmd list` to see discovered models.")
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
    ctx_size = int(getattr(args, "resolved_ctx_size", args.ctx_size))
    chat_template = getattr(args, "resolved_chat_template", None)
    if chat_template is None:
        chat_template = infer_chat_template(model, name, getattr(args, "chat_template", "auto"))
    cmd = [
        sys.executable,
        str(SERVER),
        "--model", str(model),
        "--name", name,
        "--runner", str(runner),
        "--host", str(args.host),
        "--port", str(args.port),
        "--ctx-size", str(ctx_size),
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
    if chat_template:
        cmd += ["--chat-template", chat_template]
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
    if getattr(args, "paged_kv", False):
        cmd.append("--paged-kv")
        cmd += ["--kv-page-size", str(args.kv_page_size)]
    if getattr(args, "kv_cache_dtype", "f32"):
        cmd += ["--kv-cache-dtype", str(args.kv_cache_dtype)]
    if args.no_warmup:
        cmd.append("--no-warmup")
    if args.quiet:
        cmd.append("--quiet")
    for extra in args.extra_arg or []:
        cmd += ["--extra-arg", extra]
    return cmd


def desired_server_config(args: argparse.Namespace, model: Path, runner: Path, name: str) -> dict[str, Any]:
    ctx_size = resolve_ctx_size(getattr(args, "ctx_size", "auto"), model, runner, getattr(args, "arch", None))
    chat_template = infer_chat_template(model, name, getattr(args, "chat_template", "auto"))
    kv_cache_dtype = str(getattr(args, "kv_cache_dtype", "f32"))
    args.resolved_ctx_size = ctx_size
    args.resolved_chat_template = chat_template
    return {
        "model": name,
        "model_path": str(model),
        "ctx_size": ctx_size,
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "chat_template": chat_template,
        "paged_kv": bool(getattr(args, "paged_kv", False)),
        "kv_page_size": int(getattr(args, "kv_page_size", 256)),
        "kv_cache_dtype": kv_cache_dtype,
    }


def server_config_matches(current: dict[str, Any], desired: dict[str, Any]) -> tuple[bool, str]:
    for key, wanted in desired.items():
        got = current.get(key)
        if key == "model_path" and got is None:
            continue
        if isinstance(wanted, float):
            try:
                same = abs(float(got) - wanted) < 1e-6
            except Exception:
                same = False
        else:
            same = got == wanted
        if not same:
            return False, f"{key}: current={got!r} desired={wanted!r}"
    return True, ""


def ensure_up(args: argparse.Namespace, model_value: str | None = None) -> dict[str, Any]:
    model = resolve_model(model_value or getattr(args, "model", None))
    name = args.name or model_name_from_path(model)
    runner = ensure_runner(args)
    desired = desired_server_config(args, model, runner, name)
    warn_experimental_runtime(args)
    current = health(args.host, args.port)
    if current and current.get("ok"):
        same, reason = server_config_matches(current, desired)
        if same:
            log(f"already ready: http://{args.host}:{args.port} model={name}")
            return {"model": name, "model_path": str(model), "ctx_size": desired["ctx_size"], "already_running": True}
        state = read_state()
        if int(state.get("port") or 0) == args.port and state.get("pid"):
            log(f"restarting launcher-owned server on port {args.port}: {reason}")
            stop_owned_server()
        else:
            raise SystemExit(
                f"port {args.port} already serves a different config ({reason}); use --port or stop that server first"
            )

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
        "ctx_size": desired["ctx_size"],
        "max_new_tokens": desired["max_new_tokens"],
        "chat_template": desired["chat_template"],
        "paged_kv": desired["paged_kv"],
        "kv_page_size": desired["kv_page_size"],
        "kv_cache_dtype": desired["kv_cache_dtype"],
        "command": cmd,
        "started_at": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "stdout_log": str(STDOUT_LOG),
        "stderr_log": str(STDERR_LOG),
    })
    ready = wait_ready(args.host, args.port, args.startup_timeout + max(5.0, args.warmup_tokens * 5.0))
    log(f"ready: http://{args.host}:{args.port} model={ready.get('model')}")
    log(f"ctx_size={ready.get('ctx_size', desired['ctx_size'])} paged_kv={ready.get('paged_kv', desired['paged_kv'])} kv_cache_dtype={ready.get('kv_cache_dtype', desired['kv_cache_dtype'])}")
    log(f"logs: {STDERR_LOG}")
    return {"model": name, "model_path": str(model), "ctx_size": desired["ctx_size"], "pid": proc.pid}


def command_up(args: argparse.Namespace) -> int:
    ensure_up(args, args.model)
    return 0


def command_serve(args: argparse.Namespace) -> int:
    model = resolve_model(args.model)
    name = args.name or model_name_from_path(model)
    runner = ensure_runner(args)
    desired_server_config(args, model, runner, name)
    warn_experimental_runtime(args)
    cmd = server_command(args, model, runner, name)
    log(f"serving foreground: http://{args.host}:{args.port} model={name} ctx_size={args.resolved_ctx_size}")
    return subprocess.call(cmd, cwd=str(ROOT))


def command_down(args: argparse.Namespace) -> int:
    stopped = stop_owned_server()
    log("stopped" if stopped else "no launcher-owned server pidfile found")
    return 0


def command_status(args: argparse.Namespace) -> int:
    current = health(args.host, args.port)
    state = read_state()
    if current and current.get("ok"):
        log(f"ready: http://{args.host}:{args.port} model={current.get('model')} worker_alive={current.get('worker_alive')} ctx_size={current.get('ctx_size')} paged_kv={current.get('paged_kv')} kv_cache_dtype={current.get('kv_cache_dtype')}")
    else:
        log(f"not responding on http://{args.host}:{args.port}")
    if state:
        log(f"pid={state.get('pid')} model_path={state.get('model_path')} stderr={state.get('stderr_log')}")
    return 0


def command_list(_: argparse.Namespace) -> int:
    models = discover_models()
    if not models:
        log("no .gguf models discovered under models/")
        return 1
    for idx, path in enumerate(models, start=1):
        name = model_name_from_path(path)
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"{idx}. {name}\t{rel}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    model = resolve_model(args.model)
    name = args.name or model_name_from_path(model)
    runner = ensure_runner(args)
    native = model_context_length(model, runner, getattr(args, "arch", None))
    resolved = resolve_ctx_size(getattr(args, "ctx_size", "auto"), model, runner, getattr(args, "arch", None))
    rel = model.relative_to(ROOT) if model.is_relative_to(ROOT) else model
    log(f"model={name}")
    log(f"path={rel}")
    log(f"native_context_length={native if native is not None else 'unknown'}")
    log(f"auto_ctx_limit={auto_ctx_limit()}")
    log(f"resolved_ctx_size={resolved}")
    if native and resolved < native:
        log("use `motifcl run --ctx-size max` for the model-native window; keep `auto` for the safer 8K default")
    if getattr(args, "paged_kv", False):
        log(f"paged_kv=on kv_page_size={args.kv_page_size}")
    else:
        log("paged_kv=off (enable with --paged-kv only for measured long-context runs)")
    log(f"kv_cache_dtype={args.kv_cache_dtype}")
    warn_experimental_runtime(args)
    return 0


def command_install(args: argparse.Namespace) -> int:
    directory = Path(args.bin_dir).expanduser().resolve()
    shim = write_shim(directory)
    path_changed = False
    if os.name == "nt":
        path_changed = add_user_path_windows(directory)
    elif not path_contains(directory):
        log(f"warning: {directory} is not in PATH; add it to your shell profile")
    log(f"installed command shim: {shim}")
    if path_changed:
        log("added shim directory to user PATH; open a new terminal if this one does not see `motifcl`")
    elif path_contains(directory):
        log("shim directory is already in PATH")
    return 0


def command_uninstall(args: argparse.Namespace) -> int:
    directory = Path(args.bin_dir).expanduser().resolve()
    removed = False
    for name in ("motifcl.cmd", "motifcl"):
        path = directory / name
        if path.exists():
            path.unlink()
            removed = True
            log(f"removed: {path}")
    if not removed:
        log(f"no motifcl shim found in {directory}")
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
    p.add_argument("--ctx-size", default="auto",
                   help="context window: auto (default 8K or model max if smaller), max/native/model, or integer")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0, help="server default temperature")
    p.add_argument("--top-k", type=int, default=0, help="server default top_k")
    p.add_argument("--top-p", type=float, default=1.0, help="server default top_p")
    p.add_argument("--arch", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--chat-template", default="auto",
                   help="chat template for prompt wrapping: auto, none/raw, gemma, chatml, llama3, mistral")
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--no-prefill", action="store_true")
    p.add_argument("--force-prefill", action="store_true")
    p.add_argument("--disable-adaptive-prefill", action="store_true")
    p.add_argument("--cpu-sampling", action="store_true")
    p.add_argument("--paged-kv", action="store_true", help="experimental paged KV cache path")
    p.add_argument("--kv-page-size", type=int, default=256, help="paged KV page size")
    p.add_argument("--kv-cache-dtype", default="f32", choices=["f32", "q8", "q8_0", "q4", "q4_0"],
                   help="experimental compressed regular KV cache dtype")
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

    inspect = sub.add_parser("inspect", help="show resolved context settings for a local model")
    add_common(inspect)
    inspect.add_argument("model", nargs="?", help="model path or discovered alias; default: newest local GGUF")
    inspect.set_defaults(func=command_inspect)

    install = sub.add_parser("install", help="install `motifcl` into the user PATH")
    install.add_argument("--bin-dir", type=Path, default=default_shim_dir())
    install.set_defaults(func=command_install)

    uninstall = sub.add_parser("uninstall", help="remove the installed `motifcl` command shim")
    uninstall.add_argument("--bin-dir", type=Path, default=default_shim_dir())
    uninstall.set_defaults(func=command_uninstall)
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
