"""Python package wrapper for the MotifCL native extension."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _configure_runtime_paths() -> None:
    package_dir = Path(__file__).resolve().parent
    kernel_dir = package_dir / "kernels"
    if kernel_dir.is_dir() and "MOTIFCL_KERNEL_DIR" not in os.environ:
        os.environ["MOTIFCL_KERNEL_DIR"] = str(kernel_dir)

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates = [
        package_dir,
        Path(sys.prefix) / "Library" / "bin",
        Path(r"C:\Strawberry\c\bin"),
        Path(r"C:\msys64\mingw64\bin"),
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            try:
                os.add_dll_directory(str(candidate))
            except OSError:
                pass


_configure_runtime_paths()

try:
    from ._motifcl import *
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Failed to import motifcl native extension. Ensure the wheel was built for "
        "this Python version and required OpenCL/compiler runtime DLLs are available."
    ) from exc

__all__ = [name for name in globals() if not name.startswith("_")]
