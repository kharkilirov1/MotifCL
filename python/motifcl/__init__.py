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


def _install_tensor_dunders() -> None:
    def _require_tensor(other):
        if not isinstance(other, Tensor):
            raise TypeError("MotifCL Tensor operators currently require another Tensor")
        return other

    def _add(self, other):
        return add(self, _require_tensor(other))

    def _radd(self, other):
        return add(_require_tensor(other), self)

    def _sub(self, other):
        return sub(self, _require_tensor(other))

    def _rsub(self, other):
        return sub(_require_tensor(other), self)

    def _mul(self, other):
        if isinstance(other, (int, float)):
            return scale(self, float(other))
        return mul(self, _require_tensor(other))

    def _rmul(self, other):
        if isinstance(other, (int, float)):
            return scale(self, float(other))
        return mul(_require_tensor(other), self)

    def _matmul(self, other):
        return matmul(self, _require_tensor(other))

    def _neg(self):
        return scale(self, -1.0)

    Tensor.__add__ = _add
    Tensor.__radd__ = _radd
    Tensor.__sub__ = _sub
    Tensor.__rsub__ = _rsub
    Tensor.__mul__ = _mul
    Tensor.__rmul__ = _rmul
    Tensor.__matmul__ = _matmul
    Tensor.__neg__ = _neg


class no_grad:
    """Context manager that disables autograd recording inside the block."""

    def __enter__(self):
        self._previous = is_grad_enabled()
        set_grad_enabled(False)
        return self

    def __exit__(self, exc_type, exc, tb):
        set_grad_enabled(self._previous)
        return False


_install_tensor_dunders()

from . import nn as nn
from . import functional as functional
from . import optim as optim
from . import serialization as serialization

__all__ = [name for name in globals() if not name.startswith("_")]
