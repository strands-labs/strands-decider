"""llama.cpp as the backend of strands_decider.hidden_engine.HiddenStateEngine, through
shim.c (libdecider.so)."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

_I = ctypes.POINTER(ctypes.c_int)


def _ints(xs: list[int]) -> Any:
    return (ctypes.c_int * len(xs))(*xs)


class LlamaBackend:
    def __init__(self, lib_path: str, model_path: str, threads: int):
        lib = ctypes.CDLL(lib_path)
        lib.dh_load.argtypes = [ctypes.c_char_p, ctypes.c_int, _I]
        lib.dh_load.restype = ctypes.c_void_p
        lib.dh_forward.argtypes = [
            ctypes.c_void_p, _I, ctypes.c_int, _I, _I, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
        ]  # fmt: skip
        lib.dh_forward.restype = ctypes.c_int
        n_embd = ctypes.c_int(0)
        self.handle = lib.dh_load(model_path.encode(), threads, ctypes.byref(n_embd))
        if not self.handle:
            raise RuntimeError(f"llama.cpp could not load {model_path}")
        self.lib = lib
        self.n_embd = n_embd.value

    def forward(
        self, state: list[int], questions: list[list[int]], rows: list[list[int]]
    ) -> list[np.ndarray]:
        lens = [len(q) for q in questions]
        out = np.empty((sum(lens), self.n_embd), dtype=np.float32)
        rc = self.lib.dh_forward(
            self.handle,
            _ints(state),
            len(state),
            _ints([t for q in questions for t in q]),
            _ints(lens),
            len(lens),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"dh_forward failed with {rc}")
        hidden = np.split(out, np.cumsum(lens)[:-1])
        return [h[r] for h, r in zip(hidden, rows, strict=True)]
