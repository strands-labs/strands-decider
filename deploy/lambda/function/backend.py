"""llama.cpp as the backend of strands_decider.hidden_engine.HiddenStateEngine, through
shim.c (libdecider.so). LlamaVisionBackend also reads images, with a vision tower beside the
model; LlamaBackend has no forward_images, so the engine refuses images over it.

Calls run one at a time, as HiddenStateEngine's Backend contract asks of a backend that has
not been shown to run them concurrently: each call already uses a thread per core. Lambda sends
an execution environment one request at a time, so there the lock never waits.
"""

from __future__ import annotations

import ctypes
import threading
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
        self._lock = threading.Lock()

    def forward(
        self, state: list[int], questions: list[list[int]], rows: list[list[int]]
    ) -> list[np.ndarray]:
        lens = [len(q) for q in questions]
        out = np.empty((sum(lens), self.n_embd), dtype=np.float32)
        with self._lock:
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


_U8P = ctypes.POINTER(ctypes.c_ubyte)


class LlamaVisionBackend(LlamaBackend):
    def __init__(self, lib_path: str, model_path: str, mmproj_path: str, threads: int):
        super().__init__(lib_path, model_path, threads)
        self.lib.dh_load_mm.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.dh_load_mm.restype = ctypes.c_int
        self.lib.dh_forward_mm.argtypes = [
            ctypes.c_void_p, _I, _I, ctypes.POINTER(_U8P), _I, _I, ctypes.c_int,
            _I, _I, ctypes.c_int, ctypes.POINTER(ctypes.c_float), _I,
        ]  # fmt: skip
        self.lib.dh_forward_mm.restype = ctypes.c_int
        rc = self.lib.dh_load_mm(self.handle, mmproj_path.encode())
        if rc != 0:
            raise RuntimeError(f"llama.cpp could not load the vision tower {mmproj_path} ({rc})")

    def forward_images(
        self,
        segments: list[list[int]],
        images: list[tuple[bytes, int, int]],
        questions: list[list[int]],
        rows: list[list[int]],
    ) -> tuple[list[np.ndarray], list[int]]:
        if len(segments) != len(images) + 1:
            raise ValueError(f"{len(segments)} text segments around {len(images)} images")
        for i, (rgb, w, h) in enumerate(images):
            if len(rgb) != w * h * 3:  # the shim reads w * h * 3 bytes
                raise ValueError(f"image {i} is {len(rgb)} bytes, not {w}x{h} RGB")
        buffers = [ctypes.create_string_buffer(rgb, len(rgb)) for rgb, _, _ in images]
        pixels = (_U8P * len(images))(*[ctypes.cast(b, _U8P) for b in buffers])
        lens = [len(q) for q in questions]
        out = np.empty((sum(lens), self.n_embd), dtype=np.float32)
        tokens = (ctypes.c_int * len(images))()
        with self._lock:
            rc = self.lib.dh_forward_mm(
                self.handle,
                _ints([t for segment in segments for t in segment]),
                _ints([len(segment) for segment in segments]),
                pixels,
                _ints([w for _, w, _ in images]),
                _ints([h for _, _, h in images]),
                len(images),
                _ints([t for q in questions for t in q]),
                _ints(lens),
                len(lens),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                tokens,
            )
        if rc != 0:
            raise RuntimeError(f"dh_forward_mm failed with {rc}")
        hidden = np.split(out, np.cumsum(lens)[:-1])
        return [h[r] for h, r in zip(hidden, rows, strict=True)], list(tokens)
