"""Stream the GGUF from S3 into memory (a memfd), so no weight byte touches disk.

The 2 GB Q8_0 model does not fit the 512 MB /tmp that SnapStart allows. llama.cpp
mmaps the model from a path, and a /proc/self/fd/<n> path backed by a memfd works.
The memfd stays open for the life of the process, because llama holds a view of it.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

log = logging.getLogger("decider.loader")

CHUNK = 64 * 1024 * 1024
WORKERS = 32
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
_libc.memfd_create.restype = ctypes.c_int

# Keep the memfd open for the process lifetime: llama mmaps the model from it.
_fds: list[int] = []
_s3 = boto3.client("s3", config=Config(max_pool_connections=WORKERS))


def _memfd(name: str) -> int:
    fd = _libc.memfd_create(name.encode(), 0)
    if fd == -1:
        err = ctypes.get_errno()
        raise OSError(err, f"memfd_create failed: {os.strerror(err)}")
    return fd


def _download_part(bucket: str, key: str, fd: int, start: int, end: int) -> None:
    body = _s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")["Body"]
    pos = start
    for chunk in body.iter_chunks(8 * 1024 * 1024):
        os.pwrite(fd, chunk, pos)
        pos += len(chunk)


def download_to_memfd(bucket: str, key: str) -> str:
    """Parallel ranged GET of one S3 object into a memfd. Returns /proc/self/fd/<n>."""
    size = _s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    fd = _memfd(os.path.basename(key))
    try:
        os.ftruncate(fd, size)
    except OSError as e:
        os.close(fd)
        raise RuntimeError(f"not enough memory for {key} ({size / 2**20:.0f} MiB)") from e
    parts = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda p: _download_part(bucket, key, fd, *p), parts))
    _fds.append(fd)
    log.info(
        "staged s3://%s/%s (%.0f MiB) into memory in %.1fs",
        bucket,
        key,
        size / 2**20,
        time.perf_counter() - t0,
    )
    return f"/proc/self/fd/{fd}"
