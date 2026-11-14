"""Test tiers (README, Tests).

- CPU: ``pytest tests`` on any machine. Without vLLM the tests of the request checks, the segment plan, the cache keys,
  ``decision_config.json`` and the version gate run; with vLLM 0.33 installed (no GPU needed) the tests of the model
  classes, the model state, the attention backend, the engine-core and scheduler wrappers, vLLM's block hasher and the
  server run too. A test that needs a GPU is marked ``requires_cuda`` and skips without one.
- GPU: ``tests/gpu/ci.sh`` on one GPU (tiny random checkpoints, servers, a transformers reference).
"""

from __future__ import annotations

import pytest


def _cuda() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda: needs a CUDA GPU; skipped without one")


def pytest_collection_modifyitems(config, items):
    if _cuda():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU (requires_cuda)")
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip)
