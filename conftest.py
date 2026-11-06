"""Shared test setup.

Keeps the suite quiet and deterministic: the Hub's progress bars and warnings
otherwise bury the actual test output, and an unseeded torch makes every test that
draws random tensors produce different numbers run to run.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True)
def _deterministic():
    try:
        import torch
    except ModuleNotFoundError:  # the torch-free tests (tests/test_core_*.py) run without it
        yield
        return
    torch.manual_seed(0)
    yield
