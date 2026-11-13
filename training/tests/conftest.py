"""Fixtures of the token-readout tests. They run on the CPU without network; the few that need Gemma 4's tokenizer
read it from STRANDS_DECIDER_GEMMA4_TOKENIZER (a tokenizer.json of google/gemma-4-*-it, as for tests/test_core_*.py)
and are skipped without it."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from trainer_support import FIXTURES, PINNED_TOKENIZER_SHA256


@pytest.fixture(scope="session")
def previous():
    """Outputs of the trainer the released models were trained with, on fixed inputs (fixtures/README.md)."""
    return json.loads((FIXTURES / "previous_trainer.json").read_text())


@pytest.fixture(scope="session")
def gemma4_tokenizer():
    path = os.environ.get("STRANDS_DECIDER_GEMMA4_TOKENIZER")
    if not path:
        pytest.skip("set STRANDS_DECIDER_GEMMA4_TOKENIZER to a Gemma 4 tokenizer.json")
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


@pytest.fixture(scope="session")
def pinned_gemma4_tokenizer(gemma4_tokenizer):
    """The Gemma 4 tokenizer the render fixture was made with; a test that needs it skips with another."""
    digest = hashlib.sha256(
        Path(os.environ["STRANDS_DECIDER_GEMMA4_TOKENIZER"]).read_bytes()
    ).hexdigest()
    if digest != PINNED_TOKENIZER_SHA256:
        pytest.skip(
            f"the render fixture was made with tokenizer {PINNED_TOKENIZER_SHA256[:12]}, not {digest[:12]}"
        )
    return gemma4_tokenizer
