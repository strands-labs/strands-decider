"""Shared by the torch-free core's tests (tests/test_core_*.py).

The render goldens and properties run against Gemma 4's real tokenizer. It is 32 MB, so it
is not committed: point STRANDS_DECIDER_GEMMA4_TOKENIZER at a copy of `tokenizer.json`
(google/gemma-4-E2B-it at GEMMA4_REVISION, the file every Gemma 4 size ships), and the
tests check its sha256 before they trust it. Without it those tests skip; the tests that
need only text, numbers or a toy tokenizer always run.

    STRANDS_DECIDER_GEMMA4_TOKENIZER=/path/to/tokenizer.json pytest tests/test_core_*.py

CI's `core tokenizer (gemma 4)` job (.github/workflows/tests.yml) fetches that file and the
E2B chat template at GEMMA4_REVISION, checks both sha256s, and runs these tests without torch.

STRANDS_DECIDER_GEMMA4_TEMPLATES lists the Gemma 4 base models' chat_template.jinja files
(os.pathsep between them) for the tests that compare strands-chat with them, which also
need jinja2 (and transformers, for the apply_chat_template comparison). A decision checkpoint
ships no chat template, and its prompts are never rendered with one.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "core"
GEMMA4_REPO = "google/gemma-4-E2B-it"
GEMMA4_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
GEMMA4_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
TOKENIZER_ENV = "STRANDS_DECIDER_GEMMA4_TOKENIZER"
TEMPLATES_ENV = "STRANDS_DECIDER_GEMMA4_TEMPLATES"


@lru_cache(maxsize=1)
def _gemma4_path() -> str | None:
    p = os.environ.get(TOKENIZER_ENV)
    if not p:
        return None
    if os.path.isdir(p):
        p = os.path.join(p, "tokenizer.json")
    with open(p, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != GEMMA4_TOKENIZER_SHA256:
        raise AssertionError(
            f"{p} has sha256 {digest}; the render goldens are for {GEMMA4_REPO}@{GEMMA4_REVISION} "
            f"({GEMMA4_TOKENIZER_SHA256})"
        )
    return p


def gemma4_tokenizer_path() -> str:
    p = _gemma4_path()
    if p is None:
        pytest.skip(f"set {TOKENIZER_ENV} to Gemma 4's tokenizer.json ({GEMMA4_REPO})")
    return p


@lru_cache(maxsize=1)
def _load_gemma4(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def gemma4_tokenizer():
    """Gemma 4's tokenizer, or a skip. One instance per session: it is 262k tokens."""
    return _load_gemma4(gemma4_tokenizer_path())


def decision_config_fixture() -> dict:
    """The scrubbed E2B decision_config.json: the schema's fields only (codes and token ids
    as the checkpoint has them; `prompt` and `max_length_validated` are test values)."""
    return json.loads((FIXTURES / "decision_config.json").read_text(encoding="utf-8"))


def toy_tokenizer():
    """A word-level tokenizer with Gemma 4's control tokens and every code as one token:
    enough to drive render and evaluate where the real tokenizer is not available."""
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

    from strands_decider.core import CODES

    words = ["<unk>", "\n", " ", ".", "<", ">", "/", '"', "=", "—", "answer", "model", "user"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words + list(CODES)))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|\s|[^\w\s]"), behavior="isolated")
    tok.add_special_tokens(
        [AddedToken(s, special=True) for s in ("<bos>", "<|turn>", "<turn|>", "<eos>")]
    )
    return tok
