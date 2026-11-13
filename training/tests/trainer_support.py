"""Shared by the token-readout tests and their conftest: where the fixtures are, and whether this environment is the
one the previous trainer's fixtures were made in."""

import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
# The tokenizer the render fixture was made with (fixtures/README.md).
PINNED_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
# The previous trainer's fixtures were made with Python 3.12 and torch 2.14 (training/requirements.txt). Python 3.12
# made `sum()` of floats a compensated sum, and another torch may round a float32 reduction differently in its last
# bit, so elsewhere the tests that compare those floats allow that rounding and no more.
SAME_PYTHON_SUM = sys.version_info >= (3, 12)


def same_torch() -> bool:
    import torch

    return torch.__version__.split("+")[0] == "2.14.0"
