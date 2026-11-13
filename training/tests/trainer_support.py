"""Shared by the token-readout tests and their conftest: where the fixtures are, and whether the tests compare with the
previous trainer's outputs bit for bit.

The previous trainer's fixtures were made with the versions of training/requirements.txt (Python 3.12, torch 2.14.0,
numpy 2.5.3). The floats they record go through Python's `sum()` (which Python 3.12 made a compensated sum), numpy's
`exp` and `log`, libm's `cos` and torch's float32 reductions, all of which may round differently in the last bit under
other versions or on other CPUs. So the tests compare bit for bit only when STRANDS_DECIDER_EXACT_FIXTURES=1, which the
CI job with the recipe's pins sets and which requires those versions; everywhere else they allow that last bit's
rounding and no more. Plans, row lengths, targets and microbatches are integers or exact, and are compared exactly
everywhere.
"""

import os
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
# The tokenizer the render fixture was made with (fixtures/README.md).
PINNED_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
RECIPE_VERSIONS = {"torch": "2.14.0", "numpy": "2.5.3"}


def exact() -> bool:
    """True when the tests must match the fixtures bit for bit; refuses the setting under other versions."""
    if os.environ.get("STRANDS_DECIDER_EXACT_FIXTURES") != "1":
        return False
    import numpy
    import torch

    found = {"torch": torch.__version__.split("+")[0], "numpy": numpy.__version__}
    if sys.version_info[:2] != (3, 12) or found != RECIPE_VERSIONS:
        raise RuntimeError(
            f"STRANDS_DECIDER_EXACT_FIXTURES=1 needs Python 3.12 and {RECIPE_VERSIONS}, not "
            f"{sys.version_info[:2]} and {found} (training/requirements.txt)"
        )
    return True
