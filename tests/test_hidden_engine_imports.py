"""hidden_engine must import without torch or transformers: it exists to serve where they
do not fit. Run in a fresh interpreter, so modules other tests imported do not count."""

from __future__ import annotations

import subprocess
import sys


def test_hidden_engine_imports_without_torch():
    code = (
        "import sys, strands_decider.hidden_engine; "
        "assert not {'torch', 'transformers'} & set(sys.modules), sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
