"""What a run records about the software it ran on."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

PACKAGES = ("strands-decider", "torch", "transformers", "peft", "accelerate", "safetensors")


def library_versions() -> dict[str, str]:
    """Python's version and every installed one of PACKAGES."""
    found = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            found[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return found
