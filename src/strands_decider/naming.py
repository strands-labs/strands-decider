"""Public model names, `strands-decider-{size}-{base}-v{N}-{YYMM}` (docs/naming.md).

    python -m strands_decider.naming BASE_MODEL N YYMM     (prints the release name)

size is the base model's own size token as its Hub id writes it (2B, E2B, 26B-A4B); base is
one token per base family; N counts the releases on that base family (v1 is the first); YYMM is
the release month.
The names released before this convention (strands-decider-2B-hobson-v19, -v21) keep theirs.
"""

from __future__ import annotations

import re
import sys

# Hub id prefix of a base family -> its token in the public name.
BASES = {"Qwen/Qwen3.5-": "qwen3.5", "google/gemma-4-": "gemma4"}
SIZE = r"E?\d+B(?:-A\d+B)?"
YYMM = r"\d{2}(?:0[1-9]|1[0-2])"
NAME = re.compile(rf"strands-decider-({SIZE})-({'|'.join(map(re.escape, BASES.values()))})"
                  rf"-v([1-9]\d*)-({YYMM})")


def release_name(base_model: str, generation: int, yymm: str) -> str:
    """The public name of release `generation` on `base_model`'s family, released in `yymm`."""
    prefix = next((p for p in BASES if base_model.startswith(p)), "")
    size = re.match(SIZE, base_model[len(prefix):]) if prefix else None
    if not size:
        raise ValueError(f"{base_model}: not a base this naming knows ({', '.join(BASES)})")
    name = f"strands-decider-{size.group(0)}-{BASES[prefix]}-v{generation}-{yymm}"
    if not NAME.fullmatch(name):
        raise ValueError(f"{name}: generation must be 1 or more and yymm a month (2610)")
    return name


def parse(name: str) -> tuple[str, str, int, str]:
    """(size, base, generation, yymm) of a public name, or ValueError. Takes `org/name` too."""
    m = NAME.fullmatch(name.rsplit("/", 1)[-1])
    if not m:
        raise ValueError(f"{name}: not strands-decider-{{size}}-{{base}}-v{{N}}-{{YYMM}}")
    return m.group(1), m.group(2), int(m.group(3)), m.group(4)


if __name__ == "__main__":
    print(release_name(sys.argv[1], int(sys.argv[2]), sys.argv[3]))
