"""The 255 answer codes of the decision interface (docs/spec.md, section 2).

A question with k options binds its options, in rendered order, to the first k codes. The
readout has one row per code, so a code means "the option listed with this code in this
prompt" and nothing else.

The list is the spreadsheet-column sequence A..Z, AA..AZ, ..., IA..IV (256 names) without
GZ, which Gemma 4's tokenizer splits into two tokens (G, Z). Every other name is one token.
"""

from __future__ import annotations

from string import ascii_uppercase

# Names that are not one token in Gemma 4's vocabulary, so cannot be codes.
EXCLUDED = ("GZ",)


def _column_names(n: int) -> list[str]:
    names = list(ascii_uppercase) + [a + b for a in ascii_uppercase for b in ascii_uppercase]
    return names[:n]


CODES: tuple[str, ...] = tuple(c for c in _column_names(256) if c not in EXCLUDED)
MAX_CODES = len(CODES)
assert MAX_CODES == 255 and CODES[-1] == "IV"


def codes_for(k: int) -> tuple[str, ...]:
    """The codes of a question with k options: the first k, in order."""
    if not 1 <= k <= MAX_CODES:
        raise ValueError(f"a question has 1 to {MAX_CODES} options, not {k}")
    return CODES[:k]
