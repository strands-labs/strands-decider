"""Whitespace, as docs/spec.md defines it (Rendering content).

The template strips request text of whitespace at its ends, and collapses each run of it in an
option's name and description to one space. Whitespace is the 29 code points of Python's
`str.isspace`, the same set on every Python from 3.10 (Unicode 13 to 15.1), written out so
that the rule never depends on the interpreter's Unicode tables. U+200B ZERO WIDTH SPACE and
U+FEFF, the byte order mark, are not whitespace.
"""

from __future__ import annotations

import re

WHITESPACE = (
    "\t\n\x0b\x0c\r"  # U+0009 to U+000D
    "\x1c\x1d\x1e\x1f"  # U+001C to U+001F, the information separators
    " \x85\xa0"  # SPACE, NEXT LINE, NO-BREAK SPACE
    "\u1680"  # OGHAM SPACE MARK
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"  # EN QUAD to HAIR SPACE
    "\u2028\u2029"  # LINE SEPARATOR, PARAGRAPH SEPARATOR
    "\u202f\u205f\u3000"  # NARROW NO-BREAK, MEDIUM MATHEMATICAL and IDEOGRAPHIC SPACE
)
_RUN = re.compile(f"[{re.escape(WHITESPACE)}]+")


def strip(text: str) -> str:
    """`text` without whitespace at either end."""
    return text.strip(WHITESPACE)


def one_line(text: str) -> str:
    """`text` stripped, with each run of whitespace inside it, newlines included, as one
    space: an option is always one line."""
    return _RUN.sub(" ", strip(text))
