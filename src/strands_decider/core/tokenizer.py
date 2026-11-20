"""A checkpoint's tokenizer, set to encode request text as plain text and checked at load.

Gemma 4's sizes ship slightly different tokenizer.json files, so a runtime checks once, at load,
the ids the prompt and the readout depend on. A wrong tokenizer would otherwise feed the model
wrong ids and give wrong answers with no error.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .codes import CODE_TOKEN_IDS, CODES, CONTROL_TOKEN_IDS

if TYPE_CHECKING:
    from tokenizers import Tokenizer

TOKENIZER_FILE = "tokenizer.json"


def prepare_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """Sets `tokenizer`, in place, to encode text with special-token matching, padding and
    truncation off, as `render` needs, and checks it gives Gemma 4's control-token and code ids."""
    tokenizer.encode_special_tokens = True
    tokenizer.no_padding()
    tokenizer.no_truncation()
    wrong = {
        s: tokenizer.token_to_id(s)
        for s, i in CONTROL_TOKEN_IDS.items()
        if tokenizer.token_to_id(s) != i
    }
    if wrong:
        raise ValueError(f"the tokenizer's control tokens are not Gemma 4's: it has {wrong}")
    # Special-token matching off doesn't stop a non-special added token from matching text.
    plain = [t.content for t in tokenizer.get_added_tokens_decoder().values() if not t.special]
    if plain:
        raise ValueError(
            f"the tokenizer has added tokens that request text could match: {plain[:5]}"
        )
    encoded = [e.ids for e in tokenizer.encode_batch(list(CODES), add_special_tokens=False)]
    bad = [c for c, ids, want in zip(CODES, encoded, CODE_TOKEN_IDS, strict=True) if ids != [want]]
    if bad:
        raise ValueError(f"the tokenizer doesn't encode codes {bad[:5]} as Gemma 4's single tokens")
    return tokenizer


def load_tokenizer(checkpoint: str | os.PathLike[str]) -> Tokenizer:
    """The checkpoint's tokenizer.json, prepared and checked (`prepare_tokenizer`)."""
    from tokenizers import Tokenizer

    return prepare_tokenizer(
        Tokenizer.from_file(os.path.join(os.fspath(checkpoint), TOKENIZER_FILE))
    )
