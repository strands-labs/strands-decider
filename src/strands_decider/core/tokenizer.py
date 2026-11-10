"""A checkpoint's tokenizer, checked against the interface (docs/spec.md, From text to token ids).

The tokenizer that is normative for a checkpoint is its own: its base's `tokenizer.json` at
`provenance.base_revision`. Gemma 4's sizes ship slightly different files (the spec lists
them), so a runtime checks, when it loads a checkpoint, the ids the prompt and the readout
depend on: each control token at the id the spec gives, each code one token whose id is the
checkpoint's `token_ids` entry, and every added token special, so that request text, which
is encoded with special-token matching off, can never produce one.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .codes import CODES
from .config import DecisionConfig
from .images import IMAGE_CLOSE, IMAGE_OPEN, IMAGE_SOFT
from .prompt import BOS, TURN_CLOSE, TURN_OPEN, _Encoder

if TYPE_CHECKING:
    from tokenizers import Tokenizer

TOKENIZER_FILE = "tokenizer.json"

# The control tokens' ids in Gemma 4's tokenizer (docs/spec.md, From text to token ids). The
# image tokens' ids are also config.json's boi_token_id, image_token_id and eoi_token_id.
CONTROL_TOKEN_IDS: dict[str, int] = {
    BOS: 2,
    TURN_OPEN: 105,
    TURN_CLOSE: 106,
    IMAGE_OPEN: 255999,
    IMAGE_SOFT: 258880,
    IMAGE_CLOSE: 258882,
}


def check_tokenizer(tokenizer: Tokenizer, config: DecisionConfig) -> None:
    """That `tokenizer` gives the ids the checkpoint `config` describes was trained with:
    each control token at its id (`CONTROL_TOKEN_IDS`), each code exactly one token whose
    id is `config.token_ids`' entry for it, and no added token that is not special.
    ValueError otherwise. A runtime calls it once, when it loads the checkpoint
    (`load_tokenizer` does)."""
    wrong = {
        s: tokenizer.token_to_id(s)
        for s, i in CONTROL_TOKEN_IDS.items()
        if tokenizer.token_to_id(s) != i
    }
    if wrong:
        want = {s: CONTROL_TOKEN_IDS[s] for s in wrong}
        raise ValueError(
            f"the tokenizer's control tokens are not Gemma 4's: it has {wrong}, the interface "
            f"needs {want}"
        )
    # Encoded as request text is: by the private copy, special-token matching off. The copy
    # also refuses a tokenizer with added tokens that are not special.
    encoded = _Encoder(tokenizer).texts(list(CODES))
    bad = [
        c for c, ids, want in zip(CODES, encoded, config.token_ids, strict=True) if ids != [want]
    ]
    if bad:
        shown = {c: encoded[CODES.index(c)] for c in bad[:5]}
        raise ValueError(
            f"the tokenizer does not encode codes {bad[:5]}{'...' if len(bad) > 5 else ''} as "
            f"decision_config.json's token_ids say: it gives {shown}"
        )


def load_tokenizer(
    checkpoint: str | os.PathLike[str], config: DecisionConfig | None = None
) -> Tokenizer:
    """The checkpoint's `tokenizer.json`, checked by `check_tokenizer` against `config` (or
    the directory's decision_config.json)."""
    from tokenizers import Tokenizer

    d = os.fspath(checkpoint)
    cfg = config if config is not None else DecisionConfig.load(d)
    tok: Tokenizer = Tokenizer.from_file(os.path.join(d, TOKENIZER_FILE))
    check_tokenizer(tok, cfg)
    return tok
