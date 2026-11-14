"""A checkpoint's tokenizer, checked against the interface (docs/spec.md, section 3.2).

The tokenizer that is normative for a checkpoint is its own: its base's `tokenizer.json` at
`provenance.base_revision`. Gemma 4's sizes ship slightly different files (tests/core_support.py
lists them), so a runtime checks, when it loads a checkpoint, the ids the prompt and the readout
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
from .prompt import (
    BOS,
    TURN_CLOSE,
    TURN_OPEN,
    _make_plain_text_copy,
    _restore_settings,
    plain_text_tokenizer,
    release_plain_text_tokenizer,
)

if TYPE_CHECKING:
    from tokenizers import Tokenizer

TOKENIZER_FILE = "tokenizer.json"

# The control tokens' ids in Gemma 4's tokenizer (docs/spec.md, section 3.2). The
# image tokens' ids are also config.json's boi_token_id, image_token_id and eoi_token_id.
CONTROL_TOKEN_IDS: dict[str, int] = {
    BOS: 2,
    TURN_OPEN: 105,
    TURN_CLOSE: 106,
    IMAGE_OPEN: 255999,
    IMAGE_SOFT: 258880,
    IMAGE_CLOSE: 258882,
}


# Each code's token id in Gemma 4's tokenizer, in code order (docs/spec.md, section 2): the same
# in all four of its known tokenizer.json files.
# fmt: off
CODE_TOKEN_IDS: tuple[int, ...] = (
    236776, 236799, 236780, 236796, 236788, 236811, 236823, 236814, 236777, 236863, 236855,
    236798, 236792, 236797, 236806, 236791, 236935, 236794, 236773, 236774, 236836, 236847,
    236824, 236917, 236874, 236953, 8686, 3066, 2714, 2466, 14196, 11558, 2807, 20211,
    12553, 57902, 9978, 1565, 1936, 1536, 70713, 2663, 80055, 1425, 2371, 1375, 22258, 7008,
    35066, 26428, 5790, 25862, 6552, 20804, 5901, 11217, 12051, 28671, 54684, 48367, 11526,
    127017, 74610, 8017, 16070, 10589, 8791, 17991, 159610, 12357, 5580, 24219, 8462, 82085,
    75007, 121922, 19470, 153766, 5635, 26204, 3129, 6329, 4660, 13880, 17956, 2293, 9575,
    81265, 20045, 5094, 10087, 9353, 2457, 7618, 85549, 6459, 6786, 1692, 33186, 20741,
    63746, 82279, 40736, 129399, 7172, 6270, 12622, 21738, 4070, 7332, 65899, 54634, 9982,
    43793, 12536, 30455, 23390, 48774, 10009, 10904, 137414, 7560, 8925, 18540, 11965,
    80850, 42771, 39399, 87207, 164367, 37553, 56343, 5856, 2413, 10421, 28135, 63426,
    101722, 73344, 102109, 46570, 3387, 3158, 1439, 6864, 13237, 38254, 1089, 1740, 2406,
    30722, 22486, 108534, 2201, 6594, 79526, 4815, 38131, 11962, 35023, 17179, 5495, 40807,
    75536, 8537, 191318, 49512, 7750, 37413, 32479, 12041, 22591, 123622, 11869, 16164,
    7627, 44519, 61688, 27330, 36991, 57954, 175992, 10032, 8242, 11925, 29138, 8809, 15015,
    30389, 32701, 17508, 118670, 105774, 9179, 24464, 36875, 10677, 23172, 202788, 11124,
    15683, 22618, 26943, 60572, 59141, 161184, 34144, 14983, 61026, 24084, 14563, 2808,
    37030, 98100, 40325, 26722, 139168, 31581, 17287, 42025, 74035, 10858, 18835, 98264,
    17372, 11767, 4080, 80858, 80657, 47281, 186240, 32207, 158347, 5299, 7692, 2011, 1735,
    7742, 6354, 2996, 83145, 4490, 107007, 19994, 3334, 2783, 1204, 6462, 4012, 79463, 3359,
    2249, 1768, 87317, 3767,
)
# fmt: on
assert len(CODE_TOKEN_IDS) == len(CODES) == len(set(CODE_TOKEN_IDS))


def check_tokenizer(
    tokenizer: Tokenizer, config: DecisionConfig, *, hand_over: bool = False
) -> None:
    """That `tokenizer` gives the ids the checkpoint `config` describes was trained with:
    each control token at its id (`CONTROL_TOKEN_IDS`), each code exactly one token whose
    id is `config.token_ids`' entry for it, which must be section 2's (`CODE_TOKEN_IDS`),
    and no added token that is not special. ValueError otherwise. A runtime calls it once,
    when it loads the checkpoint (`load_tokenizer` does); `render` does not call it.

    The tokenizer is checked as it is now, through a fresh plain-text copy that is then
    dropped, not through the copy `render` may have cached. The copy is transient but
    doubles peak memory (about 170 MB more for Gemma 4's); with `hand_over=True` the
    tokenizer is instead handed over to the core (`plain_text_tokenizer`) and checked in
    place, with no copy, and `render` then uses it as it is."""
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
    if tuple(config.token_ids) != CODE_TOKEN_IDS:
        bad = [
            c
            for c, i, want in zip(CODES, config.token_ids, CODE_TOKEN_IDS, strict=True)
            if i != want
        ]
        raise ValueError(
            f"decision_config.json's token_ids are not Gemma 4's for codes {bad[:5]} "
            "(docs/spec.md, section 2)"
        )
    # Encoded as request text is: plain text, special-token matching off, by a fresh copy,
    # which also refuses a tokenizer with added tokens that are not special.
    if hand_over:
        saved = (tokenizer.encode_special_tokens, tokenizer.padding, tokenizer.truncation)
        copy = plain_text_tokenizer(tokenizer)
    else:
        copy = _make_plain_text_copy(tokenizer)
    encoded = [list(e.ids) for e in copy.encode_batch(list(CODES), add_special_tokens=False)]
    bad = [
        c for c, ids, want in zip(CODES, encoded, config.token_ids, strict=True) if ids != [want]
    ]
    if bad:
        if hand_over:
            # A hand-over whose check fails changes nothing: the core lets go of the
            # tokenizer and its settings are restored.
            release_plain_text_tokenizer(tokenizer)
            _restore_settings(tokenizer, *saved)
        shown = {c: encoded[CODES.index(c)] for c in bad[:5]}
        raise ValueError(
            f"the tokenizer does not encode codes {bad[:5]}{'...' if len(bad) > 5 else ''} as "
            f"decision_config.json's token_ids say: it gives {shown}"
        )


def load_tokenizer(
    checkpoint: str | os.PathLike[str],
    config: DecisionConfig | None = None,
    *,
    hand_over: bool = False,
) -> Tokenizer:
    """The checkpoint's `tokenizer.json`, checked by `check_tokenizer` against `config` (or
    the directory's decision_config.json). With `hand_over=True` it is handed over to the
    core as it is checked, so that neither the check nor `render` makes a copy of it."""
    from tokenizers import Tokenizer

    d = os.fspath(checkpoint)
    cfg = config if config is not None else DecisionConfig.load(d)
    tok: Tokenizer = Tokenizer.from_file(os.path.join(d, TOKENIZER_FILE))
    check_tokenizer(tok, cfg, hand_over=hand_over)
    return tok
