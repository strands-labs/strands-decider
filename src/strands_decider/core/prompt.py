"""A request as token ids: the prompt template of docs/spec.md.

Each question becomes one prompt, `state part + question part`. The state part is
identical for every question of a request and ends at S, its token count; a runtime
encodes it once and shares it. The question part holds the question and its options,
each option labelled with its code (`A. billing — payments`), and the prompt's last token
is where the readout reads the answer.

Two variants, `strands-plain` and `strands-chat`, share one body:

    plain:  <bos>                 + body + "\\n<answer>"
    chat:   <bos><|turn>user\\n     + body + "<turn|>\\n<|turn>model\\n"

Images go first inside `<state>`, one line each: `<|image>`, N `<|image|>` placeholders
(N from `images.soft_tokens`) and `<image|>`, then the state text's line.

A prompt is a sequence of control tokens, emitted by id, and text runs, each encoded on
its own as plain text: a special-token string inside the request (say `<turn|>` in a
state) is text, never a control token. The state part and the question part are encoded
separately, so no token can straddle S. With Gemma 4's tokenizer the result is also what
encoding the whole prompt in one call gives, for any request without special-token
strings; tests/test_core_prompt.py proves both.

Rendering never changes the caller's tokenizer: text is encoded by a private copy whose
settings are fixed once, so renders on several threads, and other users of the same
tokenizer, cannot change how request text is encoded. A tokenizer whose added tokens are not
all special is refused, since request text could match them.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .codes import codes_for
from .images import IMAGE_CLOSE, IMAGE_OPEN, IMAGE_SOFT, EncodedImage
from .request import Question, Request, parse_request
from .text import one_line, strip

if TYPE_CHECKING:
    from tokenizers import Tokenizer

PLAIN = "strands-plain"
CHAT = "strands-chat"
VARIANTS = (PLAIN, CHAT)

# Gemma 4's control tokens, by their strings.
BOS = "<bos>"
TURN_OPEN = "<|turn>"
TURN_CLOSE = "<turn|>"

HEADERS = {
    "noul": "Decide whether the statement is true of the state.",
    "choice": "Select exactly one option.",
    "score": "Rate the state against the ordered levels below (lowest first).",
}
ANSWER_LINE = "Answer with the code of one option."
# Between an option's name and its description: space, U+2014 EM DASH, space (as v1).
DESCRIPTION_SEPARATOR = " — "

# (is_control, string): a control token by its string, or a run of text.
Piece = tuple[bool, str]


def render_content(content: Any) -> str:
    """A state, instruction or description as text: v1's `prompting.render_content`.

    A string is stripped of whitespace (`text.WHITESPACE`, the spec's); structured data is
    indented JSON with its key order kept, as Python's `json.dumps(indent=2,
    ensure_ascii=False)` writes it, so the same state always renders, and so tokenises, the
    same way. docs/spec.md spells out the JSON, number by number and escape by escape, and
    `request.check_json` refuses what it cannot render alike everywhere.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return strip(content)
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


def state_block(state: Any) -> str:
    """`<state>\\n{state}\\n</state>\\n`, or `<state>\\n</state>\\n` for an empty state."""
    text = render_content(state)
    return f"<state>\n{text}\n</state>\n" if text else "<state>\n</state>\n"


def state_pieces(state: Any, images: Sequence[EncodedImage] = ()) -> list[Piece]:
    """The state block as pieces. Without images it is `state_block(state)`, one text run.
    With images, each image's line comes first, in order: `<|image>`, N `<|image|>` and
    `<image|>` as control tokens, the lines joined by a line feed; then the state text's
    line, unless the state is empty:

        <state>\\n<|image>...<image|>\\n<|image>...<image|>\\n{state}\\n</state>\\n
        <state>\\n<|image>...<image|>\\n</state>\\n                       (empty state)
    """
    if not images:
        return [(False, state_block(state))]
    text = render_content(state)
    pieces: list[Piece] = [(False, "<state>\n")]
    for i, image in enumerate(images):
        if i:
            pieces.append((False, "\n"))
        pieces += [(True, IMAGE_OPEN), *[(True, IMAGE_SOFT)] * image.tokens, (True, IMAGE_CLOSE)]
    pieces.append((False, f"\n{text}\n</state>\n" if text else "\n</state>\n"))
    return pieces


@dataclass(frozen=True)
class QuestionBlock:
    text: str
    request_options: tuple[str, ...]  # the answer values in the request's order
    options: tuple[str, ...]  # options[k]: the answer value bound to the k-th code
    descriptions: tuple[str, ...]  # options[k]'s description, as render_content gives it
    codes: tuple[str, ...]


def question_block(question: Question, option_order: Sequence[int] | None = None) -> QuestionBlock:
    """The question part's body, from `<question` to `</question>`.

    `option_order` permutes the options before they take their codes (training
    augmentation); the binding it produces is returned, so callers never track it. It is a
    sequence of the option indices, each once: any order of a choice's options or a noul's
    two, and a score's levels in order or reversed (as v1's training reverses them).
    """
    names = list(question.options)
    descs = [render_content(d) for d in question.criteria]
    identity = list(range(len(names)))
    if option_order is None:
        order = identity
    else:
        if isinstance(option_order, (str, bytes, Mapping)) or not isinstance(
            option_order, Sequence
        ):
            raise ValueError(
                f"option_order must be a sequence of option indices, not {option_order!r}"
            )
        order = list(option_order)
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in order):
            raise ValueError(f"option_order must hold integer option indices, not {order!r}")
        if sorted(order) != identity:
            raise ValueError("option_order must be a permutation of the option indices")
        if question.kind == "score" and order not in (identity, identity[::-1]):
            raise ValueError(
                "a score's levels are ordered: option_order may reverse them, as v1's training "
                f"does, but not shuffle them ({order!r})"
            )
    codes = codes_for(len(names))
    lines = []
    for code, i in zip(codes, order, strict=True):
        name, desc = one_line(names[i]), one_line(descs[i])
        line = f"{code}." + (f" {name}" if name else "")
        lines.append(line + (f"{DESCRIPTION_SEPARATOR}{desc}" if desc else ""))

    parts = [f'<question type="{question.kind}">', HEADERS[question.kind]]
    instructions = render_content(question.instructions)
    if instructions:
        parts.append(instructions)
    parts += ["<options>", *lines, "</options>", ANSWER_LINE, "</question>"]
    return QuestionBlock(
        text="\n".join(parts),
        request_options=tuple(names),
        options=tuple(names[i] for i in order),
        descriptions=tuple(descs[i] for i in order),
        codes=codes,
    )


def prompt_pieces(
    variant: str, state: str | Sequence[Piece], body: str
) -> tuple[list[Piece], list[Piece]]:
    """The state part's and the question part's pieces for `variant`. `state` is the state
    block, as text or as `state_pieces` gives it (which starts and ends with text)."""
    sp = [(False, state)] if isinstance(state, str) else list(state)
    if variant == PLAIN:
        return [(True, BOS), *sp], [(False, body + "\n<answer>")]
    if variant == CHAT:
        return (
            [(True, BOS), (True, TURN_OPEN), (False, "user\n" + sp[0][1]), *sp[1:]],
            [
                (False, body),
                (True, TURN_CLOSE),
                (False, "\n"),
                (True, TURN_OPEN),
                (False, "model\n"),
            ],
        )
    raise ValueError(f"unknown prompt variant {variant!r}; this runtime implements {VARIANTS}")


@dataclass(frozen=True)
class RenderedImage:
    """One image of the state, in the request's order: the file the request sent (`data`,
    decoded from base64), its format, its header size, the sha256 of `data`, and where it
    sits in the state ids. `state_ids[start:end]` are its `tokens` (N) `<|image|>`
    placeholders, which the runtime's vision encoder fills in; `<|image>` is at `start - 1`
    and `<image|>` at `end`."""

    data: bytes = field(repr=False)
    format: str
    width: int
    height: int
    sha256: str
    start: int
    tokens: int

    @property
    def end(self) -> int:
        return self.start + self.tokens

    @property
    def span(self) -> tuple[int, int]:
        """[start, end): the placeholders in the state ids."""
        return (self.start, self.end)


@dataclass(frozen=True)
class RenderedQuestion:
    """One question's prompt: `ids = state_ids + question_ids`, S = `state_len`.

    The readout reads the hidden state at `last_index`, the prompt's last token. Its k
    options are bound to `codes` in order: `options[k]` is the answer value ("false" or
    "true", a choice name, or a score level) whose code is `codes[k]`, and
    `code_token_ids[k]` is that code's token id. `request_options` holds the same values
    in the request's order, which differs from `options` only under an option order.
    `images` are the state's images, in order, with their spans in `state_ids`.
    """

    name: str
    kind: str
    variant: str
    state_ids: tuple[int, ...]
    question_ids: tuple[int, ...]
    codes: tuple[str, ...]
    code_token_ids: tuple[int, ...]
    options: tuple[str, ...]
    request_options: tuple[str, ...]
    descriptions: tuple[str, ...]
    state_text: str
    question_text: str
    images: tuple[RenderedImage, ...] = ()

    @property
    def state_len(self) -> int:
        return len(self.state_ids)

    @property
    def ids(self) -> tuple[int, ...]:
        return self.state_ids + self.question_ids

    @property
    def text(self) -> str:
        return self.state_text + self.question_text

    @property
    def last_index(self) -> int:
        return len(self.state_ids) + len(self.question_ids) - 1

    @property
    def option_to_code(self) -> dict[str, str]:
        return dict(zip(self.options, self.codes, strict=True))


@dataclass(frozen=True)
class RenderedRequest:
    variant: str
    model: str
    questions: dict[str, RenderedQuestion] = field(default_factory=dict)

    @property
    def state_ids(self) -> tuple[int, ...]:
        return next(iter(self.questions.values())).state_ids

    @property
    def images(self) -> tuple[RenderedImage, ...]:
        """The state's images, in order, with their spans in `state_ids`."""
        return next(iter(self.questions.values())).images

    @property
    def input_tokens(self) -> int:
        """Tokens a shared-state runtime encodes: the state once, then every question."""
        return len(self.state_ids) + sum(len(q.question_ids) for q in self.questions.values())


def _text(pieces: Sequence[Piece]) -> str:
    return "".join(s for _, s in pieces)


# Keyed by id with the tokenizer itself held, so an id is never reused while its entry lives:
# tokenizers before 0.23 cannot be weakly referenced. Least recently used out, because each
# copy costs about 170 MB; a process that serves all five Gemma 4 sizes, each with its own
# tokenizer, should raise the bound to five.
_PLAIN_TEXT: OrderedDict[int, tuple[Tokenizer, Tokenizer]] = OrderedDict()
_PLAIN_TEXT_MAX = 4
# Guards _PLAIN_TEXT and _PENDING only; a copy is made outside it, under its tokenizer's lock
# in _PENDING, so a first render of one tokenizer never blocks renders of another.
_PLAIN_TEXT_LOCK = threading.Lock()
_PENDING: dict[int, tuple[Tokenizer, threading.Lock]] = {}


def _make_plain_text_copy(tok: Tokenizer) -> Tokenizer:
    """A copy of `tok` with special-token matching, padding and truncation off, checked to
    encode every added token's string as text. Refuses a tokenizer with an added token that
    is not special: special-token matching off does not stop such a token from matching
    request text."""
    plain = [t.content for t in tok.get_added_tokens_decoder().values() if not t.special]
    if plain:
        raise ValueError(
            f"the tokenizer has added tokens that are not special, {plain[:5]}: request text "
            "would be matched to them, which the interface forbids (docs/spec.md, From text "
            "to token ids)"
        )
    copy: Tokenizer = type(tok).from_str(tok.to_str())
    copy.no_padding()
    copy.no_truncation()
    copy.encode_special_tokens = True
    added = {
        t.content: copy.token_to_id(t.content) for t in tok.get_added_tokens_decoder().values()
    }
    strings = list(added)
    for s, enc in zip(strings, copy.encode_batch(strings, add_special_tokens=False), strict=True):
        if list(enc.ids) == [added[s]]:
            raise ValueError(
                f"request text {s!r} encodes to its control token's id: this version of "
                "tokenizers does not honour encode_special_tokens, or the vocabulary holds the "
                "string as an ordinary token"
            )
    return copy


def _plain_text_copy(tok: Tokenizer) -> Tokenizer:
    """A private copy of `tok` that encodes all text as plain text: special-token matching,
    padding and truncation off, set once and never changed, so it is safe to share across
    threads. One copy per tokenizer object, made on first use (Gemma 4's takes about 0.3 s
    and 170 MB), outside the cache's lock; the four most recently used tokenizers' copies are
    kept. A tokenizer changed after its first render keeps its first copy: pass the changed
    one as a new object."""
    with _PLAIN_TEXT_LOCK:
        entry = _PLAIN_TEXT.get(id(tok))
        if entry is not None and entry[0] is tok:
            _PLAIN_TEXT.move_to_end(id(tok))
            return entry[1]
        pending = _PENDING.get(id(tok))
        if pending is None or pending[0] is not tok:
            pending = (tok, threading.Lock())
            _PENDING[id(tok)] = pending
    with pending[1]:  # one copy per tokenizer, however many first renders race to make it
        with _PLAIN_TEXT_LOCK:
            entry = _PLAIN_TEXT.get(id(tok))
            if entry is not None and entry[0] is tok:
                _PLAIN_TEXT.move_to_end(id(tok))
                return entry[1]
        try:
            copy = _make_plain_text_copy(tok)
        except BaseException:
            with _PLAIN_TEXT_LOCK:
                if _PENDING.get(id(tok)) is pending:
                    del _PENDING[id(tok)]
            raise
        with _PLAIN_TEXT_LOCK:  # cached and no longer pending in one step
            _PLAIN_TEXT[id(tok)] = (tok, copy)
            _PLAIN_TEXT.move_to_end(id(tok))
            while len(_PLAIN_TEXT) > _PLAIN_TEXT_MAX:
                _PLAIN_TEXT.popitem(last=False)
            if _PENDING.get(id(tok)) is pending:
                del _PENDING[id(tok)]
        return copy


class _Encoder:
    """Encodes pieces as the spec says: control tokens by id, text runs as plain text.

    It reads the caller's tokenizer once, to copy it, and never changes it. Text goes
    through the private copy of `_plain_text_copy`, whose special-token matching is off
    for good, so request text can never become a control token, whatever another thread
    does to the caller's tokenizer meanwhile, and renders may run on any number of threads.
    """

    def __init__(self, tok: Tokenizer):
        self.tok = _plain_text_copy(tok)

    def control(self, s: str) -> int:
        i = self.tok.token_to_id(s)
        if i is None:
            raise ValueError(f"the tokenizer has no {s} token; these prompts need Gemma 4's")
        return int(i)

    def texts(self, texts: list[str]) -> list[list[int]]:
        return [list(e.ids) for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def pieces(self, groups: list[list[Piece]]) -> list[list[int]]:
        runs = [s for g in groups for c, s in g if not c]
        encoded = iter(self.texts(runs))
        out: list[list[int]] = []
        for g in groups:
            ids: list[int] = []
            for c, s in g:
                ids.extend([self.control(s)] if c else next(encoded))
            out.append(ids)
        return out

    def codes(self, codes: Sequence[str]) -> tuple[int, ...]:
        ids = self.texts(list(codes))
        bad = [c for c, x in zip(codes, ids, strict=True) if len(x) != 1]
        if bad:
            raise ValueError(f"the tokenizer splits codes {bad}; every code must be one token")
        return tuple(x[0] for x in ids)


def _rendered_images(
    state_ids: tuple[int, ...], images: Sequence[EncodedImage], enc: _Encoder
) -> tuple[RenderedImage, ...]:
    """Each image's span in the state ids. Request text never encodes to a control token,
    so the placeholders are the only `<|image|>` ids, in runs of N between `<|image>` and
    `<image|>`, one run per image in order."""
    if not images:
        return ()
    soft, open_, close = (enc.control(t) for t in (IMAGE_SOFT, IMAGE_OPEN, IMAGE_CLOSE))
    out: list[RenderedImage] = []
    pos = 0
    for image in images:
        start = state_ids.index(soft, pos)
        end = start + image.tokens
        if (
            state_ids[start - 1] != open_
            or state_ids[start:end] != (soft,) * image.tokens
            or state_ids[end] != close
        ):
            raise AssertionError("an image's placeholders are not where the template put them")
        out.append(
            RenderedImage(
                data=image.data,
                format=image.format,
                width=image.width,
                height=image.height,
                sha256=image.sha256,
                start=start,
                tokens=image.tokens,
            )
        )
        pos = end
    return tuple(out)


@dataclass(frozen=True)
class _Prepared:
    request: Request
    state_pieces: list[Piece]
    blocks: dict[str, QuestionBlock]
    question_pieces: dict[str, list[Piece]]


def _prepare(
    request: Any, variant: str, option_orders: Mapping[str, Sequence[int]] | None
) -> _Prepared:
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; this runtime implements {VARIANTS}")
    req = parse_request(request)
    orders = dict(option_orders or {})
    unknown = set(orders) - set(req.questions)
    if unknown:
        raise ValueError(f"option_orders names unknown questions {sorted(unknown)}")
    sblock = state_pieces(req.state, req.images)
    blocks = {n: question_block(q, orders.get(n)) for n, q in req.questions.items()}
    return _Prepared(
        request=req,
        state_pieces=prompt_pieces(variant, sblock, "")[0],
        blocks=blocks,
        question_pieces={n: prompt_pieces(variant, sblock, b.text)[1] for n, b in blocks.items()},
    )


def render_text(
    request: Any, variant: str, *, option_orders: Mapping[str, Sequence[int]] | None = None
) -> tuple[str, dict[str, str]]:
    """The prompt as text, without a tokenizer: the state part, and each question's part.
    Control tokens appear as their strings (`<bos>`, `<|turn>`, `<turn|>`, and an image's
    `<|image>`, N `<|image|>` and `<image|>`)."""
    prep = _prepare(request, variant, option_orders)
    return _text(prep.state_pieces), {n: _text(p) for n, p in prep.question_pieces.items()}


def render(
    request: Any,
    tokenizer: Tokenizer,
    variant: str,
    *,
    option_orders: Mapping[str, Sequence[int]] | None = None,
    max_length: int | None = None,
) -> RenderedRequest:
    """Render every question of `request` for `variant`.

    `request` is the JSON object of `POST /v1/systemone` (or a v1 `SystemOneRequest`).
    `tokenizer` is a `tokenizers.Tokenizer`, Gemma 4's; render never changes it and may run
    on several threads at once with the same one. `option_orders` maps a question
    name to a permutation of its options (training augmentation). A prompt longer than
    `max_length` tokens is refused, never truncated.
    """
    prep = _prepare(request, variant, option_orders)
    names = list(prep.blocks)
    kmax = max(len(b.codes) for b in prep.blocks.values())

    enc = _Encoder(tokenizer)
    encoded = enc.pieces([prep.state_pieces, *(prep.question_pieces[n] for n in names)])
    code_ids = enc.codes(codes_for(kmax))

    state_ids = tuple(encoded[0])
    state_text = _text(prep.state_pieces)
    images = _rendered_images(state_ids, prep.request.images, enc)
    out: dict[str, RenderedQuestion] = {}
    for n, qids in zip(names, encoded[1:], strict=True):
        b = prep.blocks[n]
        if max_length is not None and len(state_ids) + len(qids) > max_length:
            raise ValueError(
                f"question {n!r}: prompt of {len(state_ids) + len(qids)} tokens "
                f"({len(state_ids)} state + {len(qids)} question) exceeds the limit of "
                f"{max_length} tokens"
            )
        out[n] = RenderedQuestion(
            name=n,
            kind=prep.request.questions[n].kind,
            variant=variant,
            state_ids=state_ids,
            question_ids=tuple(qids),
            codes=b.codes,
            code_token_ids=code_ids[: len(b.codes)],
            options=b.options,
            request_options=b.request_options,
            descriptions=b.descriptions,
            state_text=state_text,
            question_text=_text(prep.question_pieces[n]),
            images=images,
        )
    return RenderedRequest(variant=variant, model=prep.request.model, questions=out)
