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
import operator
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .codes import codes_for
from .images import IMAGE_CLOSE, IMAGE_OPEN, IMAGE_SOFT, EncodedImage, decode_images
from .request import Question, Request, parse_request

if TYPE_CHECKING:
    from pydantic import BaseModel
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

    A string is stripped of whitespace (`str.strip`'s); structured data is indented JSON with
    its key order kept, as Python's `json.dumps(indent=2, ensure_ascii=False)` writes it, so
    the same state always renders, and so tokenises, the same way. A request's content is
    checked to render alike everywhere (`request.Request`); content nested too deeply to
    write is a ValueError here too.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    try:
        return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)
    except RecursionError:
        raise ValueError("content nested too deeply to render") from None


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


def _indices(option_order: Any) -> list[int]:
    """An option order as a list of ints: any iterable of integers that `operator.index`
    takes, numpy's among them (a permutation from `rng.permutation(k)`), but not bools, and
    not the characters of a string or the keys of a mapping."""
    if isinstance(option_order, (str, bytes, Mapping)):
        raise ValueError(f"option_order must be a sequence of option indices, not {option_order!r}")
    try:
        items = list(option_order)
    except TypeError:
        raise ValueError(
            f"option_order must be a sequence of option indices, not {option_order!r}"
        ) from None
    order = []
    for i in items:
        try:
            if type(i).__name__ in ("bool", "bool_"):
                raise TypeError
            order.append(operator.index(i))
        except TypeError:
            raise ValueError(
                f"option_order must hold integer option indices, not {items!r}"
            ) from None
    return order


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
        order = _indices(option_order)
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
        # Each run of whitespace, newlines included, becomes one space: an option is one line.
        name, desc = " ".join(names[i].split()), " ".join(descs[i].split())
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
# copy costs about 170 MB; eight is more than a process serving all five Gemma 4 sizes needs.
# A caller that cannot spare the memory hands its tokenizer over (`plain_text_tokenizer`).
_PLAIN_TEXT: OrderedDict[int, tuple[Tokenizer, Tokenizer]] = OrderedDict()
_PLAIN_TEXT_MAX = 8
# Tokenizers handed over by `plain_text_tokenizer`, by id, held: render uses them as they are.
_OWNED: dict[int, Tokenizer] = {}
# Guards _PLAIN_TEXT and _PENDING only; a copy is made outside it, under its tokenizer's lock
# in _PENDING, so a first render of one tokenizer never blocks renders of another.
_PLAIN_TEXT_LOCK = threading.Lock()
_PENDING: dict[int, tuple[Tokenizer, threading.Lock]] = {}


def _refuse_plain_added_tokens(tok: Tokenizer) -> None:
    """Special-token matching off does not stop an added token that is not special from
    matching request text, so a tokenizer with one is refused."""
    plain = [t.content for t in tok.get_added_tokens_decoder().values() if not t.special]
    if plain:
        raise ValueError(
            f"the tokenizer has added tokens that are not special, {plain[:5]}: request text "
            "would be matched to them, which the interface forbids (docs/spec.md, section "
            "3.2)"
        )


def _set_plain_text(tok: Tokenizer) -> None:
    """Special-token matching, padding and truncation off, then checked: every added token's
    string must encode as text, not as its own id."""
    tok.no_padding()
    tok.no_truncation()
    tok.encode_special_tokens = True
    added = {t.content: tok.token_to_id(t.content) for t in tok.get_added_tokens_decoder().values()}
    strings = list(added)
    for s, enc in zip(strings, tok.encode_batch(strings, add_special_tokens=False), strict=True):
        if list(enc.ids) == [added[s]]:
            raise ValueError(
                f"request text {s!r} encodes to its control token's id: this version of "
                "tokenizers does not honour encode_special_tokens, or the vocabulary holds the "
                "string as an ordinary token"
            )


def _is_plain_text(tok: Tokenizer) -> bool:
    return bool(tok.encode_special_tokens) and tok.padding is None and tok.truncation is None


def _make_plain_text_copy(tok: Tokenizer) -> Tokenizer:
    """A copy of `tok`, set to encode all text as plain text (`_set_plain_text`)."""
    _refuse_plain_added_tokens(tok)
    copy: Tokenizer = type(tok).from_str(tok.to_str())
    _set_plain_text(copy)
    return copy


def plain_text_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """Hand `tokenizer` over to the core, so that `render` encodes with it directly and makes
    no private copy of it (Gemma 4's costs about 170 MB, which matters on AWS Lambda). It is
    set in place to encode all text as plain text: special-token matching, padding and
    truncation off, checked as a copy is; if the check fails it is left as it was, and the
    ValueError raised. Returns the tokenizer.

    Once handed over it is the core's: the caller must not change its settings or its
    vocabulary, and must not use it to encode text with special-token matching on; `render`
    refuses it if its settings have changed. The core holds it, by a strong reference, until
    `release_plain_text_tokenizer` or the end of the process.

    Peak memory: rendering with it makes no copy, but `check_tokenizer` and `load_tokenizer`
    make a transient one for their check unless they are asked to hand the tokenizer over
    (`hand_over=True`), when they check it in place and make none."""
    _refuse_plain_added_tokens(tokenizer)
    saved = (tokenizer.encode_special_tokens, tokenizer.padding, tokenizer.truncation)
    try:
        _set_plain_text(tokenizer)
    except BaseException:
        _restore_settings(tokenizer, *saved)
        raise
    with _PLAIN_TEXT_LOCK:
        _OWNED[id(tokenizer)] = tokenizer
        _PLAIN_TEXT.pop(id(tokenizer), None)
    return tokenizer


def release_plain_text_tokenizer(tokenizer: Tokenizer) -> None:
    """Take back a tokenizer handed over by `plain_text_tokenizer`: the core drops its
    reference, and `render` would copy it again. Its settings stay plain text."""
    with _PLAIN_TEXT_LOCK:
        if _OWNED.get(id(tokenizer)) is tokenizer:
            del _OWNED[id(tokenizer)]


def _restore_settings(
    tok: Tokenizer,
    encode_special_tokens: bool,
    padding: dict[str, Any] | None,
    truncation: dict[str, Any] | None,
) -> None:
    tok.encode_special_tokens = encode_special_tokens
    if padding is None:
        tok.no_padding()
    else:
        tok.enable_padding(**padding)
    if truncation is None:
        tok.no_truncation()
    else:
        tok.enable_truncation(**truncation)


def _plain_text_copy(tok: Tokenizer) -> Tokenizer:
    """A private copy of `tok` that encodes all text as plain text: special-token matching,
    padding and truncation off, set once and never changed, so it is safe to share across
    threads. One copy per tokenizer object, made on first use (Gemma 4's takes about 0.3 s
    and 170 MB), outside the cache's lock; the eight most recently used tokenizers' copies are
    kept. A tokenizer changed after its first render keeps its first copy: pass the changed
    one as a new object. A tokenizer handed over by `plain_text_tokenizer` is used as it is."""
    with _PLAIN_TEXT_LOCK:
        if _OWNED.get(id(tok)) is tok:
            if not _is_plain_text(tok):
                raise ValueError(
                    "a tokenizer handed over by plain_text_tokenizer has had its settings "
                    "changed: special-token matching, padding and truncation must stay off"
                )
            return tok
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
    images: tuple[EncodedImage, ...]
    questions: dict[str, Question]
    state_pieces: list[Piece]
    blocks: dict[str, QuestionBlock]
    question_pieces: dict[str, list[Piece]]


def _prepare(
    request: dict[str, Any] | BaseModel,
    variant: str,
    option_orders: Mapping[str, Sequence[int]] | None,
) -> _Prepared:
    if variant not in VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; this runtime implements {VARIANTS}")
    req = parse_request(request)
    images = decode_images(req.images)
    questions = {n: q.normalised() for n, q in req.questions.items()}
    orders = dict(option_orders or {})
    unknown = set(orders) - set(questions)
    if unknown:
        raise ValueError(f"option_orders names unknown questions {sorted(unknown)}")
    sblock = state_pieces(req.state, images)
    blocks = {n: question_block(q, orders.get(n)) for n, q in questions.items()}
    return _Prepared(
        request=req,
        images=images,
        questions=questions,
        state_pieces=prompt_pieces(variant, sblock, "")[0],
        blocks=blocks,
        question_pieces={n: prompt_pieces(variant, sblock, b.text)[1] for n, b in blocks.items()},
    )


def render_text(
    request: dict[str, Any] | BaseModel,
    variant: str,
    *,
    option_orders: Mapping[str, Sequence[int]] | None = None,
) -> tuple[str, dict[str, str]]:
    """The prompt as text, without a tokenizer: the state part, and each question's part.
    Control tokens appear as their strings (`<bos>`, `<|turn>`, `<turn|>`, and an image's
    `<|image>`, N `<|image|>` and `<image|>`)."""
    prep = _prepare(request, variant, option_orders)
    return _text(prep.state_pieces), {n: _text(p) for n, p in prep.question_pieces.items()}


def render(
    request: dict[str, Any] | BaseModel,
    tokenizer: Tokenizer,
    variant: str,
    *,
    option_orders: Mapping[str, Sequence[int]] | None = None,
    max_length: int | None = None,
) -> RenderedRequest:
    """Render every question of `request` for `variant`: the state's token ids, each
    question's, its codes and their token ids, and each image's span.

    Args:
        request: The request: the JSON object of `POST /v1/systemone`, as `json.loads` reads
            it, a v1 `SystemOneRequest`, or a `Request` from `parse_request`. It is checked
            as `parse_request` checks it, and then its images (`images.decode_images`).
        tokenizer: A `tokenizers.Tokenizer`, Gemma 4's: the checkpoint's own. Render never
            changes it, and may run on several threads at once with the same one.
        variant: The prompt, `strands-plain` or `strands-chat` (`VARIANTS`): the
            checkpoint's decision_config.json `prompt`.
        option_orders: For training augmentation, a permutation of a question's option
            indices, by question name, in the order the options are rendered; a score's
            levels may only be reversed. A question it does not name keeps the request's
            order.
        max_length: The most tokens a prompt may have, or None for no limit. A longer
            prompt is refused, never truncated.

    Returns:
        The rendered request: its questions in the request's order, each with the shared
        state ids, its own ids, its codes and their token ids, and the state's images.

    Raises:
        ValueError: When the request or one of its images is malformed (`parse_request`,
            `images.decode_images`), the variant is unknown, `option_orders` names an
            unknown question or is not a permutation, the tokenizer lacks a control token,
            splits a code, has an added token that is not special or one that request text
            encodes to, or was handed over by `plain_text_tokenizer` and has had its
            settings changed since, or a prompt is longer than `max_length`.
        ImportError: When the request has images and Pillow is not installed.
    """
    prep = _prepare(request, variant, option_orders)
    names = list(prep.blocks)
    kmax = max(len(b.codes) for b in prep.blocks.values())

    enc = _Encoder(tokenizer)
    encoded = enc.pieces([prep.state_pieces, *(prep.question_pieces[n] for n in names)])
    code_ids = enc.codes(codes_for(kmax))

    state_ids = tuple(encoded[0])
    state_text = _text(prep.state_pieces)
    images = _rendered_images(state_ids, prep.images, enc)
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
            kind=prep.questions[n].kind,
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
