"""A v2 request as token ids: the prompt template of docs/v2-spec.md.

Each question becomes one prompt, `state part + question part`. The state part is
identical for every question of a request and ends at S, its token count; a runtime
encodes it once and shares it. The question part holds the question and its options,
each option labelled with its code (`A. billing — payments`), and the prompt's last token
is where the readout reads the answer.

Two variants, `strands-v2-plain` and `strands-v2-chat`, share one body:

    plain:  <bos>                 + body + "\\n<answer>"
    chat:   <bos><|turn>user\\n     + body + "<turn|>\\n<|turn>model\\n"

A prompt is a sequence of control tokens, emitted by id, and text runs, each encoded on
its own as plain text: a control-token string inside the request (say `<turn|>` in a
state) is text, never a control token. The state part and the question part are encoded
separately, so no token can straddle S. With Gemma 4's tokenizer the result is also what
encoding the whole prompt in one call gives, for any request without control-token
strings; tests/test_v2_prompt.py proves both.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .codes import codes_for
from .request import Question, Request, parse_request

if TYPE_CHECKING:
    from tokenizers import Tokenizer

PLAIN = "strands-v2-plain"
CHAT = "strands-v2-chat"
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

    A string is stripped; structured data is indented JSON with its key order kept, so
    the same state always renders, and so tokenises, the same way.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


def one_line(text: str) -> str:
    """Whitespace runs, newlines included, as one space: an option is always one line."""
    return " ".join(text.split())


def state_block(state: Any) -> str:
    """`<state>\\n{state}\\n</state>\\n`, or `<state>\\n</state>\\n` for an empty state."""
    text = render_content(state)
    return f"<state>\n{text}\n</state>\n" if text else "<state>\n</state>\n"


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
    augmentation); the binding it produces is returned, so callers never track it.
    """
    names = list(question.options)
    descs = [render_content(d) for d in question.criteria]
    order = list(range(len(names))) if option_order is None else list(option_order)
    if sorted(order) != list(range(len(names))):
        raise ValueError("option_order must be a permutation of the option indices")
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


def prompt_pieces(variant: str, state: str, body: str) -> tuple[list[Piece], list[Piece]]:
    """The state part's and the question part's pieces for `variant`."""
    if variant == PLAIN:
        return [(True, BOS), (False, state)], [(False, body + "\n<answer>")]
    if variant == CHAT:
        return (
            [(True, BOS), (True, TURN_OPEN), (False, "user\n" + state)],
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
class RenderedQuestion:
    """One question's prompt: `ids = state_ids + question_ids`, S = `state_len`.

    The readout reads the hidden state at `last_index`, the prompt's last token. Its k
    options are bound to `codes` in order: `options[k]` is the answer value ("false" or
    "true", a choice name, or a score level) whose code is `codes[k]`, and
    `code_token_ids[k]` is that code's token id. `request_options` holds the same values
    in the request's order, which differs from `options` only under an option order.
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
    def input_tokens(self) -> int:
        """Tokens a shared-state runtime encodes: the state once, then every question."""
        return len(self.state_ids) + sum(len(q.question_ids) for q in self.questions.values())


def _text(pieces: Sequence[Piece]) -> str:
    return "".join(s for _, s in pieces)


class _Encoder:
    """Encodes pieces with special-token matching off, restoring the tokenizer after.

    Like v1's `prompting.fit_window`, this turns the tokenizer's padding and truncation
    off, so it must not run on one tokenizer from two threads at once.
    """

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self._was = False

    def __enter__(self) -> _Encoder:
        self.tok.no_padding()
        self.tok.no_truncation()
        self._was = bool(self.tok.encode_special_tokens)
        self.tok.encode_special_tokens = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.tok.encode_special_tokens = self._was

    def control(self, s: str) -> int:
        i = self.tok.token_to_id(s)
        if i is None:
            raise ValueError(f"the tokenizer has no {s} token; the v2 prompts need Gemma 4's")
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
    sblock = state_block(req.state)
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
    Control tokens appear as their strings (`<bos>`, `<|turn>`, `<turn|>`)."""
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
    `tokenizer` is a `tokenizers.Tokenizer`, Gemma 4's. `option_orders` maps a question
    name to a permutation of its options (training augmentation). A prompt longer than
    `max_length` tokens is refused, never truncated.
    """
    prep = _prepare(request, variant, option_orders)
    names = list(prep.blocks)
    kmax = max(len(b.codes) for b in prep.blocks.values())

    with _Encoder(tokenizer) as enc:
        encoded = enc.pieces([prep.state_pieces, *(prep.question_pieces[n] for n in names)])
        code_ids = enc.codes(codes_for(kmax))

    state_ids = tuple(encoded[0])
    state_text = _text(prep.state_pieces)
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
        )
    return RenderedRequest(variant=variant, model=prep.request.model, questions=out)
