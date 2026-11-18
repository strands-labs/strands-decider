"""The prompt template, and a request rendered as token ids.

Each question is its own prompt: the state part, shared by every question, then the question part.
The readout reads the prompt's last token.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import groupby
from typing import TYPE_CHECKING

from .codes import CODE_TOKEN_IDS, CODES, CONTROL_TOKEN_IDS
from .images import IMAGE_CLOSE, IMAGE_OPEN, IMAGE_SOFT, EncodedImage, decode_image
from .request import Content, Request

if TYPE_CHECKING:
    from tokenizers import Tokenizer

# The prompt's name in decision_config.json. Gemma 4's chat turn, written out rather than taken
# from the chat template: the 12B, 26B-A4B and 31B templates add an empty thought channel after
# `<|turn>model\n`, which this prompt leaves out at every size.
CHAT = "strands-chat"


class Control(str):
    """A control token, encoded by its id. Every other string in a prompt is text."""


BOS, TURN_OPEN, TURN_CLOSE = Control("<bos>"), Control("<|turn>"), Control("<turn|>")
IMG_OPEN, IMG_SOFT, IMG_CLOSE = Control(IMAGE_OPEN), Control(IMAGE_SOFT), Control(IMAGE_CLOSE)

HEADERS = {
    "noul": "Decide whether the statement is true of the state.",
    "choice": "Select exactly one option.",
    "score": "Rate the state against the ordered levels below (lowest first).",
}


def state_part(state: str, images: tuple[EncodedImage, ...]) -> list[str]:
    return [
        BOS, TURN_OPEN, "user\n<state>\n",
        *[p for im in images for p in (IMG_OPEN, *[IMG_SOFT] * im.tokens, IMG_CLOSE, "\n")],
        f"{state}\n" if state else "",
        "</state>\n",
    ]  # fmt: skip


def question_part(kind: str, instructions: str, options: list[str]) -> list[str]:
    return [
        f'<question type="{kind}">\n{HEADERS[kind]}\n',
        f"{instructions}\n" if instructions else "",
        "<options>\n", *options, "</options>\n",
        "Answer with the code of one option.\n</question>",
        TURN_CLOSE, "\n", TURN_OPEN, "model\n",
    ]  # fmt: skip


def option_line(code: str, name: str, description: str) -> str:
    # "A. name — description", with v1's separator.
    name, description = " ".join(name.split()), " ".join(description.split())  # one line each
    return (
        f"{code}."
        + (f" {name}" if name else "")
        + (f" — {description}" if description else "")
        + "\n"
    )


def render_content(content: Content | None) -> str:
    """A string stripped; structured content as indented JSON in its own key order."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False)


def encode(tokenizer: Tokenizer, prompt: list[str]) -> list[int]:
    """Control tokens by id; each run of text as one plain-text encode, so request text that
    spells a control token (`<turn|>` in a state, say) stays text."""
    ids: list[int] = []
    for is_control, run in groupby(prompt, key=lambda p: isinstance(p, Control)):
        if is_control:
            ids += [CONTROL_TOKEN_IDS[c] for c in run]
        else:
            ids += tokenizer.encode("".join(run), add_special_tokens=False).ids
    return ids


@dataclass(frozen=True)
class RenderedImage:
    """An image of the state, with its N placeholders at `state_ids[start:end]`."""

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
        return (self.start, self.end)


@dataclass(frozen=True)
class RenderedQuestion:
    """One question's prompt, `state_ids + question_ids`. `options[k]` is the answer value whose
    code is `codes[k]`; `request_options` holds them in the request's order."""

    name: str
    kind: str
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


@dataclass(frozen=True)
class RenderedRequest:
    model: str
    questions: dict[str, RenderedQuestion]

    @property
    def state_ids(self) -> tuple[int, ...]:
        return next(iter(self.questions.values())).state_ids

    @property
    def images(self) -> tuple[RenderedImage, ...]:
        return next(iter(self.questions.values())).images

    @property
    def input_tokens(self) -> int:
        """Tokens a runtime that shares the state encodes: the state once, then every question."""
        return len(self.state_ids) + sum(len(q.question_ids) for q in self.questions.values())


@dataclass(frozen=True)
class _Question:
    kind: str
    codes: tuple[str, ...]
    options: tuple[str, ...]  # in rendered order: options[k] is codes[k]'s
    request_options: tuple[str, ...]
    descriptions: tuple[str, ...]
    prompt: list[str]


def _parts(
    request: Request, option_orders: dict[str, list[int]] | None
) -> tuple[list[str], tuple[EncodedImage, ...], dict[str, _Question]]:
    """The state part, the decoded image headers, and each question's part."""
    orders = option_orders or {}
    unknown = set(orders) - set(request.questions)
    if unknown:
        raise ValueError(f"option_orders names unknown questions {sorted(unknown)}")
    images = tuple(decode_image(i) for i in request.images)
    questions = {}
    for name, question in request.questions.items():
        q = question.normalised()
        order = list(orders.get(name, range(len(q.options))))
        if sorted(order) != list(range(len(q.options))):
            raise ValueError(f"question {name!r}: option_orders must permute its options")
        codes = CODES[: len(order)]
        options = tuple(q.options[i] for i in order)
        descriptions = tuple(render_content(q.criteria[i]) for i in order)
        lines = [option_line(*x) for x in zip(codes, options, descriptions, strict=True)]
        prompt = question_part(q.kind, render_content(q.instructions), lines)
        questions[name] = _Question(q.kind, codes, options, q.options, descriptions, prompt)
    return state_part(render_content(request.state), images), images, questions


def render_text(
    request: Request, *, option_orders: dict[str, list[int]] | None = None
) -> tuple[str, dict[str, str]]:
    """The prompt as text, control tokens as their strings: the state part and each question's."""
    state, _, questions = _parts(request, option_orders)
    return "".join(state), {n: "".join(q.prompt) for n, q in questions.items()}


def render(
    request: Request,
    tokenizer: Tokenizer,
    *,
    option_orders: dict[str, list[int]] | None = None,
    max_length: int | None = None,
) -> RenderedRequest:
    """Every question of `request` as token ids, with the shared state ids, its codes and their
    ids, and each image's placeholder span. `tokenizer` must encode with special-token matching
    off (`load_tokenizer` and `prepare_tokenizer` set it). `option_orders` permutes a question's
    options for training. A prompt longer than `max_length` is refused, never truncated."""
    if not tokenizer.encode_special_tokens:
        raise ValueError(
            "the tokenizer must encode with special-token matching off, so request text can't "
            "become a control token: load it with load_tokenizer or prepare_tokenizer"
        )
    state, encoded_images, questions = _parts(request, option_orders)
    state_ids = tuple(encode(tokenizer, state))
    starts = [i + 1 for i, t in enumerate(state_ids) if t == CONTROL_TOKEN_IDS[IMAGE_OPEN]]
    images = tuple(
        RenderedImage(im.data, im.format, im.width, im.height, im.sha256, start, im.tokens)
        for im, start in zip(encoded_images, starts, strict=True)
    )
    out = {}
    for name, q in questions.items():
        question_ids = tuple(encode(tokenizer, q.prompt))
        if max_length is not None and len(state_ids) + len(question_ids) > max_length:
            raise ValueError(
                f"question {name!r}: prompt of {len(state_ids) + len(question_ids)} tokens "
                f"({len(state_ids)} state + {len(question_ids)} question) exceeds the limit of "
                f"{max_length} tokens"
            )
        out[name] = RenderedQuestion(
            name=name,
            kind=q.kind,
            state_ids=state_ids,
            question_ids=question_ids,
            codes=q.codes,
            code_token_ids=CODE_TOKEN_IDS[: len(q.codes)],
            options=q.options,
            request_options=q.request_options,
            descriptions=q.descriptions,
            state_text="".join(state),
            question_text="".join(q.prompt),
            images=images,
        )
    return RenderedRequest(model=request.model, questions=out)
