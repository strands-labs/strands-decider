"""Rendering for training. Every row goes through `strands_decider.core` (docs/spec.md section 3), the library a
runtime renders requests with, so the trainer and a server see the same token ids.

`render_question` renders one row and `prompt_tokens` counts its tokens; `left_pad` stacks rendered rows into a batch;
`code_token_ids` reads the 255 codes' token ids off the core. The templates (`TEMPLATES`) put a body inside:

    strands-plain   <bos>{body}\\n<answer>
    strands-chat    <bos><|turn>user\\n{body}<turn|>\\n<|turn>model\\n

The body is a `<state>` block followed by a `<question>` block, which lists the options one per line. An option's line
is `f"{code}."`, a space and the option's name, then, for an option with a description,
`strands_decider.core.prompt.DESCRIPTION_SEPARATOR` and the description. The separator is `" \\u2014 "`: a
space, U+2014 EM DASH and a space.

The core tokenizes the state, through `</state>\\n`, apart from the question and reports its length (the spec's S,
`state_len`): that is where the prefix_bidir mask divides the prompt.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from strands_decider import core

TEMPLATES = (core.PLAIN, core.CHAT)  # the spec's two template names


def core_question(question: dict) -> dict:
    """`question` adjusted to pass the core's request validation. Two kinds of training question fail it although
    the spec renders an equivalent: one with no instructions, which get "" and so print nothing, and a score with
    levels that are not strings, each of which is replaced by `core.render_content` of it. Nothing else is changed,
    so whatever else the core rejects stays rejected."""
    q = dict(question)
    if q.get("instructions") is None:
        q["instructions"] = ""
    if q.get("type") == "score" and isinstance(q.get("criteria"), list):
        q["criteria"] = [c if isinstance(c, str) else core.render_content(c) for c in q["criteria"]]
    return q


def bound_options(question: dict) -> list[str]:
    """What the core reports each code stands for, in code order: the criteria names of a choice, "0", "1", ... for
    a score's levels, "false" and "true" for a noul."""
    if question["type"] == "choice":
        return list(question["criteria"])
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


def render_question(row: dict, tokenizer: Any, prompt: str) -> core.RenderedQuestion:
    """The core's rendering of `row` as a request of one question. ValueError if the template cannot render the row,
    or if the core's codes would stand for other options than the trainer's."""
    if row.get("images"):
        raise ValueError("this recipe trains on text only, and the row has images")
    rendered = core.render(
        {"state": row["state"], "questions": {"q": core_question(row["question"])}},
        tokenizer,
        prompt,
    )
    question = rendered.questions["q"]
    expected = bound_options(row["question"])
    if list(question.options) != expected:
        raise ValueError(
            f"the core's codes stand for {list(question.options)[:4]}..., the trainer's for "
            f"{expected[:4]}..."
        )
    return question


def prompt_tokens(row: dict, tokenizer: Any, prompt: str) -> int:
    """How many tokens `row` renders to under template `prompt`, the state's and the question's together."""
    return len(render_question(row, tokenizer, prompt).ids)


def code_token_ids(tokenizer: Any, prompt: str) -> list[int]:
    """The 255 codes' token ids, in code order, read off the core's rendering of a 255-option probe question."""
    probe = core.render(
        {
            "state": "",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "Choose a code.",
                    "criteria": {c: None for c in core.CODES},
                }
            },
        },
        tokenizer,
        prompt,
    ).questions["q"]
    return list(probe.code_token_ids)


def left_pad(
    rendered: Sequence[core.RenderedQuestion], pad_id: int, max_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three tensors for a batch of rendered rows: input_ids and attention_mask of shape [rows, width], every row
    padded on the left to the longest one, and for each row the position of its first question token, which is its
    padding plus its state's length. Raises if the longest row is over `max_length`, since no prompt is cut short."""
    width = max(len(q.ids) for q in rendered)
    if width > max_length:
        raise ValueError(
            f"a prompt of {width} tokens is over max_length {max_length}; prompts are never cut"
        )
    ids = torch.full((len(rendered), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rendered), width), dtype=torch.long)
    ends = []
    for i, q in enumerate(rendered):
        n = len(q.ids)
        ids[i, width - n :] = torch.tensor(q.ids, dtype=torch.long)
        mask[i, width - n :] = 1
        ends.append(width - n + q.state_len)
    return ids, mask, torch.tensor(ends, dtype=torch.long)
