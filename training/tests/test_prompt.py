from types import SimpleNamespace

import pytest

from strands_decider import core
from training import prompt


def test_core_question_mapping():
    q = prompt.core_question(
        {"type": "score", "criteria": ["low", 2, {"x": 1}], "instructions": None}
    )
    assert q["instructions"] == "" and q["criteria"] == ["low", "2", core.render_content({"x": 1})]
    assert (
        prompt.core_question({"type": "noul", "instructions": "Is it?"})["instructions"] == "Is it?"
    )


def test_bound_options():
    assert prompt.bound_options({"type": "choice", "criteria": {"b": None, "a": None}}) == [
        "b",
        "a",
    ]
    assert prompt.bound_options({"type": "score", "criteria": ["x", "y", "z"]}) == ["0", "1", "2"]
    assert prompt.bound_options({"type": "noul"}) == ["false", "true"]


def test_the_option_separator_is_the_documented_one():
    """prompt.py's docstring and training/README.md describe the option line by the core's constant and its value."""
    from strands_decider.core import prompt as core_prompt

    assert core_prompt.DESCRIPTION_SEPARATOR == " \u2014 "
    for text in (prompt.__doc__,):
        assert (
            "strands_decider.core.prompt.DESCRIPTION_SEPARATOR" in text and '`" \\u2014 "`' in text
        )
        assert " - <description>" not in text and "\u2014" not in text  # no dash typed, no hyphen


@pytest.mark.parametrize("template", prompt.TEMPLATES)
def test_option_lines_render_as_documented(gemma4_tokenizer, template):
    tokenizer = gemma4_tokenizer
    row = {
        "state": "s",
        "question": {
            "type": "choice",
            "instructions": "Pick.",
            "criteria": {"billing": "payments", "shipping": None},
        },
    }
    text = tokenizer.decode(
        list(prompt.render_question(row, tokenizer, template).ids), skip_special_tokens=False
    )
    assert "\nA. billing \u2014 payments\nB. shipping\n" in text


def test_left_pad_pads_on_the_left_and_places_the_boundary():
    rendered = [
        SimpleNamespace(ids=(5, 6, 7, 8), state_len=2),
        SimpleNamespace(ids=(1, 2, 3, 4, 5, 6), state_len=3),
    ]
    ids, mask, ends = prompt.left_pad(rendered, 0, 16)
    assert ids.tolist() == [[0, 0, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6]]
    assert mask.tolist() == [[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    assert ends.tolist() == [4, 3]
    with pytest.raises(ValueError):
        prompt.left_pad(rendered, 0, 5)


CUE = {core.PLAIN: [107, 236820, 14433, 236813], core.CHAT: [106, 107, 105, 4368, 107]}


@pytest.mark.parametrize("template", prompt.TEMPLATES)
def test_rendering_with_the_gemma_tokenizer(gemma4_tokenizer, template):
    tokenizer = gemma4_tokenizer
    row = {
        "state": "Order #A-1182 arrived damaged.",
        "question": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"C": "4", "A": "3", "B": None},
        },
    }
    q = prompt.render_question(row, tokenizer, template)
    assert list(q.ids[-len(CUE[template]) :]) == CUE[template]
    assert list(q.options) == ["C", "A", "B"] and len(q.codes) == 3
    assert prompt.prompt_tokens(row, tokenizer, template) == len(q.ids)
    ids = prompt.code_token_ids(tokenizer, template)
    assert len(ids) == 255 and len(set(ids)) == 255
    with pytest.raises(ValueError):
        prompt.render_question({**row, "images": ["x.png"]}, tokenizer, template)
