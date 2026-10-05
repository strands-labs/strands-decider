"""The teacher must see exactly the prompt its benchmark score was measured on."""
from __future__ import annotations

from typing import ClassVar

import pytest

from strands_decider.data.format import Example
from strands_decider.data.teacher import LETTERS, render, to_row

NOUL = Example(kind="noul", state="WIN a FREE cruise", instructions="Is this spam?",
               options=[["false", "genuine"], ["true", "promotional"]], label=1, task="spam")
CHOICE = Example(kind="choice", state="Arsenal won 3-1.", instructions="Which section?",
                 options=[["World", "international"], ["Sports", ""]], label=1, task="ag_news")
SCORE = Example(kind="score", state="Great food.", instructions="How positive?",
                options=[["0", "bad"], ["1", "ok"], ["2", "good"]], label=2, task="yelp")


def test_rows_match_the_jevbench_semif_adapter_mapping():
    row, order = to_row(NOUL)
    # true first, as the adapter lists it; order maps teacher position -> canonical index
    assert [o["id"] for o in row["options"]] == ["true", "false"]
    assert [o["description"] for o in row["options"]] == ["true: promotional", "false: genuine"]
    assert order == [1, 0]
    row, order = to_row(CHOICE)
    # an empty description falls back to the option name, as the adapter does
    assert [o["description"] for o in row["options"]] == ["World: international", "Sports: Sports"]
    assert order == [0, 1]
    row, _ = to_row(SCORE)
    assert [o["description"] for o in row["options"]] == ["0: bad", "1: ok", "2: good"]


def test_too_many_options_get_no_teacher():
    many = Example(kind="choice", state="s", instructions="q",
                   options=[[f"o{i}", ""] for i in range(len(LETTERS) + 1)], label=0)
    assert to_row(many) is None


@pytest.mark.parametrize("ex", [NOUL, CHOICE, SCORE])
def test_prompt_is_byte_identical_to_semif(ex):
    semif = pytest.importorskip("semif_phase1.direct")
    transformers = pytest.importorskip("transformers")
    try:  # same chat template family as the teacher; skip if not cached offline
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not cached")
    row, _ = to_row(ex)
    ids, _, _ = semif.encode_prompt(tok, row, 4096)
    assert tok.encode(render(tok, row), add_special_tokens=False) == ids


class _FakeLLM:
    """vLLM's LLM as label_vllm uses it: the restricted, renormalised distribution over the
    allowed letters, here a softmax over logits made from the prompt length."""

    seen: ClassVar[list] = []

    def __init__(self, normalised: bool = True, **kw):
        self.normalised, self.kw = normalised, kw

    def generate(self, prompts, params, use_tqdm=False):
        import math
        from types import SimpleNamespace

        out = []
        for prompt, sp in zip(prompts, params, strict=True):
            ids = prompt["prompt_token_ids"]
            _FakeLLM.seen.append((ids, sp.allowed_token_ids, sp.logprobs))
            logits = [(len(ids) % 7) * (k + 1) / 10 for k in range(len(sp.allowed_token_ids))]
            z = math.log(sum(math.exp(x) for x in logits)) + (0 if self.normalised else 1.0)
            lp = {t: SimpleNamespace(logprob=x - z) for t, x in zip(sp.allowed_token_ids, logits, strict=True)}
            out.append(SimpleNamespace(outputs=[SimpleNamespace(logprobs=[lp])]))
        return out


def _fake_vllm(monkeypatch, normalised=True):
    import sys
    from types import ModuleType, SimpleNamespace

    import transformers
    from tiny_qwen35 import tokenizer

    tok = tokenizer()
    tok.chat_template = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<assistant>{% endif %}")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    vllm = ModuleType("vllm")
    vllm.LLM = lambda **kw: _FakeLLM(normalised, **kw)
    vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    inputs = ModuleType("vllm.inputs")
    inputs.TokensPrompt = dict
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    _FakeLLM.seen = []
    return tok


def test_vllm_labels_are_the_letter_softmax_in_canonical_order(monkeypatch):
    import math

    from strands_decider.data.teacher import label_vllm, letter_ids, prompts

    tok = _fake_vllm(monkeypatch)
    examples = [NOUL, CHOICE, SCORE]
    got: dict[int, list[float]] = {}
    out = label_vllm("m", "r", examples, skip={1}, sink=lambda i, p: got.__setitem__(i, p))
    letters = letter_ids(tok)
    want_rows = [r for r in prompts(tok, examples, 4096) if r[0] != 1]
    assert [(ids, allowed) for ids, allowed, _ in _FakeLLM.seen] == \
        [(ids, letters[:len(order)]) for _, ids, order in want_rows]
    for i, ids, order in want_rows:
        logits = [(len(ids) % 7) * (k + 1) / 10 for k in range(len(order))]
        z = sum(math.exp(x) for x in logits)
        teacher_order = [math.exp(x) / z for x in logits]
        assert out[i] == pytest.approx([teacher_order[order.index(c)] for c in range(len(order))])
        assert got[i] == out[i]
    assert out[1] is None and 1 not in got  # skipped: labelled by an earlier run


def test_vllm_that_does_not_renormalise_is_refused(monkeypatch):
    from strands_decider.data.teacher import label_vllm

    _fake_vllm(monkeypatch, normalised=False)
    with pytest.raises(RuntimeError, match="processed logprobs"):
        label_vllm("m", "r", [NOUL])
