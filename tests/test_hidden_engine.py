"""HiddenStateEngine must answer as SystemOneEngine does, given the same hidden states.

The backend here is the torch engine's own torso, run one full forward per question, so
any difference is the hidden-state engine's: the window fitting, the offsets, the rows it
asks for, the numpy pointer head, the readouts or the saved assets. A tiny random-weight
Qwen3.5 checkpoint is built in a temp dir; nothing is downloaded.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from tokenizers import Tokenizer  # noqa: E402

from strands_decider.hidden_engine import (  # noqa: E402
    HiddenStateEngine,
    NumpyPointerHead,
    save_assets,
)
from strands_decider.infer import EngineConfig, SystemOneEngine, load_engine  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    PointerHead,
    StrandsDeciderConfig,
    StrandsDeciderModel,
    ensure_bos,
)
from strands_decider.prompting import fit_window, render_question, render_state  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

POLICY = (
    "Loans run 21 days and renew twice online unless another patron holds the item. "
    "Accounts with fines above ten dollars cannot renew until the fine is paid. "
)
QUESTIONS = {
    "team": ChoiceQuestion(
        instructions="Which team should handle this?",
        criteria={
            "circulation": "renewals and holds",
            "billing": "fines",
            "reference": "research help",
        },
    ),
    "urgent": NoulQuestion(instructions="Does the patron convey urgency?"),
    "mood": ScoreQuestion(
        instructions="How frustrated is the patron?", criteria=["calm", "annoyed", "angry"]
    ),
}
# Over 384 tokens, the most of a 512-token window a question may claim: trimmed from the
# front, it loses its first options, and both engines refuse it.
LONG_CHOICE = ChoiceQuestion(
    instructions="Which shelf holds the book?",
    criteria={f"shelf {i}": "fiction and poetry, rows A to F" for i in range(60)},
)
# Over 384 tokens of instructions and two options: trimmed from the front to the
# question's reserve, it keeps its options and is answered.
LONG_NOUL = NoulQuestion(
    instructions="Read the whole policy before you answer. " * 60 + "Is it renewable?"
)
REQUESTS = {
    "one question": SystemOneRequest(
        state=POLICY + "Can I renew the cookbook?", questions={"team": QUESTIONS["team"]}
    ),
    "three questions": SystemOneRequest(
        state=POLICY * 6 + "My renewal failed and I leave Friday!", questions=QUESTIONS
    ),
    # About 1,100 tokens against a 512-token window: both engines trim the state.
    "trimmed state": SystemOneRequest(state=POLICY * 30 + "Help!", questions=QUESTIONS),
    "trimmed question": SystemOneRequest(state=POLICY, questions={"renewable": LONG_NOUL}),
}


def _tokenizer(bos: bool):
    """A word-level tokenizer covering every rendered prompt, optionally adding a BOS."""
    from tokenizers import models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    texts = [render_state(r.state) for r in REQUESTS.values()]
    texts += [render_question(q).text for q in [*QUESTIONS.values(), LONG_CHOICE, LONG_NOUL]]
    words = {w for text in texts for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(text)}
    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", "<bos>", *sorted(words)])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        bos_token="<bos>" if bos else None,
    )
    return ensure_bos(hf) if bos else hf


def _checkpoint(tmp_path, bos: bool):
    tok = _tokenizer(bos)
    torch.manual_seed(0)
    base_cfg = transformers.Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    base = tmp_path / "base"
    transformers.Qwen3_5ForCausalLM(base_cfg).save_pretrained(base)
    cfg = StrandsDeciderConfig(
        base_model=str(base),
        head_type="pointer",
        pointer_dim=16,
        max_length=512,
        torch_dtype="float32",
        lora_r=4,
        lora_alpha=8,
        temperature=0.8,
        temperature_by_kind={"choice": 0.6, "score": 1.5},
        ordinal_smoothing=0.1,
        force_bos=bos,
    )
    model = StrandsDeciderModel(
        cfg, transformers.Qwen3_5ForCausalLM.from_pretrained(base).model, tok
    )
    model.attach_lora()
    with torch.no_grad():  # lora_B starts at zero: make the adapter change the torso
        for name, p in model.torso.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.1)
    path = tmp_path / "ckpt"
    model.save_pretrained(str(path))
    return path


class TorsoBackend:
    """The torch engine's torso as a Backend: one full forward of state + question each."""

    def __init__(self, model: StrandsDeciderModel):
        self.model = model
        self.calls: list[tuple[list[int], list[list[int]], list[list[int]]]] = []

    @torch.inference_mode()
    def forward(self, state, questions, rows):
        self.calls.append((state, questions, rows))
        out = []
        for q, r in zip(questions, rows, strict=True):
            ids = torch.tensor([state + q])
            hidden = self.model.encode(ids, torch.ones_like(ids))[0, len(state) :]
            out.append(hidden[r].float().numpy())
        return out


def _hidden_engine(torch_engine: SystemOneEngine, assets, **config) -> HiddenStateEngine:
    return HiddenStateEngine.from_assets(
        str(assets),
        TorsoBackend(torch_engine.model),
        model_name=torch_engine.cfg.model_name,
        **config,
    )


def _probabilities(response):
    return {
        name: [a.noul] if a.type == "noul" else list(a.probabilities.values())
        for name, a in response.answers.items()
    }


@pytest.fixture(scope="module", params=[False, True], ids=["no-bos", "bos"])
def engines(request, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bos" if request.param else "nobos")
    torch_engine = load_engine(str(_checkpoint(tmp, request.param)), device="cpu")
    assets = tmp / "assets"
    save_assets(torch_engine.model, str(assets), "tiny")
    # Saved tokenizers can carry padding and truncation (Qwen3.5's pads a batch to its
    # longest): the engine must count every token of every question regardless.
    tok = Tokenizer.from_file(str(assets / "tokenizer.json"))
    tok.enable_padding(pad_id=0, pad_token="<pad>")
    tok.enable_truncation(max_length=8)
    tok.save(str(assets / "tokenizer.json"))
    return torch_engine, assets


@pytest.mark.parametrize("eps", [1e-5, 1e-2])
def test_numpy_head_matches_pointer_head(eps):
    torch.manual_seed(1)
    head = PointerHead(hidden_size=32, dim=8).eval()
    head.norm.eps = eps
    with torch.no_grad():
        head.norm.weight.normal_()
        head.norm.bias.normal_()
        decide, options = torch.randn(32) * 0.01, torch.randn(5, 32) * 0.01
        want = head(decide[None], options[None])[0].numpy()
    got = NumpyPointerHead({k: v.numpy() for k, v in head.state_dict().items()}, 8, eps)
    np.testing.assert_allclose(
        got.logits(decide.numpy(), options.numpy()), want, rtol=1e-5, atol=1e-5
    )


def test_assets_hold_the_head_and_settings(engines):
    torch_engine, assets = engines
    cfg = torch_engine.model.config
    saved = json.loads((assets / "head_config.json").read_text())
    assert saved == {
        "format": 1,
        "readout": "pointer",
        "model_name": "tiny",
        "pointer_dim": cfg.pointer_dim,
        "eps": torch_engine.model.head.norm.eps,
        "max_length": cfg.max_length,
        "temperature": cfg.temperature,
        "temperature_by_kind": cfg.temperature_by_kind,
        "ordinal_smoothing": cfg.ordinal_smoothing,
        "truncation_side": "right",
    }
    assert _hidden_engine(torch_engine, assets).cfg.model_name == torch_engine.cfg.model_name


@pytest.mark.parametrize("name", list(REQUESTS))
def test_answers_as_torch_does(engines, name):
    torch_engine, assets = engines
    expected = torch_engine.evaluate(REQUESTS[name])
    got = _hidden_engine(torch_engine, assets).evaluate(REQUESTS[name].model_dump())
    assert (got.model, got.usage) == (expected.model, expected.usage)
    want, have = _probabilities(expected), _probabilities(got)
    assert have.keys() == want.keys()
    for q in want:
        assert have[q] == pytest.approx(want[q], abs=1.5e-4), q  # both round to 4 places
    for q, answer in expected.answers.items():
        if answer.type == "choice":
            assert got.answers[q].choice == answer.choice


@pytest.mark.parametrize("name", ["trimmed state", "trimmed question"])
def test_fit_matches_system_one_engine(engines, name):
    torch_engine, assets = engines
    state = render_state(REQUESTS[name].state)
    texts = [render_question(q).text for q in REQUESTS[name].questions.values()]
    s, q, offs = _hidden_engine(torch_engine, assets).fit(state, texts)
    want_s, want_q, want_offs = torch_engine._fit(state, texts)
    assert (s, q) == (want_s, want_q)
    assert offs == [[tuple(o) for o in x] for x in want_offs]
    assert len(s) + max(len(x) for x in q) <= torch_engine.model.config.max_length
    if torch_engine.tok.bos_token_id is not None:
        assert s[0] == torch_engine.tok.bos_token_id


def test_backend_is_asked_for_the_option_ends_then_the_last_token(engines):
    torch_engine, assets = engines
    engine = _hidden_engine(torch_engine, assets)
    engine.evaluate(REQUESTS["three questions"])
    ((_, questions, rows),) = engine.backend.calls
    for q, r, question in zip(questions, rows, QUESTIONS.values(), strict=True):
        assert len(r) == render_question(question).n_slots + 1
        assert r[-1] == len(q) - 1 and r[:-1] == sorted(r[:-1])


def test_strict_window_refuses_as_torch_does(engines):
    torch_engine, assets = engines
    strict_torch = SystemOneEngine(
        torch_engine.model, EngineConfig(device="cpu", strict_window=True)
    )
    strict = _hidden_engine(torch_engine, assets, strict_window=True)
    for engine in (strict_torch, strict):
        with pytest.raises(ValueError, match="exceeds the context window of 512 tokens"):
            engine.evaluate(REQUESTS["trimmed state"])
    assert strict.evaluate(REQUESTS["one question"]).answers.keys() == {"team"}


def test_question_trimmed_into_its_options_is_refused_before_the_forward(engines):
    torch_engine, assets = engines
    request = SystemOneRequest(state="Where is it?", questions={"shelf": LONG_CHOICE})
    engine = _hidden_engine(torch_engine, assets)
    for e in (torch_engine, engine):
        with pytest.raises(ValueError, match="truncated through its option list"):
            e.evaluate(request)
    assert not engine.backend.calls


def test_request_limit_and_images_are_refused(engines):
    torch_engine, assets = engines
    engine = _hidden_engine(torch_engine, assets, max_request_tokens=100)
    with pytest.raises(ValueError, match="exceeds the limit of 100 tokens"):
        engine.evaluate(REQUESTS["three questions"])
    with pytest.raises(ValueError, match="images"):
        engine.evaluate(REQUESTS["one question"].model_copy(update={"images": ["aGk="]}))
    assert not engine.backend.calls


def test_concurrent_requests_answer_as_sequential_ones(engines):
    torch_engine, assets = engines
    engine = _hidden_engine(torch_engine, assets)
    requests = list(REQUESTS.values()) * 3
    sequential = [engine.evaluate(r).answers for r in requests]
    with ThreadPoolExecutor(6) as pool:
        assert list(pool.map(lambda r: engine.evaluate(r).answers, requests)) == sequential


class _WrongRows:
    """A backend that returns the wrong rows: every position of each question, one row
    too few, or one array too few."""

    def __init__(self, inner: TorsoBackend, mistake: str):
        self.inner, self.mistake = inner, mistake

    def forward(self, state, questions, rows):
        if self.mistake == "every position":
            rows = [list(range(len(q))) for q in questions]
        out = self.inner.forward(state, questions, rows)
        if self.mistake == "a row too few":
            return [h[:-1] for h in out]
        if self.mistake == "an array too few":
            return out[:-1]
        return out


@pytest.mark.parametrize(
    ("mistake", "message"),
    [
        ("every position", r"backend returned shape \(\d+, 64\) for question 'team'; expected"),
        (
            "a row too few",
            r"backend returned shape \(3, 64\) for question 'team'; expected \(4, hidden\)",
        ),
        ("an array too few", "backend returned 2 arrays for 3 questions"),
    ],
)
def test_a_backend_returning_the_wrong_rows_is_named(engines, mistake, message):
    torch_engine, assets = engines
    engine = HiddenStateEngine.from_assets(
        str(assets), _WrongRows(TorsoBackend(torch_engine.model), mistake)
    )
    with pytest.raises(ValueError, match=message):
        engine.evaluate(REQUESTS["three questions"])


def _resave(assets, tmp_path, **changes):
    """A copy of the assets with head_config.json's keys changed (None removes one)."""
    import shutil

    copy = tmp_path / "assets"
    shutil.copytree(assets, copy)
    config = json.loads((copy / "head_config.json").read_text())
    config.update(changes)
    config = {k: v for k, v in config.items() if v is not None}
    (copy / "head_config.json").write_text(json.dumps(config))
    return copy


def test_assets_from_a_newer_release_load_with_a_warning(engines, tmp_path):
    torch_engine, assets = engines
    newer = _resave(assets, tmp_path, format=2, a_later_setting=True)
    with pytest.warns(UserWarning, match="newer strands-decider"):
        engine = _hidden_engine(torch_engine, newer)
    expected = _hidden_engine(torch_engine, assets).evaluate(REQUESTS["one question"])
    assert engine.evaluate(REQUESTS["one question"]) == expected


def test_assets_from_before_the_format_key_load(engines, tmp_path):
    torch_engine, assets = engines
    older = _resave(assets, tmp_path, format=None, readout=None, eps=None, truncation_side=None)
    expected = _hidden_engine(torch_engine, assets).evaluate(REQUESTS["one question"])
    assert _hidden_engine(torch_engine, older).evaluate(REQUESTS["one question"]) == expected


def test_assets_for_another_readout_are_refused(engines, tmp_path):
    torch_engine, assets = engines
    with pytest.raises(ValueError, match="not for a pointer head"):
        _hidden_engine(torch_engine, _resave(assets, tmp_path, readout="slot"))


@pytest.mark.parametrize("side", ["right", "left"])
def test_a_long_state_is_trimmed_on_the_tokenizers_side(engines, tmp_path, monkeypatch, side):
    """fit_window trims the state as the Hugging Face tokenizer does with its
    truncation_side, and save_assets carries that side to the hidden-state engine."""
    torch_engine, _ = engines
    monkeypatch.setattr(torch_engine.tok, "truncation_side", side)
    state = render_state(REQUESTS["trimmed state"].state)
    question = render_question(QUESTIONS["urgent"]).text
    reserve = len(torch_engine.tok(question, add_special_tokens=False)["input_ids"])
    want = torch_engine.tok(
        state, add_special_tokens=True, truncation=True, max_length=512 - reserve
    )["input_ids"]
    s, _, _ = fit_window(
        torch_engine.tok.backend_tokenizer, state, [question], 512, truncation_side=side
    )
    assert s == want
    assert torch_engine._fit(state, [question])[0] == want
    save_assets(torch_engine.model, str(tmp_path / "assets"), "tiny")
    assert _hidden_engine(torch_engine, tmp_path / "assets").fit(state, [question])[0] == want


def test_the_head_uses_the_saved_eps(engines, tmp_path):
    torch_engine, assets = engines
    assert _hidden_engine(torch_engine, _resave(assets, tmp_path, eps=0.01)).head.eps == 0.01
