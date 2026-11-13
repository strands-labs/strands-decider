"""v1's pointer path computes what main computed before the one trainer: the numbered renders, the collator's
training batches, the model's losses on them, and the engine's answers on both serving paths, pinned to values
recorded on main at 87d0fd5. The pointer head of training/train.py renders and batches through these same pieces
(test_pointer.py checks that it does).

Carried over from #124's test_token_readout.py, which pinned the pointer path the same way.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from tiny_support import _examples, _tiny_model_factory, _tokenizer_file

from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderConfig
from strands_decider.prompting import build_prompt
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from training.data.collate import CollatorConfig, SystemOneCollator
from training.data.format import load_examples, write_jsonl

QUESTIONS = {
    "team": ChoiceQuestion(
        instructions="Which team?", criteria={"billing": "money", "tech": None, "sales": "deals"}
    ),
    "urgent": NoulQuestion(instructions="It is urgent."),
    "mood": ScoreQuestion(instructions="How angry?", criteria=["calm", "annoyed", "furious"]),
}
# The trace on main at 87d0fd5.
MAIN_RENDERS = "bb75baf142839488b12ab5331f4d7b7c804d48d90c3e574cf71a8a9afc39960f"
MAIN_BATCHES = "7d2ad3f1c9b937565d247346e57ba050e9858f26b43e9aedd32a2d06bf94d44d"
MAIN_LOSSES = [1.2146382, 1.0987360, 1.3282480, 1.2259021, 1.1267848, 1.0307044]
MAIN_PROBS = [0.2482, 0.3714, 0.3804, 0.5091, 0.4687, 0.2211, 0.3102] * 2  # cached, batched


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("pointer_main")
    rows = _examples(150, seed=1)
    write_jsonl(str(d / "a.jsonl"), rows)
    _tokenizer_file(str(d / "tokenizer.json"), rows)
    return d


def _pointer_trace(corpus):
    """(sha256 of numbered renders, sha256 of six training batches, their losses, the engine's probabilities on both
    serving paths)."""
    renders = hashlib.sha256()
    for ex in _examples(150, seed=1):
        for order in (None, list(reversed(range(ex.n_options)))):
            renders.update(build_prompt(ex.state, ex.to_question(), option_order=order)[0].encode())
    rows = load_examples([str(corpus / "a.jsonl")])
    torch.manual_seed(0)
    model = _tiny_model_factory(str(corpus / "tokenizer.json"))(
        StrandsDeciderConfig(base_model="tiny", pointer_dim=16)
    )
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(max_length=2048), train=True)
    batches, losses = hashlib.sha256(), []
    for i in range(0, 48, 8):
        b = coll(rows[i : i + 8])
        for k in sorted(b):
            batches.update(f"{k}{b[k].dtype}{tuple(b[k].shape)}".encode())
            batches.update(b[k].contiguous().numpy().tobytes())
        with torch.no_grad():
            out = model(**{k: b[k] for k in b if k not in ("teacher", "has_teacher")})
        losses.append(float(out["loss"]))
    engine = SystemOneEngine(model, EngineConfig(device="cpu"))
    probs = []
    for prefix_cache in (True, False):
        engine.cfg.use_prefix_cache = prefix_cache
        got = engine.evaluate(SystemOneRequest(state="Payouts failing.", questions=QUESTIONS))
        for a in got.answers.values():
            probs += [a.noul] if a.type == "noul" else list(a.probabilities.values())
    return renders.hexdigest(), batches.hexdigest(), losses, probs


def test_the_pointer_path_is_what_main_computed(corpus):
    renders, batches, losses, probs = _pointer_trace(corpus)
    assert (renders, batches) == (MAIN_RENDERS, MAIN_BATCHES)
    assert losses == pytest.approx(MAIN_LOSSES, rel=1e-5)
    assert probs == pytest.approx(MAIN_PROBS, abs=2e-4)  # answers are rounded to 4 places
