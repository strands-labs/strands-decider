"""End-to-end checks that need real weights and a GPU.

Skipped automatically without CUDA. These are slow (they download and run a 0.6B
model, Qwen3-0.6B-Base unless STRANDS_DECIDER_TEST_BASE names another) but they cover the two
things unit tests cannot: that the torso actually
loads headless, and that the shared-prefix cache is numerically equivalent to the
naive path. The second one matters -- a subtly wrong cache would not crash, it
would just silently answer differently under load.
"""

from __future__ import annotations

import os

import pytest
import torch

from strands_decider.data.format import Example

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

SMALL_BASE = os.environ.get("STRANDS_DECIDER_TEST_BASE", "Qwen/Qwen3-0.6B-Base")


@pytest.fixture(scope="module")
def model():
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    cfg = StrandsDeciderConfig(base_model=SMALL_BASE, num_slots=8, max_length=512, use_lora=False)
    m = StrandsDeciderModel.from_pretrained_base(cfg)
    return m.to("cuda").eval()


def test_torso_loads_without_lm_head(model):
    assert not hasattr(model.torso, "lm_head")
    assert model.head.proj.out_features == 8


def test_forward_produces_masked_distribution(model):
    from strands_decider.data.collate import CollatorConfig, SystemOneCollator

    ex = Example(
        kind="choice",
        state="Help! My payouts have been failing for 3 days.",
        instructions="Route this ticket.",
        options=[["billing", "money"], ["technical", "bugs"], ["sales", "pricing"]],
        label=0,
        task="smoke",
    )
    coll = SystemOneCollator(
        model.tokenizer, CollatorConfig(num_slots=8, max_length=512), train=False
    )
    batch = {k: v.to("cuda") for k, v in coll([ex]).items()}
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            n_slots=batch["n_slots"],
        )
    probs = out["log_probs"].exp()
    assert probs[0, :3].sum().item() == pytest.approx(1.0, abs=1e-4)
    assert probs[0, 3:].sum().item() == pytest.approx(0.0, abs=1e-6)


@pytest.fixture(scope="module")
def fp32_model():
    """An fp32 copy, so cache equivalence can be asserted exactly.

    In bf16 the two paths differ by ~2e-3 purely from accumulation order (the
    batched path multiplies different tensor shapes). fp32 removes that, which
    turns a fuzzy comparison into a real correctness check.
    """
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    cfg = StrandsDeciderConfig(
        base_model=SMALL_BASE, num_slots=8, max_length=512,
        use_lora=False, torch_dtype="float32",
    )
    return StrandsDeciderModel.from_pretrained_base(cfg).to("cuda").eval()


def _routing_request():
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

    return SystemOneRequest(
        state=(
            "Help! My payouts have been failing for 3 days and support has not "
            "replied to either of my emails. I am close to switching providers."
        ),
        questions={
            "is_urgent": NoulQuestion(instructions="Does this convey urgency?"),
            "department": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={
                    "billing": "payments, invoices, payouts",
                    "technical": "bugs, outages, API errors",
                    "sales": "pricing and upgrades",
                },
            ),
            "frustration": ScoreQuestion(
                instructions="How frustrated is the writer?",
                criteria=["Calm", "Frustrated", "Very angry"],
            ),
        },
    )


def _max_answer_delta(a, b):
    deltas = [abs(a.answers["is_urgent"].noul - b.answers["is_urgent"].noul)]
    for opt, p in a.answers["department"].probabilities.items():
        deltas.append(abs(p - b.answers["department"].probabilities[opt]))
    deltas.append(abs(a.answers["frustration"].score - b.answers["frustration"].score))
    return max(deltas)


def test_prefix_cache_is_exact_in_fp32(fp32_model):
    """The shared-state cache must be an optimisation, not a behaviour change.

    Asserted in fp32 where it should hold to rounding, not approximately.
    """
    from strands_decider.infer import EngineConfig, SystemOneEngine

    request = _routing_request()
    cached = SystemOneEngine(fp32_model, EngineConfig(use_prefix_cache=True)).evaluate(request)
    naive = SystemOneEngine(fp32_model, EngineConfig(use_prefix_cache=False)).evaluate(request)
    assert _max_answer_delta(cached, naive) < 1e-4


def test_prefix_cache_close_enough_in_bf16(model):
    """bf16 is the serving dtype; the residual gap must stay far below any
    threshold a caller would route on."""
    from strands_decider.infer import EngineConfig, SystemOneEngine

    request = _routing_request()
    cached = SystemOneEngine(model, EngineConfig(use_prefix_cache=True)).evaluate(request)
    naive = SystemOneEngine(model, EngineConfig(use_prefix_cache=False)).evaluate(request)
    assert _max_answer_delta(cached, naive) < 1e-2


def test_prefix_cache_saves_tokens(model):
    """The whole point: N questions should not re-encode the state N times."""
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.schema import NoulQuestion, SystemOneRequest

    state = "A fairly long support ticket. " * 60
    questions = {
        f"q{i}": NoulQuestion(instructions=f"Is claim {i} supported by the state?")
        for i in range(6)
    }
    request = SystemOneRequest(state=state, questions=questions)

    cached = SystemOneEngine(model, EngineConfig(use_prefix_cache=True)).evaluate(request)
    naive = SystemOneEngine(model, EngineConfig(use_prefix_cache=False)).evaluate(request)

    assert cached.usage.input_tokens < naive.usage.input_tokens / 3


def test_answers_respect_api_shape(model):
    from strands_decider.infer import SystemOneEngine
    from strands_decider.schema import ScoreQuestion

    engine = SystemOneEngine(model)
    resp = engine.ask(
        "The food was fine but the wait was long.",
        {
            "mood": ScoreQuestion(
                instructions="How positive is this review?",
                criteria=["negative", "mixed", "positive"],
            )
        },
    )
    ans = resp.answers["mood"]
    assert ans.type == "score"
    assert 0.0 <= ans.score <= 2.0
    assert ans.legend == {"0": "negative", "1": "mixed", "2": "positive"}
    assert set(ans.probabilities) == {"0", "1", "2"}
    assert 0.0 <= ans.confidence <= 1.0


def test_lora_checkpoint_save_load_roundtrip(tmp_path):
    """Saving and reloading a LoRA checkpoint must reproduce identical answers.

    StrandsDeciderModel.load bypasses __init__ (to avoid stacking a second adapter on top
    of the one being restored) and reattaches the head by hand. That is exactly the
    kind of path that can silently load a randomly-initialised head, so compare
    actual outputs rather than just asserting the files exist.
    """
    import torch

    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    cfg = StrandsDeciderConfig(
        base_model=SMALL_BASE, num_slots=8, max_length=512,
        use_lora=True, lora_r=4, torch_dtype="float32",
    )
    model = StrandsDeciderModel.from_pretrained_base(cfg)

    # Perturb the head so a failure to restore it cannot pass by coincidence.
    with torch.no_grad():
        for p in model.head.parameters():
            p.add_(torch.randn_like(p) * 0.3)

    ckpt = str(tmp_path / "ckpt")
    model.save_pretrained(ckpt)

    request = _routing_request()
    before = SystemOneEngine(model.to("cuda").eval(), EngineConfig()).evaluate(request)

    reloaded = StrandsDeciderModel.load(ckpt)
    after = SystemOneEngine(reloaded.to("cuda").eval(), EngineConfig()).evaluate(request)

    assert _max_answer_delta(before, after) < 1e-4
    assert reloaded.config.num_slots == 8
    assert reloaded.config.lora_r == 4


def test_temperature_is_applied_on_load(tmp_path):
    """A fitted temperature must survive the checkpoint and actually soften output."""
    import json
    import os

    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
    from strands_decider.schema import ChoiceQuestion, SystemOneRequest

    cfg = StrandsDeciderConfig(
        base_model=SMALL_BASE, num_slots=8, max_length=512,
        use_lora=False, torch_dtype="float32",
    )
    model = StrandsDeciderModel.from_pretrained_base(cfg)
    ckpt = str(tmp_path / "ckpt")
    model.save_pretrained(ckpt)

    request = SystemOneRequest(
        state="The delivery was three days late and nobody answered the phone.",
        questions={
            "dept": ChoiceQuestion(
                instructions="Which team?",
                criteria={"logistics": "shipping", "support": "contact", "billing": "money"},
            )
        },
    )

    sharp = SystemOneEngine(
        StrandsDeciderModel.load(ckpt).to("cuda").eval(), EngineConfig()
    ).evaluate(request)

    # Raise the temperature: the distribution must flatten, so confidence drops.
    path = os.path.join(ckpt, "hobson_config.json")
    with open(path, encoding="utf-8") as fh:
        stored = json.load(fh)
    stored["temperature"] = 5.0
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stored, fh)

    soft_model = StrandsDeciderModel.load(ckpt)
    assert soft_model.config.temperature == 5.0
    soft = SystemOneEngine(soft_model.to("cuda").eval(), EngineConfig()).evaluate(request)

    assert soft.answers["dept"].confidence < sharp.answers["dept"].confidence


def _chunking_request():
    from strands_decider.schema import ChoiceQuestion, SystemOneRequest

    # Distinct option counts per question, so a mispairing shows up as a probability
    # dict with the wrong *keys* rather than merely different numbers.
    questions = {
        f"q{i}": ChoiceQuestion(
            instructions=f"Question {i}?",
            criteria={f"opt{i}_{j}": "" for j in range(2 + (i % 5))},
        )
        for i in range(11)
    }
    return questions, SystemOneRequest(state="Some state to evaluate.", questions=questions)


def _chunk_pair(model, max_batch_a=3, max_batch_b=64):
    from strands_decider.infer import EngineConfig, SystemOneEngine

    questions, request = _chunking_request()
    a = SystemOneEngine(model, EngineConfig(max_batch=max_batch_a)).evaluate(request)
    b = SystemOneEngine(model, EngineConfig(max_batch=max_batch_b)).evaluate(request)
    return questions, a, b


def test_chunking_pairs_answers_with_the_right_questions(model):
    """More questions than max_batch must still map each answer to its own question.

    The chunking loop slices three parallel lists; an off-by-one would pair a
    question's name with another question's distribution -- silent and nasty. Checked
    by option-set identity, which is independent of float precision.
    """
    questions, chunked, single = _chunk_pair(model)
    assert set(chunked.answers) == set(questions)
    for name, q in questions.items():
        assert set(chunked.answers[name].probabilities) == set(q.criteria), name
        assert set(single.answers[name].probabilities) == set(q.criteria), name


def test_chunking_is_exact_in_fp32(fp32_model):
    """Batch size must not change the answer. Asserted where it should hold to rounding.

    In bf16 the same comparison drifts ~3e-3 purely because a different max_batch
    yields a different padding width and so a different accumulation order.
    """
    questions, chunked, single = _chunk_pair(fp32_model)
    for name in questions:
        assert chunked.answers[name].probabilities == pytest.approx(
            single.answers[name].probabilities, abs=1e-4
        )


def test_chunking_close_enough_in_bf16(model):
    """bf16 is the serving dtype; drift must stay far below any routing threshold."""
    questions, chunked, single = _chunk_pair(model)
    for name in questions:
        assert chunked.answers[name].probabilities == pytest.approx(
            single.answers[name].probabilities, abs=1e-2
        )


def _long_state(n_chars: int = 12000) -> str:
    """A policy-document-sized state, like JevBench's long_policy family."""
    clause = ("Clause {i}: refunds require a receipt issued within 30 days, unless the "
              "item was purchased on promotion, in which case store credit applies. ")
    out = []
    i = 0
    while sum(len(x) for x in out) < n_chars:
        out.append(clause.format(i=i))
        i += 1
    return "".join(out)


def test_long_state_does_not_starve_the_question(model):
    """The regression this guards: a state longer than the window used to consume all
    of it, leaving the question a floor of 8 tokens -- too few to hold the option list,
    so the model chose among options it could not see. Every JevBench long_policy task
    hit that path."""
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.prompting import render_question, render_state
    from strands_decider.schema import ChoiceQuestion

    eng = SystemOneEngine(model, EngineConfig())
    q = ChoiceQuestion(
        instructions="Under the stated policy, what is the correct outcome?",
        criteria={
            "full_refund": "money back in full",
            "store_credit": "credit only, no cash",
            "reject": "no remedy available",
            "partial": "refund minus a restocking fee",
        },
    )
    rq = render_question(q)
    wanted = len(eng.tok(rq.text, add_special_tokens=False)["input_ids"])

    s, kept, _ = eng._fit(render_state(_long_state()), [rq.text])

    assert len(kept[0]) == wanted, "the whole question should survive a long state"
    assert len(s) > 0, "the state must not be squeezed to nothing"
    assert len(s) + len(kept[0]) <= model.config.max_length

    # The option names are what the slots bind to; losing them makes the task blind.
    decoded = eng.tok.decode(kept[0])
    for name in q.criteria:
        assert name in decoded


def test_question_reserve_is_capped_so_state_survives(model):
    """A pathological question must not take the entire window.

    Both inputs are oversized here on purpose. With a short state the cap is
    untestable, because the state simply does not fill the budget it is given.
    """
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.prompting import render_state

    eng = SystemOneEngine(model, EngineConfig(max_question_fraction=0.75))
    huge = "Is this permitted under the policy? " * 2000
    s, kept, _ = eng._fit(render_state(_long_state()), [huge])

    cap = int(model.config.max_length * 0.75)
    assert len(kept[0]) == cap, "an oversized question should be trimmed to the cap"
    # The state gets everything the cap leaves, and it is long enough to use it.
    assert len(s) == model.config.max_length - cap
    assert len(s) + len(kept[0]) <= model.config.max_length


def test_strict_window_refuses_instead_of_truncating(model):
    """--strict-window: a prompt over the window is refused, one that fits is kept whole."""
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.prompting import render_state

    eng = SystemOneEngine(model, EngineConfig(strict_window=True))
    with pytest.raises(ValueError, match="context window"):
        eng._fit(render_state(_long_state()), ["Is this permitted?"])

    short = render_state("A short state.")
    question = "Is this permitted under the policy? " * 10
    s, kept, _ = eng._fit(short, [question])
    assert len(kept[0]) == len(eng.tok(question, add_special_tokens=False)["input_ids"])
    assert s == eng.tok(short, add_special_tokens=True)["input_ids"]


def test_long_state_answer_is_still_well_formed(model):
    """End to end: a long state must still yield a valid distribution over the options."""
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.schema import ChoiceQuestion, SystemOneRequest

    eng = SystemOneEngine(model, EngineConfig())
    req = SystemOneRequest(
        state=_long_state(),
        questions={
            "outcome": ChoiceQuestion(
                instructions="What is the correct outcome?",
                criteria={"refund": "money back", "credit": "store credit", "reject": "none"},
            )
        },
    )
    ans = eng.evaluate(req).answers["outcome"]
    assert set(ans.probabilities) == {"refund", "credit", "reject"}
    assert sum(ans.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_both_paths_agree_on_a_long_state(model):
    """_fit changed tokenisation on both paths; they must still match."""
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.schema import ChoiceQuestion, SystemOneRequest

    req = SystemOneRequest(
        state=_long_state(),
        questions={
            "outcome": ChoiceQuestion(
                instructions="What is the correct outcome?",
                criteria={"refund": "money back", "credit": "store credit", "reject": "none"},
            )
        },
    )
    cached = SystemOneEngine(model, EngineConfig(use_prefix_cache=True)).evaluate(req)
    naive = SystemOneEngine(model, EngineConfig(use_prefix_cache=False)).evaluate(req)
    for k, v in cached.answers["outcome"].probabilities.items():
        assert v == pytest.approx(naive.answers["outcome"].probabilities[k], abs=1e-2)
