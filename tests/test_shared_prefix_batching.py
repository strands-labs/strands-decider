"""Cross-request shared-prefix batching: multi-question requests share one prefix forward
and one suffix forward across requests of equal state length.

The forward's correctness condition is exactly the one `_slot_probs_shared_prefix` pins:
state cache is forked (not shared) per row and suffixes are right-padded with their
attention masks. These tests pin that the cross-request path matches the per-request
shared-prefix path on tiny random torsos of all three families; that different state
lengths bucket separately; that an unforkable cache falls back cleanly; and that one bad
request does not poison its neighbours.
CPU only.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from strands_decider.infer import (  # noqa: E402
    EngineConfig,
    SystemOneEngine,
    UnforkableCache,
    plan_prefix_groups,
)
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)


def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["<state> <question> <options> <answer> choice score noul"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<pad>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")


def _torso(kind, vocab):
    torch.manual_seed(0)
    if kind == "qwen3":
        return transformers.Qwen3Model(
            transformers.Qwen3Config(
                vocab_size=vocab,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
            )
        )
    if kind == "qwen3_5":
        if not hasattr(transformers, "Qwen3_5TextConfig"):
            pytest.skip("this transformers has no Qwen3.5")
        cfg = transformers.Qwen3_5TextConfig(
            vocab_size=vocab,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ],
        )
        return transformers.Qwen3_5ForCausalLM(cfg).model.eval()
    if kind == "gemma4":
        if not hasattr(transformers, "Gemma4TextConfig"):
            pytest.skip("this transformers has no Gemma 4")
        cfg = transformers.Gemma4TextConfig(
            vocab_size=vocab,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            global_head_dim=16,
            hidden_size_per_layer_input=8,
            vocab_size_per_layer_input=vocab,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
            sliding_window=8,
            num_kv_shared_layers=1,
            final_logit_softcapping=30.0,
            tie_word_embeddings=True,
            pad_token_id=0,
        )
        torso = transformers.Gemma4ForCausalLM(cfg).model.eval()
        torso.set_attn_implementation("eager")
        return torso
    raise ValueError(kind)


def _model(kind="qwen3"):
    tok = _tokenizer()
    cfg = StrandsDeciderConfig(
        base_model="tiny",
        head_type="pointer",
        pointer_dim=16,
        use_lora=False,
        torch_dtype="float32",
        max_length=512,
    )
    return StrandsDeciderModel(cfg, _torso(kind, len(tok)), tok)


def _close(a, b, tol=2e-4):
    assert a.keys() == b.keys()
    for k in a:
        assert abs(a[k] - b[k]) <= tol, (k, a[k], b[k])


def _engine(kind="qwen3", **kw):
    """Strict-window engine, so prepare() fills PreparedPrefix for multi-question requests."""
    return SystemOneEngine(
        _model(kind),
        EngineConfig(device="cpu", use_prefix_cache=True, **kw),
    )


# ---- the forward itself matches the per-request shared-prefix path ---------------


@pytest.mark.parametrize("kind", ["qwen3", "qwen3_5", "gemma4"])
def test_cross_request_matches_isolated_shared_prefix(kind):
    """Several multi-Q requests of equal state length, run together through
    evaluate_prepared, give answers equal to each run alone through evaluate()
    (which takes the per-request shared-prefix path for multi-Q)."""
    eng = _engine(kind)
    # Build four requests whose states tokenise to the same length (same text, varied
    # questions).
    text = "event logged; " * 5
    reqs = []
    for i in range(4):
        reqs.append(
            SystemOneRequest(
                state=text,
                questions={
                    "a": NoulQuestion(type="noul", instructions="is it ok?"),
                    "b": ChoiceQuestion(
                        type="choice",
                        instructions="pick one",
                        criteria={f"o{k}": "" for k in range(2 + i % 3)},
                    ),
                    "c": ScoreQuestion(
                        type="score", instructions="rate", criteria=["lo", "mid", "hi"]
                    ),
                },
            )
        )
    isolated = [eng.evaluate(r) for r in reqs]
    prepared = [eng.prepare(r) for r in reqs]
    assert all(p.prefix is not None for p in prepared), "multi-Q should fill PreparedPrefix"
    together = eng.evaluate_prepared(prepared)
    assert len(together) == len(isolated)
    for iso, tog in zip(isolated, together, strict=True):
        assert not isinstance(tog, BaseException), tog
        for name in iso.answers:
            _close(_dump(iso.answers[name]), _dump(tog.answers[name]))


def _dump(ans):
    d = ans.model_dump()
    return d.get("probabilities") or {"noul": d["noul"]}


# ---- different state lengths bucket separately -----------------------------------


def test_different_state_lengths_bucket_separately():
    """plan_prefix_groups keeps requests of different state length in different groups."""
    groups = plan_prefix_groups(
        state_lengths=[10, 10, 20, 10, 20],
        question_counts=[2, 2, 2, 2, 2],
        max_tokens=10_000,
        max_rows=100,
    )
    # Every group's lengths must agree.
    for g in groups:
        lens = {[10, 10, 20, 10, 20][i] for i in g}
        assert len(lens) == 1, g


def test_plan_prefix_groups_respects_row_cap():
    """A row cap of 5 cuts a sequence of (3, 3, 1) question counts into (3) and (3, 1)."""
    groups = plan_prefix_groups([8, 8, 8], [3, 3, 1], max_tokens=10_000, max_rows=5)
    assert groups == [[0], [1, 2]]


def test_plan_prefix_groups_respects_prefix_token_cap():
    """A prefix-token cap splits a bucket whose padded prefix tokens exceed the cap."""
    # Three requests of state_len 100, cap 250 -> at most 2 prefixes per group.
    groups = plan_prefix_groups([100, 100, 100], [1, 1, 1], max_tokens=250, max_rows=10)
    assert groups == [[0, 1], [2]]


# ---- mixed cross-request + row path + fallback-alone -----------------------------


def test_mixed_prepared_requests():
    """A batch of (prefix-path, row-path, alone-path) requests all succeed."""
    eng = _engine("qwen3")
    multi = SystemOneRequest(
        state="one two three",
        questions={
            "a": NoulQuestion(type="noul", instructions="ok?"),
            "b": NoulQuestion(type="noul", instructions="urgent?"),
        },
    )
    single = SystemOneRequest(
        state="just a single question request",
        questions={"q": NoulQuestion(type="noul", instructions="ok?")},
    )
    # Prefix-path baseline
    multi_alone = eng.evaluate(multi)
    single_alone = eng.evaluate(single)
    together = eng.evaluate_prepared([eng.prepare(multi), eng.prepare(single)])
    assert not any(isinstance(r, BaseException) for r in together)
    for name in multi.questions:
        _close(_dump(multi_alone.answers[name]), _dump(together[0].answers[name]))
    _close(_dump(single_alone.answers["q"]), _dump(together[1].answers["q"]))


# ---- unforkable cache falls back to per-request evaluate -------------------------


def test_unforkable_cache_falls_back(monkeypatch):
    """When the cache cannot be forked, the engine runs each request alone through
    evaluate (which handles unforkable caches on its own path), rather than failing
    the whole group."""
    import strands_decider.infer as infer

    def boom(cache, repeats):
        raise UnforkableCache("test: unforkable")

    eng = _engine("qwen3")
    monkeypatch.setattr(infer, "_expand_cache_per_row", boom)
    reqs = [
        SystemOneRequest(
            state=f"state {i}",
            questions={
                "a": NoulQuestion(type="noul", instructions="ok?"),
                "b": NoulQuestion(type="noul", instructions="urgent?"),
            },
        )
        for i in range(2)
    ]
    prepared = [eng.prepare(r) for r in reqs]
    assert all(p.prefix is not None for p in prepared)
    out = eng.evaluate_prepared(prepared)
    assert not any(isinstance(r, BaseException) for r in out), out
    # Sticky flag flipped so later batches do not even try the cross-request path.
    assert eng._prefix_cache_unforkable is True


# ---- exception isolation ---------------------------------------------------------


def test_exception_in_one_request_does_not_poison_others(monkeypatch):
    """If the shared forward raises a non-fatal error, we re-run each request alone,
    so one bad row cannot take out its neighbours."""
    eng = _engine("qwen3")
    calls = {"n": 0}
    orig = eng._shared_prefix_batched

    def flaky(prefixes, rendered):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return orig(prefixes, rendered)

    monkeypatch.setattr(eng, "_shared_prefix_batched", flaky)
    reqs = [
        SystemOneRequest(
            state="shared state",
            questions={
                "a": NoulQuestion(type="noul", instructions="ok?"),
                "b": NoulQuestion(type="noul", instructions="urgent?"),
            },
        )
        for _ in range(3)
    ]
    out = eng.evaluate_prepared([eng.prepare(r) for r in reqs])
    assert not any(isinstance(r, BaseException) for r in out), out


# ---- _expand_cache_per_row round-trip --------------------------------------------


def test_expand_cache_per_row_matches_single_fork():
    """For a batch-1 cache, _expand_cache_per_row([n]) must agree with _expand_cache(_, n)
    row for row."""
    import strands_decider.infer as infer

    eng = _engine("qwen3")
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    prefix_out = eng.model.torso(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True
    )
    a = infer._expand_cache(prefix_out.past_key_values, 3)
    b = infer._expand_cache_per_row(prefix_out.past_key_values, [3])
    for la, lb in zip(a.layers, b.layers, strict=True):
        for name in ("keys", "values"):
            ta, tb = getattr(la, name, None), getattr(lb, name, None)
            if isinstance(ta, torch.Tensor):
                assert torch.equal(ta, tb)
