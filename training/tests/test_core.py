"""Tests for the parts that do not need a GPU or model weights."""

from __future__ import annotations

import json

import pytest
import torch

from strands_decider.modeling import (
    PointerHead,
    StrandsDeciderConfig,
    build_head,
    masked_log_softmax,
    pool_last_token,
)
from strands_decider.prompting import (
    build_prompt,
    read_choice,
    read_noul,
    read_score,
    render_question,
    score_legend,
)
from strands_decider.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    derive_confidence,
)
from training.data.collate import CollatorConfig, SystemOneCollator
from training.data.format import Example, split_examples

# ---------------------------------------------------------------- confidence


def test_confidence_uniform_is_zero():
    assert derive_confidence([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(0.0, abs=1e-9)
    assert derive_confidence([0.25] * 4) == pytest.approx(0.0, abs=1e-9)


def test_confidence_one_hot_is_one():
    assert derive_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_confidence_matches_documented_formula():
    # (3 * 0.88 - 1) / 2
    assert derive_confidence([0.88, 0.12, 0.0]) == pytest.approx((3 * 0.88 - 1) / 2)


def test_confidence_is_scale_free_across_n():
    """A 3-option and a 10-option question at the same confidence should agree,
    otherwise a single routing threshold cannot serve both."""
    three = derive_confidence([0.7, 0.2, 0.1])
    ten = [0.0] * 10
    # Solve for p_max giving the same normalised confidence at N=10.
    p = (three * 9 + 1) / 10
    ten[0] = p
    for i in range(1, 10):
        ten[i] = (1 - p) / 9
    assert derive_confidence(ten) == pytest.approx(three, abs=1e-6)


# ---------------------------------------------------------------- prompting


def test_choice_permutation_rebinds_slots():
    q = ChoiceQuestion(
        instructions="Route this ticket.",
        criteria={"billing": "payments", "technical": "bugs", "sales": "pricing"},
    )
    base = render_question(q)
    assert base.slot_labels == ("billing", "technical", "sales")

    flipped = render_question(q, option_order=[2, 0, 1])
    assert flipped.slot_labels == ("sales", "billing", "technical")

    # The same underlying probability must be read back for the same option
    # regardless of which slot it landed in.
    assert read_choice([0.1, 0.7, 0.2], base)["technical"] == pytest.approx(0.7)
    assert read_choice([0.2, 0.1, 0.7], flipped)["technical"] == pytest.approx(0.7)


def test_noul_true_slot_found_after_permutation():
    q = NoulQuestion(instructions="Is this urgent?")
    normal = render_question(q)
    assert normal.slot_labels == ("false", "true")
    assert read_noul([0.3, 0.7], normal) == pytest.approx(0.7)

    swapped = render_question(q, option_order=[1, 0])
    assert swapped.slot_labels == ("true", "false")
    assert read_noul([0.7, 0.3], swapped) == pytest.approx(0.7)


def test_score_reversed_rubric_reads_back_correctly():
    q = ScoreQuestion(
        instructions="How angry is the customer?",
        criteria=["Calm", "Frustrated", "Very angry"],
    )
    normal = render_question(q)
    rev = render_question(q, option_order=[2, 1, 0])

    # Same belief -- almost certainly level 2 -- expressed in both renderings.
    s_normal, p_normal = read_score([0.0, 0.05, 0.95], normal)
    s_rev, p_rev = read_score([0.95, 0.05, 0.0], rev)

    assert s_normal == pytest.approx(s_rev)
    assert s_normal == pytest.approx(1.95)
    assert p_normal == pytest.approx(p_rev)
    # Probabilities are keyed by level, ascending, in both cases.
    assert list(p_normal) == ["0", "1", "2"]


def test_score_legend_is_canonical_even_when_reversed():
    q = ScoreQuestion(instructions="x", criteria=["Calm", "Frustrated", "Very angry"])
    legend = score_legend(render_question(q, option_order=[2, 1, 0]))
    assert legend == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}


def test_prompt_contains_numbered_options():
    q = ChoiceQuestion(instructions="Pick.", criteria={"a": "first", "b": "second"})
    prompt, _ = build_prompt("some state", q)
    assert "<state>" in prompt and "some state" in prompt
    assert "1. a — first" in prompt
    assert "2. b — second" in prompt
    assert prompt.rstrip().endswith("<answer>")


def test_state_renders_deterministically():
    """The shared-prefix cache is only sound if identical states tokenise identically."""
    state = {"z": 1, "a": [1, 2, {"k": "v"}]}
    p1, _ = build_prompt(state, NoulQuestion(instructions="q"))
    p2, _ = build_prompt(state, NoulQuestion(instructions="q"))
    assert p1 == p2


def test_bad_permutation_rejected():
    q = ChoiceQuestion(instructions="x", criteria={"a": "", "b": ""})
    with pytest.raises(ValueError):
        render_question(q, option_order=[0, 0])


# ---------------------------------------------------------------- schema


def test_score_level_bounds_enforced():
    with pytest.raises(ValueError):
        ScoreQuestion(instructions="x", criteria=["only one"])
    with pytest.raises(ValueError):
        ScoreQuestion(instructions="x", criteria=[str(i) for i in range(11)])


def test_choice_needs_two_options():
    with pytest.raises(ValueError):
        ChoiceQuestion(instructions="x", criteria={"only": "one"})


def test_request_accepts_empty_state_and_structured_criteria():
    """A quiz item carries the whole task in its question; an option may be structured data."""
    req = SystemOneRequest.model_validate(
        {
            "state": "",
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "Best move?",
                    "criteria": {
                        "b8g8": {"uci": "b8g8", "san": "Qg8"},
                        "b8b7": "Qb7",
                        "a1a2": None,
                    },
                }
            },
        }
    )
    rq = render_question(req.questions["move"])
    assert '"san": "Qg8"' in rq.text and "Qb7" in rq.text
    assert rq.n_slots == 3 and "3. a1a2\n" in rq.text


def test_request_parses_documented_payload():
    req = SystemOneRequest.model_validate(
        {
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "strands-decider-latest",
            "questions": {
                "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
                "department": {
                    "type": "choice",
                    "instructions": "Route it.",
                    "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"},
                },
                "frustration": {
                    "type": "score",
                    "instructions": "How frustrated?",
                    "criteria": ["Calm", "Frustrated", "Very angry"],
                },
            },
        }
    )
    assert set(req.questions) == {"is_urgent", "department", "frustration"}
    assert req.questions["frustration"].type == "score"


# ---------------------------------------------------------------- masking / pooling


def test_masked_log_softmax_ignores_unused_slots():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    lp = masked_log_softmax(logits, torch.tensor([3]))
    probs = lp.exp()
    assert probs[0, :3].sum().item() == pytest.approx(1.0, abs=1e-5)
    assert probs[0, 3:].sum().item() == pytest.approx(0.0, abs=1e-9)
    # Slots 3 and 4 have the *largest* logits; masking must still exclude them.
    assert probs.argmax().item() == 2


def test_masked_log_softmax_varies_per_row():
    logits = torch.zeros(2, 4)
    lp = masked_log_softmax(logits, torch.tensor([2, 4])).exp()
    assert lp[0, 0].item() == pytest.approx(0.5)
    assert lp[1, 0].item() == pytest.approx(0.25)


def test_pool_last_token_picks_final_real_token():
    hidden = torch.tensor(
        [
            [[1.0, 1.0], [2.0, 2.0], [9.0, 9.0]],  # last real token is index 1
            [[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]],  # no padding
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = pool_last_token(hidden, mask)
    assert pooled[0].tolist() == [2.0, 2.0]
    assert pooled[1].tolist() == [5.0, 5.0]


def test_pool_last_token_handles_prefix_cache_mask():
    """With a shared prefix the mask is longer than the forwarded tail."""
    hidden = torch.randn(1, 3, 4)
    mask = torch.ones(1, 10, dtype=torch.long)  # 7 cached + 3 forwarded
    mask[0, -1] = 0  # last forwarded token is padding
    pooled = pool_last_token(hidden, mask)
    assert torch.allclose(pooled[0], hidden[0, 1])


# ---------------------------------------------------------------- collation


def _choice_example() -> Example:
    return Example(
        kind="choice",
        state="the state",
        instructions="pick one",
        options=[["a", "first"], ["b", "second"], ["c", "third"]],
        label=1,
        task="unit",
    )


class _FakeTokenizer:
    """Character-level stand-in so collation can be tested without downloading weights."""

    pad_token = "<pad>"

    def __call__(
        self,
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors=None,
        padding_side="right",
        add_special_tokens=True,
        return_offsets_mapping=False,
    ):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [[min(ord(c), 255) for c in t[:max_length]] for t in texts]
        width = max(len(s) for s in seqs)
        ids, mask, offs = [], [], []
        for s in seqs:
            pad = width - len(s)
            ids.append(s + [0] * pad)
            mask.append([1] * len(s) + [0] * pad)
            # One character per token, so token j spans (j, j+1); padding spans (0, 0),
            # which is how a real fast tokeniser marks a non-text token.
            offs.append([[j, j + 1] for j in range(len(s))] + [[0, 0]] * pad)
        out = {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        if return_offsets_mapping:
            out["offset_mapping"] = torch.tensor(offs)
        return out


def test_collator_label_binding_is_correct():
    coll = SystemOneCollator(_FakeTokenizer(), CollatorConfig(seed=3), train=True)
    ex = _choice_example()
    gold_name = ex.options[ex.label][0]
    for _ in range(30):
        order = coll._option_order(ex)
        label = coll._remap_label(ex.label, order)
        shown = [ex.options[i][0] for i in (order or range(ex.n_options))]
        assert shown[label] == gold_name


def test_collator_does_not_shuffle_in_eval_mode():
    coll = SystemOneCollator(_FakeTokenizer(), CollatorConfig(), train=False)
    ex = _choice_example()
    for _ in range(10):
        assert coll._option_order(ex) is None


def test_score_options_never_arbitrarily_permuted():
    """Score slots are ordinal: only a full reversal is meaning-preserving."""
    coll = SystemOneCollator(
        _FakeTokenizer(),
        CollatorConfig(reverse_score_prob=1.0, seed=1),
        train=True,
    )
    ex = Example(
        kind="score",
        state="s",
        instructions="rate",
        options=[["0", "low"], ["1", "mid"], ["2", "high"]],
        label=0,
        task="unit",
    )
    order = coll._option_order(ex)
    assert order == [2, 1, 0]
    assert coll._remap_label(ex.label, order) == 2


def test_ordinal_smoothing_puts_mass_on_neighbours():
    coll = SystemOneCollator(
        _FakeTokenizer(),
        CollatorConfig(ordinal_smoothing=0.2, reverse_score_prob=0.0),
        train=True,
    )
    ex = Example(
        kind="score",
        state="s",
        instructions="rate",
        options=[["0", "a"], ["1", "b"], ["2", "c"]],
        label=1,
        task="unit",
    )
    dist = coll._target_distribution(ex, 1, 3, None)
    assert dist is not None
    assert dist[1].item() == pytest.approx(0.8)
    assert dist[0].item() == pytest.approx(0.1)
    assert dist[2].item() == pytest.approx(0.1)
    assert dist.sum().item() == pytest.approx(1.0)


def test_ordinal_smoothing_at_rubric_edge():
    """Level 0 has only one neighbour; all the smoothing mass goes there."""
    coll = SystemOneCollator(
        _FakeTokenizer(),
        CollatorConfig(ordinal_smoothing=0.2, reverse_score_prob=0.0),
        train=True,
    )
    ex = Example(
        kind="score",
        state="s",
        instructions="rate",
        options=[["0", "a"], ["1", "b"], ["2", "c"]],
        label=0,
        task="unit",
    )
    dist = coll._target_distribution(ex, 0, 3, None)
    assert dist[0].item() == pytest.approx(0.8)
    assert dist[1].item() == pytest.approx(0.2)
    assert dist[2].item() == pytest.approx(0.0)


# ---------------------------------------------------------------- data format


def test_example_rejects_out_of_range_label():
    with pytest.raises(ValueError):
        Example(kind="choice", state="s", instructions="i", options=[["a", ""], ["b", ""]], label=2)


def test_split_is_stratified_by_task():
    examples = [Example("noul", "s", "i", [["false", ""], ["true", ""]], 0, task="big")] * 200 + [
        Example("noul", "s", "i", [["false", ""], ["true", ""]], 1, task="tiny")
    ] * 4
    train, val = split_examples(examples, val_fraction=0.1, seed=0)
    # The small task must survive into both halves rather than vanishing.
    assert any(e.task == "tiny" for e in train)
    assert any(e.task == "tiny" for e in val)


def test_uniform_weights_match_unweighted_loss():
    """Weighting must be a no-op when every example has weight 1."""

    logits = torch.tensor([[2.0, 1.0, 0.0], [0.5, 2.5, 0.0]])
    labels = torch.tensor([0, 1])
    n_slots = torch.tensor([3, 3])
    lp = masked_log_softmax(logits, n_slots)

    plain = torch.nn.functional.nll_loss(lp, labels)
    per_ex = torch.nn.functional.nll_loss(lp, labels, reduction="none")
    w = torch.ones(2)
    weighted = (per_ex * w).sum() / w.sum()
    assert weighted.item() == pytest.approx(plain.item())


def test_weighting_shifts_loss_toward_heavier_example():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    labels = torch.tensor([0, 0])  # first is easy, second is badly wrong
    lp = masked_log_softmax(logits, torch.tensor([3, 3]))
    per_ex = torch.nn.functional.nll_loss(lp, labels, reduction="none")

    def weighted(w):
        w = torch.tensor(w)
        return ((per_ex * w).sum() / w.sum()).item()

    # Up-weighting the badly-wrong example must raise the loss.
    assert weighted([1.0, 4.0]) > weighted([1.0, 1.0]) > weighted([4.0, 1.0])


# ---------------------------------------------------------------- calib/test split


def _dummy(n):
    return [
        Example("noul", f"s{i}", "i", [["false", ""], ["true", ""]], i % 2, task="t")
        for i in range(n)
    ]


def test_calib_and_test_splits_are_disjoint_and_complete():
    """Fitting temperature on rows you then score inflates the calibration result."""
    from training.evaluate import partition_examples

    examples = _dummy(400)
    calib = partition_examples(examples, "calib")
    test = partition_examples(examples, "test")

    calib_states = {e.state for e in calib}
    test_states = {e.state for e in test}
    assert not (calib_states & test_states)
    assert calib_states | test_states == {e.state for e in examples}
    # Roughly even, so neither half is too small to be meaningful.
    assert 0.35 < len(calib) / len(examples) < 0.65


def test_partition_is_deterministic_across_calls():
    from training.evaluate import partition_examples

    examples = _dummy(200)
    assert [e.state for e in partition_examples(examples, "calib")] == [
        e.state for e in partition_examples(examples, "calib")
    ]


def test_partition_all_returns_everything():
    from training.evaluate import partition_examples

    examples = _dummy(50)
    assert len(partition_examples(examples, "all")) == 50


def test_partition_rejects_unknown_split():
    from training.evaluate import partition_examples

    with pytest.raises(ValueError, match="calib"):
        partition_examples(_dummy(10), "train")


def test_duplicate_examples_cannot_straddle_the_split():
    """Position-based hashing, so identical content in two rows is still split cleanly."""
    from training.evaluate import partition_examples

    same = Example("noul", "identical", "i", [["false", ""], ["true", ""]], 0, task="t")
    examples = [same] * 100
    calib = partition_examples(examples, "calib")
    test = partition_examples(examples, "test")
    assert len(calib) + len(test) == 100


def test_sample_is_representative_not_a_prefix():
    """Corpora are grouped by task; a head slice evaluates one task and calls it overall."""
    from training.evaluate import sample_examples

    grouped = [
        Example("noul", f"a{i}", "i", [["false", ""], ["true", ""]], 0, task="A")
        for i in range(500)
    ] + [
        Example("noul", f"b{i}", "i", [["false", ""], ["true", ""]], 1, task="B")
        for i in range(500)
    ]
    picked = sample_examples(grouped, 200)
    tasks = {e.task for e in picked}
    assert tasks == {"A", "B"}, "sampling collapsed onto a single task"
    share_a = sum(1 for e in picked if e.task == "A") / len(picked)
    assert 0.3 < share_a < 0.7


def test_sample_is_deterministic_and_bounded():
    from training.evaluate import sample_examples

    pool = _dummy(300)
    first = [e.state for e in sample_examples(pool, 50)]
    assert first == [e.state for e in sample_examples(pool, 50)]
    assert len(first) == 50
    # A limit at or above the corpus size returns everything, unshuffled.
    assert len(sample_examples(pool, 1000)) == 300


# ---------------------------------------------------------------- score confidence


def _smoothed(n_levels: int, gold: int, eps: float = 0.1):
    """The distribution a perfectly-fitted model emits under ordinal smoothing."""
    d = [0.0] * n_levels
    nb = [i for i in (gold - 1, gold + 1) if 0 <= i < n_levels]
    d[gold] = 1.0 - eps
    for i in nb:
        d[i] += eps / len(nb)
    if not nb:
        d[gold] = 1.0
    return d


def test_score_confidence_reaches_one_under_smoothing():
    """The bug this replaced: a perfectly-fitted score could never reach 0.9.

    Max-probability confidence caps at (L*0.9-1)/(L-1) -- 0.68 for three levels --
    so the documented ">=0.9 act automatically" band was unreachable by construction.
    """
    from strands_decider.schema import derive_score_confidence

    for n in (3, 4, 5):
        d = _smoothed(n, gold=1)
        assert derive_score_confidence(d, ordinal_smoothing=0.1) > 0.95, n
        # And the old formula demonstrably could not:
        assert derive_confidence(d) < 0.9


def test_score_confidence_adjacent_beats_bimodal():
    """Mass on neighbouring levels is precision, not doubt; mass at both ends is
    doubt of the worst kind, because the expected value falls where nothing is."""
    from strands_decider.schema import derive_score_confidence

    adjacent = [0.0, 0.5, 0.5, 0.0, 0.0]
    uniform = [0.2] * 5
    bimodal = [0.5, 0.0, 0.0, 0.0, 0.5]

    c_adj = derive_score_confidence(adjacent)
    c_uni = derive_score_confidence(uniform)
    c_bim = derive_score_confidence(bimodal)

    assert c_adj > c_uni > c_bim
    assert c_bim == pytest.approx(0.0, abs=1e-9)
    # Max-probability cannot make this distinction -- it ranks bimodal above uniform.
    assert derive_confidence(bimodal) > derive_confidence(uniform)


def test_score_confidence_one_hot_is_one_and_extremes_zero():
    from strands_decider.schema import derive_score_confidence

    assert derive_score_confidence([0.0, 1.0, 0.0]) == pytest.approx(1.0)
    assert derive_score_confidence([0.5, 0.0, 0.5]) == pytest.approx(0.0, abs=1e-9)


def test_score_confidence_is_bounded():
    from strands_decider.schema import derive_score_confidence

    for d in ([1.0, 0.0], [0.3, 0.4, 0.3], [0.0] * 4 + [1.0], _smoothed(5, 0)):
        c = derive_score_confidence(d, ordinal_smoothing=0.1)
        assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------- per-kind temperature


def test_apply_temperature_accepts_scalar_and_per_row():
    from strands_decider.modeling import apply_temperature

    logits = torch.tensor([[2.0, 1.0], [4.0, 2.0]])
    assert torch.allclose(apply_temperature(logits, 2.0), logits / 2)
    per_row = apply_temperature(logits, torch.tensor([1.0, 2.0]))
    assert torch.allclose(per_row[0], logits[0])
    assert torch.allclose(per_row[1], logits[1] / 2)


def test_apply_temperature_identity_cases():
    from strands_decider.modeling import apply_temperature

    logits = torch.randn(3, 4)
    assert torch.allclose(apply_temperature(logits, None), logits)
    assert torch.allclose(apply_temperature(logits, 1.0), logits)


def test_fit_temperature_by_kind_fits_each_separately():
    """A single global temperature lands between primitives that need different ones."""
    from training.evaluate import fit_temperature_by_kind

    torch.manual_seed(0)
    n = 400
    examples, rows, labels, slots = [], [], [], []
    for kind, scale in (("choice", 6.0), ("score", 1.0)):
        for _ in range(n):
            lab = int(torch.randint(0, 3, (1,)))
            v = torch.full((8,), -5.0)
            v[:3] = torch.randn(3)
            v[lab] += scale  # choice is sharply separable, score is not
            rows.append(v)
            labels.append(lab)
            slots.append(3)
            examples.append(
                Example(kind, "s", "i", [["0", ""], ["1", ""], ["2", ""]], lab, task=kind)
            )

    t = fit_temperature_by_kind(
        torch.stack(rows), torch.tensor(labels), torch.tensor(slots), examples
    )
    assert set(t) == {"choice", "score"}
    # The over-separated primitive needs more softening than the ambiguous one.
    assert t["choice"] != t["score"]


def test_fit_temperature_by_kind_skips_thin_primitives():
    """Too few examples to fit reliably -> omitted, so serving falls back to global."""
    from training.evaluate import fit_temperature_by_kind

    examples = [
        Example("noul", "s", "i", [["false", ""], ["true", ""]], 0, task="t") for _ in range(10)
    ]
    logits = torch.randn(10, 8)
    t = fit_temperature_by_kind(
        logits, torch.zeros(10, dtype=torch.long), torch.full((10,), 2), examples
    )
    assert t == {}


def test_ece_and_nll_objectives_can_disagree():
    """The reason per-primitive fitting is done against ECE.

    Fitting noul to NLL on the real model chose T=2.63, leaving it under-confident
    by 0.29 -- worse calibrated than not scaling at all. The objectives are not
    interchangeable, so the choice has to be deliberate.
    """
    from training.evaluate import ece_at_temperature, fit_temperature

    torch.manual_seed(0)
    n = 600
    labels, rows, examples = [], [], []
    for _ in range(n):
        lab = int(torch.randint(0, 2, (1,)))
        v = torch.full((8,), -5.0)
        v[:2] = torch.randn(2) * 0.4
        v[lab] += 3.0  # separable but noisy: confident and often, but not always, right
        rows.append(v)
        labels.append(lab)
        examples.append(Example("noul", "s", "i", [["false", ""], ["true", ""]], lab, task="t"))
    logits = torch.stack(rows)
    y = torch.tensor(labels)
    slots = torch.full((n,), 2)

    t_nll = fit_temperature(logits, y, slots, objective="nll")
    t_ece = fit_temperature(logits, y, slots, objective="ece", examples=examples)

    ece_nll = ece_at_temperature(logits, y, slots, examples, t_nll)
    ece_ece = ece_at_temperature(logits, y, slots, examples, t_ece)
    # Fitting to ECE must not be beaten by fitting to NLL, on ECE.
    assert ece_ece <= ece_nll + 1e-9


def test_fit_temperature_rejects_unknown_objective():
    from training.evaluate import fit_temperature

    with pytest.raises(ValueError, match="nll"):
        fit_temperature(
            torch.randn(4, 8),
            torch.zeros(4, dtype=torch.long),
            torch.full((4,), 2),
            objective="brier",
        )


def test_fit_temperature_ece_requires_examples():
    """ECE needs per-kind confidence, so it cannot be computed from logits alone."""
    from training.evaluate import fit_temperature

    with pytest.raises(ValueError, match="examples"):
        fit_temperature(
            torch.randn(4, 8),
            torch.zeros(4, dtype=torch.long),
            torch.full((4,), 2),
            objective="ece",
        )


# ---------------------------------------------------------------- instruction variants


def _varied_example():
    return Example(
        "choice",
        "s",
        "Which section does this belong to?",
        [["a", ""], ["b", ""]],
        0,
        task="t",
        instruction_variants=["What is this about?", "Pick a section."],
    )


def test_collator_samples_among_instruction_phrasings():
    """A head that keyed on the exact instruction string should be wrong most of
    the time -- the same argument that motivates option shuffling."""
    coll = SystemOneCollator(_FakeTokenizer(), CollatorConfig(seed=1), train=True)
    ex = _varied_example()
    seen = {coll._instruction(ex) for _ in range(200)}
    assert seen == set(ex.all_instructions())


def test_eval_always_uses_the_canonical_phrasing():
    """Otherwise successive eval runs of the same checkpoint would not be comparable."""
    coll = SystemOneCollator(_FakeTokenizer(), CollatorConfig(), train=False)
    ex = _varied_example()
    assert {coll._instruction(ex) for _ in range(20)} == {None}


def test_instruction_variation_can_be_disabled():
    coll = SystemOneCollator(
        _FakeTokenizer(),
        CollatorConfig(vary_instructions=False),
        train=True,
    )
    assert coll._instruction(_varied_example()) is None


def test_example_without_variants_is_unaffected():
    coll = SystemOneCollator(_FakeTokenizer(), CollatorConfig(seed=1), train=True)
    plain = Example("choice", "s", "only phrasing", [["a", ""], ["b", ""]], 0, task="t")
    assert coll._instruction(plain) is None
    assert plain.all_instructions() == ["only phrasing"]


def test_sampled_phrasing_reaches_the_prompt():
    """The override must flow through to_question and into the rendered text."""
    ex = _varied_example()
    prompt, _ = build_prompt(ex.state, ex.to_question("Pick a section."))
    assert "Pick a section." in prompt
    assert "Which section does this belong to?" not in prompt


def test_instruction_variants_round_trip_through_jsonl():
    ex = _varied_example()
    back = Example.from_dict(json.loads(ex.to_json()))
    assert back.instruction_variants == ex.instruction_variants
    assert back.all_instructions() == ex.all_instructions()


def test_legacy_examples_without_the_field_still_load():
    """Corpora built before instruction variants existed must keep working."""
    d = json.loads(_varied_example().to_json())
    del d["instruction_variants"]
    back = Example.from_dict(d)
    assert back.instruction_variants == []


# ---------------------------------------------------------------------------
# Pointer head: option positions, ragged widths, and the cap that disappears.
# ---------------------------------------------------------------------------


def test_option_spans_cover_each_option_line():
    """Every span must land on its own numbered line, or the head reads the wrong one."""
    q = ChoiceQuestion(
        instructions="Which team?",
        criteria={"billing": "payments", "technical": "bugs", "sales": "pricing"},
    )
    rq = render_question(q)
    assert len(rq.option_spans) == rq.n_slots
    for i, (a, b) in enumerate(rq.option_spans):
        line = rq.text[a:b]
        assert line.startswith(f"{i + 1}. ")
        assert rq.slot_labels[i] in line
        assert "\n" not in line


def test_option_spans_follow_the_permutation():
    q = ChoiceQuestion(
        instructions="Which team?",
        criteria={"billing": "payments", "technical": "bugs", "sales": "pricing"},
    )
    rq = render_question(q, option_order=[2, 0, 1])
    for i, (a, b) in enumerate(rq.option_spans):
        assert rq.slot_labels[i] in rq.text[a:b]


def test_multiline_descriptions_stay_on_one_line():
    """A newline inside a description would split an option across two lines and make
    its span ambiguous, so descriptions are collapsed before rendering."""
    q = ChoiceQuestion(
        instructions="Pick.",
        criteria={"a": "first\nsecond\n\nthird", "b": "plain"},
    )
    rq = render_question(q)
    for a, b in rq.option_spans:
        assert "\n" not in rq.text[a:b]


def test_option_token_index_finds_the_last_token_of_each_span():
    offsets = [[j, j + 1] for j in range(20)]
    idx = SystemOneCollator.option_token_index(offsets, [(2, 5), (10, 14)], base=0)
    assert idx == [4, 13]


def test_option_token_index_applies_the_base_offset():
    offsets = [[j, j + 1] for j in range(20)]
    assert SystemOneCollator.option_token_index(offsets, [(2, 5)], base=0) == [4]
    plain = SystemOneCollator.option_token_index(offsets, [(0, 3)], base=0)
    assert plain == [2]


def test_option_token_index_refuses_a_truncated_option():
    """Scoring an option from a neighbour's token would be silently wrong, so the
    collator raises instead."""
    offsets = [[j, j + 1] for j in range(5)]
    with pytest.raises(ValueError, match="truncated"):
        SystemOneCollator.option_token_index(offsets, [(40, 50)], base=0)


def test_pointer_collator_emits_padded_option_indices():
    coll = SystemOneCollator(
        _FakeTokenizer(),
        CollatorConfig(max_length=4096),
        train=False,
    )
    noul = Example(
        kind="noul",
        state="s",
        instructions="Is it urgent?",
        options=[["false", "no"], ["true", "yes"]],
        label=1,
        task="unit",
    )
    batch = coll([_choice_example(), noul])
    assert "opt_idx" in batch
    n_opts = [_choice_example().n_options, noul.n_options]
    assert batch["opt_idx"].shape == (2, max(n_opts))
    for row, k in zip(batch["opt_idx"].tolist(), n_opts, strict=True):
        assert all(i >= 0 for i in row[:k])  # real positions
        assert all(i == -1 for i in row[k:])  # padding


def test_pointer_collator_handles_any_option_count():
    """A pointer collator has no fixed-slot ceiling: opt_idx stretches to the row width."""
    wide = Example(
        kind="choice",
        state="s",
        instructions="Pick one.",
        options=[[f"o{i}", f"d{i}"] for i in range(30)],
        label=7,
        task="unit",
    )
    pointer = SystemOneCollator(
        _FakeTokenizer(),
        CollatorConfig(max_length=8192),
        train=False,
    )
    assert pointer([wide])["opt_idx"].shape[1] == 30


def test_pointer_head_scores_each_option_from_its_own_state():
    head = PointerHead(16, dim=8)
    decide = torch.randn(2, 16)
    options = torch.randn(2, 5, 16)
    logits = head(decide, options)
    assert logits.shape == (2, 5)

    # Permuting the option states permutes the logits exactly: there is no per-slot
    # parameter for a positional preference to live in.
    perm = [3, 0, 4, 1, 2]
    permuted = head(decide, options[:, perm])
    assert torch.allclose(permuted, logits[:, perm], atol=1e-6)


def test_gather_options_ignores_padding():
    from strands_decider.modeling import gather_options

    hidden = torch.randn(2, 10, 4)
    idx = torch.tensor([[1, 5, -1], [0, 2, 7]])
    out = gather_options(hidden, idx)
    assert out.shape == (2, 3, 4)
    assert torch.equal(out[0, 0], hidden[0, 1])
    assert torch.equal(out[1, 2], hidden[1, 7])


def test_build_head_returns_pointer_head():
    assert isinstance(build_head(StrandsDeciderConfig(), 32), PointerHead)


def test_slot_era_head_type_is_rejected_by_config_loader(tmp_path):
    """A checkpoint whose config names a slot head fails cleanly, not silently."""
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"base_model": "x", "head_type": "slot"}))
    with pytest.raises(ValueError, match="pointer-head"):
        StrandsDeciderConfig.from_json(str(path))
    # A legacy `head_type: "pointer"` config loads unchanged (the field is dropped).
    path.write_text(json.dumps({"base_model": "x", "head_type": "pointer", "num_slots": 24}))
    cfg = StrandsDeciderConfig.from_json(str(path))
    assert cfg.base_model == "x"


def test_head_type_is_written_to_the_exported_config():
    """Older strands-decider versions default a missing head_type to slot and crash on a
    pointer export, and out-of-tree readers may parse this field directly."""
    assert json.loads(StrandsDeciderConfig(base_model="x").to_json())["head_type"] == "pointer"


def test_collect_logits_masks_the_padded_columns():
    """A pointer head scores padded option slots from position 0, so in a mixed-width
    batch those columns hold real values. collect_logits must mask them: multistep_eval
    takes a raw argmax over every column."""
    from strands_decider.modeling import MASK_VALUE
    from training.evaluate import collect_logits

    class _Stub:
        tokenizer = _FakeTokenizer()

        class config:
            max_length = 3072

        torso = torch.nn.Module()  # no parameters: the CPU upcast has nothing to cast

        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, input_ids, attention_mask, n_slots, opt_idx, temperature):
            # Rises with the column, so without the mask the widest column always wins.
            b, k = opt_idx.shape
            return {"logits": torch.arange(k, dtype=torch.float32).expand(b, k)}

    exs = [
        Example(
            kind="choice",
            state="s",
            instructions="q",
            options=[[f"o{i}", ""] for i in range(n)],
            label=0,
            task="t",
        )
        for n in (2, 5, 3, 2)
    ]
    logits, _, slots, _ = collect_logits(_Stub(), exs, device="cpu", batch_size=4)
    assert logits.shape == (4, 5) and slots.tolist() == [2, 5, 3, 2]
    for i, n in enumerate(slots.tolist()):
        assert (logits[i, n:] == MASK_VALUE).all()
        assert (logits[i, :n] == torch.arange(n, dtype=torch.float32)).all()
        assert int(logits[i].argmax()) < n


def test_teacher_targets_follow_option_shuffling():
    """Slot k must carry the teacher's mass for the option actually shown at slot k."""
    names = ["World", "Sports", "Business", "Sci/Tech", "Health"]
    teacher = [0.05, 0.6, 0.2, 0.1, 0.05]
    coll = SystemOneCollator(_FakeTokenizer(), CollatorConfig(seed=3), train=True)
    slots = set()
    for _ in range(20):
        ex = Example(
            kind="choice",
            state="Arsenal won 3-1.",
            instructions="Which section?",
            options=[[n, ""] for n in names],
            label=1,
            task="t",
        )
        ex.teacher = teacher
        out = coll([ex])
        slot_gold = int(out["labels"][0])  # where the shuffle put the gold option
        slots.add(slot_gold)
        t = out["teacher"][0]
        assert abs(float(t[slot_gold]) - 0.6) < 1e-6
        assert abs(float(t.sum()) - 1.0) < 1e-5
        assert bool(out["has_teacher"][0])
    assert len(slots) > 1  # the options really were shuffled
