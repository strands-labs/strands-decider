"""The trainer's pointer head is v1's model (training/pointer.py), and what it saves is a v1 checkpoint.

For the same weights and rows, PointerModel builds v1's SystemOneCollator batch (ids, mask, option positions, the
augmentation's draws and targets) and gives the logits of v1's StrandsDeciderModel; a checkpoint the trainer writes
loads in v1's loader, engine, server and (on Apple silicon) MLX engine and gives the trainer's answers. CPU only: the
tiny Qwen3 torso and test-trained tokenizer of tiny_support.py.
"""

from __future__ import annotations

import json
import random

import pytest
import torch
from tiny_support import _examples, _tokenizer_file, convert, tiny_base, train_pointer

from strands_decider.modeling import StrandsDeciderModel
from strands_decider.prompting import NOUL_DEFAULT_CRITERIA as NOUL
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from training import convert as converter
from training import pointer
from training.data.collate import CollatorConfig, SystemOneCollator


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("pointer")
    examples = _examples(60, seed=1)
    _tokenizer_file(str(d / "tokenizer.json"), examples)
    return d, tiny_base(d / "base", str(d / "tokenizer.json")), examples


def _model(base, seed=0, **over):
    torch.manual_seed(seed)
    settings = dict(
        base_model=str(base), revision="0" * 40, rank=2, alpha=4, dropout=0.0, pointer_dim=16, head_dropout=0.0,
        force_bos=False, max_length=2048, ordinal_smoothing=0.1, device="cpu",
        lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )  # fmt: skip
    model = pointer.PointerModel(base, **{**settings, **over})
    with (
        torch.no_grad()
    ):  # lora_B starts at zero and the head's norm at 1 / 0: make every part matter
        for name, p in model.decider.named_parameters():
            if "lora_B" in name or "head.norm" in name:
                p.normal_(std=0.3)
    return model.eval()


def _rows(examples):
    return [converter.row_from_example(ex, f"x:{i}", "x", 0.1, []) for i, ex in enumerate(examples)]


def test_rows_round_trip_to_the_examples(tiny):
    _, _, examples = tiny
    for ex, row in zip(examples, _rows(examples), strict=True):
        back = pointer.example_from_row(row)
        assert (back.kind, back.state, back.instructions, back.options, back.label) == (
            ex.kind, ex.state, ex.instructions, ex.options, ex.label,
        )  # fmt: skip
        assert back.instruction_variants == ex.instruction_variants


@pytest.mark.parametrize("train", [False, True])
def test_encode_and_targets_are_the_collators(tiny, train):
    """Unaugmented rows give the eval collator's batch; augmented with a generator seeded s, the train collator's
    with seed s: the same option orders, phrasings, ids, positions and targets."""
    _, base, examples = tiny
    model = _model(base)
    rows = _rows(examples)
    collator = SystemOneCollator(
        model.tokenizer, CollatorConfig(max_length=2048, seed=11), train=train
    )
    rng = random.Random(11)
    for start in range(0, len(rows), 8):
        chunk = rows[start : start + 8]
        if train:
            chunk = pointer.augment(chunk, rng, reverse_score_prob=0.5)
        got = model.encode(chunk, max_length=2048)
        want = collator(examples[start : start + 8])
        for key in ("input_ids", "attention_mask", "n_slots", "opt_idx"):
            assert torch.equal(got.inputs[key], want[key]), key
        target = pointer.target_matrix(chunk, torch.device("cpu"))
        width = want["opt_idx"].shape[1]
        expected = want.get("label_dist")
        if expected is None:
            expected = torch.nn.functional.one_hot(want["labels"], width).float()
        assert torch.equal(target[:, :width], expected) and not target[:, width:].any()


def test_logits_are_v1s_for_the_same_weights(tiny, tmp_path):
    """The trainer's model and v1's StrandsDeciderModel, loaded from the checkpoint the trainer's model saves, give
    the same logits on the same batch."""
    _, base, examples = tiny
    model = _model(base)
    model.save(tmp_path / "ckpt", {"head": "pointer"})
    v1 = StrandsDeciderModel.load(str(tmp_path / "ckpt")).eval()
    batch = SystemOneCollator(v1.tokenizer, CollatorConfig(max_length=2048), train=False)(
        examples[:16]
    )
    with torch.no_grad():
        want = v1(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            n_slots=batch["n_slots"],
            opt_idx=batch["opt_idx"],
        )["logits"]
        got = model(model.encode(_rows(examples[:16]), max_length=2048))
    for r, n in enumerate(batch["n_slots"].tolist()):
        assert torch.equal(got[r, :n], want[r, :n])
        assert (got[r, n:] == pointer.MASKED).all()


def test_the_saved_config_names_the_hub_base(tiny, tmp_path):
    _, base, _ = tiny
    model = _model(base, base_model="org/some-base", revision="1" * 40)
    model.save(tmp_path / "ckpt", {"head": "pointer"})
    saved = json.loads((tmp_path / "ckpt" / "strands_decider_config.json").read_text())
    assert (saved["base_model"], saved["base_revision"], saved["head_type"]) == (
        "org/some-base",
        "1" * 40,
        "pointer",
    )
    assert (
        saved["ordinal_smoothing"],
        saved["max_length"],
        saved["lora_r"],
        saved["lora_alpha"],
    ) == (0.1, 2048, 2, 4)
    assert json.loads((tmp_path / "ckpt" / "training.json").read_text()) == {"head": "pointer"}
    for path in (tmp_path / "ckpt").rglob("*"):  # no file names the local copy of the base
        assert not path.is_file() or str(base) not in path.read_text(errors="ignore"), path
    adapter = json.loads((tmp_path / "ckpt" / "lora" / "adapter_config.json").read_text())
    assert (adapter["base_model_name_or_path"], adapter["revision"]) == ("org/some-base", "1" * 40)


REQUEST = SystemOneRequest(
    state="The parcel arrived damaged and the customer wants a refund within thirty days.",
    questions={
        "team": ChoiceQuestion(
            instructions="Which team handles this?",
            criteria={"billing": "money", "shipping": None, "sales": "deals"},
        ),
        "urgent": NoulQuestion(instructions="The customer is angry."),
        "mood": ScoreQuestion(instructions="How upset?", criteria=["calm", "annoyed", "furious"]),
    },
)


def _request_rows():
    """REQUEST's questions as the trainer's rows, in the canonical order an engine renders them (a noul without
    criteria renders with the default descriptions)."""
    from training.data.format import Example

    q = REQUEST.questions
    examples = [
        Example("choice", REQUEST.state, q["team"].instructions, [[k, v] for k, v in q["team"].criteria.items()], 0),
        Example("noul", REQUEST.state, q["urgent"].instructions, [["false", NOUL["false"]], ["true", NOUL["true"]]], 1),
        Example("score", REQUEST.state, q["mood"].instructions, [[str(i), c] for i, c in enumerate(q["mood"].criteria)], 0),
    ]  # fmt: skip
    return [converter.row_from_example(ex, f"r:{i}", "r", 0.0, []) for i, ex in enumerate(examples)]


def _answers_as_rows(answers):
    """An engine's answers in the order of _request_rows' options."""
    team, urgent, mood = answers["team"], answers["urgent"], answers["mood"]
    return [
        [team.probabilities[k] for k in ("billing", "shipping", "sales")],
        [1 - urgent.noul, urgent.noul],
        [mood.probabilities[str(i)] for i in range(3)],
    ]


@pytest.fixture(scope="module")
def trained(tiny, tmp_path_factory):
    """Three steps of the trainer's pointer head; the run directory, and the trainer's model with its weights."""
    _, base, examples = tiny
    work = tmp_path_factory.mktemp("trained")
    rows = convert(work, base, list(examples), "--val-fraction", "0.1")
    run = train_pointer(rows, base, work / "run", "--stop-after", "3", "--eval-every", "1000")
    model = _model(base)
    model.load_trained(run / "last" / "checkpoint")
    return run, model.eval()


def test_a_trained_checkpoint_answers_in_v1s_engine_as_the_trainer_does(trained):
    from strands_decider.infer import load_engine

    run, model = trained
    want = model.option_probabilities(_request_rows(), max_length=2048)
    for cache in (True, False):
        engine = load_engine(str(run / "last" / "checkpoint"), device="cpu", use_prefix_cache=cache)
        got = _answers_as_rows(engine.evaluate(REQUEST).answers)
        for g, w in zip(got, want, strict=True):
            assert g == pytest.approx(w, abs=1e-4)  # answers are rounded to 4 places


def test_a_trained_checkpoint_serves(trained):
    from fastapi.testclient import TestClient

    from strands_decider import server

    run, model = trained
    want = model.option_probabilities(_request_rows(), max_length=2048)
    app = server.create_app(str(run / "last" / "checkpoint"), device="cpu", warmup=False)
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=REQUEST.model_dump())
    assert response.status_code == 200, response.text
    from strands_decider.schema import SystemOneResponse

    got = _answers_as_rows(SystemOneResponse.model_validate(response.json()).answers)
    for g, w in zip(got, want, strict=True):
        assert g == pytest.approx(w, abs=1e-4)


def test_the_run_records_and_resumes(tiny, tmp_path):
    """A run stopped after 2 steps and resumed is the run that trained 4 steps at once."""
    _, base, examples = tiny
    rows = convert(tmp_path, base, list(examples), "--val-fraction", "0.1")
    flags = ("--eval-every", "2", "--save-every", "2", "--epochs", "2")
    whole = train_pointer(rows, base, tmp_path / "whole", *flags, "--stop-after", "4")
    part = train_pointer(rows, base, tmp_path / "part", *flags, "--stop-after", "2")
    train_pointer(rows, base, part, *flags, "--stop-after", "4", "--resume")
    lines = lambda run, name: [json.loads(x) for x in open(run / name)]  # noqa: E731
    a, b = lines(whole, "train_log.jsonl"), lines(part, "train_log.jsonl")
    assert [s["step"] for s in b] == [1, 2, 3, 4]
    for x, y in zip(a, b, strict=True):
        assert (x["loss"], x["grad_norm"], x["lr"], x["rows"]) == (
            y["loss"],
            y["grad_norm"],
            y["lr"],
            y["rows"],
        )
    from safetensors.torch import load_file

    for name in ("head.safetensors", "lora/adapter_model.safetensors"):
        p, q = (
            load_file(str(whole / "last" / "checkpoint" / name)),
            load_file(str(part / "last" / "checkpoint" / name)),
        )
        assert all(torch.equal(p[k], q[k]) for k in p)
    record = json.loads((whole / "last" / "checkpoint" / "training.json").read_text())
    assert (record["head"], record["prompt"], record["attention_mode"]) == (
        "pointer",
        "numbered",
        "causal",
    )
    assert record["schedule"] == "linear warmup, then cosine to 0.1x"
    run_config = json.loads((whole / "run_config.json").read_text())
    assert "strands_decider/modeling.py" in run_config["code_sha256"]
    evals = lines(whole, "eval_log.jsonl")
    assert [e["step"] for e in evals] == [0, 2, 4] and 0 <= evals[-1]["raw"]["accuracy"] <= 1


def test_the_head_and_the_adapter_take_v1s_optimizer_groups(tiny):
    _, base, _ = tiny
    model = _model(base)
    groups = model.parameter_groups(1e-4, 1e-3, 0.01)
    assert [(g["lr"], g["weight_decay"]) for g in groups] == [
        (1e-3, 0.01),
        (1e-3, 0.0),
        (1e-4, 0.0),
    ]
    head = {id(p) for p in model.decider.head.parameters()}
    assert all(id(p) in head for g in groups[:2] for p in g["params"])
    assert all(p.ndim > 1 for p in groups[0]["params"]) and all(
        p.ndim <= 1 for p in groups[1]["params"]
    )
    assert {id(p) for g in groups for p in g["params"]} == {
        id(p) for p in model.trainable_parameters()
    }


def test_a_prompt_over_max_length_is_refused_not_cut(tiny):
    _, base, examples = tiny
    model = _model(base)
    with pytest.raises(ValueError, match="max_length"):
        model.encode(_rows(examples[:4]), max_length=8)


def test_fixed_order_choices_keep_their_order(tiny):
    _, _, examples = tiny
    row = next(
        r
        for r in _rows(examples)
        if r["question"]["type"] == "choice" and not r["source"].get("score")
    )
    fixed = {**row, "source": {**row["source"], "fixed_order": True}}
    rng_a, rng_b = random.Random(5), random.Random(5)
    assert pointer.augment([fixed], rng_a, 0.5)[0][pointer.AUGMENTATION]["order"] is None
    pointer.augment([row], rng_b, 0.5)
    assert rng_a.random() == rng_b.random()  # the shuffle was still drawn


def test_the_seeds_of_a_run_soup_and_calibrate(tiny, tmp_path):
    """Two seeds of one pointer config soup into one v1 checkpoint (training.soup), which calibrates and loads."""
    from training.evaluate import calibrate_checkpoint
    from training.soup import soup

    _, base, examples = tiny
    rows = convert(tmp_path, base, list(examples), "--val-fraction", "0.1")
    runs = [
        train_pointer(
            rows,
            base,
            tmp_path / f"seed-{s}",
            "--seed",
            str(s),
            "--stop-after",
            "2",
            "--eval-every",
            "100",
        )
        / "last"
        / "checkpoint"
        for s in (0, 1)
    ]
    soup([str(r) for r in runs], str(tmp_path / "soup"))
    calibrate_checkpoint(str(tmp_path / "soup"), list(examples), device="cpu")
    model = StrandsDeciderModel.load(str(tmp_path / "soup"))
    assert (
        model.config.lora_r == 4 and model.config.pointer_dim == 32
    )  # two rank-2 adapters, two 16-wide heads


def test_a_run_starts_from_the_weights_v1s_trainer_started_from(tiny):
    """train.py seeds and builds the pointer model with v1's own call, so the untrained head and adapter are v1's."""
    from strands_decider.modeling import StrandsDeciderConfig

    _, base, _ = tiny
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    torch.manual_seed(3)
    v1 = StrandsDeciderModel.from_pretrained_base(
        StrandsDeciderConfig(
            base_model=str(base),
            pointer_dim=16,
            lora_r=2,
            lora_alpha=4,
            head_dropout=0.05,
            torch_dtype="float32",
            max_length=2048,
            lora_targets=targets,
        )
    )
    torch.manual_seed(3)
    ours = pointer.PointerModel(
        base, base_model=str(base), revision="0" * 40, rank=2, alpha=4, dropout=0.05, lora_targets=targets,
        pointer_dim=16, head_dropout=0.05, force_bos=False, max_length=2048, ordinal_smoothing=0.0, device="cpu",
    )  # fmt: skip
    want = {n: p for n, p in v1.named_parameters() if p.requires_grad}
    got = {n: p for n, p in ours.decider.named_parameters() if p.requires_grad}
    assert want.keys() == got.keys() and len(want) > 2
    assert all(torch.equal(want[n], got[n]) for n in want)


def test_the_pointer_head_learns_a_learnable_task(tmp_path):
    """Yes/no rows whose answer is in the state (a refund, or not): 16 epochs of 128 rows on the tiny torso take the dev
    fold from chance to above 0.9. A smoke test that the loop learns, not a measure of the recipe."""
    import random as rnd

    from training.data.format import Example

    g = rnd.Random(0)
    filler = "the order arrived late and the customer wrote twice about the parcel".split()

    def example(i: int) -> Example:
        refund = i % 2 == 0
        words = [g.choice(filler) for _ in range(g.randint(3, 8))]
        words.insert(g.randrange(len(words) + 1), "refund" if refund else "shipping")
        return Example("noul", " ".join(words), "The customer asks for a refund.",
                       [["false", "no refund"], ["true", "a refund"]], int(refund), task=f"t{i % 2}")  # fmt: skip

    examples = [example(i) for i in range(160)]
    _tokenizer_file(str(tmp_path / "tokenizer.json"), examples)
    base = tiny_base(tmp_path / "base", str(tmp_path / "tokenizer.json"))
    rows = convert(tmp_path, base, examples, "--val-fraction", "0.2")
    run = train_pointer(rows, base, tmp_path / "run", "--epochs", "16", "--lr", "3e-3", "--head-lr", "3e-2",
                        "--eval-every", "1000", "--dropout", "0", "--head-dropout", "0")  # fmt: skip
    evals = [json.loads(line) for line in open(run / "eval_log.jsonl")]
    assert evals[0]["raw"]["accuracy"] < 0.7 < 0.9 < evals[-1]["raw"]["accuracy"], evals
