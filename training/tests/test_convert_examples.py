"""training.convert --examples: v1's examples into the trainer's rows, for either head (training/convert.py)."""

from __future__ import annotations

import json
import random

import pytest
from tiny_support import _examples, _tokenizer_file, convert, tiny_base

from strands_decider.prompting import build_prompt
from training import convert as converter
from training import folds, pointer
from training.data.format import Example, split_examples, write_jsonl


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("convert")
    examples = _examples(90, seed=4)
    for ex in examples:
        ex.weight = 1.0
    _tokenizer_file(str(d / "tokenizer.json"), examples)
    return d, tiny_base(d / "base", str(d / "tokenizer.json")), examples


def _read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_each_kind_maps_to_a_row():
    noul = Example("noul", {"a": 1}, "Is it?", [["false", "no"], ["true", "yes"]], 1, task="t1")
    choice = Example("choice", "s", "Which?", [["x", None], ["y", "why"], ["z", ""]], 2, task="t2")
    score = Example("score", "s", "How?", [["0", "low"], ["1", "mid"], ["2", "high"]], 0, task="t3",
                    instruction_variants=["How much?"])  # fmt: skip
    a = converter.row_from_example(noul, "f:0", "f", 0.1, [])
    assert (a["question"], a["target"], a["label"]) == (
        {"type": "noul", "criteria": {"false": "no", "true": "yes"}, "instructions": "Is it?"}, 1.0, True,
    )  # fmt: skip
    assert (a["id"], a["family"], a["suite"], a["source"]) == (
        "f:0",
        "f:0",
        "t1",
        {"dataset": "t1", "lane": "f"},
    )
    b = converter.row_from_example(choice, "f:1", "f", 0.1, [])
    assert (b["question"]["criteria"], b["target"], b["label"]) == (
        {"x": None, "y": "why", "z": ""},
        [0, 0, 1],
        "z",
    )
    c = converter.row_from_example(score, "f:2", "f", 0.1, [])
    assert (
        c["question"]["type"] == "choice" and c["label"] == "0" and c["target"] == [0.9, 0.1, 0.0]
    )
    assert c["source"] == {"dataset": "t3", "lane": "f", "fixed_order": True, "score": True,
                           "instruction_variants": ["How much?"]}  # fmt: skip
    middle = converter.row_from_example(
        Example("score", "s", "", score.options, 1), "f:3", "f", 0.1, []
    )
    assert middle["target"] == [0.05, 0.9, 0.05]


def test_teachers_mix_into_the_target():
    ex = Example("choice", "s", "Which?", [["x", None], ["y", None]], 0)
    # two groups: weight 1 with a distribution that sums to 2 (renormalised), weight 0.5
    row = converter.row_from_example(ex, "f:0", "f", 0.0, [(1.0, [0.5, 1.5]), (0.5, [0.0, 1.0])])
    assert row["target"] == pytest.approx([(1 + 0.25) / 2.5, (0.75 + 0.5) / 2.5])
    noul = Example("noul", "s", "Is it?", [["false", ""], ["true", ""]], 0)
    assert converter.row_from_example(noul, "f:1", "f", 0.0, [(1.0, [0.2, 0.8])])[
        "target"
    ] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="teacher distribution"):
        converter.row_from_example(ex, "f:0", "f", 0.0, [(1.0, [1.0])])


def test_the_conversion_writes_what_the_trainer_reads(tiny, tmp_path):
    _, base, examples = tiny
    rng = random.Random(0)
    teacher = tmp_path / "teacher.jsonl"
    covered = sorted(rng.sample(range(len(examples)), 30))
    with open(teacher, "w") as fh:
        for i in covered:
            fh.write(
                json.dumps({"i": i, "probs": [1 / examples[i].n_options] * examples[i].n_options})
                + "\n"
            )
    out = convert(tmp_path, base, list(examples), "--teacher", "1.0", str(teacher), "--val-split-seed", "5",
                  "--val-fraction", "0.1")  # fmt: skip
    train, dev = _read(out / "train.jsonl"), _read(out / "dev.jsonl")
    manifest = json.loads((out / "train.jsonl.manifest.json").read_text())
    # v1's split, with its seed, holds out the same examples
    _kept, held = split_examples(list(examples), val_fraction=0.1, seed=5)
    assert sorted(r["id"] for r in dev) == sorted(
        f"examples.jsonl:{examples.index(e)}" for e in held
    )
    assert len(train) + len(dev) == len(examples) and manifest["train"]["rows"] == len(train)
    # the teacher reaches the training rows it covers and no dev row
    by_id = {r["id"]: r for r in train}
    for i in covered:
        row = by_id.get(f"examples.jsonl:{i}")
        if row is not None and not row["source"].get("score"):
            label = [float(j == examples[i].label) for j in range(examples[i].n_options)]
            assert row["target"] != (label[1] if row["question"]["type"] == "noul" else label)
    assert manifest["train"]["teacher_rows"] == sum(f"examples.jsonl:{i}" in by_id for i in covered)
    for row in dev:  # the dev rows keep their label's target
        i = int(row["id"].split(":")[1])
        assert (
            row["target"]
            == converter.row_from_example(examples[i], row["id"], "x", 0.1, [])["target"]
        )
    # lengths: the numbered render's tokens, the longest phrasing for training rows
    tokenizer = pointer.load_tokenizer(base, False)
    for row in train[:20]:
        ex = pointer.example_from_row(row)
        want = max(len(tokenizer(build_prompt(ex.state, ex.to_question(i))[0])["input_ids"])
                   for i in ex.all_instructions())  # fmt: skip
        assert row["source"]["input_tokens"] == want
    assert (manifest["prompt"], manifest["force_bos"], manifest["ordinal_smoothing"]) == (
        "numbered",
        False,
        0.1,
    )
    assert (
        folds.check_conversion(out / "train.jsonl", base / "tokenizer.json", "numbered")["source"]
        == "examples"
    )
    index = folds.TrainIndex(out / "train.jsonl", 0, 2048, {r["id"] for r in dev}, set(), False)
    assert len(index) == len(train)


def test_val_files_hold_out_other_examples(tiny, tmp_path):
    _, base, examples = tiny
    held = _examples(5, seed=9)
    for ex in held:
        ex.weight = 1.0
    write_jsonl(str(tmp_path / "held.jsonl"), held)
    out = convert(tmp_path, base, list(examples), "--val-files", str(tmp_path / "held.jsonl"))
    assert len(_read(out / "train.jsonl")) == len(examples)
    assert [r["id"] for r in _read(out / "dev.jsonl")] == [f"held.jsonl:{i}" for i in range(5)]


@pytest.mark.parametrize(
    "problem,message", [("weight", "weight"), ("index", "outside"), ("width", "teacher")]
)
def test_bad_inputs_stop_the_conversion(tiny, tmp_path, problem, message):
    _, base, _ = tiny
    rows = [
        Example(
            "noul",
            "s",
            "Is it?",
            [["false", ""], ["true", ""]],
            0,
            weight=0.5 if problem == "weight" else 1,
        )
    ]
    write_jsonl(str(tmp_path / "examples.jsonl"), rows)
    entry = {
        "i": 99 if problem == "index" else 0,
        "probs": [1.0] if problem == "width" else [0.5, 0.5],
    }
    (tmp_path / "teacher.jsonl").write_text(json.dumps(entry) + "\n")
    args = [
        "--examples",
        str(tmp_path / "examples.jsonl"),
        "--teacher",
        "1",
        str(tmp_path / "teacher.jsonl"),
    ]
    with pytest.raises(SystemExit, match=message):
        converter.main(
            [*args, "--base", str(base), "--out", str(tmp_path / "out"), "--prompt", "numbered"]
        )


def test_examples_convert_for_the_token_readout(gemma4_tokenizer, tmp_path):
    """With a strands-* prompt the lengths are the core's (the base needs Gemma 4's tokenizer)."""
    import os

    from training import prompt as prompts

    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_bytes(
        open(os.environ["STRANDS_DECIDER_GEMMA4_TOKENIZER"], "rb").read()
    )
    examples = _examples(20, seed=4)
    for ex in examples:
        ex.weight = 1.0
    write_jsonl(str(tmp_path / "examples.jsonl"), examples)
    converter.main(["--examples", str(tmp_path / "examples.jsonl"), "--base", str(base), "--out",
                    str(tmp_path / "out"), "--prompt", "strands-chat", "--val-fraction", "0.1"])  # fmt: skip
    for row in _read(tmp_path / "out" / "train.jsonl"):
        assert row["source"]["input_tokens"] == prompts.prompt_tokens(
            row, gemma4_tokenizer, "strands-chat"
        )
