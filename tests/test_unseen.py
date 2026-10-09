"""The unseen-family evaluation (evaluation/unseen/): the builder's draws, the answers
run.py stores, and score.py's v1.5-rule grading. Offline: synthetic dataset rows, fake
deciders, and one tiny Qwen3.5 checkpoint (tests/tiny_qwen35.py) answering as served."""

from __future__ import annotations

import json
import random
import re
import sys

import pytest
from unseen import build as B
from unseen import run as R
from unseen import score as S

from strands_decider.data.format import Example

YN = [["false", ""], ["true", ""]]


# ---- build ------------------------------------------------------------------------------


def test_choice_rows_name_repeated_options_once_and_keep_the_gold_text():
    rows = [{"question": "Where?", "answerKey": "D",
             "choices": {"label": ["A", "B", "C", "D"], "text": ["bed", "bed", "sofa", "floor"]}},
            {"question": "Which?", "answerKey": "Z", "choices": {"label": ["A"], "text": ["x"]}}]
    got = B.multiple_choice(rows, random.Random(1), "Which answer?", "unseen/x")
    assert len(got) == 1  # an answer key that is not among the labels is skipped
    assert got[0]["options"] == [["bed", "bed"], ["sofa", "sofa"], ["floor", "floor"]]
    assert got[0]["options"][got[0]["label"]][0] == "floor"
    Example.from_dict(got[0])


def test_ruletaker_draws_once_per_candidate_and_keeps_depth_five_rows_first():
    lines = [json.dumps({"task": t, "kind": "noul", "state": str(i), "instructions": "q", "options": YN,
                         "label": i % 2}) for i, t in enumerate(["ruletaker_d3"] * 3 + ["ruletaker_d5"] * 6
                                                               + ["ruletaker_natlang"] * 4 + ["emotion"])]
    rng, ref = random.Random(0), random.Random(0)
    kept = B.ruletaker(lines, rng, n=2)
    draws = [ref.random() for _ in range(10)]  # the 6 depth-5 and 4 NatLang candidates
    assert rng.random() == ref.random()  # exactly one draw per candidate, kept or not
    want = [i for i, d in zip(range(3, 13), draws, strict=True) if d < 0.5][:2]
    assert [int(r["state"]) for r in kept] == want
    assert all(r["task"].startswith("unseen/ruletaker_") for r in kept)


def test_stsb_levels_and_strategyqa_rows():
    st = [{"sentence1": "a", "sentence2": "b", "score": s} for s in (0.0, 0.31, 1.0)]
    levels = sorted(r["label"] for r in B.stsb(st, random.Random(0)))
    assert levels == [0, 2, 5] and all(len(r["options"]) == 6 for r in B.stsb(st, random.Random(0)))
    sq = [{"question": "Is ice cold?", "answer": True}, {"question": None, "answer": False}]
    got = B.strategyqa(sq, random.Random(0))
    assert [(r["label"], r["kind"]) for r in got] == [(1, "noul")]


def test_build_draws_the_families_in_order_from_one_seed(tmp_path, monkeypatch):
    data = {
        "ChilleD/StrategyQA": [{"question": f"q{i}", "answer": i % 2 == 0} for i in range(300)],
        "tau/commonsense_qa": [{"question": f"c{i}", "answerKey": "A",
                                "choices": {"label": ["A", "B"], "text": [f"x{i}", f"y{i}"]}} for i in range(200)],
        "allenai/ai2_arc": [{"question": f"a{i}", "answerKey": "B",
                             "choices": {"label": ["A", "B", "C"], "text": ["1", "2", "3"]}} for i in range(200)],
        "sentence-transformers/stsb": [{"sentence1": "s", "sentence2": "t", "score": i / 300} for i in range(300)],
    }
    monkeypatch.setattr(B, "_load", lambda name, split, config=None: [dict(r) for r in data[name]])
    held = tmp_path / "holdout.jsonl"
    held.write_text("".join(json.dumps({"task": "ruletaker_d5", "kind": "noul", "state": str(i),
                                        "instructions": "q", "options": YN, "label": 0}) + "\n" for i in range(600)))
    rows = B.build(str(held))
    tasks = [r["task"] for r in rows]
    assert [tasks.count(t) for t in ("unseen/strategyqa", "unseen/ruletaker_d5", "unseen/commonsenseqa",
                                     "unseen/arc_challenge", "unseen/stsb")] == [250, 250, 150, 150, 250]
    assert tasks == sorted(tasks, key=["unseen/strategyqa", "unseen/ruletaker_d5", "unseen/commonsenseqa",
                                       "unseen/arc_challenge", "unseen/stsb"].index)
    assert B.build(str(held)) == rows and B.build(str(held), seed=1) != rows


# ---- run --------------------------------------------------------------------------------


def _ex(kind: str) -> Example:
    opts = {"noul": YN, "choice": [["red", ""], ["blue", ""], ["green", ""]],
            "score": [["0", "low"], ["1", "mid"], ["2", "high"]]}[kind]
    return Example(kind, "A state.", "A question?", opts, 1, task=f"unseen/{kind}")


def test_answers_are_stored_in_the_rows_option_order():
    assert R.probabilities(_ex("noul"), {"type": "noul", "noul": 0.9}) == pytest.approx([0.1, 0.9])
    assert R.probabilities(_ex("choice"), {"probabilities": {"green": 0.2, "red": 0.5, "blue": 0.3}}) == [0.5, 0.3, 0.2]
    assert R.probabilities(_ex("score"), {"probabilities": {"2": 0.6, "0": 0.1, "1": 0.3}}) == [0.1, 0.3, 0.6]
    with pytest.raises(ValueError, match="no probability"):
        R.probabilities(_ex("choice"), {"probabilities": {"red": 1.0}})


def _rows(tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(e.to_json() + "\n" for e in map(_ex, ("noul", "choice", "score"))))
    return rows


def test_a_python_adapter_answers_every_row(tmp_path, monkeypatch):
    mod = tmp_path / "fake_decider.py"
    mod.write_text(
        "def make():\n"
        "    def ask(state, q):\n"
        "        if q['type'] == 'noul':\n"
        "            return {'noul': 0.25}\n"
        "        names = list(q['criteria']) if q['type'] == 'choice' else [str(i) for i in range(len(q['criteria']))]\n"
        "        return {'probabilities': {n: 1 / len(names) for n in names}}\n"
        "    return ask\n")
    monkeypatch.setattr(sys, "path", list(sys.path))
    out = tmp_path / "out"
    R.main(["--rows", str(_rows(tmp_path)), "--adapter", "python:fake_decider:make", "--path", str(tmp_path),
            "--out", str(out)])
    preds = [json.loads(x) for x in (out / "predictions.jsonl").read_text().splitlines()]
    assert [(p["kind"], p["gold"], len(p["probs"])) for p in preds] == [("noul", 1, 2), ("choice", 1, 3), ("score", 1, 3)]
    assert preds[0]["probs"] == pytest.approx([0.75, 0.25])
    meta = json.loads((out / "run_meta.json").read_text())
    assert meta["adapter"] == "python:fake_decider:make" and meta["n"] == 3 and len(meta["rows_sha256"]) == 64


def test_a_checkpoint_answers_as_the_engine_serves(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
        pytest.skip("the tiny Qwen3.5 needs transformers >= 5.18")
    from tiny_qwen35 import save_base, save_checkpoint

    from strands_decider.infer import load_engine
    from strands_decider.schema import SystemOneRequest

    tmp = tmp_path_factory.mktemp("unseen")
    ckpt = save_checkpoint(save_base(str(tmp_path_factory.mktemp("base"))), str(tmp / "ckpt"))
    rows = _rows(tmp)
    R.main(["--rows", str(rows), "--checkpoint", ckpt, "--device", "cpu", "--out", str(tmp / "out")])
    preds = [json.loads(x) for x in (tmp / "out" / "predictions.jsonl").read_text().splitlines()]
    engine = load_engine(ckpt, device="cpu")
    ex = _ex("choice")
    want = engine.evaluate(SystemOneRequest(state=ex.state, questions={"q": ex.to_question()})).answers["q"]
    assert preds[1]["probs"] == [want.probabilities[n] for n, _ in ex.options]
    meta = json.loads((tmp / "out" / "run_meta.json").read_text())
    assert meta["temperature_by_kind"] == {"noul": 0.9, "choice": 0.7} and "torch" in meta["versions"]


# ---- score ------------------------------------------------------------------------------


def _p(i, kind, gold, probs, task="t"):
    return {"id": i, "task": task, "kind": kind, "gold": gold, "probs": probs}


def test_competence_follows_the_v15_rules():
    rows = [_p(0, "noul", 1, [0.1, 0.9]), _p(1, "noul", 1, [0.3, 0.7]), _p(2, "noul", 0, [0.9, 0.1]),
            _p(3, "noul", 0, [0.1, 0.9]),
            _p(4, "choice", 0, [0.5, 0.3, 0.2]), _p(5, "choice", 1, [0.5, 0.3, 0.2]),
            _p(6, "score", 0, [1.0, 0.0, 0.0])]
    per, intelligence = S.competence(rows)
    # yes/no: right, in the band (wrong), right, wrong -> (1 - 1 + 1 - 1) / 4
    assert per["noul"] == pytest.approx(0.0)
    assert per["choice"] == pytest.approx(100 * (1 - 0.5) / 2)  # 1 and -1/2
    assert per["score"] == pytest.approx(100.0)
    assert intelligence == pytest.approx((0 + 25 + 100) / 3)
    assert S.summary(rows)["yes_no_in_band"] == 1


def test_overconfidence_and_ece():
    rows = [_p(0, "noul", 1, [0.1, 0.9]), _p(1, "noul", 0, [0.1, 0.9])]
    assert S.yes_no_overconfidence(rows) == pytest.approx(0.9 - 0.5)
    assert S.ece(rows) == pytest.approx(0.4)  # one bin: confidence 0.9, accuracy 0.5
    assert S.yes_no_overconfidence([_p(0, "choice", 0, [1.0, 0.0])]) is None


def test_vs_compares_row_by_row(tmp_path):
    base = [_p(i, "noul", 1, [0.5, 0.5]) for i in range(20)] + [_p(20 + i, "choice", 0, [0.6, 0.4]) for i in range(20)]
    sure = [_p(i, "noul", 1, [0.1, 0.9]) for i in range(20)] + base[20:]
    paths = []
    for name, rows in (("sure", sure), ("base", base)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        paths.append(str(tmp_path / name))
    S.main([paths[0], "--vs", paths[1], "--resamples", "200", "--json", str(tmp_path / "r.json")])
    vs = json.loads((tmp_path / "r.json").read_text())["vs"]
    assert vs["competence_noul"]["diff"] == 200.0  # -100 (all in the band) to +100
    assert vs["intelligence"]["ci95"][0] > 0
