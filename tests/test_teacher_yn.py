"""The stronger yes/no teacher (src/strands_decider/data/teacher_yn.py), without a model:
which rows it labels, under which index, and which labels the teacher file keeps."""
from __future__ import annotations

import json

import pytest

from strands_decider.data import teacher, teacher_yn
from strands_decider.data.format import Example, write_jsonl

YN = [["false", ""], ["true", ""]]


def _files(tmp_path):
    a = [Example(kind="noul", state="s", instructions="q", options=YN, label=1),
         Example(kind="choice", state="s", instructions="q", options=[["a", ""], ["b", ""]], label=0),
         Example(kind="noul", state="s", instructions="q", options=YN, label=0)]
    b = [Example(kind="noul", state="s", instructions="q", options=YN, label=1)]
    c = [Example(kind="noul", state="s", instructions="q", options=YN, label=0),
         Example(kind="score", state="s", instructions="q", options=[["0", ""], ["1", ""]], label=1)]
    paths = []
    for name, rows in (("a", a), ("b", b), ("c", c)):
        write_jsonl(str(tmp_path / f"{name}.jsonl"), rows)
        paths.append(str(tmp_path / f"{name}.jsonl"))
    return paths


def test_selects_the_yes_no_rows_of_the_sources_by_concatenation_index(tmp_path):
    a, b, c = _files(tmp_path)
    idx, rows = teacher_yn.select([a, b, c], [a, c])
    assert idx == [0, 2, 4]  # b's row (index 3) is not a source; 1 and 5 are not yes/no
    assert all(ex.kind == "noul" for ex in rows)
    with pytest.raises(ValueError, match="not among the train files"):
        teacher_yn.select([a, b], [c])


def test_label_writes_concatenation_indices_and_resumes(tmp_path, monkeypatch):
    a, b, c = _files(tmp_path)
    monkeypatch.setattr(teacher, "load", lambda model, revision: (None, None))
    calls = []

    def fake_label(model, tok, examples, *, sink, skip, **kw):
        calls.append(sorted(skip))
        for j in range(len(examples)):
            if j not in skip:
                sink(j, [0.25, 0.75])
                if len(calls) == 1:
                    raise KeyboardInterrupt  # the first run stops after one row

    monkeypatch.setattr(teacher, "label", fake_label)
    out = tmp_path / "raw.jsonl"
    args = ["label", "--out", str(out), "--train-files", a, b, c, "--sources", b, c]
    with pytest.raises(KeyboardInterrupt):
        teacher_yn.main(args)
    teacher_yn.main(args)
    assert calls == [[], [0]]  # the rerun skips the row the first run labelled
    assert [json.loads(x)["i"] for x in out.read_text().splitlines()] == [3, 4]


def test_build_keeps_agreeing_rows_over_the_base(tmp_path):
    a, b, c = _files(tmp_path)
    raw = tmp_path / "raw.jsonl"
    # rows 0 and 4 agree with gold, row 2 does not; row 3 also has a base label
    raw.write_text("".join(json.dumps({"i": i, "probs": p}) + "\n" for i, p in
                           ((0, [0.1, 0.9]), (2, [0.3, 0.7]), (3, [0.2, 0.8]), (4, [0.6, 0.4]))))
    base = tmp_path / "base.jsonl"
    base.write_text("".join(json.dumps({"i": i, "probs": p}) + "\n" for i, p in
                            ((1, [0.5, 0.5]), (3, [0.4, 0.6]))))
    out = tmp_path / "teacher.jsonl"
    teacher_yn.main(["build", "--raw", str(raw), "--base", str(base), "--out", str(out),
                     "--train-files", a, b, c])
    got = [json.loads(x) for x in out.read_text().splitlines()]
    assert got == [{"i": 0, "probs": [0.1, 0.9]}, {"i": 1, "probs": [0.5, 0.5]},
                   {"i": 3, "probs": [0.2, 0.8]}, {"i": 4, "probs": [0.6, 0.4]}]


def test_replay_draws_covered_rows_with_their_teacher(tmp_path):
    a, b, c = _files(tmp_path)
    tf = tmp_path / "teacher.jsonl"
    # rows 0, 3, 4 are yes/no; 1 and 5 are not; row 2 (yes/no) has no teacher label
    tf.write_text("".join(json.dumps({"i": i, "probs": p}) + "\n" for i, p in
                          ((0, [0.1, 0.9]), (1, [0.7, 0.3]), (3, [0.2, 0.8]), (4, [0.6, 0.4]),
                           (5, [0.5, 0.5]))))
    out = tmp_path / "replay.jsonl"
    teacher_yn.main(["replay", "--teacher", str(tf), "--n-yes-no", "2", "--n-other", "2",
                     "--out", str(out), "--train-files", a, b, c])
    got = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r["kind"] for r in got].count("noul") == 2 and len(got) == 4
    by_teacher = {tuple(r["teacher"]) for r in got}
    assert {(0.7, 0.3), (0.5, 0.5)} <= by_teacher  # both covered non-yes/no rows
    assert by_teacher <= {(0.1, 0.9), (0.7, 0.3), (0.2, 0.8), (0.6, 0.4), (0.5, 0.5)}
    assert all(Example.from_dict(r).kind == r["kind"] for r in got)  # still an Example
    same = tmp_path / "again.jsonl"
    teacher_yn.main(["replay", "--teacher", str(tf), "--n-yes-no", "2", "--n-other", "2",
                     "--out", str(same), "--train-files", a, b, c])
    assert same.read_text() == out.read_text()  # seeded
    with pytest.raises(ValueError, match="covers 3 yes/no"):
        teacher_yn.replay_rows([a, b, c], str(tf), 4, 0)
