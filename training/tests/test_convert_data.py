import json

import pytest

from training import folds
from training.convert import convert_row


def unified(**kw):
    row = {
        "id": "r1",
        "lane": "lane",
        "source": "src",
        "group": "g1",
        "type": "choice",
        "state": "s",
        "instructions": "Pick.",
        "options": [{"key": "a", "text": "A"}, {"key": "b", "text": None}],
        "target": [0.25, 0.75],
        "meta": {},
    }
    row.update(kw)
    return row


def test_choice_row():
    ex, why = convert_row(unified(), "as-choice")
    assert why is None
    assert ex["question"] == {
        "type": "choice",
        "criteria": {"a": "A", "b": None},
        "instructions": "Pick.",
    }
    assert ex["target"] == [0.25, 0.75] and ex["label"] == "b"
    assert (
        ex["source"] == {"dataset": "src", "lane": "lane", "mix_id": "r1"} and ex["family"] == "g1"
    )


def test_noul_row():
    ex, _ = convert_row(
        unified(
            type="noul",
            options=[{"key": "false"}, {"key": "true", "text": "yes"}],
            target=[0.8, 0.2],
            meta={"label": "true"},
        ),
        "as-choice",
    )
    assert ex["question"]["criteria"] == {"false": None, "true": "yes"}
    assert ex["target"] == 0.2 and ex["label"] is True


def test_score_rows_become_fixed_order_choices():
    row = unified(
        type="score",
        options=[{"key": "0", "text": "low"}, {"key": "1", "text": "high"}],
        target=[0.3, 0.7],
    )
    ex, _ = convert_row(row, "as-choice")
    assert ex["question"]["type"] == "choice" and list(ex["question"]["criteria"]) == [
        "low",
        "high",
    ]
    assert ex["source"]["fixed_order"] is True and ex["label"] == "high"
    assert convert_row(row, "drop") == (None, "score_dropped")


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"options": [{"key": "a"}, {"key": "a"}]}, "duplicate_option_keys"),
        ({"target": [0.5, 0.6]}, "bad_target"),
        ({"target": [1.0]}, "bad_target"),
        ({"type": "noul"}, "noul_keys"),
        ({"type": "rank"}, "unknown_type"),
        (
            {"options": [{"key": f"k{i}"} for i in range(256)], "target": [1 / 256] * 256},
            "option_count",
        ),
    ],
)
def test_rows_that_are_dropped(change, reason):
    assert convert_row(unified(**change), "as-choice") == (None, reason)


def rows_file(tmp_path, rows, name="train.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def train_row(i, n=100, **kw):
    row = {
        "id": f"t{i}",
        "suite": "s",
        "family": f"f{i}",
        "state": "x",
        "question": {"type": "choice", "criteria": {"a": None, "b": None}},
        "target": "a",
        "label": "a",
        "source": {"dataset": "d", "input_tokens": n},
    }
    row.update(kw)
    return row


def test_index_drops_long_and_image_rows(tmp_path):
    rows = [
        train_row(0),
        train_row(1, n=20000),
        train_row(2, images=["a.png"]),
        train_row(3, n=16368),
    ]
    index = folds.TrainIndex(rows_file(tmp_path, rows), 0, 16384, set(), set(), False)
    assert len(index) == 2 and index.lengths == [116, 16384]
    assert index.dropped == {"over_max_length": 1, "image_rows": 1}
    assert [r["id"] for r in index.read([0, 1]).values()] == ["t0", "t3"]


@pytest.mark.parametrize(
    "rows,protected,families,message",
    [
        ([train_row(0), train_row(0)], set(), set(), "Duplicate"),
        ([train_row(0)], {"t0"}, set(), "held-out"),
        ([train_row(0)], set(), {("d", "f0")}, "share"),
        ([train_row(0, target=[0.6, 0.6])], set(), set(), "not a probability distribution"),
    ],
)
def test_index_refuses(tmp_path, rows, protected, families, message):
    with pytest.raises(ValueError, match=message):
        folds.TrainIndex(rows_file(tmp_path, rows), 0, 16384, protected, families, False)


def test_index_refuses_rows_without_measured_lengths(tmp_path):
    row = train_row(0)
    del row["source"]["input_tokens"]
    with pytest.raises(ValueError, match="input_tokens"):
        folds.TrainIndex(rows_file(tmp_path, [row]), 0, 16384, set(), set(), False)


def test_family_overlap_can_be_allowed(tmp_path):
    index = folds.TrainIndex(
        rows_file(tmp_path, [train_row(0)]), 0, 16384, set(), {("d", "f0")}, True
    )
    assert index.family_overlap == 1


def test_conversion_manifest_is_checked(tmp_path):
    train = rows_file(tmp_path, [train_row(0)])
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    with pytest.raises(ValueError, match=r"no train\.jsonl\.manifest\.json"):
        folds.check_conversion(train, tok, "strands-chat")
    (tmp_path / "train.jsonl.manifest.json").write_text(
        json.dumps({"tokenizer_sha256": folds.sha256_file(tok), "prompt": "strands-chat"})
    )
    assert folds.check_conversion(train, tok, "strands-chat")["prompt"] == "strands-chat"
    with pytest.raises(ValueError, match="measured under"):
        folds.check_conversion(train, tok, "strands-plain")
    (tmp_path / "train.jsonl.manifest.json").write_text(json.dumps({"tokenizer_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="another tokenizer"):
        folds.check_conversion(train, tok, "strands-chat")


def test_read_fold_drops_images_before_the_limit(tmp_path):
    rows = [train_row(0, images=["x"]), train_row(1), train_row(2)]
    assert [r["id"] for r in folds.read_fold(rows_file(tmp_path, rows, "dev.jsonl"), 1)] == ["t1"]
