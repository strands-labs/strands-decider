"""The local scorers of evaluation/: the JevBench v1.5-rule proxy (jevbench/v15_proxy.py)
and the paired image comparison (vision/compare.py). Standard library and synthetic rows."""
from __future__ import annotations

import json

import pytest
from jevbench import v15_proxy as P
from vision import compare as C


def _yn(task: str, p_yes: float, correct: bool, family: str = "policy") -> dict:
    return {"task_id": task, "family": family, "probs": {"yes": p_yes, "no": 1 - p_yes},
            "predicted": "yes" if p_yes >= 0.5 else "no", "correct": correct, "ordinal_ev": None}


def _choice(task: str, correct: bool, n: int = 4) -> dict:
    return {"task_id": task, "family": "routing", "probs": {f"o{i}": 1 / n for i in range(n)},
            "correct": correct, "ordinal_ev": None}


def _score(task: str, probs: list[float], gold: int) -> dict:
    """A score result with its task's gold level attached, as `load` attaches it."""
    p = {str(i): v for i, v in enumerate(probs)}
    top = max(p, key=lambda k: p[k])
    return {"task_id": task, "family": "rubric", "probs": p, "predicted": top,
            "correct": top == str(gold), "ordinal_ev": sum(i * v for i, v in enumerate(probs)),
            "expected": gold, "labels": list(p)}


def test_yes_no_answers_inside_the_band_count_as_wrong():
    assert P.task_score(_yn("a", 0.85, True)) == 1.0
    assert P.task_score(_yn("a", 0.70, True)) == -1.0  # right, but an abstention under v1.5
    assert P.task_score(_yn("a", 0.80, True)) == 1.0  # the band is open: 0.8 is decisive
    assert P.task_score(_yn("a", 0.20, True)) == 1.0
    assert P.task_score(_yn("a", 0.21, True)) == -1.0
    assert P.task_score(_choice("c", True)) == 1.0
    assert P.task_score(_choice("c", False)) == pytest.approx(-1 / 3)  # chance-corrected


def test_proxy_weighs_the_three_types_equally():
    # uniform over 5 levels, gold 0: expected level 2, nMAE 0.5 = chance's 0.5 -> 0
    rows = [_yn("a", 0.9, True), _yn("b", 0.5, True), _choice("c", True), _score("d", [0.2] * 5, 0)]
    per, intelligence = P.proxy(rows)
    assert per == {"noul": 0.0, "choice": 100.0, "score": 0.0}
    assert intelligence == pytest.approx(100 / 3)
    m = P.metrics(rows)
    assert (m["yes_no"], m["yes_no_in_band"], m["band_correct"], m["n_correct"]) == (2, 1, 1, 4)
    assert m["band_by_family"] == {"policy": 1}


def test_score_tasks_are_graded_by_the_expected_level_not_the_top_one():
    # Both answers put the most mass on gold level 1 (right by the top level); the spread
    # one's expected level is 1.75, the peaked one's 1.0.
    spread = [0.0, 0.45, 0.35, 0.2, 0.0]
    flat, peaked = _score("f", spread, 1), _score("p", [0.0, 1.0, 0.0, 0.0, 0.0], 1)
    assert flat["correct"] and peaked["correct"]
    chance = (1 + 0 + 1 + 2 + 3) / 5 / 4  # a uniformly random level's nMAE for gold 1
    assert P.level_errors(spread, 1) == pytest.approx((0.75 / 4, chance))
    assert P.proxy([peaked])[0]["score"] == pytest.approx(100.0)
    assert P.proxy([flat])[0]["score"] == pytest.approx(100 * (1 - (0.75 / 4) / chance))
    # pooled: the mean nMAE over the mean chance nMAE, not a mean of per-task ratios
    edge = _score("e", [1.0, 0.0, 0.0, 0.0, 0.0], 0)  # right; chance nMAE 0.5 for gold 0
    assert P.proxy([flat, edge])[0]["score"] == pytest.approx(100 * (1 - (0.75 / 4) / (chance + 0.5)))
    wrong = _score("w", [0.0, 0.0, 0.0, 0.0, 1.0], 1)
    assert P.proxy([wrong])[0]["score"] == pytest.approx(100 * (1 - 0.75 / chance))  # worse than chance
    with pytest.raises(ValueError, match=r"all\.jsonl"):
        P.proxy([{k: v for k, v in flat.items() if k not in ("expected", "labels")}])


def test_load_reads_gold_levels_from_the_runs_task_file(tmp_path):
    task = {"id": "s1", "expected": 2, "labels": ["0", "1", "2"], "question": {"type": "score"}}
    result = {"task_id": "s1", "probs": {"0": 0.1, "1": 0.2, "2": 0.7}, "ordinal_ev": 1.6, "correct": True}
    (tmp_path / "all.jsonl").write_text(json.dumps(task) + "\n")
    (tmp_path / "results.jsonl").write_text(json.dumps(result) + "\n")
    rows, summary = P.load(str(tmp_path))
    assert (rows[0]["expected"], rows[0]["labels"], summary) == (2, ["0", "1", "2"], {})
    assert P.proxy(rows)[0]["score"] == pytest.approx(100 * (1 - (0.4 / 2) / (1 / 2)))
    other = tmp_path / "tasks.jsonl"
    other.write_text(json.dumps({**task, "expected": 0}) + "\n")
    assert P.load(str(tmp_path / "results.jsonl"), str(other))[0][0]["expected"] == 0


def test_paired_bootstrap_vs_a_baseline():
    base = [_yn(f"t{i}", 0.5, True) for i in range(20)] + [_choice(f"c{i}", True) for i in range(20)]
    same = P.paired_bootstrap([base], [base], resamples=200)
    assert all(v == {"diff": 0.0, "ci95": [0.0, 0.0]} for v in same.values())
    decisive = [_yn(f"t{i}", 0.9, True) for i in range(20)] + base[20:]
    got = P.paired_bootstrap([decisive, base], [base], resamples=200)
    assert got["yes_no_in_band"]["diff"] == -10.0  # the mean of -20 and 0 over the two seeds
    assert got["intelligence_proxy"]["ci95"][0] > 0
    with pytest.raises(ValueError):
        P.paired_bootstrap([base[:-1]], [base], resamples=10)


def _nb(group: int, rights: list[bool], blind_conf: float = 0.5) -> list[dict]:
    rows = []
    for k, ok in enumerate(rights):
        rows.append({"id": f"nb-{group}-{k}", "bench": "naturalbench", "group": group, "q": k // 2, "i": k % 2,
                     "gold": 0 if ok else 1, "probs": [0.7, 0.3]})
    return rows


def test_image_comparison_counts_groups_and_items():
    # group 2 is incomplete (3 of its 4 answers), so G-Acc leaves it out
    arm = {"naturalbench": _nb(0, [True] * 4) + _nb(1, [True, True, True, False]) + _nb(2, [True] * 3)}
    base = {"naturalbench": _nb(0, [True, False, True, True]) + _nb(1, [True, False, False, False])
            + _nb(2, [False] * 3)}
    got = C.paired_bootstrap([arm], [base], resamples=200)
    assert got["naturalbench_acc"]["diff"] == round(10 / 11 - 4 / 11, 4)
    assert got["naturalbench_g_acc"]["diff"] == pytest.approx(0.5)  # group 0 all right only in the arm
    same = C.paired_bootstrap([arm], [arm], resamples=50)
    assert same["naturalbench_g_acc"] == {"diff": 0.0, "ci95": [0.0, 0.0]}
