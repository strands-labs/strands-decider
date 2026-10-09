"""The corpus benchmark's record mapping must match the server's API shape.

bench/bench_corpus.py drives a live server, which pytest will not assume; these tests
pin the pure half -- corpus record -> API question, and answer -> correct/incorrect --
so a schema drift in either the corpora or the API fails here instead of silently
producing a benchmark that measures the wrong thing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1] / "bench" / "bench_corpus.py"
_spec = importlib.util.spec_from_file_location("bench_corpus", _BENCH)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

CHOICE = {  # shape of a data/synthetic choice record
    "kind": "choice", "state": "Chapter 4: Claims",
    "instructions": "What payment method must be used?",
    "options": [["wire", "mandated"], ["check", None], ["unknown", "not stated"]],
    "label": "2",
    "instruction_variants": ["Which disbursement method applies?"],
}
NOUL = {  # noul records carry true/false options and a 0/1 label
    "kind": "noul", "state": "Policy 7.1",
    "instructions": "Is this claim eligible for expedited processing?",
    "options": [["false", "a condition is unmet"], ["true", "all conditions hold"]],
    "label": "1",
}


def test_a_choice_record_maps_to_a_choice_question_with_descriptions():
    state, questions = bench.record_questions(CHOICE)
    assert state == CHOICE["state"]
    q = questions["q"]
    assert q["type"] == "choice"
    assert q["instructions"] == CHOICE["instructions"]
    assert q["criteria"] == {"wire": "mandated", "check": None, "unknown": "not stated"}
    assert bench.expected_answer(CHOICE) == "unknown"


def test_a_noul_record_maps_to_a_bare_noul_question():
    _, questions = bench.record_questions(NOUL)
    assert questions["q"] == {"type": "noul", "instructions": NOUL["instructions"]}
    assert bench.expected_answer(NOUL) is True


def test_variant_picks_a_paraphrase_and_zero_is_the_original():
    _, zero = bench.record_questions(CHOICE, variant=0)
    _, first = bench.record_questions(CHOICE, variant=1)
    assert zero["q"]["instructions"] == CHOICE["instructions"]
    assert first["q"]["instructions"] == "Which disbursement method applies?"
    _, no_variants = bench.record_questions(NOUL, variant=3)  # no variants: unchanged
    assert no_variants["q"]["instructions"] == NOUL["instructions"]


def test_check_scores_choice_and_noul_answers_against_the_label():
    good, conf = bench.check({"choice": "unknown", "confidence": 0.9}, CHOICE)
    assert good and 0.85 < conf <= 0.95
    bad, _ = bench.check({"choice": "wire", "confidence": 0.4}, CHOICE)
    assert not bad
    yes, strong = bench.check({"noul": 0.95}, NOUL)  # label 1: true
    no, _ = bench.check({"noul": 0.2}, NOUL)
    assert yes and not no and strong > 0.8


def test_score_questions_carry_a_rubric_over_the_real_state():
    state, questions = bench.score_question(CHOICE)
    assert state == CHOICE["state"]
    q = questions["q"]
    assert q["type"] == "score"
    assert q["criteria"] == ["confusing", "mixed", "clear"]
    assert q["instructions"]


def test_load_records_reads_the_eval_corpora_with_limit():
    records = bench.load_records()
    assert records, "data/synthetic/*_eval.jsonl not found"
    assert all(r["kind"] in ("noul", "choice") for r in records)
    assert all(set(r) >= {"kind", "state", "instructions", "options", "label"} for r in records)
    limited = bench.load_records(50)
    assert len(limited) == 50
    # every sampled record must serialise: the benchmark posts it as JSON
    json.dumps(limited[0])
