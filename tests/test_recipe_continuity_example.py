"""Offline checks for examples/recipe_continuity: the page data, layout candidates, the linking
rule, the question shape asked, and the HTTP call, with a stub decider or a local stub server."""

from __future__ import annotations

import importlib.util
import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "recipe_continuity"

# The hand-labelled recipes on the page, each in reading order.
EXPECTED = [
    ["k01", "k02", "k03", "k04"],  # shakshuka: title, then on into the next column
    ["k05", "k06", "k07"],  # herb omelette
    ["k13"],  # three separate tips
    ["k14"],
    ["k15"],
    ["k08"],  # labneh
    ["k09"],  # manakish
    ["k10", "k11"],  # spinach pies, split mid-sentence
    ["k12"],  # mint tea
    ["k00"],  # the page header, in no recipe
]


@pytest.fixture(scope="module")
def rc() -> Any:
    spec = importlib.util.spec_from_file_location(
        "recipe_continuity", EXAMPLE / "recipe_continuity.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def page() -> dict[str, Any]:
    return json.loads((EXAMPLE / "arabic_cookbook.json").read_text(encoding="utf-8"))


def oracle(page: dict[str, Any], asked: list[str]) -> Any:
    """A stub decider that answers from the hand labels, recording each question type asked.

    It finds the boxes by matching the quoted snippets in the state against the box texts.
    """
    recipe = {b["id"]: b["recipe"] for b in page["boxes"]}
    text = {b["id"]: b["text"] for b in page["boxes"]}

    def box(snippet: str) -> str:
        return next(i for i, t in text.items() if t.startswith(snippet) or t.endswith(snippet))

    def ask(state: str, questions: dict[str, Any]) -> dict[str, Any]:
        a = box(re.search(r'Block A ends: "(.*?)"', state, re.S).group(1))  # type: ignore[union-attr]
        if "same" in questions:
            asked.append("noul")
            b = box(re.search(r'starts: "(.*?)"$', state, re.S).group(1))  # type: ignore[union-attr]
            return {"same": {"noul": 0.9 if recipe[a] == recipe[b] else 0.1}}
        asked.append("choice")
        names = [k for k in questions["next"]["criteria"] if k != "none"]
        probs = {k: 0.9 if recipe[a] == recipe[k] else 0.01 for k in names}
        probs["none"] = 0.05 if max(probs.values()) > 0.5 else 0.9
        return {"next": {"probabilities": probs}}

    return ask


@pytest.fixture
def stub_server(rc: Any, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """A local HTTP server on a free port; set seen["status"] and seen["reply"] to answer."""
    seen: dict[str, Any] = {"status": 200, "reply": {"answers": {}}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = json.dumps(seen["reply"]).encode()
            self.send_response(seen["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # keep test output quiet

    server = HTTPServer(("127.0.0.1", 0), Handler)
    poll = {"poll_interval": 0.05}  # a short poll keeps shutdown() fast
    threading.Thread(target=server.serve_forever, kwargs=poll, daemon=True).start()
    monkeypatch.setattr(rc, "URL", f"http://127.0.0.1:{server.server_port}")
    try:
        yield seen
    finally:
        server.shutdown()
        server.server_close()


def test_page_matches_the_readme(page: dict[str, Any]) -> None:
    ids = [b["id"] for b in page["boxes"]]
    labels = {b["recipe"] for b in page["boxes"]}
    assert len(ids) == len(set(ids)) == 16
    assert len(labels - {None}) == 9
    assert page["page"]["direction"] == "rtl"


def test_candidates_follow_columns_right_to_left(rc: Any, page: dict[str, Any]) -> None:
    body = [b for b in page["boxes"] if b["id"] != "k00"]
    cands = rc.candidates(body, rtl=True)
    assert cands["k02"] == ["k03"]  # directly below, same column
    assert cands["k03"] == ["k04"]  # bottom of the right-hand column -> top of the next one left
    assert cands["k15"] == ["k08"]  # bottom of the middle column -> top of the left one
    assert "k12" not in cands  # last box of the last column


def test_candidates_follow_columns_left_to_right(rc: Any) -> None:
    boxes = [
        {"id": "a", "bbox": [0, 0, 100, 50]},
        {"id": "b", "bbox": [0, 60, 100, 300]},  # under a
        {"id": "c", "bbox": [120, 0, 220, 100]},  # top of the second column
        {"id": "d", "bbox": [120, 200, 220, 300]},  # same column as c, but too far below it
        {"id": "e", "bbox": [121, 400, 220, 450]},  # a lower part of that column, 1 px off
        {"id": "f", "bbox": [240, 0, 340, 100]},  # a third column, further away
    ]
    cands = rc.candidates(boxes, rtl=False)
    assert cands["a"] == ["b"]
    assert cands["b"] == ["c", "d", "e"]  # the tops of the nearest column: a choice is needed
    assert cands["c"] == cands["d"] == cands["e"] == ["f"]
    assert "f" not in cands


def test_candidates_tolerate_ocr_jitter(rc: Any) -> None:
    boxes = [
        {"id": "a", "bbox": [500, 0, 900, 300]},
        {"id": "b", "bbox": [500, 298, 900, 400]},  # overlaps a by 2 px vertically
        {"id": "c", "bbox": [95, 0, 505, 200]},  # next column, 5 px into the gutter
    ]
    assert rc.candidates(boxes, rtl=True) == {"a": ["b"], "b": ["c"]}


def test_header_is_found_by_layout_on_a_shifted_page(rc: Any, page: dict[str, Any]) -> None:
    def shift(b: dict[str, Any]) -> dict[str, Any]:
        x0, y0, x1, y1 = b["bbox"]
        return {**b, "bbox": [x0 + 500, y0 + 500, x1 + 500, y1 + 500]}

    shifted = {"page": page["page"], "boxes": [shift(b) for b in page["boxes"]]}
    assert rc.run(shifted, "noul", oracle(page, [])) == EXPECTED
    one_column = {  # every box is wide, but none lies over two columns
        "boxes": [
            {"id": "a", "bbox": [50, 100, 950, 200], "text": "a"},
            {"id": "b", "bbox": [50, 210, 950, 300], "text": "b"},
        ]
    }
    assert rc.run(one_column, "noul", lambda s, q: {"same": {"noul": 0.9}}) == [["a", "b"]]


def test_run_leaves_the_page_header_out(rc: Any, page: dict[str, Any]) -> None:
    groups = rc.run(page, "noul", oracle(page, []))
    assert groups[-1] == ["k00"]
    assert all("k00" not in g for g in groups[:-1])


def test_settle_gives_each_box_one_next_and_one_previous(rc: Any) -> None:
    boxes = [{"id": i} for i in ("a", "b", "c", "d")]
    # b -> c loses to the stronger a -> c; a -> b loses because a already has a next box.
    found = [(0.9, "a", "c"), (0.6, "b", "c"), (0.7, "a", "b"), (0.8, "c", "d")]
    assert rc.settle(boxes, found) == [["a", "c", "d"], ["b"]]


def test_settle_never_closes_a_loop(rc: Any) -> None:
    boxes = [{"id": i} for i in ("a", "b", "c")]
    assert rc.settle(boxes, [(0.9, "a", "b"), (0.8, "b", "a")]) == [["a", "b"], ["c"]]
    assert rc.settle(boxes, [(0.9, "a", "a")]) == [["a"], ["b"], ["c"]]
    found = [(0.9, "a", "b"), (0.8, "b", "c"), (0.7, "c", "a")]
    assert rc.settle(boxes, found) == [["a", "b", "c"]]


@pytest.mark.parametrize(
    ("mode", "shapes"), [("auto", {"noul"}), ("noul", {"noul"}), ("choice", {"choice"})]
)
def test_run_rebuilds_every_recipe_with_a_perfect_decider(
    rc: Any, page: dict[str, Any], mode: str, shapes: set[str]
) -> None:
    asked: list[str] = []
    groups = rc.run(page, mode, oracle(page, asked))
    assert groups == EXPECTED  # same recipes, each in reading order
    assert set(asked) == shapes  # every box on this page has at most one candidate


def test_choice_keeps_the_top_candidate_unless_none_wins(rc: Any) -> None:
    by_id = {i: {"id": i, "bbox": [0, 0, 10, 10], "text": i} for i in ("a", "b", "c", "x", "y")}
    cands = {"a": ["b", "c"], "x": ["y", "c"]}
    answers = {
        "a": {"b": 0.25, "c": 0.45, "none": 0.30},  # c wins below 0.5: still the top option
        "x": {"y": 0.30, "c": 0.20, "none": 0.50},  # none wins: no link
    }

    def ask(state: str, questions: dict[str, Any]) -> dict[str, Any]:
        a = re.search(r'Block A ends: "(.*?)"', state).group(1)  # type: ignore[union-attr]
        assert set(questions["next"]["criteria"]) == {*cands[a], "none"}
        return {"next": {"probabilities": answers[a]}}

    assert rc.claims(by_id, cands, "choice", ask) == [(0.45, "a", "c")]


def test_auto_ranks_with_a_choice_then_decides_with_a_noul(rc: Any) -> None:
    by_id = {i: {"id": i, "bbox": [0, 0, 10, 10], "text": i} for i in ("a", "b", "c", "x", "y")}
    cands = {"a": ["b", "c"], "x": ["y"]}
    asked: list[tuple[str, str]] = []

    def ask(state: str, questions: dict[str, Any]) -> dict[str, Any]:
        a = re.search(r'Block A ends: "(.*?)"', state).group(1)  # type: ignore[union-attr]
        if "next" in questions:
            asked.append(("choice", a))
            assert set(questions["next"]["criteria"]) == {"b", "c"}  # no "none" when ranking
            return {"next": {"probabilities": {"b": 0.3, "c": 0.7}}}
        b = re.search(r'starts: "(.*?)"$', state).group(1)  # type: ignore[union-attr]
        asked.append(("noul", f"{a}->{b}"))
        return {"same": {"noul": 0.8 if b == "c" else 0.2}}

    assert rc.claims(by_id, cands, "auto", ask) == [(0.8, "a", "c")]
    assert asked == [("choice", "a"), ("noul", "a->c"), ("noul", "x->y")]


@pytest.mark.parametrize(
    ("probs", "expected"),
    [
        ({"b": 0.5, "none": 0.5}, []),  # a tie with "none": no link
        ({"b": 0.15, "zz": 0.8, "none": 0.05}, [(0.15, "a", "b")]),  # unknown keys are ignored
        ({}, []),
    ],
)
def test_choice_edge_cases(rc: Any, probs: dict[str, float], expected: list[Any]) -> None:
    by_id = {i: {"id": i, "bbox": [0, 0, 10, 10], "text": i} for i in ("a", "b", "c")}
    ask = lambda state, questions: {"next": {"probabilities": probs}}  # noqa: E731
    assert rc.claims(by_id, {"a": ["b", "c"]}, "choice", ask) == expected


def test_ask_server_posts_to_systemone(rc: Any, stub_server: dict[str, Any]) -> None:
    stub_server["reply"] = {"answers": {"same": {"type": "noul", "noul": 0.7}}}
    questions = {"same": {"type": "noul", "instructions": "نفس الوصفة؟"}}
    answers = rc.ask_server("نص عربي", questions)  # Arabic text must survive the round trip
    assert stub_server["path"] == "/v1/systemone"
    assert stub_server["body"] == {"state": "نص عربي", "questions": questions}
    assert answers == {"same": {"type": "noul", "noul": 0.7}}


def test_ask_server_reports_http_errors(rc: Any, stub_server: dict[str, Any]) -> None:
    stub_server["status"] = 422
    stub_server["reply"] = {"detail": "bad question"}
    with pytest.raises(RuntimeError, match=r"HTTP 422.*bad question"):
        rc.ask_server("state", {})


@pytest.mark.parametrize("reply", [{"no_answers": {}}, ["not", "a", "dict"]])
def test_ask_server_reports_unexpected_replies(
    rc: Any, stub_server: dict[str, Any], reply: Any
) -> None:
    stub_server["reply"] = reply
    with pytest.raises(RuntimeError, match="unexpected reply"):
        rc.ask_server("state", {})


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://127.0.0.1", "127.0.0.1:8099"])
def test_ask_server_refuses_non_http_addresses(
    rc: Any, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setattr(rc, "URL", url)
    with pytest.raises(ValueError, match="http"):
        rc.ask_server("state", {})
