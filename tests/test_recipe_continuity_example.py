"""Offline checks for examples/recipe_continuity: layout candidates, the linking rule, the
question shape asked, and the HTTP call, with a stub decider or a local stub server."""

from __future__ import annotations

import importlib.util
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "recipe_continuity"


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
    """A stub decider that answers from the labels, recording each question type it was asked."""
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


def test_candidates_follow_columns_right_to_left(rc: Any, page: dict[str, Any]) -> None:
    text = [b for b in page["boxes"] if b["id"] != "k00"]
    cands = rc.candidates(text, rtl=True)
    assert cands["k02"] == ["k03"]  # directly below, same column
    assert cands["k03"] == ["k04"]  # bottom of the right-hand column -> top of the next one left
    assert cands["k15"] == ["k08"]
    assert "k12" not in cands  # last box of the last column


def test_settle_gives_each_box_one_continuation(rc: Any) -> None:
    boxes = [{"id": i} for i in ("a", "b", "c")]
    groups = rc.settle(boxes, [(0.9, "a", "c"), (0.6, "b", "c"), (0.7, "a", "b")])
    assert sorted(map(tuple, groups)) == [("a", "c"), ("b",)]


@pytest.mark.parametrize(
    ("mode", "shapes"), [("auto", {"noul"}), ("noul", {"noul"}), ("choice", {"choice"})]
)
def test_run_rebuilds_every_recipe_with_a_perfect_decider(
    rc: Any, page: dict[str, Any], mode: str, shapes: set[str]
) -> None:
    asked: list[str] = []
    groups = rc.run(page, mode, oracle(page, asked))
    truth: dict[str | None, set[str]] = {}
    for b in page["boxes"]:
        truth.setdefault(b["recipe"], set()).add(b["id"])
    assert sorted(map(sorted, groups)) == sorted(map(sorted, truth.values()))
    assert set(asked) == shapes  # every box on this page has at most one candidate


def test_ask_server_posts_to_systemone(rc: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen["path"] = self.path
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            reply = json.dumps({"answers": {"same": {"type": "noul", "noul": 0.7}}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, format: str, *args: Any) -> None:
            pass  # keep test output quiet

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(rc, "URL", f"http://127.0.0.1:{server.server_port}")
        questions = {"same": {"type": "noul", "instructions": "Same?"}}
        answers = rc.ask_server("some state", questions)
    finally:
        server.shutdown()
    assert seen["path"] == "/v1/systemone"
    assert seen["body"] == {"state": "some state", "questions": questions}
    assert answers == {"same": {"type": "noul", "noul": 0.7}}


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://127.0.0.1", "127.0.0.1:8099"])
def test_ask_server_refuses_non_http_addresses(
    rc: Any, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    monkeypatch.setattr(rc, "URL", url)
    with pytest.raises(ValueError, match="http"):
        rc.ask_server("state", {})
