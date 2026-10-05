"""The image data builders (data/image/) and the image dedupe check (data/checks/), on
synthetic and rendered images; nothing is downloaded.

* the builders write rows that load as Examples, with an image that decodes as the
  server decodes it, a label inside the options, and the provenance fields training reads;
* yes/no questions come both as nouls with the server's default criteria and as yes/no
  choices, and a row with duplicate option names is refused;
* the dedupe check hashes an image as the server decodes it, finds a resized copy, and
  marks an unreadable file instead of failing.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import os
import random
import sys

import pytest

from strands_decider.data.format import Example
from strands_decider.prompting import NOUL_DEFAULT_CRITERIA
from strands_decider.vision import read_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data", "image"))
sys.path.insert(0, os.path.join(ROOT, "data", "checks"))
common = importlib.import_module("common")

pytestmark = pytest.mark.skipif(importlib.util.find_spec("PIL") is None,
                                reason='the image builders need Pillow: pip install "strands-decider[vision]"')


def _check_rows(root, rows: list[dict]) -> None:
    for r in rows:
        ex = Example.from_dict(r)
        assert 0 <= ex.label < ex.n_options and r["ablation"] is False
        assert r["source"] and r["source_id"] and r["images"]
        for p in r["images"]:
            with open(os.path.join(root, p), "rb") as fh:
                read_image(fh.read())
        if ex.kind == "noul":
            assert r["options"] == [[k, v] for k, v in NOUL_DEFAULT_CRITERIA.items()]  # as served bare


def test_yesno_rows_are_nouls_or_yes_no_choices():
    rng = random.Random(1)
    rows = [common.yesno("Is it red?", yes, rng, task="t", source="s", images=["a.png"], source_id="a")
            for yes in (True, False) * 50]
    assert {r["kind"] for r in rows} == {"noul", "choice"}
    for r, yes in zip(rows, (True, False) * 50, strict=True):
        answer = r["options"][r["label"]][0]
        assert answer == ("true" if yes else "false") if r["kind"] == "noul" else answer == ("yes" if yes else "no")
    with pytest.raises(ValueError):
        common.row("choice", "q", [["a", ""], ["a", ""]], 0, task="t", source="s", images=[], source_id="x")


def test_document_builder_rows(tmp_path):
    documents = importlib.import_module("documents")
    os.makedirs(tmp_path / "docs")
    rows = [r for i in range(6) for r in documents.make((i, str(tmp_path), 1000 + i))]
    assert rows
    _check_rows(tmp_path, rows)


def test_chart_builder_rows(tmp_path):
    pytest.importorskip("matplotlib")
    charts = importlib.import_module("charts")
    os.makedirs(tmp_path / "charts")
    rows = [r for i in range(8) for r in charts.make((i, str(tmp_path), 7_000_000 + i))]
    assert rows
    _check_rows(tmp_path, rows)


def test_dedupe_hashes_as_the_server_decodes_and_finds_a_resized_copy(tmp_path):
    pytest.importorskip("imagehash")
    from PIL import Image, ImageDraw

    dedupe = importlib.import_module("dedupe_images")
    img = Image.new("RGB", (320, 240), "white")
    ImageDraw.Draw(img).rectangle((40, 30, 200, 180), fill="navy")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    small = io.BytesIO()
    img.resize((160, 120)).save(small, "JPEG")
    (tmp_path / "bad.png").write_bytes(b"not an image")
    _, a = dedupe.ph(("a", buf.getvalue()))
    _, b = dedupe.ph(("b", small.getvalue()))
    assert bin(a ^ b).count("1") <= dedupe.THRESH
    assert dedupe.ph(("bad", str(tmp_path / "bad.png"))) == ("bad", None)
