"""publish.py against a fake Hub: release selection, folder naming, pruning. Run: pytest web/convert"""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import publish
import pytest

POINTER = {"base_model": "Qwen/Qwen3.5-2B-Base", "head_type": "pointer", "use_lora": True}
FILES = {f: f"blob-{f}" for f in (*publish.INPUTS, "hobson_config.json", "README.md")}


class FakeHub:
    """Releases: {repo: (day created, {file: blob id}, decider config)}; weights repo: a set of paths."""

    def __init__(self, releases=None, weights=(), commits=1):
        self.releases, self.files, self.commits, self.calls = (
            releases or {},
            set(weights),
            commits,
            [],
        )

    def list_models(self, author, filter, full):
        return [
            SimpleNamespace(id=r, created_at=datetime(2026, 10, day))
            for r, (day, _, _) in self.releases.items()
            if r.startswith(author + "/")
        ]

    def model_info(self, repo, files_metadata):
        files = self.releases[repo][1]
        siblings = [SimpleNamespace(rfilename=f, blob_id=b) for f, b in files.items()]
        return SimpleNamespace(sha=f"{len(repo):040d}", siblings=siblings)

    def download(self, repo, filename, tmp):
        if repo == "Qwen/Qwen3.5-2B-Base":
            content = {"model_type": "qwen3_5"}
        elif repo == "Other/base":
            content = {"model_type": "llama"}
        elif filename == "provenance.json":
            content = {"base_model": "Qwen/Qwen3.5-2B-Base", "base_model_revision": "abc"}
        else:
            content = self.releases[repo][2]
        path = tmp / f"{abs(hash((repo, filename)))}.json"
        path.write_text(json.dumps(content))
        return str(path)

    def repo_exists(self, repo):
        return bool(self.files)

    def file_exists(self, repo, path):
        return path in self.files

    def list_repo_files(self, repo):
        return sorted(self.files)

    def list_repo_commits(self, repo):
        return [None] * self.commits

    def delete_folder(self, folder, repo_id, commit_message):
        self.calls.append(("delete", folder))
        self.files = {f for f in self.files if not f.startswith(folder + "/")}
        self.commits += 1

    def super_squash_history(self, repo_id, commit_message):
        self.calls.append(("squash",))
        self.commits = 1


@pytest.fixture
def hub(monkeypatch, tmp_path):
    def make(**kw):
        h = FakeHub(**kw)
        monkeypatch.setattr(
            publish, "hf_hub_download", lambda repo, f, revision: h.download(repo, f, tmp_path)
        )
        return h

    return make


def org(name):
    return f"StrandsAgents/{name}"


def test_newest_convertible_release_wins(hub):
    h = hub(
        releases={
            org("strands-decider-2B-hobson-v19"): (1, FILES, POINTER),
            org("strands-decider-2B-hobson-v20"): (9, FILES, POINTER),
            org("strands-decider-2B-hobson-v21-parent"): (12, FILES, POINTER),  # a parent
            org("strands-decider-2B-webgpu"): (13, FILES, POINTER),  # the converted weights
            org("strands-decider-2B-slot"): (14, FILES, {**POINTER, "head_type": "slot"}),
            org("strands-decider-2B-llama"): (15, FILES, {**POINTER, "base_model": "Other/base"}),
            org("strands-decider-2B-uploading"): (16, {"README.md": "x"}, POINTER),
            org("other-model"): (17, FILES, POINTER),
        }
    )
    r = publish.resolve(h)
    assert r["release"] == org("strands-decider-2B-hobson-v20")
    assert r["folder"].startswith("strands-decider-2B-hobson-v20/")
    assert r["folder"].endswith("/" + publish.converter_id())
    assert r["exists"] == "false"
    assert r["url"].endswith(f"/resolve/main/{r['folder']}/")


def test_folder_follows_the_files_convert_reads(hub):
    rel = org("strands-decider-2B-x")
    h = hub(releases={rel: (1, dict(FILES), POINTER)})
    first = publish.resolve(h, rel)["folder"]
    h.releases[rel][1]["README.md"] = "edited card"
    assert publish.resolve(h, rel)["folder"] == first
    h.releases[rel][1]["head.safetensors"] = "retrained head"
    assert publish.resolve(h, rel)["folder"] != first


def test_existing_folder_and_unconvertible_explicit_release(hub):
    rel, dense = org("strands-decider-2B-x"), org("strands-decider-2B-y")
    h = hub(releases={rel: (1, FILES, POINTER), dense: (2, FILES, {**POINTER, "use_lora": False})})
    h.files = {f"{publish.resolve(h, rel)['folder']}/manifest.json"}
    assert publish.resolve(h, rel)["exists"] == "true"
    with pytest.raises(SystemExit):
        publish.resolve(h, dense)


def test_served_folder_reads_the_built_page():
    html = '<meta name="decider-weights" content="https://huggingface.co/o/r/resolve/main/rel/abc/123/">'
    assert publish.served_folder(html) == "rel/abc/123"
    assert publish.served_folder('<meta name="decider-weights" content="weights/">') is None


def test_prune_keeps_current_and_previous_then_squashes(hub):
    h = hub(weights={f"{f}/manifest.json" for f in ("a/1/x", "b/2/x", "c/3/x")} | {"README.md"})
    assert publish.prune(h, ["c/3/x", "b/2/x"]) == ["a/1/x"]
    assert h.calls == [("delete", "a/1/x"), ("squash",)]
    assert "README.md" in h.files


def test_prune_squashes_a_history_left_from_a_failed_run(hub):
    h = hub(weights={"a/1/x/manifest.json"}, commits=3)
    assert publish.prune(h, ["a/1/x"]) == []
    assert h.calls == [("squash",)]
    assert publish.prune(h, ["a/1/x"]) == []
    assert h.calls == [("squash",)]  # one commit now: nothing to do


def test_prune_refuses_when_no_kept_folder_exists(hub):
    h = hub(weights={"a/1/x/manifest.json"})
    with pytest.raises(SystemExit):
        publish.prune(h, ["typo/1/x"])
    assert h.calls == []
