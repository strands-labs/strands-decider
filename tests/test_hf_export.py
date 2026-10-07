"""The Hugging Face-format export: no pickles, same tensors, a card the Hub's own parsers
read, deterministic bytes, and JevBench results tied to the checkpoint they served."""

import json
import os
import re
import subprocess
import tempfile

import pytest
import torch

from strands_decider import hf_export
from strands_decider.modeling import load_head_state


def _ckpt(root, window=4096):
    os.makedirs(os.path.join(root, "lora"))
    json.dump({"base_model": hf_export.BASE_MODEL, "head_type": "pointer", "num_slots": 24,
               "max_length": window},
              open(os.path.join(root, "hobson_config.json"), "w"), indent=2)
    json.dump({"seed": 0}, open(os.path.join(root, "train_config.json"), "w"))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        open(os.path.join(root, f), "w").write("{}")
    g = torch.Generator().manual_seed(0)
    head = {"norm.weight": torch.randn(8, generator=g), "q.weight": torch.randn(4, 8, generator=g),
            "k.bias": torch.randn(4, generator=g)}
    torch.save(head, os.path.join(root, "slot_head.pt"))
    from safetensors.torch import save_file

    save_file({"base_model.model.layers.0.lora_A.weight": torch.randn(2, 8, generator=g)},
              os.path.join(root, "lora", "adapter_model.safetensors"))
    json.dump({"peft_type": "LORA", "r": 2}, open(os.path.join(root, "lora", "adapter_config.json"), "w"))
    return head


def _run(root):
    os.makedirs(os.path.join(root, "logs"))
    stages = [{"stage": "train", "start_utc": "2026-09-28T00:00:00Z", "end_utc": "2026-09-28T00:30:00Z",
               "wall_s": 1800, "git_rev": "abc1234", "host_shape": "p5.48xlarge", "gpu_name": "H100"}]
    open(os.path.join(root, "stages.jsonl"), "w").write("".join(json.dumps(s) + "\n" for s in stages))
    open(os.path.join(root, "logs", "eval.log"), "w").write(
        "overall n=6,000 acc=0.641 ece=0.052 nll=0.886 T={}\n"
        "ckpt  (0 rows over the window skipped)\n  musique                    0.884  (n=1,199)\n"
        "ckpt  (0 rows over the window skipped)\n  gen:numbers                0.500  (n=2)\n"
        "ckpt  (0 rows over the window skipped)\n  adequacy_hs2               0.700  (n=10)\n"
        "ckpt  (0 rows over the window skipped)\n  adequacy_hs2               0.800  (n=20)\n")


def _jev(root, fp, window):
    os.makedirs(root)
    json.dump({"checkpoint_files_sha256_16": fp}, open(os.path.join(root, "run_meta.json"), "w"))
    json.dump({"n_correct": 167, "n_attempted": 231, "accuracy": 167 / 231, "brier_mean": 0.35,
               "ece": {"ece": 0.05}}, open(os.path.join(root, "summary.json"), "w"))
    json.dump({"max_length": window}, open(os.path.join(root, "health.json"), "w"))


def _export(tmp_path, ckpt, out, jdirs=(), wcfg=(), replace=False, repo_url=hf_export.REPO_URL,
            hub_id=None, redact=(), reports=None):
    run = tmp_path / "run"
    if not run.exists():
        _run(str(run))
    stage = tmp_path / "stage"
    import shutil

    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    hf_export.build(str(ckpt), str(stage), str(run), reports, list(map(str, jdirs)), list(wcfg),
                    "hobson-2b-test", "test-run1", "final", repo_url, hub_id, list(redact))
    return hf_export.publish(str(stage), str(out), replace)


def test_export_is_pickle_free_loadable_and_deterministic(tmp_path):
    ckpt = tmp_path / "ckpt"
    head = _ckpt(str(ckpt))
    jev = tmp_path / "jev"
    _jev(str(jev), hf_export.fingerprint(str(ckpt)), 4096)
    out = tmp_path / "out"
    assert _export(tmp_path, ckpt, out, [jev]) == "written"
    hf_export.verify(str(out))
    assert not os.path.exists(out / "slot_head.pt")
    assert not any(hf_export.is_pickle(str(out / p)) for p in hf_export.files(str(out)))
    back = load_head_state(str(out))
    assert set(back) == set(head) and all(torch.equal(back[k], head[k]) for k in head)
    assert _export(tmp_path, ckpt, out, [jev]) == "unchanged"  # same bytes twice
    from huggingface_hub import metadata_load

    meta = metadata_load(str(out / "README.md"))
    assert meta["license"] == "apache-2.0" and "license_name" not in meta
    assert meta["base_model"] == hf_export.BASE_MODEL
    summ = json.load(open(out / "eval" / "summary.json"))
    assert summ["jevbench"][0]["n_correct"] == 167
    assert summ["internal"]["eval: held-out short tasks"]["nll"] == 0.886
    assert "eval: [2] gen:numbers" in summ["internal"]


def test_every_eval_set_keeps_its_own_name_on_the_hub(tmp_path):
    """The Hub groups eval results by (task, dataset type, config, split, revision), not
    by dataset name. Two arms or two sets with the same key merge under the first name.
    Same-named sets from two eval blocks keep their `[n]` tag, so their names stay apart."""
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    arms = []
    for w in (4096, 3072):
        cfg = json.load(open(ckpt / "hobson_config.json"))
        wcfg = tmp_path / f"w{w}.json"
        wcfg.write_text(json.dumps(dict(cfg, max_length=w), indent=2) + "\n")
        fp = hf_export.fingerprint(str(ckpt), {"hobson_config.json": wcfg.read_bytes()})
        _jev(str(tmp_path / f"jev-{w}"), fp, w)
        arms.append(tmp_path / f"jev-{w}")
    out = tmp_path / "out"
    _export(tmp_path, ckpt, out, arms, [str(tmp_path / "w4096.json"), str(tmp_path / "w3072.json")])
    from huggingface_hub import metadata_load
    from huggingface_hub.repocard_data import model_index_to_eval_results

    _, results = model_index_to_eval_results(metadata_load(str(out / "README.md"))["model-index"])
    internal = hf_export.headline(json.load(open(out / "eval" / "summary.json"))["internal"])
    assert {"eval: held-out short tasks", "eval: musique", "eval: [3] adequacy_hs2",
            "eval: [4] adequacy_hs2"} <= set(internal)
    assert len(results) == 2 + len(internal)
    assert {r.dataset_name for r in results} == {
        "JevBench public, served at 4096", "JevBench public, served at 3072", *internal}
    assert {r.dataset_config for r in results if r.dataset_type == "hobson-internal"} == set(internal)
    assert (out / "README.md").read_text().count("| eval: [3] adequacy_hs2 |") == 1
    assert (out / "README.md").read_text().count("| eval: [4] adequacy_hs2 |") == 1


def test_a_different_export_is_refused(tmp_path):
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    out = tmp_path / "out"
    _export(tmp_path, ckpt, out)
    torch.save({"norm.weight": torch.zeros(8)}, ckpt / "slot_head.pt")
    with pytest.raises(SystemExit, match="different export"):
        _export(tmp_path, ckpt, out)


def test_jevbench_must_have_served_this_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    wrong = tmp_path / "jev-wrong"
    _jev(str(wrong), "0123456789abcdef", 4096)
    with pytest.raises(SystemExit, match="neither this one"):
        _export(tmp_path, ckpt, tmp_path / "out", [wrong])
    # A copy served at another window whose other files were symlinks: jevbench.sh
    # hashed only its edited config.
    cfg = json.load(open(ckpt / "hobson_config.json"))
    copy_cfg = tmp_path / "w3072.json"
    copy_cfg.write_text(json.dumps(dict(cfg, max_length=3072), indent=2) + "\n")
    import hashlib

    fp = hashlib.sha256(f"{hf_export.sha256(str(copy_cfg))}  ./hobson_config.json\n".encode()).hexdigest()[:16]
    w3072 = tmp_path / "jev-3072"
    _jev(str(w3072), fp, 3072)
    _export(tmp_path, ckpt, tmp_path / "out2", [w3072], [str(copy_cfg)])
    arm = json.load(open(tmp_path / "out2" / "eval" / "summary.json"))["jevbench"][0]
    assert arm["window"] == 3072 and "not hashed" in arm["served"]


def test_verify_catches_a_changed_file(tmp_path):
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    out = tmp_path / "out"
    _export(tmp_path, ckpt, out)
    (out / "history.json").write_text("tampered")
    with pytest.raises(SystemExit, match="MANIFEST"):
        hf_export.verify(str(out))


def test_verify_accepts_what_the_hub_and_a_download_add(tmp_path):
    """A Hub repo's first commit adds .gitattributes, and `hf download --local-dir`
    writes .cache/huggingface/. Neither is in the manifest. Any other extra file is."""
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    out = tmp_path / "out"
    _export(tmp_path, ckpt, out)
    (out / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
    os.makedirs(out / ".cache" / "huggingface" / "download")
    (out / ".cache" / "huggingface" / "download" / "README.md.metadata").write_text("x\n")
    hf_export.verify(str(out))
    (out / "extra.txt").write_text("not in the manifest")
    with pytest.raises(SystemExit, match="MANIFEST"):
        hf_export.verify(str(out))


def test_a_destination_that_is_not_an_export_is_kept_unless_replace(tmp_path):
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    out = tmp_path / "out"
    os.makedirs(out)
    (out / "precious.txt").write_text("not an export")
    with pytest.raises(SystemExit, match="not a strands-decider export"):
        _export(tmp_path, ckpt, out)
    assert (out / "precious.txt").exists()
    assert _export(tmp_path, ckpt, out, replace=True) == "replaced"
    assert not (out / "precious.txt").exists()
    hf_export.verify(str(out))
    os.remove(out / "MANIFEST.sha256")  # an unfinished copy by this module
    assert _export(tmp_path, ckpt, out) == "replaced"
    torch.save({"norm.weight": torch.zeros(8)}, ckpt / "slot_head.pt")
    assert _export(tmp_path, ckpt, out, replace=True) == "replaced"  # a different export


def test_the_checkpoint_is_never_the_destination(tmp_path):
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    before = hf_export.fingerprint(str(ckpt))
    for out in (ckpt, tmp_path, ckpt / "hf"):
        with pytest.raises(SystemExit, match="overlaps"):
            hf_export.main(["export", str(ckpt), str(out), "--run-id", "test-run1", "--replace"])
    assert hf_export.fingerprint(str(ckpt)) == before


def test_another_spelling_of_the_checkpoint_is_never_the_destination(tmp_path):
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    if not os.path.exists(tmp_path / "CKPT"):
        pytest.skip("the file system is case-sensitive")
    before = hf_export.fingerprint(str(ckpt))
    parent = tmp_path.with_name(tmp_path.name.upper())  # holds the checkpoint
    for out in (tmp_path / "CKPT", tmp_path / "CKPT" / "hf", parent):
        with pytest.raises(SystemExit, match="overlaps"):
            hf_export.main(["export", str(ckpt), str(out), "--run-id", "test-run1", "--replace"])
    assert hf_export.fingerprint(str(ckpt)) == before


def test_a_linked_copy_of_the_destination_is_never_exported_onto_it(tmp_path):
    real, linked = tmp_path / "ckpts" / "real", tmp_path / "linked"
    _ckpt(str(real))
    os.makedirs(linked)
    for name in os.listdir(real):
        os.symlink(real / name, linked / name)
    before = hf_export.fingerprint(str(real))
    for out in (real, real.parent):
        with pytest.raises(SystemExit, match="overlaps"):
            hf_export.main(["export", str(linked), str(out), "--run-id", "test-run1", "--replace"])
    assert hf_export.fingerprint(str(real)) == before


def test_no_other_input_and_not_the_stage_is_the_destination(tmp_path, monkeypatch):
    ckpt, run, tmp = tmp_path / "ckpt", tmp_path / "run", tmp_path / "tmp"
    _ckpt(str(ckpt))
    _run(str(run))
    os.makedirs(tmp)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))  # the stage is made inside `tmp`
    for out in (run, run / "logs", tmp):
        with pytest.raises(SystemExit, match="overlaps"):
            hf_export.main(["export", str(ckpt), str(out), "--run-dir", str(run), "--run-id",
                            "test-run1", "--replace"])
    assert (run / "stages.jsonl").exists() and (run / "logs" / "eval.log").exists()


def test_an_empty_destination_directory_is_written(tmp_path):
    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    _ckpt(str(ckpt))
    os.makedirs(out)
    assert _export(tmp_path, ckpt, out) == "written"
    hf_export.verify(str(out))


@pytest.mark.parametrize("there, ls_rc, ls_err, replace, want", [
    ({"precious.txt": "x"}, 0, "", False, "not a strands-decider export"),
    ({}, 254, "AccessDenied", False, "cannot list"),
    ({}, 1, "", False, "written"),
    ({"precious.txt": "x"}, 0, "", True, "replaced"),
    ({"provenance.json": None}, 0, "", False, "replaced"),  # an unfinished copy by this module
])
def test_an_s3_destination_is_listed_before_a_sync_delete(tmp_path, monkeypatch, there, ls_rc,
                                                           ls_err, replace, want):
    ckpt, stage = tmp_path / "ckpt", tmp_path / "stage"
    _ckpt(str(ckpt))
    os.makedirs(stage)
    hf_export.build(str(ckpt), str(stage), None, None, [], [], "hobson-2b-test", "test-run1", "final")
    there = {n: (text or (stage / n).read_text()) for n, text in there.items()}
    calls = []

    def run(cmd, **kw):  # stands in for the AWS CLI
        calls.append(list(cmd))
        if cmd[:3] == ["aws", "s3", "ls"]:
            return subprocess.CompletedProcess(cmd, ls_rc, "".join(f"x {n}\n" for n in there), ls_err)
        if cmd[:3] == ["aws", "s3", "cp"] and cmd[-1] == "-":
            name = cmd[3].rsplit("/", 1)[1]
            return subprocess.CompletedProcess(cmd, 0 if name in there else 1, there.get(name, ""), "")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(hf_export.subprocess, "run", run)
    dest = "s3://bucket/hf/hobson-2b-test/test-run1"
    if want in ("written", "replaced"):
        assert hf_export.publish(str(stage), dest, replace) == want
        assert ["aws", "s3", "sync", "--only-show-errors", "--delete", "--exclude", "MANIFEST.sha256",
                str(stage), dest + "/"] in calls
    else:
        with pytest.raises(SystemExit, match=want):
            hf_export.publish(str(stage), dest, replace)
        assert not any("sync" in c for c in calls)


@pytest.mark.parametrize("base", ["Qwen/Qwen3-1.7B-Base", None])
def test_a_checkpoint_on_another_base_is_refused(tmp_path, base):
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    cfg = json.load(open(ckpt / "hobson_config.json"))
    cfg.pop("base_model")
    if base:
        cfg["base_model"] = base
    json.dump(cfg, open(ckpt / "hobson_config.json", "w"))
    with pytest.raises(SystemExit, match="base_model"):
        _export(tmp_path, ckpt, tmp_path / "out")


def test_the_card_is_release_ready(tmp_path):
    """The card names the public code repository, the published package and CLI, the
    Apache-2.0 license, the data inventory and the recipe; nothing of the private past."""
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    _export(tmp_path, ckpt, tmp_path / "out")
    _export(tmp_path, ckpt, tmp_path / "out-url", repo_url="https://example.org/decider",
            hub_id="org/decider-2b")
    for out, url, model in ((tmp_path / "out", hf_export.REPO_URL, "<this folder>"),
                            (tmp_path / "out-url", "https://example.org/decider", "org/decider-2b")):
        readme, licence = (out / "README.md").read_text(), (out / "LICENSE.md").read_text()
        for text in (readme, licence):
            assert url in text
            assert {m.rstrip(".,)") for m in re.findall(r"https?://github\.com/\S+", text)} <= {url}
            assert hf_export.TEACHER in text and "ContractNLI" in text and "data/sources.md" in text
            for gone in ("Not published", "private storage", "hobson serve", "HobsonModel",
                         "pip install -e", "not decided", "abc1234"):  # the run's git_rev
                assert gone not in text
        title = "# decider-2b\n" if model != "<this folder>" else "# hobson-2b-test\n"
        assert title in readme and "test-run1" not in readme  # no run id in the public title
        assert "pip install strands-decider" in readme and f"strands-decider ask {model} " in readme
        assert f"strands-decider serve {model} --port 8000" in readme
        assert f'StrandsDeciderModel.load("{model}")' in readme
        assert "data/README.md" in readme and "training/README.md" in readme
        assert "## Limitations" in readme and "evaluation/README.md" in readme
        assert "| JevBench public, window" not in readme  # no arm in this export
        assert "| eval: musique | 0.884 | 1,199 |" in readme
        assert licence.startswith("# License\n") and "Apache License 2.0" in licence
        assert "Version 2.0, January 2004" in licence and licence.rstrip().endswith("limitations under the License.")
    from huggingface_hub import metadata_load

    meta = metadata_load(str(tmp_path / "out" / "README.md"))
    assert {"tasksource/Boardgame-QA", "hotpotqa/hotpot_qa"} <= set(meta["datasets"])
    assert meta["license"] == "apache-2.0" and "strands-decider" in meta["tags"]


def test_scrub_removes_host_and_cloud_details_and_keeps_the_rest():
    text = ('ckpt /opt/work/v19/checkpoints/recipe-w3072 and /home/me/x.csv and /Users/me/y/ '
            'on i-0123456789abcdef0 in us-west-2 (s3://my-bucket-123456789012-us-west-2/results/) '
            'account 123456789012; keep p5.48xlarge, 0.722943722943, data/raw/x.zip, ./cfg.json, '
            'https://github.com/strands-labs/strands-decider and commit abc1234')
    got = hf_export.scrub(text, ["abc1234"])
    assert got == ('ckpt <redacted>/recipe-w3072 and <redacted>/x.csv and <redacted>/y '
                   'on <redacted> in <redacted> (<redacted>) '
                   'account <redacted>; keep p5.48xlarge, 0.722943722943, data/raw/x.zip, ./cfg.json, '
                   'https://github.com/strands-labs/strands-decider and commit <redacted>')
    assert hf_export.scrub_json({"cost_usd": 0.0, "charged_usd": None, "price_input_per_m": 1,
                                 "ledger": [], "cost_basis": "x", "n": 3, "gpu": "H100",
                                 "tiers": {"easy": [1, 2]}, "paths": ["/opt/a/b"]}) == {
        "n": 3, "gpu": "H100", "tiers": {"easy": [1, 2]}, "paths": ["<redacted>/b"]}


def test_run_records_are_scrubbed_and_the_model_files_are_not(tmp_path):
    """Host paths, cost fields and the --redact text leave every copied record (JevBench
    outputs, eval reports, logs, stages.jsonl, provenance.json); a record with nothing to
    remove keeps its bytes; the checkpoint's own files are copied byte for byte."""
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    fp = hf_export.fingerprint(str(ckpt))
    jev = tmp_path / "jev"
    _jev(str(jev), fp, 4096)
    meta = json.load(open(jev / "run_meta.json"))
    meta.update(checkpoint="/opt/hobson/scratch/work/v19/checkpoints/hobson-2b-recipe",
                hobson_pkg="/opt/hobson/code/v19/src/hobson", gpu="NVIDIA H100 80GB HBM3, 595.91.07")
    json.dump(meta, open(jev / "run_meta.json", "w"), indent=2)
    json.dump({"a": {"source": "/opt/hobson/code/v19/report/data/jevbench_results.csv"},
               "slices": {"all": {"n": 231}}}, open(jev / "paired.json", "w"))
    (jev / "results.jsonl").write_text(
        json.dumps({"task_id": "t-1", "correct": True, "cost_usd": None, "charged_usd": 0.0}) + "\n")
    (jev / "manifest.json").write_text(json.dumps({"endpoint": "http://127.0.0.1:8098", "charged_usd": 0.0,
                                                   "cost_basis": "no_billable_account_public_endpoint"}))
    reports = tmp_path / "reports"
    os.makedirs(reports)
    (reports / "recipe_heldout.json").write_text('{"overall": {"n": 6000}}')  # kept byte for byte
    run = tmp_path / "run"
    _run(str(run))
    (run / "logs" / "eval.log").write_text("loading /home/ubuntu/ckpt on i-0abcdef1234567890\n"
                                           + (run / "logs" / "eval.log").read_text())
    out = tmp_path / "out"
    _export(tmp_path, ckpt, out, [jev], redact=["abc1234"], reports=reports)
    hf_export.verify(str(out))
    for p in hf_export.files(str(out)):
        if p.startswith(("eval/", "training/", "provenance", "history", "train_config")):
            text = (out / p).read_text()
            assert "/opt/" not in text and "/home/" not in text and "abc1234" not in text, p
            assert "i-0abcdef1234567890" not in text and "_usd" not in text and "cost" not in text, p
    for p in ("README.md", "LICENSE.md"):
        assert "abc1234" not in (out / p).read_text()
    got = json.load(open(out / "eval" / "jevbench-w4096" / "run_meta.json"))
    assert got["checkpoint"] == "<redacted>/hobson-2b-recipe" and got["hobson_pkg"] == "<redacted>/hobson"
    assert got["gpu"] == "NVIDIA H100 80GB HBM3, 595.91.07" and got["checkpoint_files_sha256_16"] == fp
    assert json.load(open(out / "eval" / "jevbench-w4096" / "paired.json"))["a"]["source"] == (
        "<redacted>/jevbench_results.csv")
    assert json.loads((out / "eval" / "jevbench-w4096" / "results.jsonl").read_text()) == {
        "task_id": "t-1", "correct": True}
    assert (out / "eval" / "jevbench-w4096" / "manifest.json").read_text() == '{\n  "endpoint": "http://127.0.0.1:8098"\n}\n'
    assert (out / "eval" / "internal" / "recipe_heldout.json").read_bytes() == (reports / "recipe_heldout.json").read_bytes()
    assert (out / "eval" / "internal" / "eval.log").read_text().startswith("loading <redacted>/ckpt on <redacted>\n")
    stages = [json.loads(x) for x in (out / "training" / "stages.jsonl").read_text().splitlines()]
    assert stages[0]["git_rev"] == "<redacted>" and stages[0]["host_shape"] == "p5.48xlarge"
    assert stages[0]["wall_s"] == 1800
    prov = json.load(open(out / "provenance.json"))
    assert prov["code_commit"] == "<redacted>" and prov["host_shape"] == "p5.48xlarge"
    for p in ("hobson_config.json", "train_config.json", "tokenizer.json", "tokenizer_config.json",
              "lora/adapter_config.json", "lora/adapter_model.safetensors"):
        assert (out / p).read_bytes() == (ckpt / p).read_bytes(), p
    # The same export without the literal keeps the run's commit: --redact is explicit.
    _export(tmp_path, ckpt, tmp_path / "out2", [jev], reports=reports)
    assert json.load(open(tmp_path / "out2" / "provenance.json"))["code_commit"] == "abc1234"


def test_multistep_eval_lines_parse_back_whatever_the_name_length(tmp_path, monkeypatch, capsys):
    """internal_evals, and any script that copies its regex, split a per-task line on two or more
    spaces between the name and the value. multistep_eval pads names to 26 characters, so
    a name of 26 or more characters used to leave one space and drop out of the card."""
    import sys
    from types import SimpleNamespace

    import multistep_eval  # evaluation/multistep_eval.py, on sys.path through pyproject.toml

    from strands_decider.data.format import Example
    from strands_decider.modeling import MASK_VALUE, StrandsDeciderModel

    long_name = "gen:a_name_of_thirty_one_chars!"
    assert len(long_name) >= 26
    rows = [Example(kind="choice", state="s", instructions="q", options=[["a", ""], ["b", ""]],
                    label=i % 2, task=task)
            for task, i in [(long_name, 0), (long_name, 1), (long_name, 0), ("hotpotqa", 1),
                            ("hotpotqa", 0)]]
    data = tmp_path / "eval.jsonl"
    data.write_text("".join(ex.to_json() + "\n" for ex in rows))

    def fake_collect(model, examples, *, device="cuda", batch_size=16, max_length=3072):
        out = torch.full((len(examples), 2), MASK_VALUE)
        for i, ex in enumerate(examples):
            out[i, ex.label if i != 1 else 1 - ex.label] = 0.0  # row 1 is wrong
        return out, None, torch.tensor([2] * len(examples)), examples

    monkeypatch.setattr(multistep_eval, "collect_logits", fake_collect)
    monkeypatch.setattr(StrandsDeciderModel, "load", staticmethod(lambda path, **k: SimpleNamespace(
        config=SimpleNamespace(max_length=3072), tokenizer=lambda text: {"input_ids": [0]})))
    monkeypatch.setattr(sys, "argv", ["multistep_eval", "ckpt", "--data", str(data)])
    multistep_eval.main()
    out = capsys.readouterr().out
    assert len(out.splitlines()) == 3  # the header and one line per task
    # Both parsers split the blocks on the header's ending, so pin it too.
    assert out.splitlines()[0].endswith("rows over the window skipped)")

    os.makedirs(tmp_path / "logs")
    (tmp_path / "logs" / "eval.log").write_text(out)
    assert hf_export.internal_evals(str(tmp_path)) == {
        f"eval: {long_name}": {"accuracy": 0.667, "n": 3},
        "eval: hotpotqa (held out)": {"accuracy": 1.0, "n": 2},
    }


@pytest.mark.parametrize("size", ["E2B", "E4B", "12B", "26B-A4B"])
def test_a_gemma4_checkpoint_is_described_as_gemma4(tmp_path, size):
    """The card, LICENSE.md and provenance.json name the checkpoint's own base, its decoder
    class and its recipe's teacher; a Gemma base needs the card example from the caller."""
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    cfg = json.load(open(ckpt / "hobson_config.json"))
    cfg["base_model"] = f"google/gemma-4-{size}-it"
    json.dump(cfg, open(ckpt / "hobson_config.json", "w"), indent=2)
    with pytest.raises(SystemExit, match="--example"):
        _export(tmp_path, ckpt, tmp_path / "no-example")
    stage = tmp_path / "stage2"
    os.makedirs(stage)
    hf_export.build(str(ckpt), str(stage), str(tmp_path / "run"), None, [], [], "n", "r", "final",
                    hub_id=f"org/strands-decider-{size}-gemma4", example="Output:\n\n```\nx\n```\n")
    readme, licence = (stage / "README.md").read_text(), (stage / "LICENSE.md").read_text()
    from huggingface_hub import metadata_load

    meta = metadata_load(str(stage / "README.md"))
    assert meta["base_model"] == f"google/gemma-4-{size}-it" and "gemma4" in meta["tags"]
    assert "qwen3.5" not in meta["tags"]
    prov = json.load(open(stage / "provenance.json"))
    assert prov["base_model_revision"] == hf_export.BASES[meta["base_model"]].revision
    assert "Qwen/Qwen3.5-4B" in readme and "Qwen/Qwen3.5-4B" in licence
    assert ("Gemma4UnifiedForConditionalGeneration" in readme) == (size == "12B")
    assert ("not the 128 experts" in readme) == (size == "26B-A4B")
    assert "Output:\n\n```\nx\n```\n\nOr serve it" in readme and "v19 reference checkpoint" not in readme
    assert "{" not in readme.split("## Training")[1]  # every paragraph's fields filled
    hf_export.verify(str(stage))


def test_an_unknown_base_is_refused(tmp_path):
    ckpt = tmp_path / "ckpt"
    _ckpt(str(ckpt))
    cfg = json.load(open(ckpt / "hobson_config.json"))
    cfg["base_model"] = "org/some-other-base"
    json.dump(cfg, open(ckpt / "hobson_config.json", "w"), indent=2)
    with pytest.raises(SystemExit, match=re.escape("describes Qwen/Qwen3.5-2B-Base, google/gemma-4-E2B-it")):
        _export(tmp_path, ckpt, tmp_path / "out")
