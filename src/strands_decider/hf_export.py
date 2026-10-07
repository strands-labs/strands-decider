"""A strands-decider checkpoint as a Hugging Face model-repo folder, for S3 and the Hub.

    python -m strands_decider.hf_export export CKPT OUT [--run-dir D] [--jevbench D ...]
                                    [--name N] [--run-id R] [--role final|parent]
                                    [--replace] [--repo-url URL] [--hub-id ORG/NAME]
                                    [--redact TEXT ...]
    python -m strands_decider.hf_export verify DIR
    python -m strands_decider.hf_export index ROOT          (INDEX.json + INDEX.md of every export)

CKPT, OUT, --run-dir and --jevbench may be local paths or s3:// URIs. The folder holds no
pickles: `slot_head.pt` becomes `head.safetensors` (same tensors, checked), which
`StrandsDeciderModel.load` reads. Everything else strands-decider needs is copied byte for byte. A model
card with Hub metadata, provenance, the run's timings and evals, and a sha256 manifest
(written last) complete it. The run records (timings, eval reports, JevBench outputs) are
copied through `scrub`, which removes host paths, cloud identifiers, cost fields and any
`--redact` text: the folder is for a public repository. Exporting the same inputs twice
gives the same bytes. OUT must be empty or hold an export by this module. The exporter
refuses a different export or other files at OUT unless you give --replace. It always
refuses an OUT that is an input or the temporary stage, holds one, or is inside one.
Nothing here talks to the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from .modeling import CONFIG_NAME, LEGACY_CONFIG_NAME, config_path

FORMAT = "hobson-hf-export/1"
# The checkpoint's config json is copied and required too, under whichever of CONFIG_NAME
# and LEGACY_CONFIG_NAME the checkpoint holds (`config_path`).
COPY = ["train_config.json", "history.json", "tokenizer.json",
        "tokenizer_config.json", "chat_template.jinja",
        "lora/adapter_config.json", "lora/adapter_model.safetensors"]
REQUIRED = ["lora/adapter_config.json", "lora/adapter_model.safetensors",
            "tokenizer.json", "slot_head.pt"]
# The Hub datasets that the corpus build (data/recipes.py), data/multistep.py and the recipe's
# downloads read. hotpotqa/hotpot_qa is for evaluation only.
DATASETS = ["ccdv/arxiv-classification", "clinc/clinc_oos", "community-datasets/yahoo_answers_topics",
            "dair-ai/emotion", "fancyzhx/ag_news", "fancyzhx/dbpedia_14",
            "google-research-datasets/paws", "google/boolq", "google/civil_comments",
            "legacy-datasets/banking77", "mteb/amazon_massive_intent", "nyu-mll/glue",
            "osyvokon/pavlick-formality-scores", "papluca/language-identification",
            "qiaojin/PubMedQA", "raquiba/Sarcasm_News_Headline", "sealuzh/app_reviews",
            "SetFit/20_newsgroups", "SetFit/sst5", "SetFit/subj", "SetFit/TREC-QC",
            "tals/vitaminc", "tasksource/ruletaker", "ucberkeley-dlab/measuring-hate-speech",
            "ucirvine/sms_spam", "Yelp/yelp_review_full", "nvidia/HelpSteer2",
            "tasksource/Boardgame-QA", "hotpotqa/hotpot_qa"]


@dataclass(frozen=True)
class Base:
    """What the card, LICENSE.md and provenance.json say about one base model and the
    recipe trained on it. Every checkpoint names its base in its config json; a base
    without an entry in BASES is refused rather than described wrongly."""

    model: str
    revision: str
    revision_note: str
    teacher: str  # its output distributions are training targets (data/teacher.py, data/distill.py)
    tags: list[str]
    decoder: str  # where the PEFT adapter sits, as the card says it
    example: str  # the card's `strands-decider ask` output; empty: `export --example` gives it
    teacher_route: str  # how the teacher's distributions reach the model
    trained_by: str  # the card's "Trained by" paragraph: {host} {gpu} {wall} {train}
    retrain: str  # the card's "To retrain" paragraph


QWEN = Base(
    model="Qwen/Qwen3.5-2B-Base",
    # Hub `main` at every run so far (2026-04-23); training hosts did not pin it.
    revision="b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
    revision_note="inferred: Hub main at training time; hosts did not pin it",
    teacher="Qwen/Qwen3.5-4B",
    tags=["strands-decider", "decision-model", "hobson", "lora", "peft", "qwen3.5",
          "classification", "calibration", "typed-decisions"],
    decoder="the Qwen3.5 text decoder\n(`transformers.Qwen3_5ForCausalLM(...).model`)",
    example="""The output of the v19 reference checkpoint (a retrain gives somewhat different numbers
with the same answers):

```
noul_0 noul = 0.829
choice_0 -> billing (confidence 0.769)
  billing                  0.846
  retail                   0.090
  sales                    0.064
score_0 score = 1.10 (confidence 0.519)
  0: calm                                     0.163
  1: frustrated                               0.574
  2: depressed                                0.263
```""",
    teacher_route="They are training\n  targets, directly or through a parent model that the recipe trains first.",
    trained_by="""Trained by `training/recipe.sh all` of the code repository on a `{host}`
host ({gpu}): recipe wall clock {wall} s, training stage
{train} s. Stage timings: `training/stages.jsonl`; data hashes:
`training/data_sha256.txt`; configs: `train_config.json`, `training/configs/`.""",
    retrain="""To retrain, run the same recipe on a Linux or WSL2 host with NVIDIA GPUs: about 11 hours
on one RTX 3090, or about 1 h 10 min on eight H100s with `NGPU=8 FAST=1` through the AWS
runner.""",
)


def _gemma4(size: str, revision: str) -> Base:
    """A Gemma 4 -it torso with the recipe of the Gemma 4 releases: v26's c0001 recipe (the
    Qwen3.5-4B teacher, a frozen-KL anchor toward the v19 checkpoint) on that torso."""
    cls = "Gemma4UnifiedForConditionalGeneration" if size == "12B" else "Gemma4ForConditionalGeneration"
    moe = "; it covers attention and the dense MLP, not the 128 experts or their router" \
        if size == "26B-A4B" else ""
    return Base(
        model=f"google/gemma-4-{size}-it",
        revision=revision,
        revision_note="Hub main when the training configs were written (2026-10-06); the loader "
        "did not pin it",
        teacher="Qwen/Qwen3.5-4B",
        tags=["strands-decider", "decision-model", "lora", "peft", "gemma4", "classification",
              "calibration", "typed-decisions"],
        decoder=f"the Gemma 4 text decoder\n(`transformers.{cls}(...).model.language_model`){moe}",
        example="",
        teacher_route="They are training\n  targets, next to a frozen-KL anchor toward the v19 "
        "checkpoint's own distributions\n  (a Qwen3.5-2B decider; `kl_frozen_reference`).",
        trained_by="""Trained by `strands-decider train` of the code repository under `torchrun` on a `{host}`
host ({gpu}): recipe wall clock {wall} s, training stage {train} s. Configs:
`train_config.json`, `training/configs/`.""",
        retrain="""To retrain, train with the recipe in `train_config.json` on one 8-GPU H100 host, after
`training/recipe.sh build fetch multistep generated adequacy distill`, with the v19
checkpoint where its `kl_frozen_reference` points.""",
    )


BASES = {b.model: b for b in (
    QWEN,
    _gemma4("E2B", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    _gemma4("E4B", "ee0ef6023621cff504d758262d4e04895a5af4a2"),
    _gemma4("12B", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    _gemma4("26B-A4B", "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"),
)}
# Kept for callers that named the original single base.
BASE_MODEL, BASE_REVISION, TEACHER, TAGS = QWEN.model, QWEN.revision, QWEN.teacher, QWEN.tags
REPO_URL = "https://github.com/strands-labs/strands-decider"
PICKLE_EXT = (".pt", ".pth", ".pkl", ".pickle", ".bin", ".ckpt")


# ---------------------------------------------------------------- files and S3

def _sh(*cmd: str) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def _local(src: str, tmp: str, name: str) -> str:
    """A local copy of `src` (a directory), fetched from S3 if it is an s3:// URI."""
    if not src.startswith("s3://"):
        if not os.path.isdir(src):
            raise SystemExit(f"{src}: not a directory")
        return src
    dst = os.path.join(tmp, name)
    _sh("aws", "s3", "sync", "--only-show-errors", src.rstrip("/") + "/", dst)
    if not os.path.isdir(dst) or not os.listdir(dst):
        raise SystemExit(f"{src}: nothing there")
    return dst


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files(root: str) -> list[str]:
    """Every file under `root`, relative, '/'-separated, byte-sorted; symlinks followed,
    as `find -L` does in jevbench.sh, so a linked copy hashes the files it points at."""
    out = []
    for d, _, names in os.walk(root, followlinks=True):
        for n in names:
            out.append(os.path.relpath(os.path.join(d, n), root).replace(os.sep, "/"))
    return sorted(out, key=lambda p: p.encode())


def fingerprint(root: str, override: dict[str, bytes] | None = None) -> str:
    """`checkpoint_files_sha256_16` as evaluation/jevbench/jevbench.sh computes it (`find -L`): sha256sum of
    every file but *.log, sorted by path, then the sha256 of that list, 16 hex digits.
    `override` replaces some files' bytes (a config-only copy of the checkpoint)."""
    lines = []
    for p in files(root):
        if p.endswith(".log"):
            continue
        data = (override or {}).get(p)
        h = hashlib.sha256(data).hexdigest() if data is not None else sha256(os.path.join(root, p))
        lines.append(f"{h}  ./{p}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()[:16]


def is_pickle(path: str) -> bool:
    if path.endswith(PICKLE_EXT):
        return True
    with open(path, "rb") as fh:
        head = fh.read(4)
    if head[:1] == b"\x80" and head[1:2] in (b"\x02", b"\x03", b"\x04", b"\x05"):
        return True
    if head == b"PK\x03\x04":  # torch.save's zip container holds data.pkl
        try:
            with zipfile.ZipFile(path) as z:
                return any(n.endswith(".pkl") for n in z.namelist())
        except zipfile.BadZipFile:
            return False
    return False


# ---------------------------------------------------------------- redaction

# The run records come from a training host and a benchmark harness, and the export goes
# to a public repository. These rules remove what the code repository does not already
# show: where the host kept its files, which cloud account, region, bucket or instance
# it was, and what anything cost. An instance TYPE (p5.48xlarge), a GPU name and the
# stage timings stay: training/aws/README.md states them.
REDACTED = "<redacted>"
HOST_PATH = re.compile(r"(?<![\w./:-])/(?:opt|home|Users|mnt|tmp|root|var|srv|scratch|workspace|"
                       r"efs|fsx|nvme|ephemeral|data|work)/[^\s\"'`,;)\]}]*")
CLOUD_ID = re.compile(r"s3://[^\s\"'`,;)\]}]*"                       # a bucket or an S3 URI
                      r"|(?<![\w.])\d{12}(?![\d.])"                   # an AWS account id
                      r"|\b(?:i|cr|cbo|vol|subnet|vpc|sg|ami)-[0-9a-f]{8,17}\b"  # EC2 ids
                      r"|\b(?:us|eu|ap|sa|ca|me|af|il)-(?:gov-)?[a-z]+-\d\b")  # a region
COST_KEY = re.compile(r"usd|cost|price|ledger|bill", re.I)


def scrub(text: str, literals: Sequence[str] = ()) -> str:
    """`text` with host paths cut to `<redacted>/<basename>`, cloud identifiers and the
    `literals` (a private repository's commit, for one) replaced by `<redacted>`."""
    text = HOST_PATH.sub(lambda m: f"{REDACTED}/{m.group().rstrip('/').rsplit('/', 1)[1]}", text)
    text = CLOUD_ID.sub(REDACTED, text)
    for lit in literals:
        text = text.replace(lit, REDACTED)
    return text


def scrub_json(obj: Any, literals: Sequence[str] = ()) -> Any:
    """`scrub` on every string of a parsed JSON value; keys about money are dropped."""
    if isinstance(obj, dict):
        return {k: scrub_json(v, literals) for k, v in obj.items() if not COST_KEY.search(k)}
    if isinstance(obj, list):
        return [scrub_json(v, literals) for v in obj]
    return scrub(obj, literals) if isinstance(obj, str) else obj


def _record(src: str, dst: str, literals: Sequence[str] = ()) -> None:
    """Copy a run record (json, jsonl, text) with `scrub` applied. A record that needs no
    change keeps its bytes, so an unchanged input gives an unchanged output."""
    raw = open(src, "rb").read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _copy(src, dst)
    if src.endswith(".json"):
        obj = json.loads(text)
        new = scrub_json(obj, literals)
        out = text if new == obj else json.dumps(new, indent=2) + "\n"
    elif src.endswith(".jsonl"):
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        new = scrub_json(lines, literals)
        out = text if new == lines else "".join(json.dumps(row) + "\n" for row in new)
    else:
        out = scrub(text, literals)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(out.encode("utf-8"))


# ---------------------------------------------------------------- the head

def convert_head(ckpt: str, out: str) -> tuple[str, str]:
    """slot_head.pt -> head.safetensors, proven equal. Returns (pickle sha, safetensors sha)."""
    from safetensors.torch import load_file, save_file

    src = os.path.join(ckpt, "slot_head.pt")
    state = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise SystemExit(f"{src}: not a flat state dict of tensors")
    pickle_sha = sha256(src)
    dst = os.path.join(out, "head.safetensors")
    # One metadata key only: safetensors writes the metadata map in hash order, so more
    # keys would make the file's bytes differ between runs. The source pickle's sha256 is
    # in provenance.json.
    save_file({k: v.contiguous() for k, v in state.items()}, dst, metadata={"format": "pt"})
    back = load_file(dst)
    if set(back) != set(state) or not all(
        back[k].dtype == state[k].dtype and back[k].shape == state[k].shape and torch.equal(back[k], state[k])
        for k in state
    ):
        raise SystemExit("head.safetensors does not equal slot_head.pt")
    return pickle_sha, sha256(dst)


# ---------------------------------------------------------------- evaluations

def jevbench_arm(ckpt: str, jdir: str, cfg: dict, window_configs: list[str]) -> dict:
    """Read one JevBench output dir and prove it served this checkpoint. Either the served
    files are exactly these (same fingerprint), or they are these weights with only the
    saved window changed (a config-only copy); anything else is refused."""
    meta = json.load(open(os.path.join(jdir, "run_meta.json")))
    summ = json.load(open(os.path.join(jdir, "summary.json")))
    health = json.load(open(os.path.join(jdir, "health.json")))
    want, window = meta.get("checkpoint_files_sha256_16"), health.get("max_length")
    if fingerprint(ckpt) == want:
        served = "these files"
    else:
        cands = [open(p, "rb").read() for p in window_configs]
        other = dict(cfg, max_length=window)
        cands += [(json.dumps(other, indent=i) + nl).encode() for i in (2, None) for nl in ("\n", "")]
        cands = [c for c in cands if json.loads(c) == other]
        cfg_name = os.path.basename(config_path(ckpt))
        only_cfg = [hashlib.sha256(f"{hashlib.sha256(c).hexdigest()}  ./{cfg_name}\n"
                                   .encode()).hexdigest()[:16] for c in cands]
        if any(fingerprint(ckpt, {cfg_name: c}) == want for c in cands):
            served = f"these files, saved window changed to {window}"
        elif want in only_cfg:
            # A copy whose other files were symlinks: jevbench.sh's `find -type f` hashed
            # only the edited config, so the weights are linked, not proven by hash.
            served = f"a copy with window {window}; weights symlinked, not hashed"
        else:
            raise SystemExit(f"{jdir}: served checkpoint {want} is neither this one nor a "
                             f"window-only copy of it")
    ece = summ.get("ece")
    return {"arm": f"w{window}", "window": window, "served": served, "dir": jdir,
            "n_correct": summ.get("n_correct"), "n": summ.get("n_attempted"),
            "accuracy": summ.get("accuracy"), "brier": summ.get("brier_mean"),
            "ece": ece.get("ece") if isinstance(ece, dict) else ece,
            "schema_validity_strict": summ.get("schema_validity_strict")}


LINE = re.compile(r"^\s{2}(\S.*?)\s{2,}(\d\.\d{3})\s+\(n=([\d,]+)\)\s*$")
OVERALL = re.compile(r"^overall n=([\d,]+) acc=(\d\.\d+) ece=(\d\.\d+) nll=(\d\.\d+)")


def internal_evals(run_dir: str | None) -> dict[str, dict]:
    """What the run's eval logs report: `strands-decider eval`'s `overall` line (held-out short
    tasks) and each `name  0.xxx  (n=...)` line of evaluation/multistep_eval.py. One log can
    hold several multistep_eval runs; each starts a numbered block (`[2]`, ...), so
    same-named sets from different eval files do not collide."""
    out: dict[str, dict] = {}
    logs = os.path.join(run_dir, "logs") if run_dir else ""
    for name in sorted(os.listdir(logs)) if os.path.isdir(logs) else []:
        if not (name.startswith("eval") and name.endswith(".log")):
            continue
        block = 0
        for line in open(os.path.join(logs, name), encoding="utf-8", errors="replace"):
            line = line.rstrip("\n")
            if line.endswith("rows over the window skipped)"):
                block += 1
            elif m := OVERALL.match(line):
                out[f"{name[:-4]}: held-out short tasks"] = {
                    "accuracy": float(m.group(2)), "n": int(m.group(1).replace(",", "")),
                    "ece": float(m.group(3)), "nll": float(m.group(4))}
            elif m := LINE.match(line):
                tag = f"[{block}] " if block > 1 else ""
                out[f"{name[:-4]}: {tag}{m.group(1)}"] = {"accuracy": float(m.group(2)),
                                                         "n": int(m.group(3).replace(",", ""))}
    return out


# ---------------------------------------------------------------- card, licence, provenance

LICENSE_TEXT = """# License

This model is released under the Apache License 2.0, the same license as the
strands-decider code at {repo}. The full text follows the notes below.

- **Base model:** `{base}` is released under Apache-2.0 (its Hub license tag and LICENSE).
  This repository holds a LoRA adapter and a readout head trained on it, not its weights.
- **Training data:** the recipe reads public Hub datasets whose Hub license tags include
  Apache-2.0, CC-BY, CC-BY-SA, CC0-1.0, MIT, `other`, `unknown` and none: {datasets}.
  `hotpotqa/hotpot_qa` is for evaluation only. The recipe also uses ContractNLI and MuSiQue
  from their authors' releases, and synthetic rows that open-weight language models generated
  and checked. The output distributions of the teacher model `{teacher}` are also training
  targets. `data/sources.md` in the code repository lists each source with its license and
  attribution where the repository records one. Read a source's own terms before you
  redistribute data built from it.

---

{apache}"""


def apache_license() -> str:
    """The Apache-2.0 text, from the installed strands-decider distribution's LICENSE."""
    from importlib.metadata import PackageNotFoundError, distribution

    try:
        dist = distribution("strands-decider")
    except PackageNotFoundError as e:
        raise SystemExit("strands-decider is not installed, so its LICENSE is not available") from e
    text = dist.read_text("licenses/LICENSE") or dist.read_text("LICENSE")
    if not text or "Apache License" not in text:
        raise SystemExit("the installed strands-decider distribution has no Apache LICENSE file")
    return text


def headline(internal: dict[str, dict]) -> dict[str, dict]:
    """The sets a card lists; the per-skill `gen:` lines stay in eval/summary.json."""
    return {k: v for k, v in internal.items() if "gen:" not in k or "adequacy" in k}


def card(name: str, run_id: str, role: str, prov: dict, arms: list[dict],
         internal: dict[str, dict], repo_url: str = REPO_URL, hub_id: str | None = None,
         base: Base = QWEN) -> str:
    from huggingface_hub import EvalResult, ModelCardData

    # The Hub groups results by (task, dataset type, config, split, revision), not by
    # dataset name. Without a config of its own, every result of one type merges under
    # the first name. The set's key is the config and the name: its `[n]` eval-block tag
    # keeps same-named sets from different blocks apart.
    results = [EvalResult(task_type="text-classification", dataset_type="jevbench",
                          dataset_name=f"JevBench public, served at {a['window']}",
                          dataset_config=a["arm"], metric_type="accuracy",
                          metric_value=round(a["accuracy"], 4),
                          metric_name=f"accuracy ({a['n_correct']}/{a['n']})")
               for a in arms]
    results += [EvalResult(task_type="text-classification", dataset_type="hobson-internal",
                           dataset_name=k, dataset_config=k, metric_type="accuracy",
                           metric_value=v["accuracy"], metric_name=f"accuracy (n={v['n']})")
                for k, v in headline(internal).items()]
    data = ModelCardData(base_model=base.model, base_model_relation="adapter",
                         library_name="peft", pipeline_tag="text-classification", tags=base.tags,
                         license="apache-2.0", datasets=DATASETS, model_name=name,
                         eval_results=results or None)
    rows = "\n".join(f"| JevBench public, window {a['window']} | {a['n_correct']}/{a['n']} "
                     f"| {a['brier']:.3f} | {a['ece']:.3f} | {a['served']} |" for a in arms)
    ints = "\n".join(f"| {k} | {v['accuracy']:.3f} | {v['n']:,} |" for k, v in headline(internal).items())
    model = hub_id or "<this folder>"
    # The title is the public name: the Hub repo's, or the export's. The run id stays in
    # the run records.
    title = (hub_id.rsplit("/", 1)[-1] if hub_id else name) + (" (parent)" if role == "parent" else "")
    return f"""---
{data.to_yaml()}
---

# {title}

Strands decider is a small, fast decision model for agentic AI. Unlike an LLM, which can
generate arbitrary text, a decision model picks between sets of options and rates things on
a scale, and every answer carries a calibrated confidence. It fits the decisions inside an
agentic workflow: model routing, tool selection, argument checking, triage, guardrails,
evaluations, and the rote decisions of a hybrid agent that leaves the hard ones to an LLM.

This repository holds one trained checkpoint: a LoRA adapter on `{base.model}` plus a
small readout head that scores the options of a typed question (`noul`, a yes/no question;
`choice`, one of N options; `score`, a level on an ordered scale). The code, the training
recipe, the data inventory and the evaluations are at {repo_url}, under the same
Apache-2.0 license as this model (see `LICENSE.md`).

## Use

```bash
pip install strands-decider
```

Ask one or more questions about a state. Pass `--device cuda`, `mps` or `cpu`; without it,
the best available device is used. The base weights download from the Hub at first use.

```bash
strands-decider ask {model} \\
  --state "Help! My payouts have been failing for 3 days! " \\
  --choice "Which team should handle this?=billing,sales,retail" \\
  --noul "Does this convey urgency?" \\
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

{base.example}

Or serve it over HTTP. The server binds to `127.0.0.1` and has no authentication: use it
for local experiments.

```bash
strands-decider serve {model} --port 8000
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{{
  "state": "Help! My payouts have been failing for 3 days!",
  "questions": {{"is_urgent": {{"type": "noul", "instructions": "Does this convey urgency?"}}}}
}}'
```

In Python, `strands_decider.modeling.StrandsDeciderModel.load("{model}")`. The adapter in
`lora/` is a standard PEFT adapter on {base.decoder}; the head is `head.safetensors`.

## Results

| evaluation | tasks right | Brier | ECE | served |
| --- | --- | --- | --- | --- |
{rows or '| (none recorded) | | | | |'}

| internal set | accuracy | n |
| --- | --- | --- |
{ints or '| (none recorded) | | |'}

JevBench is the external benchmark (231 public tasks); the internal sets are this recipe's
held-out short tasks, multi-step documents and answer-adequacy judgements.
`evaluation/README.md` in the code repository describes each set, and its results pages
give the figures by version. Per-skill results and the raw reports and logs are in
`eval/summary.json` and `eval/internal/`.

## Limitations

- **Questions are read less than documents.** With the state and options fixed, a changed
  question often gets the same answer. Phrase a question so the obvious reading is the
  intended one.
- **Long, multi-step documents are the weak spot**: JevBench's hard tier scores far below
  its easy tier.
- **`score` and `noul` transfer poorly** to rubrics and yes/no tasks unlike the training
  mix. Train on a rubric resembling yours.
- **Calibration is one temperature per primitive**, fitted on held-out short
  classification. The confidence bands are established there only: measure on your own
  traffic before you trust a threshold.
- **Trained on public datasets**, so it inherits their domains and their label noise.

`evaluation/README.md`, section "Limitations", has the measured figures behind each point.

## Training data

The `datasets` list in the metadata holds the Hub datasets that the recipe reads.
`hotpotqa/hotpot_qa` is for evaluation only. The recipe also uses these sources:

- ContractNLI and MuSiQue, from their authors' releases (not from the Hub).
- Synthetic rows that open-weight language models generated and checked.
- The output distributions of the frozen teacher model `{base.teacher}`. {base.teacher_route}

In the code repository, `data/sources.md` lists every source with its revision, role,
license and attribution, and `data/README.md` states what is committed, what is downloaded
and the reproduction contract.

## Training

{base.trained_by.format(host=prov.get('host_shape'), gpu=prov.get('gpu'),
                        wall=prov.get('pipeline_wall_s'), train=prov.get('train_wall_s'))}

{base.retrain} `training/README.md` has the setup, the stages and the hardware notes.

## Provenance

`provenance.json`: base model and revision (inferred: the hosts did not pin one), and the
sha256 of the config, adapter, head and source pickle. `MANIFEST.sha256` lists every file;
`python -m strands_decider.hf_export verify <folder>` checks them. The run records keep
their timings and results; host paths, cloud identifiers and cost fields are removed.
"""


def provenance(name: str, run_id: str, role: str, ckpt: str, out: str, stages: list[dict],
               pickle_sha: str, base: Base = QWEN) -> dict:
    train = [s for s in stages if s.get("stage") in ("parent", "train")]
    mine = [s for s in train if s.get("stage") == ("parent" if role == "parent" else "train")]
    starts = [s["start_utc"] for s in stages if "start_utc" in s]
    ends = [s["end_utc"] for s in stages if "end_utc" in s]
    from datetime import datetime

    def t(x: str) -> datetime:
        return datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ")

    return {"format": FORMAT, "name": name, "run_id": run_id, "role": role,
            "code_commit": next((s.get("git_rev") for s in stages if s.get("git_rev")), "unknown"),
            "base_model": base.model, "base_model_revision": base.revision,
            "base_model_revision_note": base.revision_note,
            "hobson_config_sha256": sha256(config_path(ckpt)),
            "train_config_sha256": sha256(os.path.join(ckpt, "train_config.json"))
            if os.path.exists(os.path.join(ckpt, "train_config.json")) else None,
            "adapter_sha256": sha256(os.path.join(ckpt, "lora/adapter_model.safetensors")),
            "head_sha256": sha256(os.path.join(out, "head.safetensors")),
            "source_head_pickle_sha256": pickle_sha,
            "source_checkpoint_fingerprint": fingerprint(ckpt),
            "host_shape": next((s.get("host_shape") for s in stages if s.get("host_shape")), None),
            "gpu": next((s.get("gpu_name") for s in stages if s.get("gpu_name")), None),
            "train_wall_s": mine[0].get("wall_s") if mine else None,
            "pipeline_wall_s": int((max(map(t, ends)) - min(map(t, starts))).total_seconds())
            if starts and ends else None}


# ---------------------------------------------------------------- build, check, publish

JEV_FILES = ["summary.json", "run_meta.json", "manifest.json", "health.json", "paired.json",
             "paired.txt", "results.jsonl"]


def _copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def check(out: str) -> None:
    """The rules an export must meet: no pickles, a card the Hub's own parsers read, and
    the Hub's rule for `license: other`."""
    from huggingface_hub import metadata_load
    from huggingface_hub.repocard_data import model_index_to_eval_results

    bad = [p for p in files(out) if is_pickle(os.path.join(out, p))]
    if bad:
        raise SystemExit(f"pickle files in the export: {bad}")
    meta = metadata_load(os.path.join(out, "README.md"))
    if not meta or meta.get("base_model") not in BASES:
        raise SystemExit("README.md: card metadata missing or wrong base_model")
    if meta.get("license") == "other" and not (meta.get("license_name") and meta.get("license_link")):
        raise SystemExit("README.md: license 'other' needs license_name and license_link")
    if meta.get("model-index"):
        model_index_to_eval_results(meta["model-index"])
    for p in ("head.safetensors", "lora/adapter_model.safetensors", "provenance.json", "LICENSE.md"):
        if not os.path.exists(os.path.join(out, p)):
            raise SystemExit(f"export lacks {p}")
    if not os.path.exists(config_path(out)):
        raise SystemExit(f"export lacks {CONFIG_NAME} (or legacy {LEGACY_CONFIG_NAME})")


def _hub_adds(p: str) -> bool:
    """A Hub repo's first commit adds .gitattributes; `hf download --local-dir` writes
    .cache/huggingface/. Neither is part of the export."""
    return p == ".gitattributes" or p.startswith(".cache/huggingface/")


def manifest(out: str) -> str:
    return "".join(f"{sha256(os.path.join(out, p))}  {p}\n"
                   for p in files(out) if p != "MANIFEST.sha256" and not _hub_adds(p))


# Copied through `scrub`; the config, tokenizer, template and adapter files byte for byte.
RECORDS = {"train_config.json", "history.json"}


def build(ckpt: str, out: str, run_dir: str | None, reports: str | None, jdirs: list[str],
          window_configs: list[str], name: str, run_id: str, role: str,
          repo_url: str = REPO_URL, hub_id: str | None = None, redact: Sequence[str] = (),
          example: str | None = None) -> dict:
    cfg_path = config_path(ckpt)
    missing = [p for p in REQUIRED if not os.path.exists(os.path.join(ckpt, p))]
    if not os.path.exists(cfg_path):
        missing.insert(0, CONFIG_NAME)
    if missing:
        raise SystemExit(f"{ckpt}: not a complete checkpoint, missing {missing}")
    cfg = json.load(open(cfg_path))
    # The card and provenance.json name the base and its revision (BASES). StrandsDeciderModel.load
    # reads the base from the config json, and without the key it uses the dataclass default.
    if cfg.get("base_model") not in BASES:
        raise SystemExit(f"{ckpt}: base_model is {cfg.get('base_model')!r}, but this exporter "
                         f"describes {', '.join(BASES)} only")
    base = BASES[cfg["base_model"]]
    if example is not None:
        base = replace(base, example=example.rstrip("\n"))
    elif not base.example:
        raise SystemExit(f"{base.model}: no card example for this base; give --example with the "
                         "output of `strands-decider ask` on this checkpoint")
    for p in [os.path.basename(cfg_path), *COPY]:
        src = os.path.join(ckpt, p)
        if not os.path.exists(src):
            continue
        if p in RECORDS:
            _record(src, os.path.join(out, p), redact)
        else:
            _copy(src, os.path.join(out, p))
    pickle_sha, _ = convert_head(ckpt, out)
    stages: list[dict] = []
    if run_dir:
        for src, dst in (("stages.jsonl", "stages.jsonl"), ("sha256.txt", "data_sha256.txt")):
            if os.path.exists(os.path.join(run_dir, src)):
                _record(os.path.join(run_dir, src), os.path.join(out, "training", dst), redact)
        cdir = os.path.join(run_dir, "configs")
        for p in files(cdir) if os.path.isdir(cdir) else []:
            if p.endswith(".yaml"):
                _record(os.path.join(cdir, p), os.path.join(out, "training/configs", p), redact)
        sj = os.path.join(out, "training", "stages.jsonl")  # the scrubbed copy
        if os.path.exists(sj):
            stages = [json.loads(line) for line in open(sj) if line.strip()]
    arms = [jevbench_arm(ckpt, d, cfg, window_configs) for d in jdirs]
    for a in arms:
        for f in JEV_FILES:
            if os.path.exists(os.path.join(a["dir"], f)):
                _record(os.path.join(a["dir"], f), os.path.join(out, "eval", f"jevbench-{a['arm']}", f),
                        redact)
    # A parent is not what the run's evals measured: they scored the final model.
    final = role == "final"
    internal = internal_evals(run_dir) if final else {}
    for d, keep in ((reports if final else None, lambda p: p.endswith(".json")),
                    (os.path.join(run_dir, "logs") if run_dir and final else None,
                     lambda p: p.startswith("eval") and p.endswith(".log"))):
        if not (d and os.path.isdir(d)):
            continue
        for p in files(d):
            if keep(p):
                _record(os.path.join(d, p), os.path.join(out, "eval/internal", p), redact)
    summary = {"name": name, "run_id": run_id, "role": role,
               "jevbench": [{k: v for k, v in a.items() if k != "dir"} for a in arms],
               "internal": internal}
    with open(_mk(os.path.join(out, "eval", "summary.json")), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    prov = provenance(name, run_id, role, ckpt, out, stages, pickle_sha, base)
    with open(os.path.join(out, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2, sort_keys=True)
    with open(os.path.join(out, "LICENSE.md"), "w") as fh:
        fh.write(LICENSE_TEXT.format(base=base.model, datasets=", ".join(DATASETS),
                                     teacher=base.teacher, repo=repo_url, apache=apache_license()))
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write(card(name, run_id, role, prov, arms, internal, repo_url, hub_id, base))
    check(out)
    with open(os.path.join(out, "MANIFEST.sha256"), "w") as fh:
        fh.write(manifest(out))
    return summary


def _mk(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _read_at(dest: str, name: str) -> str | None:
    """The text of `dest/name`, or None if it is not there."""
    if dest.startswith("s3://"):
        got = subprocess.run(["aws", "s3", "cp", dest.rstrip("/") + "/" + name, "-"],
                             capture_output=True, text=True)
        return got.stdout if got.returncode == 0 else None
    p = os.path.join(dest, name)
    return open(p).read() if os.path.isfile(p) else None


def _dest_state(dest: str) -> tuple[str | None, bool, bool]:
    """(manifest text or None, whether anything is there, whether this module wrote it)."""
    if dest.startswith("s3://"):
        ls = subprocess.run(["aws", "s3", "ls", dest.rstrip("/") + "/"], capture_output=True, text=True)
        # An empty prefix gives exit code 1 and no output. Other results can hide objects.
        if ls.returncode not in (0, 1) or (ls.returncode == 1 and ls.stderr.strip()):
            raise SystemExit(f"{dest}: cannot list it: {ls.stderr.strip()}")
        present = bool(ls.stdout.strip())
    else:
        present = os.path.isdir(dest) and bool(os.listdir(dest))
    if not present:
        return None, False, False
    theirs = _read_at(dest, "MANIFEST.sha256")
    # provenance() writes this line, and a copy writes provenance.json before the manifest.
    ours = theirs is not None or f'"format": "{FORMAT}"' in (_read_at(dest, "provenance.json") or "")
    return theirs, True, ours


def _overlaps(a: str, b: str) -> bool:
    """True if the path or s3:// prefix `a` is `b`, holds `b`, or is inside `b`."""
    if a.startswith("s3://") or b.startswith("s3://"):
        a, b = a.rstrip("/") + "/", b.rstrip("/") + "/"
        return a.startswith(b) or b.startswith(a)
    a, b = os.path.realpath(a), os.path.realpath(b)
    if os.path.commonpath([a, b]) in (a, b):
        return True

    def under(x: str, top: str) -> bool:
        """True if `x` or a directory above it is the same directory as `top`."""
        while not (os.path.exists(x) and os.path.samefile(x, top)):
            if os.path.dirname(x) == x:
                return False
            x = os.path.dirname(x)
        return True

    # A case-insensitive file system gives one directory more than one spelling.
    return (os.path.exists(a) and under(b, a)) or (os.path.exists(b) and under(a, b))


def publish(stage: str, dest: str, replace: bool = False) -> str:
    """Copy a built export to `dest`, manifest last. The same manifest there: no-op. This
    module's files without a manifest (an unfinished copy): replaced. A different export,
    or files that are not an export: refused, unless `replace` is set."""
    mine = open(os.path.join(stage, "MANIFEST.sha256")).read()
    theirs, present, ours = _dest_state(dest)
    if theirs == mine:
        return "unchanged"
    if theirs is not None and not replace:
        raise SystemExit(f"{dest} holds a different export. Give --replace to replace it.")
    if present and not ours and not replace:
        raise SystemExit(f"{dest} holds files that are not a strands-decider export. Give --replace "
                         f"to delete them and write the export there.")
    if dest.startswith("s3://"):
        d = dest.rstrip("/") + "/"
        _sh("aws", "s3", "sync", "--only-show-errors", "--delete", "--exclude", "MANIFEST.sha256", stage, d)
        _sh("aws", "s3", "cp", "--only-show-errors", os.path.join(stage, "MANIFEST.sha256"), d + "MANIFEST.sha256")
    else:
        if present:
            shutil.rmtree(dest)
        shutil.copytree(stage, dest, ignore=shutil.ignore_patterns("MANIFEST.sha256"),
                        dirs_exist_ok=True)  # `dest` can be an empty directory
        shutil.copyfile(os.path.join(stage, "MANIFEST.sha256"), os.path.join(dest, "MANIFEST.sha256"))
    return "replaced" if present else "written"


def verify(path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = _local(path, tmp, "export")
        want = open(os.path.join(d, "MANIFEST.sha256")).read()
        if manifest(d) != want:
            raise SystemExit(f"{path}: files do not match MANIFEST.sha256")
        check(d)


def index(root: str) -> str:
    """INDEX.json and INDEX.md at `root` (local or s3://.../hf): one row per export."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        if root.startswith("s3://"):
            _sh("aws", "s3", "sync", "--only-show-errors", "--exclude", "*",
                "--include", "*/provenance.json", "--include", "*/eval/summary.json",
                root.rstrip("/") + "/", tmp)
            base = tmp
        else:
            base = root
        for p in files(base):
            if not p.endswith("/provenance.json") or p.count("/") != 2:
                continue
            d = os.path.dirname(p)
            prov = json.load(open(os.path.join(base, p)))
            sp = os.path.join(base, d, "eval", "summary.json")
            summ = json.load(open(sp)) if os.path.exists(sp) else {}
            rows.append({"path": d, "name": prov["name"], "run_id": prov["run_id"],
                         "role": prov["role"], "code_commit": prov["code_commit"],
                         "host_shape": prov.get("host_shape"),
                         "jevbench": {j["arm"]: j["n_correct"] for j in summ.get("jevbench", [])},
                         "internal": {k: v["accuracy"] for k, v in headline(summ.get("internal", {})).items()},
                         "published": False})
        rows.sort(key=lambda r: (r["name"], r["run_id"]))
        md = ["# strands-decider model exports (Hugging Face format)", "",
              "None is published. Each folder can be pushed with `huggingface_hub.upload_folder` "
              "once publication is approved.", "",
              "| export | role | code | host | JevBench public | key internal sets |", "| --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            jev = ", ".join(f"{k} {v}/231" for k, v in sorted(r["jevbench"].items())) or "-"
            keys = [k for k in r["internal"] if any(w in k for w in ("short tasks", "musique", "hotpotqa", "adequacy"))
                    and "answerable" not in k]
            ints = "; ".join(f"{k} {r['internal'][k]:.3f}" for k in keys) or "-"
            md.append(f"| `{r['path']}` | {r['role']} | `{r['code_commit']}` | {r['host_shape']} | {jev} | {ints} |")
        for n, body in (("INDEX.json", json.dumps(rows, indent=2) + "\n"), ("INDEX.md", "\n".join(md) + "\n")):
            with open(os.path.join(tmp, n), "w") as fh:
                fh.write(body)
            if root.startswith("s3://"):
                _sh("aws", "s3", "cp", "--only-show-errors", os.path.join(tmp, n), root.rstrip("/") + "/" + n)
            else:
                shutil.copyfile(os.path.join(tmp, n), os.path.join(root, n))
        return "\n".join(md)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m strands_decider.hf_export", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("ckpt")
    ex.add_argument("out")
    ex.add_argument("--run-dir", help="the run's stages.jsonl, sha256.txt, configs/, logs/")
    ex.add_argument("--reports", help="the run's eval reports (*.json)")
    ex.add_argument("--jevbench", action="append", default=[], help="a jevbench.sh output dir")
    ex.add_argument("--window-config", action="append", default=[],
                    help="the config json of a config-only copy a JevBench arm served")
    ex.add_argument("--name", help="default: hobson-2b-<run-id prefix>[-parent]")
    ex.add_argument("--run-id", help="default: the checkpoint's results/<run-id>/ in its path")
    ex.add_argument("--role", choices=["final", "parent"], default="final")
    ex.add_argument("--replace", action="store_true",
                    help="replace what OUT holds: a different export, or files that are not an export")
    ex.add_argument("--repo-url", default=REPO_URL,
                    help="the strands-decider code's repository, named in the card and LICENSE.md")
    ex.add_argument("--hub-id", help="the Hub repo id (org/name) the card's examples name")
    ex.add_argument("--redact", action="append", default=[], metavar="TEXT",
                    help="text to replace by <redacted> in every run record (a private commit id)")
    ex.add_argument("--example", metavar="FILE",
                    help="markdown for the card's `strands-decider ask` output: required for a "
                         "base other than Qwen3.5-2B-Base, whose v19 example is built in")
    ve = sub.add_parser("verify")
    ve.add_argument("path")
    ix = sub.add_parser("index", help="write INDEX.json and INDEX.md for every export under ROOT")
    ix.add_argument("root")
    a = ap.parse_args(argv)
    if a.cmd == "index":
        print(index(a.root))
        return
    if a.cmd == "verify":
        verify(a.path)
        print(f"{a.path}: ok")
        return
    m = re.search(r"results/([^/]+)/", a.ckpt)
    run_id = a.run_id or (m.group(1) if m else None)
    if not run_id:
        raise SystemExit("--run-id is required when the checkpoint path has no results/<run-id>/")
    name = a.name or f"hobson-2b-{run_id.split('-')[0]}" + ("-parent" if a.role == "parent" else "")
    with tempfile.TemporaryDirectory() as tmp:
        # --replace deletes what OUT holds, so OUT must not overlap anything the export uses.
        # A linked copy points at files elsewhere, so check where each file really is too.
        for src in [a.ckpt, a.run_dir, a.reports, *a.jevbench, *a.window_config, tmp]:
            if not src:
                continue
            used = [src] + ([os.path.join(src, f) for f in files(src)]
                            if not src.startswith("s3://") and os.path.isdir(src) else [])
            if any(_overlaps(u, a.out) for u in used):
                raise SystemExit(f"{a.out}: overlaps {src}, which the export uses. Export to another place.")
        ckpt = _local(a.ckpt, tmp, "ckpt")
        run_dir = _local(a.run_dir, tmp, "run") if a.run_dir else None
        reports = _local(a.reports, tmp, "reports") if a.reports else None
        jdirs = [_local(j, tmp, f"jev{i}") for i, j in enumerate(a.jevbench)]
        wcfg = []
        for i, w in enumerate(a.window_config):
            if w.startswith("s3://"):
                _sh("aws", "s3", "cp", "--only-show-errors", w, os.path.join(tmp, f"wcfg{i}.json"))
                w = os.path.join(tmp, f"wcfg{i}.json")
            wcfg.append(w)
        stage = os.path.join(tmp, "export")
        os.makedirs(stage)
        summary = build(ckpt, stage, run_dir, reports, jdirs, wcfg, name, run_id, a.role, a.repo_url,
                        a.hub_id, a.redact, open(a.example).read() if a.example else None)
        state = publish(stage, a.out, a.replace)
    jev = ", ".join(f"{j['arm']} {j['n_correct']}/{j['n']}" for j in summary["jevbench"]) or "no JevBench"
    print(f"{a.out}: {state} ({name} {run_id}, {jev}, {len(summary['internal'])} internal sets)")


if __name__ == "__main__":
    main()
