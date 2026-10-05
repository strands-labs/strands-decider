# Data

This folder holds the data that the recipe trains and evaluates on, and what made it.
Only `synthetic/`, `generators/`, `checks/`, `SHA256SUMS`, the READMEs and `sources.md` are committed.
`data/` is also the recipe's build directory: `training/recipe.sh` downloads the ContractNLI,
MuSiQue and HelpSteer2 releases into `data/raw/`, the Hub datasets into the Hugging Face
cache, and writes the corpora into `data/`; git ignores those files.
Reproducing the recipe needs no paid API. To check every recorded file, run
`sha256sum -c data/SHA256SUMS` from the repository root.

- [`sources.md`](sources.md): every input of the recipe, with its revision, role,
  transformation, licence and attribution.
- [`generators/`](generators/README.md): the generators, their exports and the model backends.
  They need an API key, so the recipe never runs them.
- [`synthetic/`](synthetic/): the committed generated rows and frozen teacher distributions.
- [`checks/`](checks/): two scripts that re-derive the synthetic labels from the rendered documents.
- [`SHA256SUMS`](SHA256SUMS): the manifest that `recipe.sh` checks before the stages that
  need a recorded file.

Paths are relative to the repository root unless they are links. A module path such as
`data/recipes.py` is relative to `src/strands_decider/`, but a data file such as `data/train_v5.jsonl`,
and the scripts under `data/generators/` and `data/checks/`, are relative to the repository
root.

## Committed and downloaded data

**Synthetic data is committed; public data is not.** Every model-produced label and
every generated row the recipes train on is committed as the exact file used. The v20
paraphrases are in `data/generators/gen_paraphrases/paraphrases.jsonl`. Everything else is in
`data/synthetic/`: the generated document questions (v16, v18), the answer-adequacy items (v19), the
instruction-flip pairs (v20), the frozen Qwen3.5-4B's distributions on the short-task
corpus and the multi-step rows, and v14's replay distributions — about 47 MB. The raw
generator output behind them, every verifier answer included, is in `data/generators/gen_*/`,
and `recipe.sh` records the commands that turned one into the other. The rebuild from
the raw exports is not byte-identical today, which is why the processed files are
committed ([Reproduction contract](#reproduction-contract) gives the measured difference).
Public sources (the classification datasets, ContractNLI, MuSiQue,
BoardgameQA, HotpotQA, HelpSteer2) are downloaded and converted by `build`, `fetch`,
`multistep` and `adequacy`. Reproducing a recipe needs no paid model-API calls. It does
need the public downloads above, and the base and teacher models from Hugging Face.

One exception: the Qwen3.5-27B yes/no labels that the candidate models' configs train on
(`training/recipe.sh teacher_yn`, `src/strands_decider/data/teacher_yn.py`;
[strands-decider-2B-hobson-v20](../docs/models/strands-decider-2B-hobson-v20.md#retrain))
are not committed; the step relabels them. Two labellings were recorded, each
`data/teacher_yn_qwen35-27b.jsonl` with 58,246 rows (45,337 kept 27B rows of 54,857
labelled, over v14's 12,909 replay rows): `configs/experiments/v19-yn27b*.yaml` trained on
sha256 `5c381fb0237f464842fe2e8aa7301da9d33bfd53b9ef69d7943a997ca30fca0c`, and every later
config on `c72780dc4ee144cb7199ffb7f3d30be62be50dc5754721a2ce8d13b2b3b7725e`. A relabel
batches the rows differently, so its bf16 probabilities, and the hash, need not match byte
for byte. The labels are keyed by position in the concatenation of the config's
`train_files`, so they fit only the training files they were made from:
`data/adequacy_hs2.jsonl`, built from HelpSteer2, is not in `data/SHA256SUMS`, and the
recorded hashes hold only for a build from the same HelpSteer2 snapshot.

## Data sources and licences

The base corpus uses public classification datasets through `data/recipes.py`. The
multi-step rows (v14) come from:

- **ContractNLI** (Koreeda & Manning, Findings of EMNLP 2021; Hitachi America): CC BY 4.0.
- **MuSiQue** (Trivedi et al., TACL 2022): CC BY 4.0.
- **BoardgameQA** (Kazemi et al., NeurIPS 2023; via `tasksource/Boardgame-QA`): CC BY 4.0.
- **HotpotQA** (Yang et al., EMNLP 2018): CC BY-SA 4.0; evaluation only, never trained on.

The raw releases are downloaded into `data/raw/` (git-ignored) and converted by
`python -m strands_decider.data.multistep`.

The generated rows (v16, v18) were written by **Qwen3.6-27B** and verified by
**Qwen3.5-397B-A17B**, both open-weight under Apache-2.0, through OpenRouter; the
exports are committed in `data/generators/gen_v16/`, `data/generators/gen_pilot_qwen/`,
`data/generators/gen_mixed_pilot/` and `data/generators/gen_weak/`. v15, not part of the current recipe, used
**ShARC** (Saeidi et al., EMNLP 2018): CC BY-SA 3.0; and **ConditionalQA** (Sun et al.,
ACL 2022), whose repository states no licence.

The answer-adequacy rows and evaluation set (v19) use **HelpSteer2** (Wang et al.,
"HelpSteer2: Open-source dataset for training top-performing reward models", NVIDIA,
2024, [arXiv:2406.08673](https://arxiv.org/abs/2406.08673);
[nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its ratings are human (Scale
AI annotators) and its responses were written by NVIDIA's in-house models. Changes
made here: single-turn prompts only; each response's helpfulness and correctness
ratings thresholded into an adequate / inadequate label, with the middle band dropped;
the two classes balanced; each prompt and response wrapped as a request and response
with the question "does the response adequately answer the request?"
(`src/strands_decider/data/adequacy.py`). The raw release is downloaded into
`data/raw/helpsteer2/` (git-ignored) and is not redistributed. The generated adequacy
items (`data/generators/gen_adequacy/`) come from the same Apache-2.0 models as the generated
documents.
## Reproduction contract

Frozen targets attach to corpus rows by position. The teacher and replay files hold one
distribution for each row of the concatenated training files, and no file carries a row
key. So a target file is correct only for the exact corpus bytes that it was labelled on.
The training code checks the option count of each row, not the identity of the row.

A fresh build gave these hashes on a Mac and on AWS hosts:

| File | sha256 |
| --- | --- |
| `data/train_v5.jsonl` | `3d17d1484116d7f1f2f27d3d13714c71d60589b165a60e72513abad967b763fd` |
| `data/train_v5.holdout.jsonl` | `352cab8ff8b70a718dad12c7c90265d353dc79ffe329b0f7d87f0140fb0abe56` |
| `data/holdout_v5_norule.jsonl` | `f5796ad2301903ffa144c8ae34731f23ea45ba91d99b2f5ab27ffd551fbd1a63` |
| `data/multistep_v14.jsonl` | `19e769cf7da6ff712d729b2e88776269cecdfcfd8feb8baf5434646d0b8e2665` |

The committed 4B teacher file aligns with this `data/train_v5.jsonl` by row index (`i`). None of
its 70,408 rows has a different option count, and `distill` keeps exactly the 59,525 rows
that the v20 preregistration names. The file was labelled on the 91,408 short-task rows with at most 16 options; the
21,000 `score` rows (`yelp_stars` 6,000, `sst5_sentiment` 6,000, `app_reviews` 5,000,
`formality` 4,000) were removed before publication, pending a check of the rights on the
Yelp data, and `distill` never kept them, so its output is the same. v12
(`configs/experiments/v12.yaml`) trained on the full file; to rebuild it, run
`python -m strands_decider.data.teacher` with the command in that config. Whether this `data/train_v5.jsonl` is byte-identical to
the original v5 file is not verified, because no hash of the original file was recorded.

The committed `generated_v16.jsonl`, `generated_v16_eval.jsonl` and
`teacher_v5_qwen35-4b.jsonl` use CRLF line endings. The other committed files use LF.
`.gitattributes` keeps the committed bytes unchanged (`-text`). A JSON reader gets the same
rows from either line ending. For the two generated files, an LF rebuild with
`data/generated.py` gives the same bytes as the committed files without their CR
characters.

`data/SHA256SUMS` records the hashes above, the hashes of the committed `data/synthetic/`
files and the hashes of the four raw downloads. `recipe.sh` checks the files it needs
against this manifest. It checks them before each stage that attaches targets by row
position or trains on the corpus: `teacher`, `parent`, `replay`, `distill` and `train`.
`calibrate` and `eval` check `data/holdout_v5_norule.jsonl`, and `eval` also checks
`data/multistep_v14_eval.jsonl`. `recipe.sh` also checks the four downloads after `fetch`.
A missing entry or a different hash stops the recipe. The Hugging Face datasets load
without a pinned revision. An upstream change to a training dataset or to BoardgameQA changes `data/train_v5.jsonl` or
`data/multistep_v14.jsonl`, so it stops the recipe at this check. An upstream change to a
held-out task or to HotpotQA changes `data/holdout_v5_norule.jsonl` or
`data/multistep_v14_eval.jsonl`, so it stops `calibrate` or `eval`. No stage reads
`data/train_v5.holdout.jsonl` directly. To check every recorded file, run
`sha256sum -c data/SHA256SUMS` from the repository root.
