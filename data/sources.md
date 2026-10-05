# Source inventory

Paths are relative to the repository root unless they are links. A module path such as
`data/recipes.py` is relative to `src/strands_decider/`, but a data file such as `data/train_v5.jsonl`,
and the scripts under `data/generators/` and `data/checks/`, are relative to the repository
root.

This table lists every input of the current recipe (`recipe.sh`), the v19 route and the v20
route. The licence column copies only the licences that this repository states. "Not
recorded here" means that the repository does not state one. Read the source's own terms
before you redistribute data built from it.

| Source | Pinned revision | Role | Transformation | Committed or downloaded | Licence | Attribution |
| --- | --- | --- | --- | --- | --- | --- |
| 21 short-task classification datasets on Hugging Face ([Short-task datasets](#short-task-datasets)) | No: `load_dataset` gets no revision | Train | `strands-decider data build` (`data/recipes.py`) turns each row into a typed question. Large label sets are subsampled to at most 24 options, and each task is capped at a fixed row count. | Downloaded | Not recorded here | The dataset card of each Hugging Face id |
| 7 held-out short tasks ([Short-task datasets](#short-task-datasets)) | No | Held out: `emotion`, `hate_severity`, `massive_intent` and `sarcasm` calibrate and evaluate. The three RuleTaker depths are held out of training only. | The same build. `--holdout` writes them to `data/train_v5.holdout.jsonl`. `data/holdout_v5_norule.jsonl` drops the RuleTaker depths. | Downloaded | Not recorded here | The dataset card of each Hugging Face id |
| ContractNLI (release zip) | No (hash-checked after `fetch`) | Train and evaluation | `data/multistep.py`: 17 claims per NDA, balanced per claim | Downloaded by `recipe.sh fetch` | CC BY 4.0 | Koreeda & Manning (Findings of EMNLP 2021), Hitachi America |
| MuSiQue, full v1.0 (release zip) | No (hash-checked after `fetch`) | Train and evaluation | `data/multistep.py`: each question with its answerable and unanswerable versions | Downloaded by `recipe.sh fetch` | CC BY 4.0 | Trivedi et al., TACL 2022 |
| BoardgameQA (`tasksource/Boardgame-QA`) | No | Train and evaluation | `data/multistep.py`: at most a quarter of the multi-step rows | Downloaded | CC BY 4.0 | Kazemi et al., NeurIPS 2023 |
| HotpotQA (`hotpotqa/hotpot_qa`, distractor) | No | Evaluation only | `data/multistep.py`: comparison questions as yes/no or a choice between two titles | Downloaded | CC BY-SA 4.0 | Yang et al., EMNLP 2018 |
| HelpSteer2 (`nvidia/HelpSteer2`) | No: `resolve/main` (hash-checked after `fetch`) | Train and evaluation | `data/adequacy.py`: ratings thresholded to adequate or inadequate, classes balanced | Downloaded by `recipe.sh fetch`, not redistributed | CC BY 4.0 | Wang et al. (NVIDIA, 2024), arXiv:2406.08673 |
| Generated document questions, v16 and v18 (`data/synthetic/generated_v16*.jsonl`, `generated_v18*.jsonl`) | Yes: committed files | Train, and evaluation on held-out domains | Written by Qwen3.6-27B and kept when Qwen3.5-397B-A17B agreed, through OpenRouter. `data/generated.py` balances them. | Committed, with the raw exports in `data/generators/gen_*/` | Models: Apache-2.0. Generated text: not recorded here. | Qwen3.6-27B, Qwen3.5-397B-A17B |
| Generated adequacy items, v19 (`data/synthetic/adequacy_gen*.jsonl`) | Yes: committed files | Train and evaluation | The same writer and verifier. Each inadequate response carries one assigned defect. | Committed (`data/generators/gen_adequacy/`) | Models: Apache-2.0. Generated text: not recorded here. | Qwen3.6-27B, Qwen3.5-397B-A17B |
| Instruction-flip pairs, v20 (`data/synthetic/flips_v20*.jsonl`) | Yes: committed files | Train (v20) and evaluation | The same writer and verifier. Each short policy is asked twice, with instructions that reverse the answer. | Committed (`data/generators/gen_flips/`) | Models: Apache-2.0. Generated text: not recorded here. | Qwen3.6-27B, Qwen3.5-397B-A17B |
| Question paraphrases, v20 (`data/generators/gen_paraphrases/paraphrases.jsonl`) | Yes: committed file | Train (v20) and the paired consistency evaluation | Written by Qwen3.6-27B and checked by Qwen3.5-397B-A17B. `recipe.sh generated` attaches them as instruction variants. | Committed | Models: Apache-2.0. Generated text: not recorded here. | Qwen3.6-27B, Qwen3.5-397B-A17B |
| Catch-all rows, v20 (`data/catchall_v20*.jsonl`) | Follows the short-task corpus | Train (v20) and evaluation | `data/catchall.py` adds "other" or "none of these" options to the corpus's choice rows | Derived by `recipe.sh catchall`, not committed | Follows the short-task datasets | Not applicable |
| Frozen-teacher distributions from `Qwen/Qwen3.5-4B` | Yes: revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (`data/teacher.py`) | Training targets | v19 route: `recipe.sh teacher` labels the multi-step rows, and the parent trains toward them. v20 route: `distill` uses the committed labels of the short-task corpus (`data/synthetic/teacher_v5_qwen35-4b.jsonl`). | The model is downloaded. The short-task labels are committed, without the 21,000 `score` rows (Yelp, SST-5, app_reviews, formality), removed pending a rights check; `python -m strands_decider.data.teacher` regenerates them. `data/synthetic/teacher_multistep_v14.jsonl` is committed too, but no step reads it. | Not recorded here | Qwen |
| Replay distributions | v19 route: regenerated. v20 route: yes, committed file. | Training targets | v19 route: `recipe.sh replay` labels the multi-step rows with the parent checkpoint's own distributions, and v19 trains toward them. v20 route: `distill` uses v14's distributions (`data/synthetic/replay_v14_multistep.jsonl`). | Regenerated, or committed | Not applicable | Not applicable |
| Frozen-teacher distributions from `Qwen/Qwen3.5-27B` (the configs with `teacher_file: data/teacher_yn_qwen35-27b.jsonl`) | Yes: revision `fc05daec18b0a78c049392ed2e771dde82bdf654` (`data/teacher_yn.py`) | Training targets | `recipe.sh teacher_yn` labels the yes/no rows of v19's training files and keeps those that agree with gold, over v14's replay distributions | The model is downloaded; the labels are regenerated, not committed | Apache-2.0 | Qwen |
| Image training set (`data/image/`): VQAv2, COCO train2014, Multimodal-Mind2Web, TabFact, and charts and documents rendered here | Mind2Web: revision `1b4c6a8cf9f77b7a5e0d641959935c80c4a05889`; TabFact: commit `2ab782ba42b5808076ac91fec846473aa5315a79`; VQAv2 and COCO: no | Train (image fine-tuning) | `data/image/`: minimal pairs, counting, set-of-mark screens, table facts, rendered charts and documents ([image/README.md](image/README.md)) | Downloaded and built by the `data/image/` builders, not committed | VQAv2 and COCO annotations: CC BY 4.0; COCO images: their Flickr licences; Mind2Web: CC BY 4.0 (card: OpenRAIL); TabFact: MIT (tables CC BY-SA) | Goyal et al. 2017; Lin et al. 2014; Deng et al. 2023; Chen et al. 2020 |
| Base model `Qwen/Qwen3.5-2B-Base` | No: the loader passes no revision ([Model artifact](../docs/inference.md#model-artifact)) | The torso for training and inference | LoRA adapters and the readout are trained on it | Downloaded from Hugging Face | Apache-2.0 | Qwen |
| Base models of the v21 and bake-off configs: `Qwen/Qwen3.5-2B-Base`, `openbmb/MiniCPM5-2B` | Yes: `base_revision` in each config (`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, `f97400052a43d642bbc6e9975e2397e3ae6a6b52`); v20 continues v19 at its pinned revision; the v20 and balanced packages record no base revision and load the base's default branch (trained at `b1485b2f`) | The torso | LoRA adapters and the readout are trained on it | Downloaded from Hugging Face | Apache-2.0 | Qwen; OpenBMB |
| Image encoder `google/siglip2-so400m-patch16-384` (`strands_decider.graft`) | Yes: `dd658faac399427308559e2c3ac1e99cbe43845d` | Frozen image encoder of the grafted MiniCPM5 checkpoints | A projector is trained onto it (`configs/align/`) | Downloaded from Hugging Face | Apache-2.0 | Google |
| COCO Captions 2014, train (`data/image/captions.py`) | No: the annotation zip has no revision | Train (stage-1 projector alignment) | One caption per image, 80,000 train2014 images, val2014 refused | Downloaded by `training/recipe_minicpm_vision.sh`, not committed | Annotations CC BY 4.0; images their Flickr licences | Lin et al. 2014; Chen et al. 2015 |
| Unseen-family evaluation (`evaluation/unseen/`): `ChilleD/StrategyQA`, `tau/commonsense_qa`, `allenai/ai2_arc` (ARC-Challenge), `sentence-transformers/stsb`, and the held-out RuleTaker depth-5 rows (training uses depths 0-2) | Yes: `evaluation/unseen/build.py`'s `REVISIONS`; RuleTaker follows the short-task build | Evaluation only, never trained on | 250 / 150 / 150 / 250 rows and 250 RuleTaker rows as typed questions | Downloaded, not committed | StrategyQA MIT; CommonsenseQA MIT; ARC CC BY-SA 4.0; STS-B: not recorded here | Geva et al. 2021; Talmor et al. 2019; Clark et al. 2018; Cer et al. 2017 |

## Short-task datasets

`strands-decider data build` reads these Hugging Face datasets through `data/recipes.py`. The
repository does not record their licences.

| Recipe | Hugging Face id (config) | Question type | Role |
| --- | --- | --- | --- |
| `ag_news` | `fancyzhx/ag_news` | choice | train |
| `banking77` | `legacy-datasets/banking77` | choice | train |
| `clinc150` | `clinc/clinc_oos` (`plus`) | choice | train |
| `dbpedia` | `fancyzhx/dbpedia_14` | choice | train |
| `lang_id` | `papluca/language-identification` | choice | train |
| `yahoo_topics` | `community-datasets/yahoo_answers_topics` | choice | train |
| `spam` | `ucirvine/sms_spam` | noul | train |
| `toxicity` | `google/civil_comments` | noul | train |
| `mnli_entail` | `nyu-mll/glue` (`mnli`) | noul | train |
| `boolq` | `google/boolq` | noul | train |
| `paws` | `google-research-datasets/paws` (`labeled_final`) | noul | train |
| `vitaminc` | `tals/vitaminc` | noul | train |
| `wnli` | `nyu-mll/glue` (`wnli`) | noul | train |
| `pubmed_qa` | `qiaojin/PubMedQA` (`pqa_labeled`) | noul | train |
| `yelp_stars` | `Yelp/yelp_review_full` | score | train |
| `sst5_sentiment` | `SetFit/sst5` | score | train |
| `app_reviews` | `sealuzh/app_reviews` | score | train |
| `formality` | `osyvokon/pavlick-formality-scores` | score | train |
| `ruletaker_d0`, `ruletaker_d1`, `ruletaker_d2` | `tasksource/ruletaker` (depths 0, 1 and 2) | noul | train |
| `emotion` | `dair-ai/emotion` | choice | held out: calibration and evaluation |
| `massive_intent` | `mteb/amazon_massive_intent` (`en`) | choice | held out: calibration and evaluation |
| `sarcasm` | `raquiba/Sarcasm_News_Headline` | noul | held out: calibration and evaluation |
| `hate_severity` | `ucberkeley-dlab/measuring-hate-speech` | score | held out: calibration and evaluation |
| `ruletaker_d3`, `ruletaker_d5`, `ruletaker_natlang` | `tasksource/ruletaker` (depths 3 and 5, NatLang) | noul | held out of training, not used for calibration |
