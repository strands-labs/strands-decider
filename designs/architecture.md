# Architecture

Strands Decider has two versions. A v1 model scores each option with a pointer head. A v2
model, the current version, reads the score of each option from a token readout. This
document describes the two readouts, the three question types, the training objective and
the design decisions behind them. It then gives the runtime interface of
`strands_decider.core`, which the v2 models use, and ends with the repository layout. Paths
are relative to the repository root unless they are links. The published
`strands_decider` package (inference) lives under `src/strands_decider/`; the training
code lives under `training/`. An inference module path such as `infer.py` is relative
to `src/strands_decider/`; a training module path such as `data/teacher.py` is relative
to `training/`. The scripts under `data/generators/` and `data/checks/` are relative to
the repository root.

The numbers v7 to v19 in this document are research versions from before the first release
([research/history.md](../research/history.md)), and v19 is the reference recipe of this
repository. v1 and v2 are the two releases. Their recipes are in `configs/v1/` and
`configs/v2/`, and the v1 recipes keep their answers close to v19's.

## The architecture

![Strands Decider architecture, v1 and v2. In both versions a prompt goes through a pretrained LLM torso once, and the model gives one probability for each option without text generation. v1 reads the hidden states of the options and of the answer position into a pointer head. v2 reads the hidden state at the last token of the prompt into a token readout with one row for each of 255 codes.](figures/architecture.png)

The picture shows the two versions. In both, the prompt goes through the torso once, and the
model generates no text. The two readouts differ, and each has its own section below.

### The v1 pointer head

Take a pretrained decoder LLM, **discard its language-modelling head**, and score each
option by comparing the question's answer position against that option's own hidden
state. One forward pass, no generation, no decoding loop.

A request renders to a single prompt:

```
<state>
Help! My payouts have been failing for 3 days.
</state>
<question type="choice">
Select exactly one option.
Which team should handle this?
<options>
1. billing — payments, invoices, payouts
2. technical — bugs, outages, API errors
3. sales — pricing and upgrades
</options>
</question>
<answer>
```

The torso runs once. The readout then takes the hidden state at `<answer>` as a query
and the hidden state at the **last token of each option's line** as keys — that token
has just read its whole option under causal attention — and scores one against the
others:

```python
logit_k = <q(h_answer), k(h_option_k)> / sqrt(dim)
```

Softmax over those `K` logits is the answer. That is the entire inference path.

**An option's score depends on what it says, not where it sits.** There is no
per-option parameter anywhere in a v1 model, so nothing can learn that "the first
option is usually the right one", and nothing caps how many options a question may
carry. The same weights answer a two-option yes/no and a thirty-option routing
question, with label sets never seen in training, because the prompt alone establishes
what each option means.

Three consequences worth stating plainly:

- **Runtime-defined labels.** Options are read from the request, not baked into the
  weights. No retraining to add a category.
- **No option ceiling.** The schema's 255 is the only limit.
- **Order sensitivity is small but not zero.** Reversing the option list moves the
  distribution by ~0.016, because options still attend to earlier options under causal
  attention. Eliminating it entirely needs per-option attention isolation, which is not
  implemented here.

The readout is ~1M parameters (`dim=256`); rank-16 LoRA adapters on the torso's
projections are the only other thing trained. The v14 checkpoint is ~88 MB — adapter,
readout and tokenizer — and the base weights are fetched from HuggingFace at load.

**The torso.** v13 to v19 run Qwen3.5-2B-Base (1.9B parameters), a hybrid: 18 of its 24
layers are Gated DeltaNet — linear attention carrying a recurrent state — and 6 are full
attention. LoRA covers the projections of both kinds (`lora_targets`). v7 and earlier
ran Qwen3-1.7B-Base, all full attention. The architecture is the same either way; what
the hybrid changes is serving ([Asking many questions is nearly free](inference.md#asking-many-questions-is-nearly-free)) and the
training environment ([Setup](../training/README.md#setup)). The released v1 models run
Qwen3.5-2B-Base and four Gemma 4 sizes (E2B, E4B, 12B and 26B-A4B).

### The v2 token readout

A v2 model keeps the idea: one forward pass, and a softmax over the options of the question.
It changes where the scores come from, and how the torso reads the prompt.

A request renders to a prompt in the `strands-chat` format. This format puts the request in
one user turn of the Gemma 4 chat format, and gives each option a code:

```
<bos><|turn>user
<state>
Help! My payouts have been failing for 3 days.
</state>
<question type="choice">
Select exactly one option.
Which team should handle this?
<options>
A. billing — payments, invoices, payouts
B. technical — bugs, outages, API errors
C. sales — pricing and upgrades
</options>
Answer with the code of one option.
</question><turn|>
<|turn>model
```

The readout is one linear map with one row for each of 255 codes. The codes are the
spreadsheet column names from `A` to `IV`, without `GZ`, so each code is one token of the
Gemma 4 tokenizer. The readout reads the hidden state of the last prompt token. Row `k`
gives the score of the option under code `k`, and the rows past the question's option count
are masked. The model caps the scores with the tanh soft cap that Gemma 4 has on its own
logits, and divides them by the one temperature in the checkpoint's `decision_config.json`.
A softmax over the scores that remain is the answer. A `choice` question takes at most 255
options, the same limit as in v1.

The readout has a row for each code, not for each option. Training shuffles the options of
a `choice` question, so no code stands for a fixed label, and the model must read the text
of the option next to the code. A yes/no question and a score question keep their order: a
yes/no question has the same two options in every request (false, then true), and the
levels of a score question are ordered.

Each row starts as the base model's language-modelling head row for the token of its code,
so before training the readout gives the base model's own preference among the codes. The
model does not use the rest of that head.

The attention changes too. A state token sees all of the state, in both directions, and
never the question. A question token sees the state and all of its own question, also in
both directions, and no other question. The encoding of the state thus does not depend on the question. A request
can carry several questions about one state: the torso encodes the state once, and each
question reads that encoding.

The options still come from the request, so no retraining is necessary to add a category.
The v2 models are Gemma 4 base models, from E2B to 31B, with rank-64 LoRA adapters that the
export merges into the base weights. A v2 checkpoint is thus one directory that holds the
full model. [Training a token readout](#training-a-token-readout) has the config keys and
the export.

### One mechanism, three primitives

In the two versions, all three question types are the same masked softmax, read back
differently:

| Primitive  | Options      | Returns                                        |
| ---------- | ------------ | ---------------------------------------------- |
| **Noul**   | 2 (false/true) | `P(true)`                                    |
| **Choice** | N options    | argmax + per-option probabilities + confidence |
| **Score**  | L levels (2–10, ordered) | expected value `Σ i·pᵢ` + distribution + confidence |

**Confidence is derived, not predicted.** For `choice` it is the normalised
max-probability:

```
confidence = (N · p_max − 1) / (N − 1)
```

Uniform → 0, one-hot → 1, and crucially *independent of N*, so one routing threshold
means the same thing for a 3-option and a 10-option question. Noul has no confidence
field because with two outcomes the probability already is the uncertainty.

#### The routing convention

These documents use the example bands of the public Jev API documentation
([Confidence](https://docs.typesafe.ai/confidence)). At 0.9 or above, act without a
check. From 0.5 to 0.9, confirm first. Below 0.5, send the answer to a person. That
documentation says that the right thresholds depend on the consequences of each
action. Measure them on your own traffic before you rely on them
([Summary](../evaluation/results.md#summary) gives the measured bands).

**`score` uses a different measure**, because max-probability reads *spread* as
*doubt* — and on an ordered scale, mass on adjacent levels is precision, not
confusion. A distribution split between levels 2 and 3 is confidently saying "about
2.5". So score confidence is normalised standard deviation:

```
sigma = sqrt( sum_i p_i * (i - mean)^2 )        confidence = 1 - sigma / sigma_max
```

This also fixes an ordering bug: max-probability ranks a bimodal distribution (mass
at both ends) *above* a uniform one, when bimodal is the worst case for an ordinal —
the returned expected value falls exactly where the model thinks the answer is not.

The floor is corrected for ordinal smoothing. Training deliberately puts `eps` of each
target on neighbouring levels, so a perfectly fitted model still shows sigma ~
sqrt(eps). Without that correction a 3-level score capped at 0.68 confidence, making
the ">= 0.9 act automatically" band of the [routing convention](#the-routing-convention)
unreachable by construction. The v2 recipes use no ordinal smoothing, so the correction is
zero for a v2 model.

### Training

This section describes the v19 reference recipe, a pointer-head recipe. The v1 release
recipes build on it, and [training/README.md](../training/README.md#the-v1-release-recipes)
lists their data and their teachers. The v2 recipes are different, and
[Training a token readout](#training-a-token-readout) describes them.

The loss has three parts:

- **Cross-entropy on the gold label**, with 10% of a `score` target's mass moved to
  adjacent levels: being one level off is a smaller error than being four off.
- **KL to the frozen torso's own readout** (weight 0.3): the same torso with its adapter
  disabled, reading the option-number tokens it would emit after `<answer>`. It keeps the
  model near what pretraining already knew, for one extra forward pass and no extra
  weights.
- **KL to a frozen teacher** (weight 1.0), on the multi-step rows only: Qwen3.5-4B, read
  without any training ([What the torso knows untrained](../research/history.md#what-the-torso-knows-untrained)), its option distributions
  computed once and stored (`data/teacher.py`). It is used only where it is measurably
  better than the student: distilled on short classification, where it is not, it moved
  nothing ([v12](../research/history.md#v12-distilling-a-frozen-qwen35-4b)).

v14's corpus is 100,449 rows from 21 public classification tasks — topic, intent,
sentiment, entailment, ordinal ratings, yes/no, rule chaining — plus 12,909 multi-step
rows from ContractNLI, MuSiQue and BoardgameQA ([Data sources and licences](../data/README.md#data-sources-and-licences)).
The v19 reference recipe adds 3,815 generated document questions and 6,166
answer-adequacy rows.

One detail matters more than any of the loss terms:

> **Option order is re-shuffled on every example, every epoch, and the label is
> remapped to follow.**

Without that, the heads quietly memorise "slot 0 tends to be the positive class" and
genericity is lost. With it, the only strategy that survives gradient descent is
actually reading the option text. Score questions are the exception — their slots are
ordinal, so they are only ever *reversed* (rubric and label together), never permuted.

### Training a token readout

The v2 recipes train a token readout, the 255-code readout of `strands_decider.core` read at
the last prompt token. They train toward the target of each row only: no teacher, no KL
term and no ordinal smoothing. Two keys of the training config select the readout and its
attention:

- `head_type: token_readout`. The head is one linear map, 255 outputs and no bias, on the
  hidden state of the last prompt token. Output k scores the option listed under code k of
  `strands_decider.core` (`A` to `IV`). The prompt is always `strands-chat`, rendered by
  `strands_decider.core.render`. Its control tokens and codes are Gemma 4's token ids, so
  training refuses a base whose tokenizer is not Gemma 4's. Training shuffles the options of
  a choice question only; a yes/no or score question keeps its order. Each
  output starts as the base model's LM head row of that code's token, so before training the
  head gives the base model's own preference among the codes. On Gemma 4 its logits get the
  same cap as the LM head's (`final_logit_softcapping`), saved with the checkpoint as
  `readout_softcap`. The outputs past the question's option
  count are masked.
- `attention_mode: causal` (the default) or `prefix_bidir`, which the v2 recipes set. With
  `prefix_bidir` the prompt splits where `strands_decider.core.render` says the state ends. A state token sees all of
  the state, in both directions, and never the question. A question token sees the state and
  all of its own question. The
  full-attention layers take this mask. The sliding-window layers of Gemma 4 take it too,
  without their window (512 or 1,024 tokens), as the v2 recipe trains. The
  linear-attention layers of Qwen3.5 stay causal. Other layer types are refused. It needs SDPA
  or eager attention.

A token readout starts from the LM head's rows, not at random, so it has its own defaults
for four keys: no dropout before the readout (`head_dropout: 0.0`), the adapters' learning
rate for the readout (`head_lr` equal to `lr`), no weight decay, and AdamW betas
`[0.9, 0.999]`. A pointer run keeps `head_dropout: 0.05`, `head_lr: 1e-3`,
`weight_decay: 0.01` and betas `[0.9, 0.95]`. A key that the config sets wins.

The v2 recipes also use these keys. Each one keeps the earlier behaviour unless a config
sets it.

- `lr_floor` sets where the cosine learning-rate schedule ends, as a fraction of the peak
  (the v2 recipes use 0.1). `adam_betas` sets the AdamW betas.
- `expert_lora: true` adds one LoRA adapter of rank `expert_lora_r` to every routed expert of
  a mixture-of-experts torso (Gemma 4 26B-A4B), on its gate, up and down projections. The
  router stays frozen. Token readout only.
- `attn_split: N` runs `prefix_bidir` on a micro-batch of N padded tokens or more as two
  attention calls without a mask for each row. The attention is the same up to rounding, and
  the kernel is faster.
- `bf16_autocast: true` runs the forward pass under bf16 autocast, so the products of the
  adapters and of the readout run in bf16. The trained weights and the optimizer state stay
  in fp32, and the readout's cap is computed in float32.
- `slice_char_budget: N` caps the padded size of one forward (rows times the longest row, in
  prompt characters). A micro-batch over the cap is cut into more forwards, on one GPU or
  across the ranks of `torchrun`. The rows of a step and the summed gradient do not change.
- With `save_every`, each save also writes `resume-<step>.pt` (the last two are kept). Set
  `resume_from` to one to continue a stopped run, on the same config and number of GPUs, as
  the unbroken run would have.

`configs/v2/` has one config for each Gemma 4 size, and `python -m training.data.strands_v2`
converts v2 rows into training rows
([training/README.md](../training/README.md#the-v2-recipes)).

`python -m training.train`, `training.soup` and `training.evaluate calibrate` work with a token
readout checkpoint. The engines in this repository refuse one. To serve it, export it as the
checkpoint directory that `strands_decider.core` loads (a token readout under `prefix_bidir`
only):

```bash
python -m training.export_v2 <checkpoint> <out> --base-model <hub id>
```

The export writes these files and no others:

- the base's own multimodal shards, with the adapters merged in float32 and saved in bf16,
  and the encoders unchanged;
- `readout.safetensors` and `decision_config.json`;
- the base's `config.json` and `tokenizer.json`, byte for byte, and its
  `processor_config.json` and `tokenizer_config.json` without a chat template;
- `LICENSE`, a `NOTICE` that names the base and what changed, and `README.md`, the model
  card. The card is the file given as `--model-card`. Without one, the export writes a card
  from the runs' records: the LoRA settings, and each run's seed, data file names and steps.

`decision_config.json` holds one temperature: the checkpoint's global one, or the value of
`--temperature`. The per-type temperatures that `calibrate` fits are not carried over. The
export checks the base on disk against the `base.json` digest the run wrote
(`--unrecorded-base` for a run without one), reads its output back with
`strands_decider.core`, and deletes it if a check fails. Several checkpoints of one recipe
before `<out>` export their soup: each merged, every tensor and the readout averaged with
equal weights, temperature 1. The runs may differ in seed, data, learning rate or steps.
A checkpoint that another trainer wrote with its fine-tuning already merged exports with
`--merged-decoder <dir>` in place of the checkpoints: the directory holds the text decoder's
tensors under the base's names without `model.language_model.`, and `readout.safetensors`.
It needs `--base-model`, `--base-revision`, `--max-length-validated` and `--model-card`. The
export does not write `golden_vectors.jsonl`. Serve the checkpoint with the vLLM runtime,
`strands-decider-vllm`. Set `base_revision` in the training config when the base is a local
directory.

## Design decisions worth knowing

These decisions come from the pointer-head models. Where a v2 model does something
different, the item says so.

**Causal attention, read at the right tokens.** The note sketching this architecture likened
it to BERT, and a bidirectional encoder would pool better in principle. But converting
a causal LLM to bidirectional attention requires substantial retraining to recover what
you break. Keeping attention causal uses the pretrained weights exactly as they were
trained, which matters when the adaptation budget is one 3090. Under causal attention
the `<answer>` position has read the whole prompt and each option's last token has read
that option, so they are sound positions to read. A v2 model does not keep this decision:
it reads the state in both directions, and its rank-64 adapters train with that mask
([The v2 token readout](#the-v2-token-readout)).

**Head in fp32.** The torso runs bf16; the head is tiny and a low-precision classifier
is a needless source of calibration error. A v2 readout multiplies in bf16, as in
training, and computes the cap and the softmax in float32.

**Ordinal smoothing for scores.** Being one level off is a much smaller error than
being four off, and plain cross-entropy cannot express that. Score targets put 10% of
their mass on adjacent levels. The v2 recipes use no ordinal smoothing.

**Teacher targets only where the teacher is better.** Distillation transfers the
teacher's behaviour on the prompts it is given. On short classification the frozen 4B
was no better than the student, and distilling there moved nothing on JevBench (v12);
on multi-step documents it was far better, and there it helped (v14). Measure the
teacher against the student on the prompts first. The v2 recipes use no teacher.

**Remove the shortcuts the data allows.** The model learns whatever cheaply predicts the
label: one fixed question per task taught it to ignore questions (v11), and ContractNLI's
per-claim priors would have taught it to skip the contract. Balance them away, keep
minimal pairs together (MuSiQue's answerable and unanswerable versions), and when the
student beats its teacher on a converted dataset, test for artefacts of the conversion
before believing it (v14).

**Drop, never truncate.** A prompt cut through its option list would score an option from
a neighbour's representation. Training and evaluation drop rows over the window; at
serving time the question's tokens are reserved first and the state is cut from the
front (`_fit` in `infer.py`). A v2 runtime cuts nothing: it refuses a prompt that is longer
than its length limit.

**Hybrid torsos share the prefix too.** A Gated DeltaNet layer carries convolution and
recurrent states instead of keys and values. The cache fork (`_expand_cache` in
`infer.py`) repeats those along with the keys and values, into new layer objects, dicts
and tensors — the layer updates its states and flags in place, so a fork sharing them
would corrupt the prefix — following decider-2b's `shared_prefix.py`. A layer holding
any other tensor raises `UnforkableCache`, and the engine falls back to exact batched
encoding rather than guess whether it has a batch dimension. This item is about the
Qwen3.5 torso of the pointer-head models. A v2 runtime can cache the keys and values of the
state for its questions ([The core's runtime interface](#the-cores-runtime-interface)).

## The core's runtime interface

Every v2 runtime imports `strands_decider.core`, so its code and its goldens
(`tests/fixtures/core/`) define the v2 models' decision interface.

A runtime loads a checkpoint with `DecisionConfig.load`, `load_tokenizer`, `load_readout` and,
for images, `check_image_processor`. It supplies the decoder as a `Backend` (`core/runtime.py`),
whose `forward(state, questions, images)` takes the state's token ids, each question's ids and
the state's images (each a `RenderedImage`, with its file, its sha256 and the span of its
placeholders). It runs every question after the shared state under `prefix_bidir`, seeing no
other question, and returns one row per question at its last token. A row is the final-norm
hidden state, or the codes' probabilities for a backend that applies the readout itself.

`evaluate(request, tokenizer, weight, config, backend)` answers one `Request`, which a server
builds with `Request.model_validate_json`. It refuses images when the checkpoint's `inputs` lack
`image`, renders the request, calls `forward` once and reads the answers, from hidden states when
given the readout `weight` and from probabilities when `weight` is None. A backend with no vision
encoder raises `NO_VISION` from `forward`. A runtime may batch, cache the state's keys and values
(keyed on the state's ids and each image's `sha256`), or split the work across devices, provided
each question's result is what it would be alone.

`render` needs a tokenizer that encodes request text with special-token matching off, so the text
can never become a control token. `load_tokenizer` and `prepare_tokenizer` set it up that way.

## Layout

```
src/strands_decider/              -- published package (inference only)
  schema.py       wire types
  confidence.py   the confidence formulas, without dependencies
  prompting.py    state/question rendering; slot <-> option binding
  modeling.py     LLM torso (Qwen3 or hybrid Qwen3.5) + pointer (or slot) readout;
                  masked softmax; KL reference; is_hybrid
  infer.py        serving engine; shared-state cache, including hybrid torsos
  mps_kernels.py  Gated DeltaNet chunk rule for Apple-silicon serving (no fla on macOS)
  cpu_kernels.py  the Gated DeltaNet depthwise causal conv for CPU inference
  mlx_engine.py   serving with the torso on MLX (--device mlx); the engine otherwise infer.py's
  server.py       FastAPI, POST /v1/systemone
  cli.py          the strands-decider command (ask / serve / info)
  hf_export.py    a checkpoint as a Hugging Face model folder (safetensors, card, manifest)
  naming.py       release naming helpers (used by hf_export and release scripts)
  vision.py       Qwen3.5 vision-tower engine (`serve --vision`, `ask --image`)
  core/           the decision interface without torch: prompt rendering, the code
                  readout, decision_config.json, the runtime contract

training/                         -- not published; run as `python -m training.<module>`
  train.py        LoRA + readout training loop (`python -m training.train`)
  evaluate.py     accuracy, ECE, NLL, MAE; temperature fitting; calib/test split
                  (`python -m training.evaluate eval|calibrate`)
  soup.py         average same-recipe checkpoints (`python -m training.soup`)
  distributed.py  multi-GPU training under torchrun: the 1-GPU steps, shared across ranks
  data/           corpus building and inspection (`python -m training.data …`)
    format.py     Example, JSONL io, stratified and held-out-task splits
    recipes.py    HF dataset -> training examples
    collate.py    option shuffling, tokenisation, ordinal targets, option positions
    sampling.py   length-grouped batching
    synth.py      v9 stated-rule generators (not used by the current recipes)
    synth_pairs.py  v10 minimal-pair generators (not used by the current recipes)
    question_transforms.py  v11 question-varied rows (not used by the current recipes)
    teacher.py    frozen-teacher distributions, SemIf's reading (v12, v14)
    multistep.py  ContractNLI / MuSiQue / BoardgameQA / HotpotQA converters (v14)
    policy.py     ShARC / ConditionalQA converters (v15, not used by the current recipe)
    generated.py  generated document questions -> balanced training and eval files (v16);
                  paraphrases attached to built files (v20)
    replay.py     a checkpoint's own distributions as a teacher file (v17)
    adequacy.py   HelpSteer2 -> answer-adequacy rows and evaluation set (v19)
    catchall.py   "other" / "none of these" options on the corpus's choice rows (v20)
    distill.py    teacher distributions only where the teacher agrees with gold (v20)
    shards.py     one labelling process per GPU, and the merge

configs/          train.yaml (the v19 recipe, WSL2), train-parent.yaml (its parent,
                  v14's recipe), train-v7.yaml (v7, Windows)
                  experiments/ -- v11a ... v20 (pre-registered runs)
training/         README.md -- setup, the recipe step by step, several GPUs, hardware notes
                  recipe.sh -- v19 end to end under WSL2: corpus, downloads, multi-step rows,
                  generated and adequacy rows, teacher labels, the parent, replay labels,
                  train, calibrate, eval; and the steps that retrain v20 (catchall, distill)
                  run_recipe.sh -- recipe.sh on one multi-GPU host, timed per stage, with
                  row-count checks and an optional S3 copy
                  recipe_v7.sh -- v7 end to end on Windows (recipe.sh reuses its corpus build)
                  aws/ -- the recipe on one 8-GPU EC2 host over SSM, as run for v17 and v19:
                  README.md, scripts/ (bucket, host, code sync, teardown), image/ (host setup)
evaluation/       README.md -- how the model is measured, and the results;
                  multistep_eval.py (v14's evaluation sets), pair_eval.py (v20's paired
                  evals), question_sensitivity.py (does the answer follow the question?),
                  calibrate_mix.py (the v19 recalibration experiment), pair_accuracy.py
                  (v10's minimal pairs), bench_local.py (latency by state length and
                  question count, any device), device_parity.py (one checkpoint's answers
                  on several devices, against the first);
                  jevbench/ -- jevbench.sh (the external benchmark on a served checkpoint),
                  paired.py (McNemar against a recorded run), jevbench_cold_warm.py (each
                  task twice: first request against repeat)
data/             README.md -- what is committed, what is downloaded, and the licences;
                  SHA256SUMS -- the sha256 of the committed synthetic files, the built
                  corpora and the raw downloads, which recipe.sh checks before it uses them;
                  synthetic/ -- the exact synthetic training and evaluation files and
                  model-produced labels the recipes use; everything else that the recipe
                  writes under data/ is built from public downloads and not committed;
                  generators/ -- llm_client.py (the generators' chat client: OpenRouter,
                  Amazon Bedrock, Bedrock Mantle), gen_documents_openrouter.py (v16),
                  gen_adequacy_openrouter.py (v19), gen_paraphrases_openrouter.py and
                  gen_flips_openrouter.py (v20), gen_documents.py (the Nova pilot), and
                  their exports gen_v16/, gen_weak/, gen_adequacy/, gen_paraphrases/,
                  gen_flips/, gen_mixed_pilot/, gen_pilot_qwen/ (verifier answers included);
                  checks/ -- verify_synth_labels.py, verify_synth_pairs.py (re-derive the
                  v9/v10 corpora's labels from the rendered documents)
examples/         strands/ -- a Strands Agents intervention that gates a tool call on two
                  noul questions (tool_call_intervention.py, with _client.py)
research/         README.md -- the preregistration practice and the version table;
                  preregistrations/ -- PREREGISTRATION-v9.md ... -v20.md (and -v19-seed1,
                  -v19-calmix), predictions fixed before each run, outcomes appended, and
                  README.md (the path map: the files are frozen and name old paths);
                  history.md -- the experiment history; generations.md -- the dated
                  generation-by-generation ledger; data/ -- per-case evidence (JevBench
                  tasks and results, latencies); figures/, and research/scripts/ that
                  collect the data and draw the figures
tests/            pytest only: test_core (no GPU), test_server (stubbed), test_gpu_smoke,
                  test_sampling, test_question_transforms, test_teacher, test_cpu_kernels,
                  test_multistep, test_hybrid, test_policy, test_generated,
                  test_prefix_cache, test_adequacy, test_mps_kernels, test_engine_dtype,
                  test_catchall, test_distill, test_hf_export, test_checkpoint_load,
                  test_cli_build, test_data_identity, test_configs, test_run_recipe,
                  test_llm_client, test_cli_exit,
                  test_ddp (`-m distributed` for the multi-GPU runs);
                  test_core_prompt, test_core_readout, test_core_config, test_core_images
                  (the torch-free core)
docs/             architecture.md -- this document; inference.md -- serving a checkpoint:
                  install, ask, serve, the HTTP API, a Mac
.github/          workflows/tests.yml -- the CPU tests; workflows/generators-live-test.yml
                  and .github/scripts/check_generator_rows.py -- one live batch per generator backend
```

