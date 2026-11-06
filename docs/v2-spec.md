# Strands Decider v2: the decision interface

This document defines the Strands Decider v2 decision interface. It specifies how a request becomes token ids, how the decoder attends over them, how its output becomes answers, which files a checkpoint carries, and what a runtime must do to conform. Any runtime that follows it, in any language, gives the same answers for the same checkpoint: PyTorch, vLLM, MLX, llama.cpp, or numpy behind a remote decoder.

The reference implementation of the torch-free half is `strands_decider.v2` (`src/strands_decider/v2/`). It needs only `tokenizers`, numpy and the Python standard library. Its tests are `tests/test_v2_prompt.py`, `tests/test_v2_readout.py` and `tests/test_v2_config.py`, and the render goldens are `tests/fixtures/v2/render_goldens.json`. v2 sits beside v1: the v1 engines (`infer.py`, `hidden_engine.py`, `mlx_engine.py`) and their checkpoints are unchanged.

v2 keeps v1's request and response: the same `POST /v1/systemone` body, the same three question types and limits, and the same answer fields and confidence formulas. Three things change. Options are labelled with letter codes instead of numbers, and the readout scores the codes from the hidden state at the prompt's last token, where v1's pointer head compared the answer position with each option's last token. The state is encoded bidirectionally, and each question sees the whole prompt bidirectionally (the `prefix_bidir` mask). The decoder is Gemma 4, merged to one set of bf16 weights.

Paths are relative to the repository root. Token ids are those of Gemma 4's tokenizer, which every Gemma 4 size ships unchanged: `tokenizer.json` with sha256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`, as in `google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.

## 1. The request

A request is v1's (`src/strands_decider/schema.py`, `SystemOneRequest`):

```
POST /v1/systemone
{"state": ..., "model": "strands-decider-latest", "questions": {"<name>": {...}, ...}}
```

`state` is required. It is a string, a JSON object or a JSON array, and it may be empty when the questions carry the whole task. `questions` maps a name to a question and holds at least one. `model` is an optional string. `images` is an optional list; a text-only runtime refuses a request that has any. Other fields are ignored, as v1 ignores them.

A question has a `type`, `instructions` (a string, object or array) and `criteria`:

| type | criteria | options |
| --- | --- | --- |
| `noul` | optional object with keys from `true` and `false`; each value is a description or null | two, `false` then `true` |
| `choice` | object of option name to description (string, object, array or null) | 2 to 255, in the object's order |
| `score` | array of 2 to 10 level descriptions (strings), lowest level first | one per level, named `0` to `L-1` |

A noul's criteria replace its default descriptions. A key that is absent keeps its default (`false`: "the statement does not hold for this state"; `true`: "the statement holds for this state"). A key given as null renders as a bare label. As in v1, a question may omit `type`: it is a noul when its criteria are absent or an object whose keys are a subset of `true` and `false`, a choice when they are any other object, and a score when they are an array. `strands_decider.v2.parse_request` accepts and refuses exactly what v1's pydantic models accept and refuse, and `tests/test_v2_prompt.py` checks it against them.

## 2. The codes

A question with k options binds them, in rendered order, to the first k of 255 answer codes. The codes are the spreadsheet column names `A` through `Z`, then `AA` through `IV`, without `GZ`: 256 names less one. Gemma 4's tokenizer encodes `GZ` as two tokens (`G`, `Z`), and every other name as exactly one, so every code is a single token. The list is `strands_decider.v2.CODES`.

| question | codes |
| --- | --- |
| noul | `A` is `false`, `B` is `true` |
| choice with k options | `A` is the first option in the request's order, `B` the second, and so on to the k-th code |
| score with L levels | `A` is level 0 (the lowest), `B` level 1, and so on |

Training may permute a question's options before they take codes (`render(..., option_orders=...)`). The rendered question records the binding that results, and answers are always reported in the request's terms, so serving never permutes.

The code token ids, from the checkpoints' `decision_config.json` and checked against the tokenizer by `tests/test_v2_prompt.py`:

```
A=236776 B=236799 C=236780 D=236796 E=236788 F=236811 G=236823 H=236814 I=236777 J=236863
K=236855 L=236798 M=236792 N=236797 O=236806 P=236791 Q=236935 R=236794 S=236773 T=236774
U=236836 V=236847 W=236824 X=236917 Y=236874 Z=236953 AA=8686 AB=3066 AC=2714 AD=2466
AE=14196 AF=11558 AG=2807 AH=20211 AI=12553 AJ=57902 AK=9978 AL=1565 AM=1936 AN=1536
AO=70713 AP=2663 AQ=80055 AR=1425 AS=2371 AT=1375 AU=22258 AV=7008 AW=35066 AX=26428
AY=5790 AZ=25862 BA=6552 BB=20804 BC=5901 BD=11217 BE=12051 BF=28671 BG=54684 BH=48367
BI=11526 BJ=127017 BK=74610 BL=8017 BM=16070 BN=10589 BO=8791 BP=17991 BQ=159610 BR=12357
BS=5580 BT=24219 BU=8462 BV=82085 BW=75007 BX=121922 BY=19470 BZ=153766 CA=5635 CB=26204
CC=3129 CD=6329 CE=4660 CF=13880 CG=17956 CH=2293 CI=9575 CJ=81265 CK=20045 CL=5094
CM=10087 CN=9353 CO=2457 CP=7618 CQ=85549 CR=6459 CS=6786 CT=1692 CU=33186 CV=20741
CW=63746 CX=82279 CY=40736 CZ=129399 DA=7172 DB=6270 DC=12622 DD=21738 DE=4070 DF=7332
DG=65899 DH=54634 DI=9982 DJ=43793 DK=12536 DL=30455 DM=23390 DN=48774 DO=10009 DP=10904
DQ=137414 DR=7560 DS=8925 DT=18540 DU=11965 DV=80850 DW=42771 DX=39399 DY=87207 DZ=164367
EA=37553 EB=56343 EC=5856 ED=2413 EE=10421 EF=28135 EG=63426 EH=101722 EI=73344 EJ=102109
EK=46570 EL=3387 EM=3158 EN=1439 EO=6864 EP=13237 EQ=38254 ER=1089 ES=1740 ET=2406
EU=30722 EV=22486 EW=108534 EX=2201 EY=6594 EZ=79526 FA=4815 FB=38131 FC=11962 FD=35023
FE=17179 FF=5495 FG=40807 FH=75536 FI=8537 FJ=191318 FK=49512 FL=7750 FM=37413 FN=32479
FO=12041 FP=22591 FQ=123622 FR=11869 FS=16164 FT=7627 FU=44519 FV=61688 FW=27330 FX=36991
FY=57954 FZ=175992 GA=10032 GB=8242 GC=11925 GD=29138 GE=8809 GF=15015 GG=30389 GH=32701
GI=17508 GJ=118670 GK=105774 GL=9179 GM=24464 GN=36875 GO=10677 GP=23172 GQ=202788 GR=11124
GS=15683 GT=22618 GU=26943 GV=60572 GW=59141 GX=161184 GY=34144 HA=14983 HB=61026 HC=24084
HD=14563 HE=2808 HF=37030 HG=98100 HH=40325 HI=26722 HJ=139168 HK=31581 HL=17287 HM=42025
HN=74035 HO=10858 HP=18835 HQ=98264 HR=17372 HS=11767 HT=4080 HU=80858 HV=80657 HW=47281
HX=186240 HY=32207 HZ=158347 IA=5299 IB=7692 IC=2011 ID=1735 IE=7742 IF=6354 IG=2996
IH=83145 II=4490 IJ=107007 IK=19994 IL=3334 IM=2783 IN=1204 IO=6462 IP=4012 IQ=79463
IR=3359 IS=2249 IT=1768 IU=87317 IV=3767
```

## 3. The prompt template

Strands defines the template. It keeps v1's layout (`src/strands_decider/prompting.py`): a `<state>` block, then a `<question>` block holding the instructions and an `<options>` list, then the answer cue. Each option line starts with its code.

### 3.1 Rendering content

A state, an instruction or a description renders as text by v1's `render_content`. A string is stripped of leading and trailing whitespace. An object or array becomes JSON from `json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False)`: one item per line with a two-space indent per level, `,` ending every item but the last, `: ` after each key, keys in the request's order, and non-ASCII characters kept. Null renders as the empty string. A state is empty when it renders as the empty string: `""`, or a string of whitespace. `{}` and `[]` render as `{}` and `[]` and are not empty.

### 3.2 The state block

```
<state>\n{state}\n</state>\n
```

With an empty state the content line is left out: `<state>\n</state>\n`. `\n` is one line feed (U+000A); there are no carriage returns.

### 3.3 The question block

The lines below are joined with `\n`. There is no newline after the last line, `</question>`.

```
<question type="{type}">
{header}
{instructions}
<options>
{option line 1}
...
{option line k}
</options>
Answer with the code of one option.
</question>
```

`{type}` is `noul`, `choice` or `score`. `{header}` depends on the type, and is v1's:

| type | header |
| --- | --- |
| noul | `Decide whether the statement is true of the state.` |
| choice | `Select exactly one option.` |
| score | `Rate the state against the ordered levels below (lowest first).` |

`{instructions}` is the question's instructions rendered by section 3.1. A multi-line rendering, such as JSON, stays multi-line. When the instructions render empty, their line is left out, with no blank line in its place. The line `Answer with the code of one option.` is new in v2. It makes a code the natural next token after the prompt, which matters for the chat variant of an instruction-tuned base.

### 3.4 Option lines

Option line i, for the option bound to code c, is

```
{c}. {name} — {description}
```

The separator between the name and the description is a space, U+2014 EM DASH and a space (v1's). Name and description each have every run of whitespace, newlines included, replaced by one space, and leading and trailing whitespace removed, so that an option is always one line. When the description is empty the line is `{c}. {name}`; when the name is also empty it is `{c}.`.

The name is the noul label (`false`, `true`), the choice option's name, or the score level's index (`0`, `1`, ...). The description is the noul's criterion (or its default), the choice option's description, or the score level's rubric text, rendered by section 3.1. A structured description therefore becomes JSON flattened to one line: `{"from": "e1"}` gives `{ "from": "e1" }`.

### 3.5 The two variants

Both variants share one body: the state block followed by the question block. They differ in what surrounds it. A training A/B chooses between them, and `decision_config.json` names the one a checkpoint was trained with.

`strands-v2-plain` is the body as text, with no chat template:

```
<bos>{state block}{question block}\n<answer>
```

`strands-v2-chat` is the body as the single user turn of Gemma 4's chat template, followed by the generation prompt, with thinking off and no system turn:

```
<bos><|turn>user\n{state block}{question block}<turn|>\n<|turn>model\n
```

This is what Gemma 4's `chat_template.jinja` renders for `messages=[{"role": "user", "content": body}]` with `add_generation_prompt=True` and `enable_thinking` false or absent. The template strips the content of leading and trailing whitespace; the body starts with `<state>` and ends with `</question>`, so stripping leaves it unchanged. With thinking off the template emits no system turn, and v2 adds none: the state and question are the whole instruction, and a fixed system turn would add tokens to every state without new information. Section 12 lists this as a question for the A/B.

The chat templates of the larger Gemma 4 sizes differ in one place. With thinking off, the templates of 12B, 26B-A4B and 31B append an empty thought channel, `<|channel>thought\n<channel|>`, after `<|turn>model\n`; those of E2B and E4B do not. `strands-v2-chat` leaves the empty channel out for every size, so that one prompt string serves every checkpoint and the readout always reads the same token. `tests/test_v2_prompt.py` pins each known template, by sha256, to one of the two forms. A checkpoint trained with the empty channel needs its own prompt name.

### 3.6 From text to token ids

A prompt is a sequence of control tokens and text runs. Control tokens come only from the template and are emitted by id:

| control token | id |
| --- | --- |
| `<bos>` | 2 |
| `<\|turn>` | 105 |
| `<turn\|>` | 106 |

Each text run is encoded on its own by the tokenizer as plain text: special-token matching off (in Hugging Face `tokenizers`, `encode_special_tokens = True`) and `add_special_tokens=False`. Gemma 4's tokenizer adds no tokens of its own; its post-processor adds no BOS, which is why the template emits `<bos>` itself. Text from the request therefore never produces a control token: a state containing the characters `<turn|>` encodes them as text.

The pieces of each variant, with S marking the state boundary of section 4:

```
plain:  [<bos>] "<state>\n...</state>\n" S "<question ...>...</question>\n<answer>"
chat:   [<bos>] [<|turn>] "user\n<state>\n...</state>\n" S "<question ...>...</question>" [<turn|>] "\n" [<|turn>] "model\n"
```

For a request whose text contains no control-token string, the ids equal what one call on the whole rendered string gives with special-token matching on, which is how Hugging Face's `apply_chat_template(tokenize=True)` encodes. `tests/test_v2_prompt.py` checks this on the goldens and on randomly generated requests.

### 3.7 Examples

The request (`support_ticket` in the goldens):

```json
{"state": "Help! My payouts have been failing for 3 days.",
 "questions": {
   "is_urgent":   {"type": "noul", "instructions": "Does this convey urgency?"},
   "department":  {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "payments, invoices, payouts",
                                "technical": "bugs, outages, API errors",
                                "sales": "pricing and upgrades"}},
   "frustration": {"type": "score", "instructions": "How frustrated is the writer?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}}}
```

The `department` question in `strands-v2-plain`, 95 tokens, S = 22:

```
<bos><state>
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
</question>
<answer>
```

```
state ids [0, 22):   2 236820 3255 236813 107 28368 236888 3551 127325 735 1010 26929 573 236743
                     236800 2668 236761 107 954 3255 236813 107
question ids [22, 95): 236820 15884 1722 718 17005 905 107 11323 7121 886 5086 236761 107 24249
                     2434 1374 6060 672 236881 107 236820 8194 236813 107 236776 236761 47764 2192
                     13192 236764 77365 236764 127325 107 236799 236761 8330 2192 34725 236764
                     101990 236764 9299 9825 107 236780 236761 5886 2192 21212 532 42392 107 954
                     8194 236813 107 7925 607 506 3393 529 886 5086 236761 107 954 15884 236813 107
                     236820 14433 236813
code token ids:      A 236776, B 236799, C 236780 (billing, technical, sales)
```

The same question in `strands-v2-chat`, 99 tokens, S = 25. The text ends with a newline after `model`:

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

```
state ids [0, 25):   2 105 2364 107 236820 3255 236813 107 28368 236888 3551 127325 735 1010
                     26929 573 236743 236800 2668 236761 107 954 3255 236813 107
question ids [25, 99): the plain variant's first 69 question ids (through `</question>`), then
                     106 107 105 4368 107
code token ids:      A 236776, B 236799, C 236780 (billing, technical, sales)
```

The goldens hold every question type in both variants, with structured content, an empty state, 28 options (codes to `AB`), control-token strings in the state and a ten-level score.

## 4. The state boundary S

S is the number of tokens in the state part. Tokens [0, S) are the state part and [S, L) the question part, where L is the prompt's length. The state part is everything up to and including the line feed that ends `</state>\n`; the question part starts with `<question`. Everything before S is identical for every question about the state: `<bos>`, in the chat variant `<|turn>user\n`, and the state block. A runtime encodes it once and shares it across the request's questions.

The rule: the state part and the question part are encoded separately, as section 3.6 describes, and the prompt's ids are their concatenation. No token can straddle S, because no encoding call crosses it. A runtime must use the ids of the two parts and must not re-encode the joined text.

For Gemma 4's tokenizer the rule costs nothing: the separate encodings equal the encoding of the joined text, so S always falls on a token boundary of the whole prompt. The argument has three steps.

1. A token that straddled S would contain the line feed before S and the `<` after it. In Gemma 4's vocabulary of 262,144 tokens, every token that contains a line feed consists only of line feeds (31 such tokens, up to 31 line feeds long), and none of its 514,906 merges joins a line feed to another character.
2. The normalizer replaces each space with U+2581 and does nothing else; it maps characters one to one and sees no context. The pre-tokenizer splits on spaces, which no longer exist after normalization (and a split could only add boundaries). The post-processor adds no tokens, and BPE runs without dropout.
3. So no merge across S is ever possible. BPE applies merges in rank order, and merges on one side of S never create or remove a candidate pair on the other, so each side ends with the merges it makes alone.

`tests/test_v2_prompt.py` proves this in two ways. `test_tokenizer_cannot_merge_across_a_line_end` checks the premises of steps 1 and 2 against the tokenizer file itself. `test_s_is_a_token_boundary` encodes 120 random requests in both variants with the two text runs at S joined into one, including requests full of control-token strings, line ends, tabs, non-ASCII text and option-like text, and gets the same ids as the rule. `test_ids_match_one_call_encoding` checks that a token starts at exactly S's character offset in the one-call encoding. A future tokenizer that fails these premises keeps the rule well defined (separate encoding), but its ids may then differ from those a text-only tool produces, so this spec would need a new seam for it.

## 5. Where the answer is read

The answer is read from the hidden state at the prompt's last token, position L - 1 (`pooling: "last"`). In `strands-v2-plain` that token is `>` (id 236813), the end of `<answer>`. In `strands-v2-chat` it is the line feed (id 107) after `<|turn>model`. Every question of a variant ends with the same tokens: plain with `\n<answer>` (107 236820 14433 236813), chat with `<turn|>\n<|turn>model\n` (106 107 105 4368 107).

## 6. The attention mask: `prefix_bidir`

Token i attends to token j when

```
i < S:   j < S        (the state sees the state, in both directions, and never a question)
i >= S:  j < L        (a question sees the whole prompt, in both directions)
```

The mask applies on every layer, sliding-window layers included, and replaces their window: no layer limits attention by distance. Positions are the token indices, 0 to L - 1, so every question's first token sits at position S. Each question is its own sequence over the shared state; no question sees another. Because the state never attends to a question, its keys and values are the same for every question and can be computed once. `strands_decider.v2.attention_mask(S, L)` gives the mask as booleans, row i for token i.

## 7. The readout

For each question with k options:

```
h  = the final-norm hidden state at position L - 1                  [hidden]
z  = W h        W = readout.safetensors `weight`                    [255]
z  = c * tanh(z / c)     c = config.json `final_logit_softcapping`  (30.0 for every Gemma 4 size)
z  = z / temperature     decision_config.json `temperature`
p  = softmax(z[0:k])     over the question's own k codes only
```

`W` has one row per code, in code order. The other 255 - k codes do not enter the softmax. `p[i]` is the probability of the option bound to code i, and maps back to the request's terms by section 2. Arithmetic in the reference is float64.

The answer fields are v1's response schema (`schema.py`, `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`), rounded to four decimal places as v1 rounds:

| type | fields |
| --- | --- |
| noul | `noul`: P(true) |
| choice | `choice`: the option with the highest probability (the first in request order on a tie); `probabilities`: option name to probability, in request order; `confidence` |
| score | `score`: the expected level, sum of i times p_i; `legend`: level to rubric text (stripped), ascending; `probabilities`: level to probability, ascending; `confidence` |

Confidence uses v1's formulas, which v2 keeps unchanged and shares as code (`src/strands_decider/confidence.py`). A choice's confidence is the normalised top probability, (k p_max - 1) / (k - 1): 0 for a uniform distribution, 1 for a certain one, whatever k is. A score's is one minus the normalised standard deviation over the levels, with the optional floor correction for ordinal smoothing (`ordinal_smoothing`, 0 by default). A noul has no confidence field, because its probability already is its uncertainty. A response is v1's `SystemOneResponse`: `model`, `answers` keyed by question name, and `usage` with `input_tokens` (the state once, then each question) and `output_tokens` (the number of questions).

## 8. The `decision_config.json` schema

`decision_config.json` holds what a runtime needs, beyond the decoder weights and the tokenizer, to serve a checkpoint. Format version 1:

| field | type | meaning |
| --- | --- | --- |
| `format_version` | integer, 1 | this layout |
| `prompt` | string | the template the checkpoint was trained with: `strands-v2-plain` or `strands-v2-chat` |
| `codes` | array of 255 strings | the codes of section 2, in order; anything else is an error |
| `token_ids` | array of 255 distinct non-negative integers | each code's token id, in code order |
| `temperature` | number, finite and above 0 | the divisor of section 7 |
| `attention_mode` | string, `prefix_bidir` | the mask of section 6 |
| `pooling` | string, `last` | the readout position of section 5 |
| `max_length_validated` | positive integer | the longest prompt, in tokens (S plus the question's tokens), at which the checkpoint was validated |
| `provenance` | object, optional | public origin of the checkpoint; fields below |

`provenance` has four optional string fields and no others: `base_model`, a Hub id such as `google/gemma-4-E2B-it`; `base_revision`, the base model's 40-character commit; `release`, the released model's name or Hub id; and `source`, an https URL of the public code and recipe that trained it. No provenance value may be a filesystem path (absolute, home-relative, relative with `./` or `../`, a Windows path, or a path under `/fsx`), or a URI with any scheme but https.

Validation is strict. A missing required field, an unknown field at any level, and a value of the wrong type are errors, and a runtime refuses the checkpoint. A runtime must refuse a `prompt` it does not implement, rather than fall back to another; a later variant gets a new name. A runtime refuses a prompt longer than `max_length_validated` tokens unless its operator raises the limit explicitly, and it refuses rather than truncates. Training-side records, such as data paths, hyperparameters and the inputs to a weight average, do not belong in this file.

An example (codes and token ids shortened):

```json
{
  "format_version": 1,
  "prompt": "strands-v2-chat",
  "codes": ["A", "B", "C", "...", "IV"],
  "token_ids": [236776, 236799, 236780, "...", 3767],
  "temperature": 1.0,
  "attention_mode": "prefix_bidir",
  "pooling": "last",
  "max_length_validated": 8192,
  "provenance": {"base_model": "google/gemma-4-E2B-it",
                 "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7"}
}
```

`strands_decider.v2.DecisionConfig.load` reads and validates the file; `tests/test_v2_config.py` lists the cases it refuses.

## 9. The checkpoint layout

A v2 checkpoint is a directory that holds:

| file | contents |
| --- | --- |
| `config.json` | the decoder's configuration: `architectures: ["Gemma4TextModel"]`, `model_type: gemma4_text`, `final_logit_softcapping` |
| `model.safetensors` or `model-*-of-*.safetensors` with `model.safetensors.index.json` | the Gemma 4 text decoder in bf16, any fine-tuning merged in: no adapter, no vision or audio tower, no language-model head |
| `tokenizer.json`, `tokenizer_config.json` | Gemma 4's tokenizer, unchanged |
| `chat_template.jinja` | Gemma 4's chat template, for reference; runtimes render by this spec, not by the template |
| `readout.safetensors` | one tensor, `weight`, shape [255, hidden], bf16 or f32 |
| `decision_config.json` | section 8 |
| `golden_vectors.jsonl` | the model goldens of section 10 |
| `LICENSE`, `NOTICE`, `README.md` | the terms and the model card |

A checkpoint carries no code. Runtimes implement this spec; they do not import code shipped inside a checkpoint.

A converted artifact for another runtime, such as an MLX model or a GGUF file, carries the decoder in that runtime's format and the same `readout.safetensors` and `decision_config.json`, unchanged, beside it. The readout stays in safetensors, outside the converted decoder, so that every format reads the same rows.

## 10. Golden vectors and conformance

Two kinds of golden data pin the interface.

Render goldens fix the prompt. `tests/fixtures/v2/render_goldens.json` holds, for each case and variant, the request, the state text, S, the state ids, and for each question its text, ids, codes, code token ids and bound options. They depend only on the tokenizer. A conforming runtime reproduces them exactly: the same ids, the same S, the same codes. There is no tolerance.

Model goldens fix the answers of one checkpoint. Each checkpoint release publishes `golden_vectors.jsonl` in its directory. Each line holds a request, its rendered ids per question, and the reference code probabilities per question, computed by the reference runtime with the checkpoint's bf16 weights and float32 activations. The reference runtime is the PyTorch decoder with the `prefix_bidir` mask behind `strands_decider.v2.evaluate`; it is not part of this torch-free core. The set covers every question type, both short and long states up to `max_length_validated`, and choices with many options.

A runtime conforms for a checkpoint when, on that checkpoint's model goldens:

1. its rendered ids equal the goldens' ids for every question;
2. running the weights in bf16, its top-1 code (the code with the highest probability) agrees with the reference's on at least 98% of the questions;
3. running quantized weights, it meets the bound recorded for that format in the table below.

| weight format | top-1 agreement bound |
| --- | --- |
| bf16 | 98% |
| quantized (for example MLX 8-bit or 4-bit, GGUF Q8_0 or Q4_K_M) | set from measurement when the format is first published, and recorded here |

A quantized artifact whose format has no recorded bound makes no conformance claim.

## 11. The runtime interface

A runtime runs the decoder. Given the state ids (positions [0, S)) and the question ids of each question, it runs every question as a continuation of the state under the `prefix_bidir` mask, seeing no other question, and returns the final-norm hidden state at each question's last token: an array [questions, hidden], in the order of the questions. A runtime may instead apply the readout itself and return the raw readout logits `W h`, [questions, 255], before the soft-cap; or go further and return each question's code probabilities over its own k codes.

This is v1's `hidden_engine.Backend` narrowed to v2. v1's backend returns several rows per question, one per option and one for `<answer>`, for the pointer head; v2 needs only the last token. `strands_decider.v2.Backend` states the contract:

```python
class Backend(Protocol):
    def forward(self, state: list[int], questions: list[list[int]]) -> np.ndarray:
        """[len(questions), hidden]: each question's final-norm hidden state at its last
        token, under prefix_bidir, after the shared state."""
```

`strands_decider.v2.evaluate(request, tokenizer, readout, config, backend)` joins the pieces. It renders the request with the checkpoint's prompt, calls `forward` once with the shared state and every question, reads the answers, and returns a `SystemOneResponse` as JSON. `read(..., inputs="logits")` takes logits in place of hidden states, and `answer(rendered_question, probabilities)` takes probabilities. As with v1's backend, a runtime may batch, cache the state's keys and values, or split the work across devices, provided each question's result is what it would be alone.

## 12. Open questions

These are not yet decided, and the training A/B or the first release should settle them.

1. Whether a system turn helps `strands-v2-chat`. It would sit before S, so it would cost tokens once per state, not per question.
2. Whether `strands-v2-chat` should include the empty thought channel that the 12B, 26B-A4B and 31B templates add with thinking off (section 3.5). If those sizes train better with it, they need a third prompt name.
3. The noul's code assignment: `false` is `A` and `true` is `B`, following v1's slot order. If letter position biases the base model, training with random swaps (as for choices) removes the bias from the weights either way.
4. Whether the line `Answer with the code of one option.` earns its tokens once the readout is trained.
5. Whether `ordinal_smoothing`, which v1 kept in its head configuration, belongs in `decision_config.json`.
6. The quantized-format bounds of section 10.
