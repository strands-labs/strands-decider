# The Strands Decider decision interface

This document is the contract of the decision interface that the Strands Decider v2 models use. It covers the request and the response, the files a published checkpoint carries, and what a runtime must reproduce to serve one. Paths are relative to the repository root.

The reference implementation is `strands_decider.core` (`src/strands_decider/core/`). This document gives the template's strings, and section 3.1 says what defines the text a request fills them with. `strands_decider.core.parse_request` and, for images, `strands_decider.core.decode_images` are normative for which malformed request gets which message.

Token ids are those of Gemma 4's tokenizer. A checkpoint's normative tokenizer is its base's `tokenizer.json` at `provenance.base_revision` (section 8). The ids here and in the render goldens (section 3.1) are those of the file with sha256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`, from `google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`. The other three Gemma 4 files that `tests/core_support.py` lists give the same ids on every golden.

## 1. The request and the response

A request is v1's `POST /v1/systemone` body.

| field | value |
| --- | --- |
| `state` | A string, object or array. Optional, `""` when absent. |
| `questions` | An object of at least one question, by name. |
| `model` | A string. Optional. |
| `images` | An array of at most 4 base64 image files (section 1.1). Optional. |
| `audio` | Null, or absent. Any other value is refused. |

Other fields are ignored, as in v1, and values are never coerced. A question has a `type`, `instructions` (a string, object or array) and `criteria`.

| type | criteria | options, in order |
| --- | --- | --- |
| `noul` | Optional. An object with keys among `false` and `true`, each a description or null. | `false`, `true` |
| `choice` | An object of 2 to 255 option names, each to a description (a string, object, array or null). | the object's |
| `score` | An array of 2 to 10 level descriptions (strings), lowest first. | `0` to `L-1` |

An absent noul key keeps its default description, "the statement does not hold for this state" for `false` and "the statement holds for this state" for `true`. A null description, a noul's too, means the option has none. As in v1, a question may omit `type`, and is then a noul when its criteria are absent, null or an object with keys among `false` and `true`, a choice when they are another object, and a score when they are an array.

Content, meaning a state, an instruction, an option's name or a description, nests at most 100 levels deep, where a scalar is level 0 and an object or an array is one level deeper than the deepest value in it, so an empty one is level 1. Numbers that are not finite (`1e400` reads as infinity), integers of more than 4,300 digits and lone surrogates are refused. Every JSON request v1 accepts is accepted, except content past these limits, a non-null `audio` and images that section 1.1 refuses and v1 does not. A request without a state is accepted too.

### 1.1 Images

Each image is a [PNG](https://www.w3.org/TR/png/), [JPEG](https://www.w3.org/Graphics/JPEG/itu-t81.pdf), [WebP](https://www.rfc-editor.org/rfc/rfc9649) or [GIF](https://www.w3.org/Graphics/GIF/spec-gif89a.txt) file, the first frame of an animated one. It is sent in standard base64 ([RFC 4648](https://www.rfc-editor.org/rfc/rfc4648), section 4), optionally behind a `data:image/<subtype>;base64,` prefix, with whitespace stripped from its ends and refused inside it. Each file is at most 16,777,216 bytes (16 MiB), and its header gives at most 16,777,216 pixels (4096 × 4096). A checkpoint whose `decision_config.json` `inputs` lack `image` refuses every request with images.

### 1.2 The response

A response is v1's `SystemOneResponse`, with `model`, `answers` by question name and `usage`. `usage` holds `input_tokens`, which counts the state once and then each question, and `output_tokens`, the number of questions.

Each answer has its question's `type` and these fields.

| type | answer fields |
| --- | --- |
| noul | `noul`, the probability of `true` |
| choice | `choice`, the most probable option (the first on a tie); `probabilities`, by option; `confidence` |
| score | `score`, the expected level; `legend`, each level's text; `probabilities`, by level; `confidence` |

The fields come from section 7's probabilities, in double precision and request order, rounded to four decimal places as in v1. A choice's confidence is (k p_max - 1) / (k - 1) for k options, and a score's is one minus the normalised standard deviation over its levels (`src/strands_decider/confidence.py`).

## 2. The codes

A question with k options binds them, in rendered order, to the first k of 255 codes. The codes are the spreadsheet column names `A` to `IV` without `GZ`, which Gemma 4's tokenizer splits in two, so every code is one token. Serving keeps the request's order, so `A` is a noul's `false` and a score's level 0, and answers are given in the request's terms.

The code token ids are normative. A checkpoint's `decision_config.json` gives them as `token_ids` (`strands_decider.core.CODE_TOKEN_IDS`).

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

### 3.1 The two variants

Each question is one prompt, its body wrapped by the checkpoint's variant. `\n` is a line feed.

```
strands-plain:  <bos>{state block}{question block}\n<answer>
strands-chat:   <bos><|turn>user\n{state block}{question block}<turn|>\n<|turn>model\n
```

The state block holds a line per image (section 3.3), in request order, and then the state. The question block has no newline after `</question>`.

```
<state>
{image lines}
{state}
</state>
<question type="{noul|choice|score}">
{header}
{instructions}
<options>
{code}. {name} — {description}
...
</options>
Answer with the code of one option.
</question>
```

The header is `Decide whether the statement is true of the state.` for a noul, `Select exactly one option.` for a choice and `Rate the state against the ordered levels below (lowest first).` for a score. Prompts are rendered by `strands_decider.core.render`, and the render goldens (`tests/fixtures/core/render_goldens.json`) define the exact text and token ids.

`strands-chat` is what Gemma 4's chat template renders for one user turn holding the body, with the generation prompt added and thinking off, but without the empty thought channel of the 12B, 26B-A4B and 31B templates. A runtime never builds either prompt with a chat template.

### 3.2 Token ids

Control tokens come only from the template and the image lines, and are emitted by id.

| token | id | token | id |
| --- | --- | --- | --- |
| `<bos>` | 2 | `<\|image>` | 255999 |
| `<\|turn>` | 105 | `<\|image\|>` | 258880 |
| `<turn\|>` | 106 | `<image\|>` | 258882 |

Every text run between them is encoded on its own as plain text, with nothing added and no special or added token matched. That is `encode_special_tokens = True` with `add_special_tokens=False` in Hugging Face `tokenizers`, and `add_special = false` with `parse_special = false` in llama.cpp. So `<turn|>` in a state is text. When it loads a checkpoint, a runtime refuses its tokenizer unless each control token has this id, each code is one token with its `token_ids` entry, and every added token is special (`strands_decider.core.load_tokenizer`).

### 3.3 Images

An image's line is `<|image>`, N `<|image|>` placeholders and `<image|>`. N depends on the width w and height h in the file's header and on the image processor's `patch_size` 16, `pooling_kernel_size` 3 and `max_soft_tokens` 280. It is computed in IEEE 754 double precision, one rounded operation at a time, in this order (`strands_decider.core.soft_tokens`).

```
f = sqrt(645120 / (h × w))
H = floor((f × h) / 48) × 48
W = floor((f × w) / 48) × 48
if H = 0:       H = 48,  W = min(floor(w / h) × 48, 13440)
else if W = 0:  W = 48,  H = min(floor(h / w) × 48, 13440)
N = (H / 48) × (W / 48)
```

At aspect ratios such as 35:2 the double's last bit decides the floor. `tests/fixtures/core/image_soft_tokens.json` gives N for 4,906 sizes.

A runtime decodes the file as v1 does (`strands_decider.vision.decode_image`). It refuses a file that fails to decode and one whose decoded size is neither the header's nor its transpose (`strands_decider.core.check_decoded_size`). It resizes the pixels to the W × H of the decoded, oriented size with `strands_decider.core.resize_image`, Pillow's bicubic filter on the width and then on the height. That filter is cubic convolution with a = -0.5, widened by the downscale factor, with weights in 22-bit fixed point, and `spec_resize` in `tests/core_support.py` writes it in numpy. The runtime then runs the checkpoint's image processor with `do_resize=False`, checks that the processor counts N soft tokens, and puts the vision encoder's N vectors in place of the placeholders.

### 3.4 Example

The `department` question of the `support_ticket` golden, in `strands-plain`, is 95 tokens with S = 22.

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

The golden gives its ids.

## 4. The state boundary S

The state part is `<bos>`, `<|turn>user\n` in `strands-chat`, and the state block through its `</state>\n`. S is its length in tokens, the question part is the rest, and L is the prompt's length. A runtime encodes the two parts separately and concatenates their ids, so no token straddles S. It counts S from what it encoded, never by searching for `</state>`, which a state may hold. Every question of a request shares the state part.

## 5. Where the answer is read

The answer is read from the final-norm hidden state at the prompt's last token, position L - 1 (`pooling: "last"`). That is the `>` of `<answer>` (id 236813) in `strands-plain`, and the line feed after `<|turn>model` (id 107) in `strands-chat`.

## 6. The attention mask: `prefix_bidir`

Token i attends to token j as follows.

```
i < S:   j < S     the state sees the whole state, never a question
i >= S:  j < L     a question sees the whole prompt
```

The mask applies on every layer, sliding-window layers included, with no window, and positions are the token indices. Image placeholders are state tokens like any other. No question sees another, and the state never sees a question, so its keys and values serve every question. A runtime that keeps them across requests keys them on the state's ids and each image's sha256, in order (`RenderedImage.sha256`), and reuses them only when the whole state matches, since the state attends both ways.

## 7. The readout

For a question with k options, the readout is as follows.

```
h = bf16(final-norm hidden state at L - 1)                       [hidden]
z = bf16(W h)          W = readout.safetensors `weight`, bf16     [255]
z = tanh(z / c) * c    float32; c = final_logit_softcapping (30.0 for Gemma 4)
z = z / T              float32; T = temperature
p = softmax(z[0:k])    float32, over the question's own k codes
```

`W` has a row per code, in code order, and p[i] is the probability of the option bound to code i.

### 7.1 The precision regime

1. h is rounded to bf16, to nearest with ties to even, so a float32 decoder's h is rounded. W is bf16. The products are summed in float32 or wider, and z is rounded to bf16.
2. The soft-cap and the temperature are computed in float32, one rounded operation at a time, and each is applied once.
3. The softmax is taken in float32.

`strands_decider.core.read` is the reference, in numpy.

## 8. The `decision_config.json` schema

| field | type | meaning |
| --- | --- | --- |
| `interface` | `"strands-decider"` | marks the file as this interface's |
| `format_version` | `1` | this layout |
| `prompt` | `"strands-plain"` or `"strands-chat"` | the variant it was trained with |
| `inputs` | `["text"]` or `["text", "image"]`, in any order | what a request may carry |
| `codes` | the 255 codes, in order | section 2 |
| `token_ids` | 255 distinct integers from 0 to 2^31 - 1 | each code's token id |
| `temperature` | number above 0 | T, section 7 |
| `final_logit_softcapping` | number above 0 | c, section 7 |
| `attention_mode` | `"prefix_bidir"` | section 6 |
| `pooling` | `"last"` | section 5 |
| `max_length_validated` | integer from 1 to 2^31 - 1 | the longest prompt validated, in tokens |
| `provenance` | object | below |

`provenance` holds `base_model` and `base_revision`, which are required, and `release` and `source`, which are optional. None is null, and there are no others. The checkpoint's `config.json`, `processor_config.json`, tokenizer and every part not fine-tuned are the base's at `base_revision`. Each value matches its pattern in full, in ASCII.

| field | pattern |
| --- | --- |
| `base_model`, a Hub id | `[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+` |
| `base_revision`, the base's commit | `[0-9a-f]{40}` |
| `release`, the released model | `([A-Za-z0-9][A-Za-z0-9_.-]*/)?[A-Za-z0-9][A-Za-z0-9_.-]*` |
| `source`, the public training code and recipe | `https://[!-~]+` |

Validation is strict. A runtime reads `interface` first, since other decision-model formats use the same file names, then `format_version`. A missing or unknown field, a value of the wrong type and a `prompt` the runtime does not implement are errors. Integers have no fraction or exponent. `temperature` and `final_logit_softcapping` must be normal float32 values, and `final_logit_softcapping / temperature` finite in float32. A runtime refuses, and never truncates, a prompt longer than `max_length_validated` unless its operator raises the limit. `tests/fixtures/core/decision_config.json` is an example.

## 9. The checkpoint layout

A decision checkpoint is its base model's own multimodal checkpoint, with any fine-tuning merged into the base's tensors under their names, and with these files.

| file | contents |
| --- | --- |
| `config.json` | The base's. Any `final_logit_softcapping` it gives, under `text_config` or at the top level, must equal `decision_config.json`'s. |
| `model.safetensors`, or shards with `model.safetensors.index.json` | The base's tensors in bf16, fine-tuning merged in. No adapter, and no `lm_head`, which Gemma 4 ties to the embedding. |
| `processor_config.json` | The base's. Image requests need `patch_size` 16, `pooling_kernel_size` 3, `max_soft_tokens` 280 and `image_seq_length` 280 (`strands_decider.core.check_image_processor`). |
| `tokenizer.json`, `tokenizer_config.json` | The base's, with no `chat_template`. |
| `readout.safetensors` | One tensor, `weight`, [255, hidden], bf16. |
| `decision_config.json` | Section 8. |
| `golden_vectors.jsonl` | The model goldens, section 10. |
| `LICENSE`, `NOTICE`, `README.md` | The terms, and the model card, which says what was trained. |

The base's tensors fall under these prefixes.

| part | E2B, E4B | 12B | 26B-A4B, 31B |
| --- | --- | --- | --- |
| decoder and token embedding | `model.language_model.` | `model.language_model.` | `model.language_model.` |
| vision encoder | `model.vision_tower.` | `model.vision_embedder.` | `model.vision_tower.` |
| vision projection | `model.embed_vision.` | `model.embed_vision.` | `model.embed_vision.` |
| audio encoder | `model.audio_tower.` | none | none |
| audio projection | `model.embed_audio.` | `model.embed_audio.` | none |

A text-only runtime loads `model.language_model.` alone and refuses images. A checkpoint carries no code and no chat template. A converted artifact, such as a GGUF decoder with its mmproj file or an MLX model, carries `readout.safetensors`, `decision_config.json` and the image processor's settings unchanged.

## 10. Conformance

A runtime conforms for a checkpoint when it passes these goldens.

1. The render goldens, exactly, with the pinned tokenizer.
2. The readout goldens (`tests/fixtures/core/readout_goldens.json`). From each golden h its GEMM gives every z within one bf16 step, and from each golden z it gives every probability within 1e-5.
3. The checkpoint's `golden_vectors.jsonl`, at least 1,000 questions. Each line is an object with `request`, `variant`, `state_ids`, `images` (each with `sha256` and `span`) and `questions`, holding by name each question's `question_ids`, `codes` and the reference's `probabilities`. The reference is the PyTorch model in bf16 under `prefix_bidir`, read as section 7 says, and the set covers every question type, states up to `max_length_validated`, many options, and images where `inputs` holds `image`. The runtime's ids equal the goldens', and it meets the bounds below on top-1 agreement and on the mean of each question's largest probability difference, |Δp| (`strands_decider.core.agreement`).

| weight format | top-1 agreement | mean \|Δp\| |
| --- | --- | --- |
| bf16, any activation precision | at least 98% | at most 0.02 |
| MLX 8-bit (group size 64), at E2B | at least 98% | at most 0.02 |
| GGUF Q8_0, at E2B | at least 98% | at most 0.02 |

Other formats and sizes make no claim. A runtime that serves images also decodes the pixel goldens (`tests/fixtures/core/image_pixels.json`) within 1 per channel, exactly for PNG, GIF and lossless WebP, and resizes their pixels within 1 per channel of `resize_image`. One whose decoder differs by more conforms only by meeting the bounds with that decoder.
