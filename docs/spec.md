# The Strands Decider decision interface

This document defines the Strands Decider decision interface, which the Strands Decider v2 models use. It specifies how a request, its text and its images, becomes token ids, how the decoder attends over them, how its output becomes answers, which files a checkpoint carries, and what a runtime must do to conform. Any runtime that follows it, in any language, gives answers within section 10's bounds of the reference's for the same checkpoint: PyTorch, vLLM, MLX, llama.cpp, or numpy behind a remote decoder.

The reference implementation of the torch-free half is `strands_decider.core` (`src/strands_decider/core/`). It needs only `tokenizers`, numpy and the Python standard library; it reads images from their file headers and needs no imaging library. The `strands-decider` distribution declares both, but it also requires torch and transformers for v1's engines, so installing it installs torch. A torch-free install, for AWS Lambda for one, needs a separate distribution or an extra without torch; which one is not yet decided (section 12). Its tests are `tests/test_core_prompt.py`, `tests/test_core_readout.py`, `tests/test_core_config.py` and `tests/test_core_images.py`, and the render goldens are `tests/fixtures/core/render_goldens.json`. The core sits beside the v1 engines (`infer.py`, `hidden_engine.py`, `mlx_engine.py`); they and their checkpoints are unchanged.

The interface keeps v1's request and response: the same `POST /v1/systemone` body, the same three question types and limits, and the same answer fields and confidence formulas. Three things change. Options are labelled with letter codes instead of numbers, and the readout scores the codes from the hidden state at the prompt's last token, where v1's pointer head compared the answer position with each option's last token. The state is encoded bidirectionally, and each question sees the whole prompt bidirectionally (the `prefix_bidir` mask). The model is Gemma 4, merged to one set of bf16 weights. A request may also carry images, as v1's may. The base model's own vision encoder reads them, and they go first inside `<state>`, laid out as v1 lays out images for Gemma 4 (strands-decider pull request 67).

Paths are relative to the repository root. Token ids are those of Gemma 4's tokenizer. The tokenizer that is normative for a checkpoint is its own, its base model's `tokenizer.json` at `provenance.base_revision` (section 9), which a runtime checks when it loads the checkpoint (section 3.6). Gemma 4's repositories ship four slightly different `tokenizer.json` files, listed in section 3.6. The ids in this document and in the render goldens are those of the pinned file, with sha256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`, as in `google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.

## 1. The request

A request is v1's (`src/strands_decider/schema.py`, `SystemOneRequest`):

```
POST /v1/systemone
{"state": ..., "images": [...], "model": "strands-decider-latest", "questions": {"<name>": {...}, ...}}
```

`state` is required. It is a string, a JSON object or a JSON array, and it may be empty when the questions or the images carry the whole task. `questions` maps a name to a question and holds at least one. `model` is an optional string. Audio input is not supported: a request with a non-empty `audio` field is refused with `audio input is not supported`, never ignored. `images` is optional: a list of at most 4 strings, each a base64 image file, PNG, JPEG, WebP or GIF, which may carry a `data:image/...;base64,` prefix. The images are part of the state, in the order given (section 3.2). Null, or anything but a list of strings, is an error, as in v1. A runtime that cannot take images refuses a non-empty list before it decodes any image (section 11). Other fields are ignored, as v1 ignores them.

A question has a `type`, `instructions` (a string, object or array) and `criteria`:

| type | criteria | options |
| --- | --- | --- |
| `noul` | optional object with keys from `true` and `false`; each value is a description or null | two, `false` then `true` |
| `choice` | object of option name to description (string, object, array or null) | 2 to 255, in the object's order |
| `score` | array of 2 to 10 level descriptions (strings), lowest level first | one per level, named `0` to `L-1` |

A noul's criteria replace its default descriptions. A key that is absent keeps its default (`false`: "the statement does not hold for this state"; `true`: "the statement holds for this state"). A key given as null renders as a bare label. As in v1, a question may omit `type`: it is a noul when its criteria are absent or an object whose keys are a subset of `true` and `false`, a choice when they are any other object, and a score when they are an array.

The request is JSON, read as Python's `json.loads` reads it. A key that appears twice in an object keeps its last value, at the place of its first appearance. Beyond v1's checks, a request is refused when its state, an instruction, an option's name or a description holds what the template cannot render alike in every runtime (section 3.1): a number that is not finite (`NaN`, `Infinity` and `-Infinity`, which Python's parser reads), an integer of more than 4,300 digits, a string with a lone surrogate (a `\uD800` to `\uDFFF` escape that is not half of a pair, which has no UTF-8 form to tokenize), an object key that is not a string, a value that is not JSON (from a caller that builds the request in code), or nesting more than 100 levels deep. A scalar is at level 0, and an object or an array is one level deeper than the deepest value in it, so `[[1]]` is 2 levels deep. v1 renders `NaN` and `Infinity` as those words, which are not JSON; Python 3.10.7 and later refuse such integers already; and v1 renders deeper nesting until, past about 1,000 levels, it fails with a `RecursionError` (an HTTP 500). The fixed lower limit is one every runtime can meet.

`strands_decider.core.parse_request` accepts and refuses exactly what v1's pydantic models accept and refuse, except for the refusals of the paragraph above and a non-empty `audio` field, which v1 ignores; `tests/test_core_prompt.py` checks it against them.

### 1.1 Images

After the request's types are checked, its images are checked as v1's image engine checks them (`src/strands_decider/vision.py`, `decode_image`), in this order and with v1's messages, except for base64 (step 2):

1. More than 4 images: `{n} images; this server takes at most 4`. The count is checked before any image is decoded.
2. Each string loses its leading and trailing whitespace (section 3.1's 29 code points, which are Python's `str.strip`'s), then a prefix that matches `^data:image/[A-Za-z0-9_.+-]+;base64,`, whatever subtype it names. v1's prefix is `[\w.+-]+`, whose `\w` follows the interpreter's Unicode tables, so the two differ for a subtype that is not ASCII, which the core does not strip. The rest must be standard base64 (RFC 4648, section 4) and nothing else: only the 64 characters `A`–`Z`, `a`–`z`, `0`–`9`, `+` and `/`, in groups of four, the last group padded with `=` or `==` when it is short, with no whitespace or line break inside and no URL-safe alphabet. Non-zero bits before the padding are accepted. Otherwise: `image is not valid base64: {reason}`, where the reason is `a character that is not ASCII`, `a character outside the standard base64 alphabet` or `its length or padding is wrong`. This rule does not follow Python's `base64.b64decode(s, validate=True)`, which accepts different strings, and gives different messages, on Python 3.10, 3.11, 3.12 and 3.13; it is what 3.12 accepts.
3. The decoded file is a PNG, JPEG, WebP or GIF file, by its signature, whose header gives its width and height by the rules below. Otherwise: `image could not be read as PNG/JPEG/WEBP/GIF: {reason}`. A width or a height of 0 is an error.
4. Its width times its height is at most 4096 × 4096 = 16,777,216 pixels. Otherwise: `image is {w}x{h}; at most 16,777,216 pixels are accepted`.

The reasons after the colons are the core's own, for base64 and for headers, where v1 passes on Python's and Pillow's. The header gives the width and height:

| format | signature | width and height |
| --- | --- | --- |
| PNG | `89 50 4E 47 0D 0A 1A 0A` | The first 8 data bytes of the IHDR chunk: width, then height, 4-byte big-endian. The chunks are walked from byte 8 up to the first IDAT, which must be reached. Each chunk has a type of four ASCII letters, lies within the file, and has a matching CRC-32. IHDR has at least 13 data bytes. It may come anywhere before IDAT, and a later IHDR replaces an earlier one (the PNG specification puts IHDR first; Pillow does not insist). |
| JPEG | `FF D8 FF` | The last frame header before the first scan (SOS, `FFDA`) or the file's end: SOF0 to SOF3, SOF5 to SOF7, SOF9 to SOF11, SOF13 to SOF15, or DHP (`FFDE`). After its 2-byte length come the sample precision, which must be 8, then the height and the width, 2-byte big-endian, then the number of components, which must be 1, 3 or 4. Markers are read from byte 2 as Pillow reads them. A byte other than `FF` between segments is skipped, `FF FF` is fill, and `FF 00` is skipped. Markers `C8`, `D0` to `D9` and `F0` to `FD` carry no length. Every other marker in `C0` to `CF`, `DA` to `DF`, `E0` to `EF` and `FE` carries a 2-byte big-endian length that counts itself, and any other marker is an error. |
| WebP | `RIFF`, 4 bytes, `WEBP`, then a first chunk `VP8 `, `VP8L` or `VP8X` | The RIFF size (bytes 4 to 7, little-endian) is at least 12, and at most the file's length less 8. The first chunk's size (bytes 16 to 19) is at most the RIFF size less 12. `VP8 ` (lossy) checks its 3-byte little-endian frame tag at byte 20: bit 0 is 0 (a key frame), bits 1 to 3 (the profile) are at most 3, bit 4 (shown) is 1, and bits 5 to 23 (the first partition's size) are less than the chunk's size. The start code `9D 01 2A` follows at byte 23, then the width and the height, each the low 14 bits of a 2-byte little-endian integer, at bytes 26 and 28. `VP8L` (lossless) has the signature `2F` at byte 20, then a 4-byte little-endian integer whose bits 0 to 13 are the width less 1, bits 14 to 27 the height less 1, and bits 29 to 31 (the version) 0. `VP8X` (extended) has a chunk size of at least 10, and the canvas width less 1 and height less 1 as 3-byte little-endian integers at bytes 24 and 27. |
| GIF | `GIF87a` or `GIF89a` | The logical screen's width and height, 2-byte little-endian at bytes 6 and 8, each enlarged to cover the first image: its left offset plus its width, and its top offset plus its height, from the first image descriptor (`2C`). Before the descriptor come the global colour table, 3 × 2^(1 + (byte 10 & 7)) bytes if bit 7 of byte 10 is set, then extension blocks (`21`, a label, then data sub-blocks up to an empty one), which are skipped; any other byte between blocks is skipped too. Reaching the trailer (`3B`) or the file's end before a descriptor is an error. The descriptor, its local colour table and the byte after them must lie within the file. |

These are the fields Pillow reads when v1 opens an image, and `tests/test_core_images.py` checks that, wherever the core and Pillow both read a file, they read the same format and size; that files cut short are accepted and refused alike; and that the core never refuses a file Pillow opens. Pillow, through libwebp and its JPEG plugin, checks some fields past the size as well, so a corrupted file can pass these rules and still fail when the runtime opens or decodes it. The runtime then refuses the request with v1's message, `image could not be read as PNG/JPEG/WEBP/GIF: {reason}` or `image could not be decoded: {reason}`. `strands_decider.core.decode_image` implements the four checks without decoding any pixel, and `tests/test_core_images.py` checks it against v1's `decode_image` where Pillow and torch are installed.

## 2. The codes

A question with k options binds them, in rendered order, to the first k of 255 answer codes. The codes are the spreadsheet column names `A` through `Z`, then `AA` through `IV`, without `GZ`: 256 names less one. Gemma 4's tokenizer encodes `GZ` as two tokens (`G`, `Z`), and every other name as exactly one, so every code is a single token. The list is `strands_decider.core.CODES`.

| question | codes |
| --- | --- |
| noul | `A` is `false`, `B` is `true` |
| choice with k options | `A` is the first option in the request's order, `B` the second, and so on to the k-th code |
| score with L levels | `A` is level 0 (the lowest), `B` level 1, and so on |

Training may permute a question's options before they take codes (`render(..., option_orders=...)`): a choice's options in any order, a noul's two either way, and a score's levels in order or reversed, as v1's training reverses them, but not otherwise, since the score's header says the levels are listed lowest first (a reversed score keeps that header, as in v1). An order lists the option indices, integers, each once. The rendered question records the binding that results, and answers are always reported in the request's terms, so serving never permutes.

The code token ids, from the checkpoints' `decision_config.json` and checked against the tokenizer by `tests/test_core_prompt.py`:

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

Strands defines the template. It keeps v1's layout (`src/strands_decider/prompting.py`): a `<state>` block, which holds the images first, then a `<question>` block holding the instructions and an `<options>` list, then the answer cue. Each option line starts with its code.

### 3.1 Rendering content

A state, an instruction or a description renders as text by v1's `render_content`. A string is stripped of leading and trailing whitespace, and nothing else in it changes: a carriage return inside it, for one, stays. Null renders as the empty string. A state is empty when it renders as the empty string: `""`, or a string of whitespace. `{}` and `[]` render as `{}` and `[]` and are not empty.

Whitespace, here and wherever this spec collapses it, is these 29 code points, which are Python's `str.isspace` on every Python from 3.10 to 3.13: U+0009 to U+000D, U+001C to U+001F, U+0020, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F and U+3000. U+200B ZERO WIDTH SPACE and U+FEFF (a byte order mark) are not whitespace and are kept.

An object or an array becomes JSON as Python's `json.dumps(value, indent=2, ensure_ascii=False)` writes it:

- **Layout.** An empty object is `{}` and an empty array `[]`. Otherwise the opening bracket ends its line, each member or item has a line of its own, indented two spaces per level of nesting, every member or item but the last ends with `,`, and the closing bracket has a line of its own at its container's indent. A member is its key as a string, then `: `, then its value. Keys keep the request's order.
- **Strings**, keys included, are quoted with `"`. Seven characters are escaped by name: `"` as `\"`, `\` as `\\`, U+0008 as `\b`, U+0009 as `\t`, U+000A as `\n`, U+000C as `\f` and U+000D as `\r`. Every other code point below U+0020 is escaped as `\u00` and two lowercase hex digits (U+001F as `\u001f`). Nothing else is escaped: `/`, U+007F, U+2028, U+2029 and all non-ASCII text are written as they are.
- **Literals** are `true`, `false` and `null`.
- **Integers.** A JSON number written without a fraction or an exponent is an integer, written in decimal with a `-` when it is negative: `-0` is written `0`.
- **Doubles.** Any other JSON number is a double, the IEEE 754 binary64 value nearest the number as written, ties to even, and is written as Python's `repr` writes a float: the shortest digits that read back as the same double, `d₁d₂…dₙ × 10^e` with d₁ not 0. When -4 ≤ e < 16 they are written in positional notation with at least one digit after the point: `1.0`, `1000000000000000.0`, `0.0001`, `0.30000000000000004`, `-0.0`. Otherwise they are written as d₁, then `.` and the other digits if there are any, then `e`, the exponent's sign and at least two digits of it: `1e-05`, `1.5e+16`, `1e+300`. So `1e3` is written `1000.0`, and `1.0` stays `1.0`.

The requests' refusals (section 1) leave out what these rules cannot write: numbers that are not finite, integers past 4,300 digits and lone surrogates.

### 3.2 The state block

```
<state>\n{state}\n</state>\n
```

With an empty state the content line is left out: `<state>\n</state>\n`. `\n` is one line feed (U+000A). As for carriage returns, the template adds none; request text is kept as sent.

Images come first, one line each in the request's order, then the state's line. This is v1's layout for Gemma 4 images:

```
<state>
<|image><|image|>...<|image|><image|>
{state}
</state>
```

With n images the block is `<state>\n{image 1}\n{image 2}\n...{image n}\n{state}\n</state>\n`. An image's line is `<|image>`, then N copies of `<|image|>`, then `<image|>`, with nothing between them. With images and an empty state, the state's line is left out and the last image's line is followed by `\n</state>\n`: `<state>\n{image 1}\n...{image n}\n</state>\n`. A state is empty by the rule of section 3.1. Without images the block is as above, and the question part is the same with images as without them. The three image tokens are control tokens, emitted by id (section 3.6).

N is the number of soft tokens Gemma 4's image processor gives the image (`num_soft_tokens_per_image`), one placeholder for each vector the vision encoder puts in the prompt. It depends only on the image's width w and height h, from its header (section 1.1), and on three settings of the processor, which every Gemma 4 base's `processor_config.json` sets alike: `patch_size` 16, `pooling_kernel_size` 3 and `max_soft_tokens` 280, which is also its `image_seq_length`. The processor resizes the image, keeping its aspect ratio, to the largest size whose sides are multiples of 48 pixels (3 × 16) and whose area is at most 645,120 pixels (280 × 3² × 16²). It then pools each 3 × 3 group of 16-pixel patches into one soft token. In IEEE 754 double precision, each operation rounded to nearest, as Python's floats, `math.sqrt` and `math.floor` give it:

```
f = sqrt(645120 / (h × w))          h × w is exact; the quotient, then its square root
H = floor((f × h) / 48) × 48        the resized height
W = floor((f × w) / 48) × 48        the resized width
if H = 0:     H = 48,  W = min(floor(w / h) × 48, 13440)
else if W = 0: W = 48, H = min(floor(h / w) × 48, 13440)
N = (H / 48) × (W / 48)
```

The two branches apply only to images more than 280 times as wide as they are tall, or as tall as they are wide, and both give N = 280; 13440 is 280 × 48. N is at most 280. Every square image has N = 256, a 64 × 48 image 266, a 300 × 40 image 270, and a 1 × 100 image 167. Swapping w and h leaves N unchanged, so EXIF orientation does not change it, although v1 applies the orientation before the processor sees the image and so may swap the two. The resized width and height do swap: section 11.1 computes them from the decoded, oriented size. These are the steps of `get_aspect_ratio_preserving_size` in transformers' `Gemma4ImageProcessor`, which E2B, E4B, 26B-A4B and 31B use, and `Gemma4UnifiedImageProcessor`, which the 12B uses. The two processors give the vision encoder different inputs: 16-pixel patches with their positions, padded to 2,520, against 3 × 3 groups merged into 48-pixel patches, padded to 280. But they size the image with the same function and settings, and so count N alike, and one rule serves every size. `strands_decider.core.soft_tokens` implements it. `tests/fixtures/core/image_soft_tokens.json` holds N as both processors gave it for 3,200 sizes, from 1 × 1 to 1 × 16,777,216: a grid of common and boundary sides, both sides of the 280:1 ratio, every size whose area makes f exactly 1, and random sizes. `tests/test_core_images.py` checks the core against the table and, where transformers is installed, against both processors on random sizes.

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

`{instructions}` is the question's instructions rendered by section 3.1. A multi-line rendering, such as JSON, stays multi-line. When the instructions render empty, their line is left out, with no blank line in its place. The line `Answer with the code of one option.` is not in v1's prompt. It makes a code the natural next token after the prompt, which matters for the chat variant of an instruction-tuned base.

### 3.4 Option lines

Option line i, for the option bound to code c, is built from three parts, in order:

1. `{c}.`, always;
2. ` {name}`, a space and the name, if the name is not empty;
3. ` — {description}`, a space, U+2014 EM DASH, a space and the description, if the description is not empty. The separator is v1's.

Name and description each have every run of whitespace, newlines included, replaced by one space, and leading and trailing whitespace removed, before the test for emptiness, so that an option is always one line and a name of only whitespace counts as empty. The four forms are therefore:

| name | description | line |
| --- | --- | --- |
| not empty | not empty | `{c}. {name} — {description}` |
| not empty | empty | `{c}. {name}` |
| empty | not empty | `{c}. — {description}` |
| empty | empty | `{c}.` |

The name is the noul label (`false`, `true`), the choice option's name, or the score level's index (`0`, `1`, ...). The description is the noul's criterion (or its default), the choice option's description, or the score level's rubric text, rendered by section 3.1. A structured description therefore becomes JSON flattened to one line: `{"from": "e1"}` gives `{ "from": "e1" }`. The collapsing applies to the whole rendered text, so it reaches inside the JSON's strings: `{"a": "x  y"}` gives `{ "a": "x y" }`. A character that the JSON escapes, such as U+001F written `\u001f`, is no longer whitespace there.

### 3.5 The two variants

Both variants share one body: the state block followed by the question block. They differ in what surrounds it. A training A/B chooses between them, and `decision_config.json` names the one a checkpoint was trained with.

`strands-plain` is the body as text, with no chat template:

```
<bos>{state block}{question block}\n<answer>
```

`strands-chat` is the body as the single user turn of Gemma 4's chat template, followed by the generation prompt, with thinking off and no system turn:

```
<bos><|turn>user\n{state block}{question block}<turn|>\n<|turn>model\n
```

This is what Gemma 4's `chat_template.jinja` renders for `messages=[{"role": "user", "content": body}]` with `add_generation_prompt=True` and `enable_thinking` false or absent. The template strips the content of leading and trailing whitespace; the body starts with `<state>` and ends with `</question>`, so stripping leaves it unchanged. With thinking off the template emits no system turn, and this interface adds none: the state and question are the whole instruction, and a fixed system turn would add tokens to every state without new information. Section 12 lists this as a question for the A/B.

The chat templates of the larger Gemma 4 sizes differ in one place. With thinking off, the templates of 12B, 26B-A4B and 31B append an empty thought channel, `<|channel>thought\n<channel|>`, after `<|turn>model\n`; those of E2B and E4B do not. `strands-chat` leaves the empty channel out for every size, so that one prompt string serves every checkpoint and the readout always reads the same token. `tests/test_core_prompt.py` pins each known template, by sha256, to one of the two forms. A checkpoint trained with the empty channel needs its own prompt name.

The chat template is where `strands-chat` comes from, not how it is rendered. A runtime must not render these prompts with Hugging Face's `apply_chat_template`, or with any chat template, for either variant, and a decision checkpoint ships no template (section 9). Gemma 4's templates do not give these prompts: the 12B, 26B-A4B and 31B templates add the empty thought channel; a template only wraps the body, which section 3 defines; `strands-plain` has no chat turn at all; and `apply_chat_template(tokenize=True)` encodes the rendered string in one call with special-token matching on, so special-token strings in request text become control tokens, which section 3.6 forbids. Runtimes render by this spec, with `strands_decider.core.render` or an implementation that reproduces the render goldens of section 10.

### 3.6 From text to token ids

A prompt is a sequence of control tokens and text runs. Control tokens come only from the template and from the request's images, and are emitted by id:

| control token | id | where |
| --- | --- | --- |
| `<bos>` | 2 | the template |
| `<\|turn>` | 105 | the template (chat) |
| `<turn\|>` | 106 | the template (chat) |
| `<\|image>` | 255999 | an image's line, first |
| `<\|image\|>` | 258880 | an image's line, N times |
| `<image\|>` | 258882 | an image's line, last |

The image ids are also `config.json`'s `boi_token_id`, `image_token_id` and `eoi_token_id` in every Gemma 4 base.

A runtime checks the checkpoint's tokenizer when it loads the checkpoint: each of these control tokens has the id in the table, each of the 255 codes encodes, as request text is encoded, to exactly one token whose id is the code's entry in `decision_config.json`'s `token_ids`, and every added token is special. A tokenizer that fails is refused. `strands_decider.core.check_tokenizer(tokenizer, config)` makes these checks, and `strands_decider.core.load_tokenizer(checkpoint)` loads a checkpoint's `tokenizer.json` and makes them.

These are the Gemma 4 `tokenizer.json` files known when this was written, by sha256 (the Hub's LFS digests), with the revisions that ship them:

| sha256 | shipped by | merges |
| --- | --- | --- |
| `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` (pinned) | `google/gemma-4-E2B-it` at `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, `google/gemma-4-26B-A4B-it` at `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, `google/gemma-4-31B-it` at `842da3794eaa0b77d5f08bae87a17459d91ff475` | 514,906 |
| `ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2` | `google/gemma-4-E2B-it` at `4adf15748de9e1a0097b6274cee9c15836c41c37` | 514,639 |
| `91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc` | `google/gemma-4-E4B-it` at `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | 514,653 |
| `9788bdb6a021b3170357cd3587b5d622b1177561ad3c5bf43d73728fd71aeacb` | `google/gemma-4-12B-it` at `43c9d014b231474fa7212dbb48a762566c2b255d` | 514,667 |

The render goldens are the pinned file's. `tests/test_core_prompt.py` checks them only against it, and checks the properties of this section and of section 4 against any of the four.

Each text run is encoded on its own by the tokenizer as plain text: special-token matching off (in Hugging Face `tokenizers`, `encode_special_tokens = True`) and `add_special_tokens=False`. Gemma 4's tokenizer adds no tokens of its own; its post-processor adds no BOS, which is why the template emits `<bos>` itself. Text from the request therefore never produces any of the tokenizer's 24 special tokens, its added tokens, all marked special:

```
<pad> <eos> <bos> <unk> <mask> <|tool> <tool|> <|tool_call> <tool_call|> <|tool_response>
<tool_response|> <|"|> <|think|> <|channel> <channel|> <|turn> <turn|> <|image> <|audio>
<|image|> <|audio|> <image|> <audio|> <|video|>
```

A state containing the characters `<turn|>`, `<|image|>`, or any other of the 24 strings, encodes them as text. The only special tokens in a prompt are the template's and, in the state, each image's line. Matching special tokens off does not stop an added token that is not special from matching text, so a tokenizer with such a token is refused. `strands_decider.core.render` encodes with a private copy of the tokenizer whose settings never change, checked when it is made to encode every added token's string as text, so this holds when several threads render with one tokenizer, or use it for other things, at the same time.

The pieces of each variant, with S marking the state boundary of section 4:

```
plain:  [<bos>] "<state>\n...</state>\n" S "<question ...>...</question>\n<answer>"
chat:   [<bos>] [<|turn>] "user\n<state>\n...</state>\n" S "<question ...>...</question>" [<turn|>] "\n" [<|turn>] "model\n"
```

With images, the state block's run is cut at each image's line. With two images, in the plain variant (in the chat variant the first run is `"user\n<state>\n"`, after `[<bos>] [<|turn>]`, as above):

```
[<bos>] "<state>\n" [<|image>] [<|image|>] × N1 [<image|>] "\n" [<|image>] [<|image|>] × N2 [<image|>] "\n{state}\n</state>\n" S ...
```

With an empty state the last run is `"\n</state>\n"`.

For a request whose text contains none of the 24 special-token strings, the ids equal what one call on the whole rendered string gives with special-token matching on, which is how Hugging Face's `apply_chat_template(tokenize=True)` encodes. With images they also equal what Hugging Face's `Gemma4Processor`, and the 12B's `Gemma4UnifiedProcessor`, give for the same string with each image's line replaced by one `<|image|>`, which the processor expands into the line. `tests/test_core_prompt.py` checks this on the goldens and on randomly generated requests, with and without images, and, where transformers is installed, against `apply_chat_template(tokenize=True)` itself with each Gemma 4 template given. `tests/test_core_images.py` checks it against both processors, where transformers, Pillow and torchvision are installed. `tests/test_core_prompt.py` also checks that requests holding all 24 strings, in every place text can go, produce none of them beyond the template's and the images' lines.

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

The `department` question in `strands-plain`, 95 tokens, S = 22:

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

The same question in `strands-chat`, 99 tokens, S = 25. The text ends with a newline after `model`:

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

A request with one image (`one_image` in the goldens): a 64 × 48 PNG, so N = 266, and the state `Checkout page after the user tapped Pay.`. Its state part in `strands-plain` is 287 tokens, S = 287, and the image's placeholders are state ids [6, 272):

```
<bos><state>
<|image><|image|>...(266 in all)...<|image|><image|>
Checkout page after the user tapped Pay.
</state>
```

```
state ids [0, 6):     2 236820 3255 236813 107 255999
state ids [6, 272):   258880, 266 times
state ids [272, 287): 258882 107 77524 3104 1308 506 2430 74060 9762 236761 107 954 3255 236813 107
```

In `strands-chat` the same state part is 290 tokens, `2 105 2364 107` before `236820 3255 236813 107 255999`, and the placeholders are [9, 275). The question parts are the text-only ones.

The goldens hold every question type in both variants, with structured content, an empty state, 28 options (codes to `AB`) and 255 (every code, `GY` followed by `HA`, through `IV`), special-token strings in the state, all 24 of them in every place request text goes (the state text of an image request included), a ten-level score, questions whose type is inferred, nouls with one criterion given, a state that forges `</state>\n<question`, and choice options with empty names, one with a description (`A. — no preference yet`) and one without (`B.`). They hold requests with images: one image; two of different sizes and formats, a JPEG (N = 270) and a WebP behind a `data:` prefix (N = 266); and a GIF (N = 256) with an empty state. They pin section 3.1's JSON with a float, both exponent forms, a large integer, `-0.0`, `true`, `false`, `null`, every kind of escape, `/`, non-ASCII text, U+2028 and nested empty containers, in a state, in instructions and flattened in descriptions; and its whitespace with NBSP, U+2028, U+001F and U+FEFF at the edges of a state and inside options.

## 4. The state boundary S

S is the number of tokens in the state part. Tokens [0, S) are the state part and [S, L) the question part, where L is the prompt's length. S is defined by the pieces of section 3.6, never by the text: the state part is the ids of the pieces before S, which are `<bos>`, in the chat variant `<|turn>`, and the state block's text runs and image lines, the last run ending with the template's own `</state>\n`; the question part is the ids of the pieces after S, which start with the template's `<question`. S is the length of the state part's ids. A runtime counts it from the pieces it encodes, and never finds it by searching the text or the ids for `</state>` or `<question`: a state may itself hold `</state>\n<question ...`, which is state text like any other (the goldens hold such a state). Everything before S is identical for every question about the state: `<bos>`, in the chat variant `<|turn>user\n`, and the state block, its images' lines included. A runtime encodes it once and shares it across the request's questions.

The rule: the state part and the question part are encoded separately, as section 3.6 describes, and the prompt's ids are their concatenation. No token can straddle S, because no encoding call crosses it. A runtime must use the ids of the two parts and must not re-encode the joined text.

For Gemma 4's tokenizer the rule costs nothing: the separate encodings equal the encoding of the joined text, so S always falls on a token boundary of the whole prompt. The argument has three steps.

1. A token that straddled S would contain the line feed before S and the `<` after it. In the pinned tokenizer's vocabulary of 262,144 tokens, every token that contains a line feed consists only of line feeds (31 such tokens, up to 31 line feeds long), and none of its 514,906 merges joins a line feed to another character.
2. The normalizer replaces each space with U+2581 and does nothing else; it maps characters one to one and sees no context. The pre-tokenizer splits on spaces, which no longer exist after normalization (and a split could only add boundaries). The post-processor adds no tokens, and BPE runs without dropout.
3. So no merge across S is ever possible. BPE applies merges in rank order, and merges on one side of S never create or remove a candidate pair on the other, so each side ends with the merges it makes alone.

`tests/test_core_prompt.py` proves this in two ways, for whichever of section 3.6's files it is given. `test_tokenizer_cannot_merge_across_a_line_end` checks the premises of steps 1 and 2 against the tokenizer file itself. `test_s_is_a_token_boundary` encodes 160 random requests in both variants with the two text runs at S joined into one, including requests full of special-token strings, line ends, tabs, non-ASCII text and option-like text; 40 of them carry one to four images. It gets the same ids as the rule. (With images the state's last run still ends with `</state>\n`.) A rendered prompt always has `</state>\n` before S and `<question` after it, so the test also varies the characters next to S directly: 1,200 random pairs of a run ending in one to three line feeds after any character and a run starting with any character but a line feed, special-token strings included, encode the same joined as apart. `test_ids_match_one_call_encoding` checks that a token starts at exactly S's character offset in the one-call encoding. A future tokenizer that fails these premises keeps the rule well defined (separate encoding), but its ids may then differ from those a text-only tool produces, so this spec would need a new seam for it.

## 5. Where the answer is read

The answer is read from the hidden state at the prompt's last token, position L - 1 (`pooling: "last"`). In `strands-plain` that token is `>` (id 236813), the end of `<answer>`. In `strands-chat` it is the line feed (id 107) after `<|turn>model`. Every question of a variant ends with the same tokens: plain with `\n<answer>` (107 236820 14433 236813), chat with `<turn|>\n<|turn>model\n` (106 107 105 4368 107).

## 6. The attention mask: `prefix_bidir`

Token i attends to token j when

```
i < S:   j < S        (the state sees the state, in both directions, and never a question)
i >= S:  j < L        (a question sees the whole prompt, in both directions)
```

The mask applies on every layer, sliding-window layers included, and replaces their window: no layer limits attention by distance. Positions are the token indices, 0 to L - 1, so every question's first token sits at position S. An image's placeholders are state tokens like any other: one position each, as Gemma 4's one-dimensional rotary positions count them, and the mask lets them see the whole state and be seen by it. So a runtime adds no image mask of its own; the bidirectional attention that some Gemma 4 sizes give the tokens of one image is contained in this one. Each question is its own sequence over the shared state; no question sees another. Because the state never attends to a question, its keys and values are the same for every question and can be computed once.

A runtime that reuses a state's keys and values beyond one request, in a prefix cache for one, keys them on the state's token ids and, in order, on the sha256 of each image's file (the bytes decoded from base64, `RenderedImage.sha256`). The placeholder ids give only an image's number of soft tokens, so two different images with the same N give the same ids, and a cache keyed on the ids alone would answer one image's questions from another image's keys and values. The same holds for any part of the state a runtime caches, down to a block of the placeholders: the key covers the files of the images whose placeholders it holds or follows. `strands_decider.core.attention_mask(S, L)` gives the mask as booleans, row i for token i. It is an illustration of the rule, for tests and small examples: a dense L × L array, O(L²) in memory (64 MB at L = 8,192). A runtime builds the form its attention kernel takes, such as the state's length and each question's span.

## 7. The readout

For each question with k options:

```
h  = bf16(the final-norm hidden state at position L - 1)                          [hidden]
z  = bf16(W h)          a bf16 GEMM; W = readout.safetensors `weight`, in bf16    [255]
z  = tanh(z / c) * c    float32; c = decision_config.json `final_logit_softcapping` (30.0 for every Gemma 4 size)
z  = z / T              float32; T = decision_config.json `temperature`
p  = softmax(z[0:k])    float32, over the question's own k codes only
```

`W` has one row per code, in code order. The other 255 - k codes do not enter the softmax. `p[i]` is the probability of the option bound to code i, and maps back to the request's terms by section 2.

### 7.1 The precision regime

The arithmetic above is the training reference's, and that of `strands-decider-vllm`, the vLLM runtime (strands-decider pull request 130). A runtime follows it step by step:

1. `W h` is a bf16 GEMM. `W` is bf16; a runtime that reads an f32 or f16 tensor rounds it to bf16 first. `h` is rounded to bf16 before the GEMM, to nearest with ties to even, as `strands-decider-vllm` rounds it (`hidden_states.to(bf16)`): a bf16 decoder's h is unchanged by it, and a float32 decoder's, such as llama.cpp's or MLX's in float32, is rounded (a wider h is rounded to float32 first). The products are accumulated in float32 or wider, and the result `z` is rounded to bf16, to nearest with ties to even. The rounding is part of the regime, not noise to be avoided: between 16 and 32 bf16's values are 0.125 apart, so it moves a logit there by up to 0.0625, more than any float32 error.
2. The soft-cap is computed in float32 as `tanh(z / c) * c`, each operation rounded to float32, and the temperature divides the result in float32.
3. The softmax over the question's k codes is taken in float32.

`strands_decider.core.read` is the reference implementation of this regime in numpy, which has no bf16. It rounds h to bf16, sums the GEMM's products in float64, rounds each sum to float32, and rounds that to bf16 through the float32's bits (`round_to_bf16`: add 0x7FFF, plus 1 if the lowest kept bit is set, and clear the low 16 bits). A GPU's float32 accumulator sums in another order, so where a sum falls next to a bf16 rounding boundary its `z` can differ by one bf16 step; section 10's bounds absorb this. `tests/test_core_readout.py` checks `read` against an independent reimplementation of `strands-decider-vllm`'s arithmetic on random bf16 inputs, logits between 16 and 32 included: `z` agrees bit for bit, and the probabilities to 1e-5. It also checks that a float32 h gives the z of its bf16 rounding, and not the z of the unrounded h.

The temperature, like the soft-cap, is applied exactly once. A runtime that applies part of the readout itself returns a named stage (section 11), and the client applies only what the stage has not. On vLLM, `strands-decider-vllm` applies the soft-cap and the checkpoint temperature in the model, so the client sends `temperature` 1.0 in its request. Under `--logprobs-mode processed_logprobs` vLLM divides the logits by the request's temperature before it takes the log-softmax, so any other value would apply a temperature a second time.

The answer fields are v1's response schema (`schema.py`, `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer`), computed from the probabilities in double precision and rounded to four decimal places, as v1 rounds them, by Python's `round(x, 4)`: the decimal with four places nearest to the double's exact value, ties to even. The sums behind `score` and `confidence` can differ in their last bit between Python versions (3.12 and later add floats with compensation) and between languages, so a field can differ from the reference's by 0.0001 where its unrounded value lies at a rounding boundary. Section 10 bounds the probabilities; it does not ask for the fields' last digit.

| type | fields |
| --- | --- |
| noul | `noul`: P(true) |
| choice | `choice`: the option with the highest probability (the first in request order on a tie); `probabilities`: option name to probability, in request order; `confidence` |
| score | `score`: the expected level, sum of i times p_i; `legend`: level to rubric text (stripped), ascending; `probabilities`: level to probability, ascending; `confidence` |

Confidence uses v1's formulas, unchanged and shared as code (`src/strands_decider/confidence.py`). A choice's confidence is the normalised top probability, (k p_max - 1) / (k - 1): 0 for a uniform distribution, 1 for a certain one, whatever k is. A score's is one minus the normalised standard deviation over the levels, with the optional floor correction for ordinal smoothing (`ordinal_smoothing`, 0 by default). A noul has no confidence field, because its probability already is its uncertainty. A response is v1's `SystemOneResponse`: `model`, `answers` keyed by question name, and `usage` with `input_tokens` (the state once, then each question) and `output_tokens` (the number of questions).

## 8. The `decision_config.json` schema

`decision_config.json` holds what a runtime needs, beyond the model's weights, its tokenizer and its image processor's configuration, to serve a checkpoint. Format version 1:

| field | type | meaning |
| --- | --- | --- |
| `interface` | string, `strands-decider` | names the file as this interface's; any other value, or none, is an error |
| `format_version` | integer, 1 | this layout |
| `prompt` | string | the template the checkpoint was trained with: `strands-plain` or `strands-chat` |
| `inputs` | array of distinct strings from `text` and `image`, holding `text` | what a request to the checkpoint may carry: `["text"]`, or `["text", "image"]` for a checkpoint that serves images |
| `codes` | array of 255 strings | the codes of section 2, in order; anything else is an error |
| `token_ids` | array of 255 distinct integers from 0 to 2^31 − 1 | each code's token id, in code order |
| `temperature` | number, finite and above 0 | the divisor of section 7 |
| `final_logit_softcapping` | number, finite and above 0 | the soft-cap c of section 7: the decoder's `final_logit_softcapping`, 30.0 for every Gemma 4 size |
| `attention_mode` | string, `prefix_bidir` | the mask of section 6 |
| `pooling` | string, `last` | the readout position of section 5 |
| `max_length_validated` | integer from 1 to 2^31 − 1 | the longest prompt, in tokens (S, image placeholders included, plus the question's tokens), at which the checkpoint was validated |
| `provenance` | object | public origin of the checkpoint; fields below |

`inputs` names what the checkpoint was validated on beyond text: a checkpoint with `image` in it has image requests among its model goldens (section 10). The array's order carries no meaning. A runtime refuses a request with images unless `image` is in it (section 11); audio would add `audio` (section 12).

`provenance` has four string fields and no others. `base_model`, a Hub id such as `google/gemma-4-E2B-it`, and `base_revision`, the base model's 40-character commit, are required: the checkpoint's `config.json`, `processor_config.json` and tokenizer files are the base's at that revision, and so is every part of the model that was not fine-tuned (section 9). `release`, the released model's name or Hub id, and `source`, an https URL of the public code and recipe that trained it, are optional; an optional field that does not apply is left out. `"provenance": null` is an error, and so is any of its fields given as null. Each field must match its pattern as a whole string, in ASCII:

| field | pattern |
| --- | --- |
| `base_model` | `[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+` |
| `base_revision` | `[0-9a-f]{40}` |
| `release` | `([A-Za-z0-9][A-Za-z0-9_.-]*/)?[A-Za-z0-9][A-Za-z0-9_.-]*` |
| `source` | `https://` and then one or more of the printable ASCII characters `!` to `~` (no space) |

`DecisionConfig` checks only these formats. Keeping private paths and hosts out of a released checkpoint is the job of the release's export checks, not of this file's loader.

`interface` is the discriminator. Other decision-model formats use the same file names, and converters, such as llama.cpp's, claim any directory that holds `decision_config.json` and `readout.safetensors`. A runtime or converter therefore identifies a decision checkpoint by `"interface": "strands-decider"`, never by its file names, and a runtime refuses a `decision_config.json` without it before it reads any other field. It reads `format_version` next, so that a file of a later layout is refused as that, not for the fields it adds.

`temperature` and `final_logit_softcapping` divide the readout's logits in float32 (section 7), so each must also lie in float32's normal range, about 1.2e-38 to 3.4e38, and `final_logit_softcapping / temperature`, the largest final logit, must be finite in float32; a value outside these is an error.

An integer is a JSON number written without a fraction or an exponent: `8192`, not `8192.0` or `8.192e3`, which are numbers of the wrong type. `format_version` is exactly `1`. Integers past the bounds in the table are errors, so that a runtime that holds them in 32 bits never overflows, and so is a number too large for a double, wherever a number is expected.

Validation is strict. A missing required field, an unknown field at any level, and a value of the wrong type are errors, and a runtime refuses the checkpoint. A runtime must refuse a `prompt` it does not implement, rather than fall back to another; a later variant gets a new name. A runtime refuses a prompt longer than `max_length_validated` tokens unless its operator raises the limit explicitly, and it refuses rather than truncates. Training-side records, such as data paths, hyperparameters and the inputs to a weight average, do not belong in this file.

An example (codes and token ids shortened):

```json
{
  "interface": "strands-decider",
  "format_version": 1,
  "prompt": "strands-chat",
  "inputs": ["text", "image"],
  "codes": ["A", "B", "C", "...", "IV"],
  "token_ids": [236776, 236799, 236780, "...", 3767],
  "temperature": 1.0,
  "final_logit_softcapping": 30.0,
  "attention_mode": "prefix_bidir",
  "pooling": "last",
  "max_length_validated": 8192,
  "provenance": {"base_model": "google/gemma-4-E2B-it",
                 "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7"}
}
```

`strands_decider.core.DecisionConfig.load` reads and validates the file; `tests/test_core_config.py` lists the cases it refuses.

## 9. The checkpoint layout

A decision checkpoint is the base model's own multimodal checkpoint, in the base's layout, with its fine-tuning merged in, and the files of this interface beside it. Training may fine-tune the decoder and, with image data, the vision encoder's projection into the decoder or the encoder itself. Whatever was trained is merged into the base's tensors, under the base's names, and the rest is the base's at `provenance.base_revision`. No tensor is required to equal the base's; the model card says which parts were trained. A checkpoint is a directory that holds:

| file | contents |
| --- | --- |
| `config.json` | the base's configuration: `architectures: ["Gemma4ForConditionalGeneration"]` and `model_type: gemma4` for E2B, E4B, 26B-A4B and 31B; `["Gemma4UnifiedForConditionalGeneration"]` and `gemma4_unified` for the 12B. It holds the base's `text_config`, `vision_config`, `audio_config` (null for 26B-A4B and 31B) and image token ids. Its `text_config.final_logit_softcapping` equals `decision_config.json`'s; a runtime takes the soft-cap from `decision_config.json` and refuses a checkpoint whose `config.json` gives a different value, under `text_config` or at its top level |
| `model.safetensors` or `model-*-of-*.safetensors` with `model.safetensors.index.json` | every tensor of the base, in bf16 under the base's names (the prefixes below), any fine-tuning merged in: no adapter. There is no `lm_head` tensor: Gemma 4 ties it to the token embedding, and the readout takes its place |
| `processor_config.json` | the base's: its image processor, `Gemma4ImageProcessor` (`Gemma4UnifiedImageProcessor` for the 12B) with `patch_size` 16, `pooling_kernel_size` 3 and `max_soft_tokens` 280, its feature extractor and its video processor. N depends on the three settings (section 3.2), so a runtime refuses image requests for a checkpoint whose image processor has other values |
| `tokenizer.json`, `tokenizer_config.json` | the base's tokenizer, unchanged; `tokenizer_config.json` carries no `chat_template` key |
| `readout.safetensors` | one tensor, `weight`, shape [255, hidden], bf16; a runtime that reads an f32 or f16 tensor rounds it to bf16 (section 7.1) |
| `decision_config.json` | section 8 |
| `golden_vectors.jsonl` | the model goldens of section 10 |
| `LICENSE`, `NOTICE`, `README.md` | the terms and the model card, which says which parts of the model were trained |

The base's tensors fall under these prefixes, and no others:

| part | E2B, E4B | 12B | 26B-A4B, 31B |
| --- | --- | --- | --- |
| the decoder, with its token embedding | `model.language_model.` | `model.language_model.` | `model.language_model.` |
| the vision encoder | `model.vision_tower.` | `model.vision_embedder.` | `model.vision_tower.` |
| the vision encoder's projection into the decoder | `model.embed_vision.` | `model.embed_vision.` | `model.embed_vision.` |
| the audio encoder | `model.audio_tower.` | none | none |
| the audio projection into the decoder | `model.embed_audio.` | `model.embed_audio.` | none |

These are the names in the bases' own files: the 26B-A4B's and 31B's `model.safetensors.index.json`, and the safetensors header of the single `model.safetensors` that E2B, E4B and the 12B ship without an index. They were read from `google/gemma-4-E2B-it` at `4adf15748de9e1a0097b6274cee9c15836c41c37` (and at `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, with the same names), `google/gemma-4-E4B-it` at `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c`, `google/gemma-4-12B-it` at `43c9d014b231474fa7212dbb48a762566c2b255d`, `google/gemma-4-26B-A4B-it` at `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, and `google/gemma-4-31B-it` at `842da3794eaa0b77d5f08bae87a17459d91ff475`. The 12B has no vision tower and no audio tower (transformers' `Gemma4UnifiedModel` is encoder-free). Its `model.vision_embedder.` is a patch embedder of a few layers, and its audio input, raw waveform frames, passes through `model.embed_audio.` alone. Transformers renames the 12B's vision tensors when it loads them (`vision_embedder.patch_dense` to `embed_vision.patch_dense`, and so on); the files keep the base's names.

A text-only runtime loads the tensors under `model.language_model.` and nothing else. With that prefix removed they are the tensors of transformers' `Gemma4TextModel` (`Gemma4UnifiedTextModel` for the 12B), configured by `config.json`'s `text_config`. Such a runtime refuses image requests (section 11).

A checkpoint carries no code. Runtimes implement this spec; they do not import code shipped inside a checkpoint.

A checkpoint carries no chat template either: no `chat_template.jinja`, unlike the Gemma 4 base models, and no `chat_template` in `tokenizer_config.json` or `processor_config.json`. A template beside the weights would be applied by tools that look for one, such as a chat-completions endpoint or a GGUF converter that embeds it, and section 3.5 explains why no template gives these prompts. Shipping a Strands template that renders `strands-chat` for every size was considered and not chosen: it would still leave the body, the plain variant and the encoding of special-token strings to the spec, and invite the one-call tokenization that gets the last wrong.

A converted artifact for another runtime carries the model in that runtime's format: for llama.cpp a GGUF file of the decoder and a multimodal projector file (mmproj) of the vision encoder and its projection, and for MLX a model that holds both. Beside them it carries the same `readout.safetensors` and `decision_config.json`, unchanged, and the image processor's settings, as `processor_config.json` or in the converted format's own fields. An artifact without the vision encoder serves text only, and its runtime refuses images. The readout stays in safetensors, outside the converted decoder, so that every format reads the same rows. The artifact needs no `config.json`: everything the readout needs beyond its weight, the soft-cap included, is in `decision_config.json`. `strands_decider.core.Readout.load` reads the weight and the soft-cap from those two files.

## 10. Golden vectors and conformance

Two kinds of golden data pin the interface.

Render goldens fix the prompt. `tests/fixtures/core/render_goldens.json` holds, for each case and variant, the request, the state text, S, the state ids, for a request with images each image's format, width, height, sha256, N and span, and for each question its text, ids, codes, code token ids and bound options. They depend only on the tokenizer and, through N, on the image processor's three settings. A conforming runtime reproduces them exactly: the same ids, the same S, the same spans, the same codes. There is no tolerance. Their images are small synthetic files in `tests/fixtures/core/images`, drawn deterministically by `tests/core_support.py` and checked in, and the goldens carry them in their requests as base64.

Model goldens fix the answers of one checkpoint. Each checkpoint release publishes `golden_vectors.jsonl` in its directory. Each line holds a request, with its images as base64, its rendered ids per question and each image's span, and the reference code probabilities per question, computed by the reference runtime: the model in bf16, weights and activations, one prompt per forward pass under the `prefix_bidir` mask, then the readout in section 7.1's precision regime. For images the reference runtime prepares the pixels as section 11.1 says, with `Gemma4ImageProcessorPil` (`Gemma4UnifiedImageProcessor` for the 12B), and runs the checkpoint's vision encoder and projection in bf16. The reference runtime is the PyTorch model with the `prefix_bidir` mask behind `strands_decider.core.evaluate`; it is not part of this torch-free core. The set covers every question type, both short and long states up to `max_length_validated`, and choices with many options. For a checkpoint whose `inputs` hold `image` it also covers image requests: one image; up to four of different sizes, formats and aspect ratios, extreme ones included; an image with an empty state; and images beside a long state.

A runtime conforms for a checkpoint when, on that checkpoint's model goldens:

1. its rendered ids equal the goldens' ids for every question;
2. running the weights in bf16, its top-1 code (the code with the highest probability) agrees with the reference's on at least 98% of the questions, and its mean |Δp| is at most 0.02;
3. running quantized weights, it meets the bounds recorded for that format in the table below.

A question's |Δp| is the largest absolute difference between the runtime's probability and the reference's for any of the question's options; the mean |Δp| is its average over the goldens' questions. The top-1 bound alone cannot see the errors that section 7.1 exists to prevent: the soft-cap and the temperature are both monotone, so a runtime that skips the cap, or applies T twice, keeps every top-1 code and still gets every `noul`, `score`, `probabilities` and `confidence` wrong. The probability bound does see them. Measured against the reference, vLLM (`strands-decider-vllm`) and transformers at all five sizes have a mean |Δp| of 0.0033 to 0.0119.

| weight format | top-1 agreement | mean \|Δp\| |
| --- | --- | --- |
| bf16 | at least 98% | at most 0.02 |
| quantized (for example MLX 8-bit or 4-bit, GGUF Q8_0 or Q4_K_M) | set from measurement when the format is first published, and recorded here | the same |

A quantized artifact whose format has no recorded bounds makes no conformance claim.

## 11. The runtime interface

A runtime runs the model. Given the state ids (positions [0, S)), the state's images with their spans, and the question ids of each question, it puts each image's vision-encoder output at its placeholders (section 11.1), runs every question as a continuation of the state under the `prefix_bidir` mask, seeing no other question, and returns one row per question, in the order of the questions. The row is one of five stages of section 7, and the runtime names which, so that nothing is applied twice:

| stage (`outputs`, `stage`) | a row holds | what the core still applies |
| --- | --- | --- |
| `hidden` | h, the final-norm hidden state at the question's last token, [hidden] | the bf16 GEMM, the soft-cap, the temperature, the softmax |
| `raw_logits` | `z = W h`, before the soft-cap: the 255 codes, or the question's first k | the rounding to bf16 (which leaves a bf16 GEMM's output unchanged), the soft-cap, the temperature, the softmax |
| `capped_logits` | `tanh(z / c) * c`: the soft-cap applied, the temperature not | the temperature, the softmax |
| `final_logits` | `tanh(z / c) * c / T`: both applied, as `strands-decider-vllm`'s model returns them at the code ids | the softmax |
| `probabilities` | the codes' probabilities: the question's k, or all 255 | renormalisation over the question's k codes |

A row of logits or probabilities holds the codes in code order, either all 255 or only the question's first k, so rows may differ in width. The core refuses capped logits beyond ±c and final logits beyond ±c / T: such a row cannot be what its stage says, which is how raw logits passed as capped usually show.

`strands-decider-vllm` is a `probabilities` runtime. A completion with `allowed_token_ids` set to the question's k code ids, `logprobs` k and `temperature` 1.0, under `--logprobs-mode processed_logprobs`, returns the log-softmax of the model's final logits over exactly those k codes; their exponentials are the row. This is the form to use. vLLM refuses `logprobs` above its `--max-logprobs` setting, 20 by default, so the server needs `--max-logprobs 255` for questions of up to 255 options (`strands-decider-vllm serve` sets it). A completion with `logprobs` 255 and no `allowed_token_ids` returns the softmax over all 255 codes, which renormalises to the same row in exact arithmetic, but not always in float32: at a small temperature the question's own codes can all lie so far below a code outside the question that their probabilities underflow to 0, and the core refuses such a row (`probabilities must ... not all be 0`). The k-wide form cannot underflow so, because its largest probability is at least 1/k.

This is v1's `hidden_engine.Backend`, narrowed. v1's backend returns several rows per question, one per option and one for `<answer>`, for the pointer head; the code readout needs only the row at the last token. `strands_decider.core.Backend` states the contract:

```python
class Backend(Protocol):
    # The stage forward returns: hidden, raw_logits, capped_logits, final_logits or probabilities.
    outputs: str

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage] = (),
    ) -> np.ndarray | Sequence[np.ndarray]:
        """One row per question, in order, of the stage `outputs` names: each question run
        under prefix_bidir after the shared state, read at its last token. `images` are the
        state's, in order, each with its file and the span of its placeholders in `state`."""
```

A backend may also have an `inputs` attribute, which says what `forward` takes: `("text",)`, or `("text", "image")` for a backend with an image path. A backend without it takes text only. It is not a member of the protocol, so that a text-only backend need not define it.

`strands_decider.core.evaluate(request, tokenizer, readout, config, backend)` joins the pieces. It refuses a backend that does not name its stage, and refuses a request with images as section 11.1 says. It renders the request with the checkpoint's prompt and checks, before the decoder runs, everything that does not need it: the stage, that the readout's soft-cap is the checkpoint's, that a `hidden` backend has a readout weight with a row per code, and that the rendered code token ids are the checkpoint's `token_ids`. It then calls `forward` once with the shared state and every question, and with the images if there are any, reads the answers from the stage the backend names, and returns a `SystemOneResponse` as JSON. Its parts are public: `read(rows, readout, config, rendered, stage=stage)` gives the answers and `question_probabilities` the code probabilities, from rows of any stage, where `stage` is required (`inputs`, its former name, still works and warns that it is deprecated); `readout_logits` is the bf16 GEMM; `answer(rendered_question, probabilities)` formats one answer. As with v1's backend, a runtime may batch, cache the state's keys and values (keyed as section 6 says), or split the work across devices, provided each question's result is what it would be alone.

### 11.1 Images

A runtime refuses a request with a non-empty `images` list, before it decodes any image, when the checkpoint's `inputs` lack `image` (`this checkpoint does not take images: its decision_config.json inputs are ["text"]`), and when the runtime has no image path: no vision encoder loaded, as with a text-only loader or an artifact without the encoder. The second refusal uses v1's text-only message, `this engine has no vision tower; images are not supported`. It also refuses image requests for a checkpoint whose image processor's settings differ from section 3.2's (section 9).

`render` gives, for each image in order, its file (`data`, the bytes decoded from base64), its format, width and height from the header, the sha256 of its file, N, and its span: state ids [`start`, `end`) are its N placeholders, with `<|image>` at `start` - 1 and `<image|>` at `end` (`strands_decider.core.RenderedImage`). The core does not decode pixels. The runtime does, and prepares them:

1. It decodes the file as v1 does (`vision.decode_image`), by the rules below, into 8-bit RGB pixels. A file that fails here is refused with v1's message, `image could not be decoded: {reason}`. The decoded size, after the orientation, must be the header's size or its transpose; any other is refused the same way (`strands_decider.core.check_decoded_size`).
2. It resizes the decoded image to W × H, computed by section 3.2's arithmetic from the decoded width and height, after the orientation, not from the header's (`strands_decider.core.resize_target`, or `check_decoded_size`, which also checks the size). An image that the orientation turns a quarter therefore has its W and H swapped: the test image `rotated-80x40.jpg` has a header of 80 × 40 and pixels of 40 × 80, and is resized to 528 × 1104, not to the 1104 × 528 of the header's shape. N is the same both ways, so the count of soft tokens cannot show this mistake. The resize is the bicubic resize below. The runtime then prepares the pixels with the checkpoint's image processor, configured by its `processor_config.json`, with the processor's own resize turned off (`do_resize=False`): scaled to [0, 1] and not normalised, and cut into 16-pixel patches with their positions, which the 12B's processor merges in 3 × 3 groups. W and H are multiples of 48, so the processor needs no further resize. The processor's count of soft tokens must be the core's N, and a runtime checks that it is.
3. The vision encoder and its projection turn the patches into N vectors, which take the place of the placeholders' token embeddings at the image's span, in order.

These are v1's decoding rules, which are Pillow's (v1 opens, loads, orients, composites and converts the image with Pillow):

- **Decoders.** PNG is inflated and unfiltered, and GIF's LZW decoded, exactly. JPEG is decoded as libjpeg-turbo decodes it with its default settings (the accurate integer inverse DCT, and fancy upsampling of subsampled chroma), and WebP as libwebp decodes it with its default settings; these are the decoders in Pillow's wheels. Of an animated GIF or WebP, only the first frame is decoded.
- **Samples.** A greyscale sample v becomes (v, v, v). A PNG's 1-, 2- and 4-bit greyscale samples are scaled to 8 bits (by 255, 85 and 17). 16-bit PNG samples keep their high byte, except in a 16-bit greyscale PNG without alpha, which Pillow opens as 16-bit integers and converts by clipping: v becomes min(v, 255), so most such images decode nearly white. A palette image takes each index's colour from its PNG's `PLTE`, or from its GIF frame's local colour table, else the global one; a GIF with no colour table, or whose table is the grey ramp (entry i is (i, i, i) for every entry), is greyscale, each index its grey level. A four-component JPEG is CMYK, its samples stored inverted, as Adobe stores them: with C, M, Y and K each 255 less the stored sample, R is (255 − K) − m(C, 255 − K), and G and B likewise with M and Y, where m(x, y) = ((t >> 8) + t) >> 8 and t = x · y + 128.
- **Transparency.** An image with an alpha channel (RGBA, or greyscale with alpha), and a palette image with transparency (a PNG's `tRNS`, in which an index without an entry is opaque, or a GIF frame's transparent index), is composited onto opaque white: a channel c with alpha a, both 0 to 255, becomes ((y >> 8) + y) >> 15 with y = 128 · (c · a + 255 · (255 − a) + 128), and 255 where a is 0. This is Pillow's `alpha_composite` in integers, within 1 of c · a / 255 + 255 − a. The `tRNS` of a greyscale or RGB PNG, and the transparent index of a greyscale GIF, are not applied: those pixels keep their colour, as v1 composites only RGBA, greyscale-with-alpha and palette images.
- **A GIF's canvas** has section 1.1's size. Before the first frame is drawn it holds the frame's transparent index, if its graphic control extension gives one, and index 0 otherwise; the frame's pixels are then drawn at its offset. So the canvas outside an offset first frame is colour 0, or white where the frame has a transparent index.
- **Orientation.** The EXIF orientation (tag 274), from the EXIF data of a JPEG's APP1 segment, a PNG's `eXIf` chunk or a WebP's `EXIF` chunk, or failing that from an XMP `tiff:Orientation`, as Pillow's `ImageOps.exif_transpose` finds it, is applied before the compositing: 2 mirrors left to right, 3 turns the image 180°, 4 mirrors top to bottom, 5 transposes (across the main diagonal), 6 turns it 90° clockwise, 7 transverses (across the other diagonal), and 8 turns it 90° anticlockwise.

The resize is Pillow's bicubic resampling, a separable cubic convolution, defined here so that a runtime without Pillow can implement it. It resizes the width first and then the height, each by one pass of the rule below over 8-bit samples, every channel alike; a pass whose size does not change is left out. A pass from n samples to m computes, for each output index i:

- the scale s = n / m, the filter scale f = max(s, 1), so that the kernel widens with the downscale, and the support r = 2f;
- the centre c = (i + 0.5) · s, the first input j₀ = max(⌊c − r + 0.5⌋, 0) and the input after the last j₁ = min(⌊c + r + 0.5⌋, n);
- for each input j from j₀ to j₁ − 1, the weight K((j − c + 0.5) / f), where K is Keys's cubic convolution kernel with a = −0.5: K(x) = (a + 2)|x|³ − (a + 3)|x|² + 1 for |x| < 1, a|x|³ − 5a|x|² + 8a|x| − 4a for 1 ≤ |x| < 2, and 0 beyond. The weights are divided by their sum, all in double precision, then made fixed-point with 22 fraction bits: each w becomes the integer nearest w · 2²², a half rounded away from zero;
- the output, from the inputs x_j, is ⌊(2²¹ + Σ x_j k_j) / 2²²⌋ in integers, clamped to 0 to 255.

This is `image.resize((W, H), Image.Resampling.BICUBIC)` in Pillow 10, 11 and 12. Pillow 13 does the height's pass first when the height shrinks by more than twice as much as the width, as it does for a tall screenshot such as 1,080 × 2,400, and its pixels then differ by up to about 12 levels; with Pillow 13 or later the rule is two calls, `image.resize((W, h)).resize((W, H))`, each one pass.

`tests/fixtures/core/image_pixels.json` holds the pixel goldens: for each test image, its decoded size and the sha256 of its RGB bytes after step 1 (rows from the top, each pixel's R, G and B), and its W × H and the sha256 after the resize, made with Pillow 12.2.0. Pillow 10.0 to 13.0 give the same bytes for these images, which are all enlarged. A runtime conforms on pixels when every channel of its decoded and of its resized image is within 1 of the reference's, the bytes the pinned Pillow gives for the same file; JPEG and WebP decoders other than libjpeg-turbo's and libwebp's can differ by more. `tests/test_core_images.py` checks v1's decode and the resize against the goldens where Pillow is installed, the resize rule above, written in numpy, against Pillow byte for byte, the decoding rules on images made for each case, and the image processor with its resize off against its own resize.

This bicubic resize is the reference for every size, the 12B included. For E2B, E4B, 26B-A4B and 31B it gives the same pixels as transformers' `Gemma4ImageProcessorPil`, the PIL backend v1 pins, on Pillow 10 to 12, and on Pillow 13 wherever that version keeps the width first. The 12B's `Gemma4UnifiedImageProcessor` has only a torchvision backend, so a 12B runtime resizes with Pillow first, as above. Other resizing, such as transformers' torchvision backend or vLLM's default processing, gives slightly different pixels, enough in v1's measurements to flip some answers, and a runtime that uses it counts the difference against section 10's bounds like any other. Hugging Face's image processors take the decoded image as a PIL image, or as an array with its channel layout given (`input_data_format`). From an array 1 or 3 pixels high or wide they cannot tell which axis holds the channels, and they count a 100 × 1 array as 192 soft tokens, not 167.

## 12. Open questions

These are not yet decided, and the training A/B or the first release should settle them.

1. Whether a system turn helps `strands-chat`. It would sit before S, so it would cost tokens once per state, not per question.
2. Whether `strands-chat` should include the empty thought channel that the 12B, 26B-A4B and 31B templates add with thinking off (section 3.5). If those sizes train better with it, they need a third prompt name.
3. The noul's code assignment: `false` is `A` and `true` is `B`, following v1's slot order. If letter position biases the base model, training with random swaps (as for choices) removes the bias from the weights either way.
4. Whether the line `Answer with the code of one option.` earns its tokens once the readout is trained.
5. Whether `ordinal_smoothing`, which v1 kept in its head configuration, belongs in `decision_config.json`.
6. The quantized-format bounds of section 10.
7. How to install the torch-free core without torch: a separate distribution, such as `strands-decider-core`, or an extra of `strands-decider` with torch and transformers moved out of its base requirements. Until then `pip install strands-decider` installs torch, and a torch-free runtime installs the core's modules with `tokenizers` and numpy by other means.
8. Audio. It is not supported: a request with a non-empty `audio` field is refused (`audio input is not supported`), never ignored. E2B and E4B have an audio encoder (`model.audio_tower.`, with `model.embed_audio.`) and the 12B projects waveform frames through `model.embed_audio.`, so audio could follow the image pattern later, as v1's Gemma 4 audio input does (strands-decider pull request 67).
9. Which base revision each release pins. A checkpoint's own tokenizer, its base's at `base_revision`, is normative for it (section 3.6). At the revisions section 9 lists, the E2B, E4B and 12B repositories ship tokenizers with 239 to 267 fewer merges than the pinned file, and E2B's vocabulary differs in two entries. When this was measured, every render golden's text then in the file, and every code, encoded to the same ids with all three. The release should fix one revision per size and check, beyond `check_tokenizer`, that its tokenizer gives the render goldens' ids.
