# Images

Strands Decider (v19) answers questions about images when it is loaded with `--vision`.
No weights change: the published checkpoint is used as is, and the images are read by the
vision tower that already ships inside Qwen3.5-2B-Base. Text-only requests to a vision
server get the same answers as from a text server.

## Using it

```bash
pip install "strands-decider[vision]"     # Pillow; needs transformers >= 5.18
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --vision --port 8000
```

A request adds `images`: a list of base64 images, PNG, JPEG, WebP or GIF (a
`data:image/...;base64,` prefix is accepted). The state may be empty.

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Checkout page after the user tapped Pay.",
  "images": ["'"$(base64 < screen.png | tr -d '\n')"'"],
  "questions": {
    "paid":   {"type": "noul",   "instructions": "Did the payment succeed?"},
    "next":   {"type": "choice", "instructions": "Which control should the agent use next?",
               "criteria": {"retry": "a retry button is shown", "back": "return to the cart",
                            "none": "nothing to do"}},
    "severe": {"type": "score",  "instructions": "How severe is the error shown?",
               "criteria": ["no error", "warning", "blocking error"]}
  }
}'
```

From the command line, `ask --image FILE` (repeatable) loads the vision tower for that call.
A server started without `--vision` answers a request with `images` with HTTP 422 rather
than ignore them. `/health` reports `"vision": true|false`. `--strict-window` and
`--max-batch` apply to image requests as they do to text ones.

Limits, fixed per server:

- Each image is scaled down (never up) so its longer side is at most 448 px: 196 tokens
  for a square image (patch 16, merge 2). At most 4 images per request.
- An image over 4096 x 4096 pixels is refused from its header, before it is decoded.
  Other formats are refused before they are parsed. EXIF rotation is applied, and
  transparent areas are read as white.
- Images count against the same 4,096-token window as the state. Without
  `--strict-window`, the questions keep their reserve, the images are kept whole, and the
  state text loses its end, as text requests do. With it, an over-long prompt gets 422.
- The vision tower adds about 331M parameters (0.66 GB in bf16).

## How it works

`StrandsDeciderModel._load_torso` loads only the text decoder of Qwen3.5. `--vision` loads
`Qwen3_5ForConditionalGeneration(...).model` instead: the ViT, the patch merger and the
same 24-layer decoder. The checkpoint's LoRA adapter was trained on the bare decoder
(`layers.N...`); inside the multimodal model the same decoder sits under
`language_model.layers.N...`, so the adapter's keys are remapped onto it
(`VisionDeciderModel.load`). The vision tower is frozen; nothing in it is adapted.

Images go inside `<state>` as Qwen vision placeholders, before the state text:

```
<state>
<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>
Checkout page after the user tapped Pay.
</state>
<question type="noul"> ... <answer>
```

Each `<|image_pad|>` stands for one merged patch. Everything after the state is text, so
the option positions, the pointer readout, the temperatures and every confidence formula
are the text model's. The shared-prefix cache works as for text: the image and state are
encoded once and forked across the questions.

Two details matter for exact answers:

- **Positions.** Qwen3.5 uses M-RoPE: an image advances the rotary positions by its grid
  size, not its token count, so text after an image sits at `token index + rope_delta`.
  The image path passes these positions explicitly for the cached prefix and the question
  suffixes. Left to transformers, a suffix-only forward builds positions from the full
  mask, and reads a `rope_deltas` that an earlier plain image forward left on the module.
- **Pixels.** The image processor is pinned to the PIL backend
  (`Qwen2VLImageProcessorPil`). `Qwen2VLImageProcessor` switches to torchvision when it
  is installed, and that changes pixel values enough to flip a few answers.

Checks (`tests/test_vision.py`, offline, on a tiny random-weight Qwen3.5 with a real
`Qwen3_5Tokenizer`): text requests to a vision engine get the text engine's answers; over
one and two images the shared-prefix path matches a plain full forward to under 1e-5; no
position state leaks into the next request; the window and decoding rules above. Each
test fails when the fix it pins is removed.

## How well it does

Measured with the published v19, no image training, against the untrained base and an
image-trained 2B decider, at 448 px (Mapika at its 768 px). Lower is better for ECE and
Brier.

| System | NaturalBench acc | NaturalBench G-Acc | NaturalBench ECE | POPE-adv acc | POPE-adv Brier | POPE-adv ECE | Image JevBench preview, exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-2B-Base, untrained option-number readout | 0.712 | 0.177 | 0.048 | 0.865 | 0.227 | 0.116 | 28/60 |
| Strands Decider (v19), `--vision` | 0.784 | 0.327 | 0.014 | 0.877 | 0.202 | 0.072 | 37/60 |
| Mapika/decider-2b-vision (image-trained) | 0.794 | 0.360 | 0.080 | 0.860 | 0.204 | 0.061 | 38/60 |

v19 against Mapika, paired bootstrap 95% CIs (v19 minus Mapika): NaturalBench ECE
-0.066 (-0.083 to -0.032), the one clear difference; NaturalBench accuracy -0.010 (-0.031
to +0.011), G-Acc -0.033 (-0.077 to +0.010), POPE accuracy +0.017 (-0.005 to +0.038) and
POPE ECE +0.010 (-0.025 to +0.053), all within noise. So v19 matches the image-trained
model on accuracy, and on NaturalBench is much better calibrated.

With the image removed every system falls to chance (NaturalBench 0.500, POPE about
0.50), so the answers come from the image. What text training does not teach shows here:
without the image, v19 still answers at a mean confidence of 0.652 on NaturalBench and
0.725 on POPE (ECE 0.152 and 0.225; Mapika 0.153 and 0.175; the untrained base 0.092 and
0.059).

- NaturalBench: the first 300 groups of the first shard (1,200 questions); each group is
  two images and two questions with opposite answers, and G-Acc counts a group only when
  all four are right.
- POPE adversarial: the first 600 items by question id.
- Image JevBench: the official items are sealed. These are the 60 published preview items
  that can be rebuilt exactly from their source rows (CLEVR-HOPE, Geometry3K, ArxivQA).
- One run per system, on one H100 each (fp32 for v19 and the base, bf16 for Mapika).

The script is `evaluation/vision/run.py`; [evaluation/vision/README.md](../evaluation/vision/README.md)
has the exact command and a link to every per-item result.
