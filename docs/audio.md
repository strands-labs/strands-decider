# Images and audio on Gemma 4 E2B and E4B

Gemma 4 E2B and E4B ship a vision encoder and an audio encoder beside the text decoder. The
Strands Decider Gemma 4 checkpoints were trained on text only, and the text loader keeps only
the decoder. With `--vision` the server keeps the vision encoder; with `--audio` it keeps both.
A request may then carry images, audio clips, or both, together with the text state, and every
question is answered in one forward pass over all of it. No weights change: the published
checkpoint is used as is.

Measured on `StrandsAgents/strands-decider-E4B-gemma4-v1-2610`
([How well it does](#how-well-it-does)). Text-only requests to an `--audio` server get the
text server's answers exactly: on the 231 JevBench v1 public tasks every probability is
bit-identical, and both servers score 187/231.

## Using it

```bash
pip install "strands-decider[audio]"     # Pillow, soundfile, scipy; needs transformers >= 5.18
strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v1-2610 --audio --port 8000
```

A request adds `audio`: a list of base64 clips, WAV, FLAC, OGG or MP3 (a
`data:audio/...;base64,` prefix is accepted). `images` works as in [vision.md](vision.md).
The state may be empty.

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Support call, customer on the premium plan.",
  "audio": ["'"$(base64 < call.wav | tr -d '\n')"'"],
  "images": ["'"$(base64 < screenshot.png | tr -d '\n')"'"],
  "questions": {
    "angry":    {"type": "noul",   "instructions": "Is the caller angry?"},
    "intent":   {"type": "choice", "instructions": "What does the caller want?",
                 "criteria": {"refund": "money back", "fix": "the error on screen fixed",
                              "cancel": "to cancel the plan"}},
    "urgency":  {"type": "score",  "instructions": "How urgent is it?",
                 "criteria": ["can wait", "today", "right now"]}
  }
}'
```

From the command line, `ask --audio FILE` (repeatable) loads the audio encoder for that call,
and `--image FILE` can be given beside it. `/health` reports `"audio": true|false`. `--audio`
implies `--vision`.

Refused rather than ignored: a server started without `--audio` answers a request with `audio`
with HTTP 422. A checkpoint whose base has no audio encoder (the Qwen3.5 models, Gemma 4
26B-A4B) refuses to load with `--audio`. `--vision` and `--audio` are built and tested for E2B
and E4B; the 12B (`gemma4_unified`) is refused.

Limits, fixed per server:

- At most 2 clips per request, each at most 30 s (Gemma 4's own limit). The duration is read
  from the file header, and a longer clip is refused before it is decoded, not cut.
- Clips are mixed down to mono and resampled to 16 kHz. Each 40 ms is one token: a 30 s clip
  is 750 tokens.
- Each image is resized by Gemma's processor to at most 280 tokens (about 645,000 pixels,
  aspect ratio kept). No pre-scaling is applied: unlike Qwen's, Gemma's token count does not
  follow the pixel count, so a smaller image would only lose detail. At most 4 images.
- Media count against the same 4,096-token window as the state. Without `--strict-window`, the
  questions keep their reserve, the media are kept whole, and the state text loses its end. With
  it, an over-long prompt gets HTTP 422.
- The encoders add about 0.47B parameters. The E4B served with `--audio` peaked at 15.4 GiB of GPU
  memory in bf16 in the measurements below (one NVIDIA L4).

## How it works

`VisionDeciderModel.load(..., audio=True)` loads `Gemma4ForConditionalGeneration(...).model`:
the vision encoder and its projection into the decoder (`embed_vision`), the audio encoder (a
conformer) and its projection (`embed_audio`), all frozen, and the same decoder the text loader
uses. Without `audio`, the model is built without an audio encoder and its weights are left
unread. The checkpoint's LoRA adapter was trained on the bare decoder (`layers.N...`); in the
multimodal model the same decoder sits under `language_model.layers.N...`, so the adapter's keys
are remapped onto it, onto the same modules.

Media go inside `<state>`, images first, then clips, then the state text:

```
<state>
<|image><|image|>...<|image|><image|>
<|audio><|audio|>...<|audio|><audio|>
Support call, customer on the premium plan.
</state>
<question type="noul"> ... <answer>
```

Each `<|image|>` or `<|audio|>` is one soft token the encoder fills in. The counts are the
processor's: `num_soft_tokens_per_image` for images; for audio, the valid mel frames (10 ms hop)
after two stride-2 convolutions, as `Gemma4Processor` counts them. Everything after the state
is text, so the option positions, the pointer readout, the temperatures and every confidence
formula are the text model's. The shared-prefix cache works as for text: the media and state
are encoded once and the cache is forked across the questions. Gemma 4 uses plain 1-D RoPE
(one position per token, media included), so a question suffix after the cached prefix takes
the default positions, and no position state is kept between requests. E2B and E4B attend
causally to image tokens.

The image processor is pinned to `Gemma4ImageProcessorPil` and the feature extractor is
`Gemma4AudioFeatureExtractor`, both read from the base model's `processor_config.json` at the
checkpoint's base revision. E2B keeps its eager attention on CUDA, as on the text path.

Checks (`tests/test_multimodal_gemma.py`, offline, on a tiny random-weight
`Gemma4ForConditionalGeneration` with sliding, full and KV-shared layers, a vision and an audio
encoder, and a real `GemmaTokenizer`):

- The torso's weights are the base's, with the audio encoder only when asked for.
- The adapter is the text checkpoint's, on the same modules.
- Text requests match the text engine.
- Audio token counts are what the encoder emits.
- Over audio, two clips of different lengths, an image with a clip, and two images, the
  shared-prefix path matches a full forward to under 1e-5. These prefixes run far past the
  sliding window and are forked across three questions.
- Long, malformed and unsupported clips are refused, and so are clips sent to engines and
  servers without an audio encoder.

## How well it does

`evaluation/multimodal/run.py` measures `StrandsAgents/strands-decider-E4B-gemma4-v1-2610`
(revision `1344d44`) with no image or audio training, on one NVIDIA L4 in bf16, three ways
(E2B, revision `d383ecf`, served and blind only):

- **served:** the server's shared-prefix path.
- **blind:** the same question with the image or clip removed.
- **untrained:** google/gemma-4-E4B-it read without the adapter, through the frozen torso's
  option-number logits.

Lower is better for Brier and ECE.

| Benchmark (chance) | E4B served: acc | Brier | ECE | E4B blind: acc | ECE | untrained base: acc | ECE | E2B served: acc | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NaturalBench (0.50) | 0.750 | 0.340 | 0.065 | 0.500 | 0.272 | 0.715 | 0.276 | 0.736 | 0.014 |
| POPE adversarial (0.50) | 0.835 | 0.241 | 0.051 | 0.500 | 0.426 | 0.798 | 0.197 | 0.833 | 0.051 |
| ESC-50, 5-way choice (0.20) | 0.490 | 0.644 | 0.055 | 0.188 | 0.162 | 0.290 | 0.669 | 0.320 | 0.126 |
| ESC-50, presence (0.50) | 0.663 | 0.423 | 0.062 | 0.500 | 0.378 | 0.595 | 0.298 | 0.515 | 0.182 |
| Speech Commands, 10-way choice (0.10) | 0.930 | 0.136 | 0.149 | 0.100 | 0.099 | n/a | n/a | 0.453 | 0.167 |
| Speech Commands, presence (0.50) | 0.940 | 0.142 | 0.147 | 0.500 | 0.439 | 0.825 | 0.149 | 0.705 | 0.032 |

NaturalBench G-Acc (all four answers of a group right): E4B 0.250, untrained base 0.190, E2B 0.233.
For comparison, the Qwen3.5 v1 2B with `--vision` scores 0.776 on NaturalBench and 0.872 on POPE on
the same items ([vision.md](vision.md#v1)). The untrained reading has no 10-way column: option
10 has no single-token number, so the base model cannot be read that way.

E4B served minus the untrained base, paired bootstrap 95% CIs:

| | accuracy | ECE |
| --- | --- | --- |
| NaturalBench | +0.035 (+0.014 to +0.055) | -0.211 (-0.230 to -0.188) |
| POPE adversarial | +0.037 (+0.007 to +0.065) | -0.146 (-0.173 to -0.112) |
| ESC-50, 5-way choice | +0.200 (+0.145 to +0.253) | -0.613 (-0.649 to -0.545) |
| ESC-50, presence | +0.068 (-0.003 to +0.133) | -0.236 (-0.294 to -0.166) |
| Speech Commands, presence | +0.115 (+0.078 to +0.155) | -0.002 (-0.052 to +0.041) |

So the text adapter carries over to images and audio. It answers more accurately than the
base model it was trained from, and it is much better calibrated. The one exception is spoken
words: there the base is already as well calibrated, and the decider is underconfident (mean
confidence 0.78 at 0.93 accuracy). Speech is strong on E4B. Environmental sounds are weaker,
and E2B is much weaker on audio than E4B.

With the medium removed, both models fall to chance, so the answers come from the image or
clip. But on yes/no questions they still answer confidently: E4B's mean confidence without the
medium is 0.77 on NaturalBench, 0.93 on POPE, 0.88 on ESC-50 presence and 0.94 on Speech
Commands presence. Its confidence does not tell you that a clip or image is missing or
unreadable, so check that it is there before you ask.

Benchmarks (each item is one question about one image or clip):

- NaturalBench: the first 300 groups of the first shard, 1,200 questions, as in
  [vision.md](vision.md).
- POPE adversarial: the first 600 items by question id.
- ESC-50 (environmental sounds, CC BY-NC 3.0, used for evaluation only): fold 1, 400 clips of
  5 s. Each clip gets two questions:
  - "Which sound is heard in the recording?" over 5 classes: its own and 4 others, seeded.
  - "Is the sound of X heard in this recording?", true for half the clips and false for the
    other half.
- Speech Commands v0.02 (CC BY 4.0): the official test set, the first 40 clips by file name
  for each of the 10 core words, 1 s each. Each clip gets two questions:
  - "Which word does the speaker say?" over the 10 words.
  - "Does the speaker say the word X?", true for half and false for half.

Latency of a served request (`VisionEngine.evaluate`, decoding included), p50 over 20 runs on
one NVIDIA L4, bf16. The image is a NaturalBench photo; the clip is a 5 s ESC-50 clip (125
tokens).

| Request | E4B, 1 question | E4B, 3 questions | E2B, 1 question | E2B, 3 questions |
| --- | --- | --- | --- | --- |
| text | 117 ms | 253 ms | 117 ms | 228 ms |
| image | 399 ms | 409 ms | 519 ms | 545 ms |
| audio | 401 ms | 407 ms | 417 ms | 449 ms |
| image + audio | 544 ms | 641 ms | 619 ms | 624 ms |

Peak GPU memory over the run was 15.4 GiB for E4B and 10.6 GiB for E2B (E2B runs eager
attention on CUDA).

The script is `evaluation/multimodal/run.py`. `evaluation/multimodal/paired.py` computes the
intervals. Every download is pinned, and the revisions and versions are in each run's
`summary.json`.
