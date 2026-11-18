# strands-decider-vllm

A vLLM plugin that serves Strands Decider checkpoints (E2B, E4B, 12B, 26B-A4B, 31B) on vLLM 0.33, with text and image input. It patches no vLLM file; vLLM loads it through the `vllm.general_plugins` entry point. It also ships a server, `strands-decider-vllm serve`, which answers the System One request (`POST /v1/systemone`, images included) with vLLM's engine in process.

## Checkpoints

A checkpoint is the base Gemma 4 model's own multimodal checkpoint with its fine-tuning merged in: the base's `config.json`, `processor_config.json` and tokenizer, the weights under the base's names with no LM head, `readout.safetensors` (one tensor `weight` of shape [255, hidden]) and `decision_config.json` (the core library's `DecisionConfig`). The 12B uses `Gemma4UnifiedForConditionalGeneration`, the other sizes `Gemma4ForConditionalGeneration`.

The plugin also serves, text-only and through `vllm serve` only, the intermediate checkpoints made before this layout: a bare `Gemma4TextModel` decoder with a `decision_config.json` that predates `interface` and `inputs`.

A checkpoint may be a local directory or a Hugging Face Hub repo id. The plugin refuses at startup a `decision_config.json` it can't serve (another interface, malformed inputs, code token ids that aren't 255 distinct vocabulary ids, no soft-cap, or a soft-cap that differs from the decoder config's).

## What the plugin does

Stock vLLM 0.33 would read the model through the LM head with Gemma 4's causal, sliding-window attention, which gives different answers (see Parity). The plugin changes five things, and only for its own architectures:

- **The readout.** Three architectures, one per layout, subclass vLLM's Gemma 4 classes and replace the LM head with the readout: a bf16 GEMM on the final-norm hidden state, then the soft-cap and the checkpoint temperature in float32, placed at the 255 codes' token ids with -inf elsewhere. A request restricted to a question's codes gets the log-softmax over exactly those codes.

  | Layout | Plugin architecture | vLLM class |
  |---|---|---|
  | multimodal, E2B, E4B, 26B-A4B, 31B | `StrandsDeciderForConditionalGeneration` | `Gemma4ForConditionalGeneration` |
  | multimodal, 12B | `StrandsDeciderUnifiedForConditionalGeneration` | `Gemma4UnifiedForConditionalGeneration` |
  | intermediate bare decoder | `StrandsDeciderForCausalLM` | `Gemma4ForCausalLM` |

- **The `prefix_bidir` mask on every layer, with no sliding window.** A request sends its state boundary S as `vllm_xargs.prefix_end`. State tokens, image placeholders included, attend to the state `[0, S)` in both directions and never to the question; question tokens attend to the whole prompt in both directions. A Model Runner V2 model state splits each prompt step into two non-causal FlashAttention calls on the paged KV cache. The sliding-window layers are built with a window of `--max-model-len` tokens, so they cut nothing. A prompt must be computed whole in one step, so chunked prefill is off, and when vLLM's scheduler would cut a prompt before one of its images (the encoder's budget or cache is full), the plugin holds the whole prompt back to the next step instead.
- **The base's "vision" mode off.** The 12B, 26B-A4B and 31B configs carry `"use_bidirectional_attention": "vision"`, which turns on vLLM's multimodal PrefixLM mode. `prefix_bidir` replaces it, so the runtime turns it off with `--hf-overrides` (the server does this), and the plugin refuses to start with it on.
- **The core's pixels.** Every image is resized with the core's `resize_image` and the processor runs with its own resize off. The plugin registers its own processor for its multimodal classes, so that vLLM counts one placeholder per 48 × 48 pixels for an image already at its size and uses `Gemma4ImageProcessorPil`, the reference's, for E2B, E4B, 26B-A4B and 31B. The pixels vLLM encodes match the reference's bit for bit (see Parity).
- **Cache keys and request checks.** Under this mask a state block's keys depend on every state token and image, and a question block's on the whole prompt, so the plugin adds a digest of the state and its images to block 0, and of the question to every block past S. A further question on a cached state reuses the state's full blocks; different states, images or boundaries share nothing. The engine core refuses, per request and before scheduling, anything the workers couldn't serve as a decision: a missing or invalid `prefix_end`, prompt embeddings, streaming input, `max_tokens` other than 1, sampling that would change the answer (temperature other than 1.0, truncation, penalties, logit bias, bad words, structured outputs), images for a text-only checkpoint, and images on an engine its server didn't start, since vLLM's own endpoints let a client choose an image's cache key. vLLM's server answers these with HTTP 400 and the plugin's with 422, and both keep serving.

The plugin also refuses to start with settings under which it would serve the mask incorrectly: chunked prefill, a step budget below `--max-model-len`, `--long-prefill-token-threshold`, micro-batching under data parallelism, `cudagraph_mode` FULL, a logprobs mode other than the processed ones, any dtype but bf16, quantization, a quantized KV cache, prompt embeddings, Model Runner V1, speculative decoding, KV or encoder-cache connectors, `--kv-sharing-fast-prefill`, pipeline or context parallelism, and a prefix-cache hash other than sha256.

Other models in the same environment keep vLLM's attention, cache keys, scheduling and request handling: each wrapper acts only on engines and requests of a Strands Decider model. On any vLLM release other than 0.33.x the plugin logs a warning and registers nothing.

## Requirements

- vLLM 0.33.x, Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`; the server sets it) and Python 3.11 or later. Tested with vLLM 0.33.0, torch 2.14.0+cu130, transformers 5.20.2, torchvision 0.29.0 and Pillow 13.0.0.
- An NVIDIA H100 80 GB is the only GPU tested: vLLM picks FlashAttention 4 for Gemma 4's two head sizes on Hopper. On Ampere or Ada GPUs vLLM picks Triton attention for the real checkpoints, which the plugin refuses.
- vLLM's x86 CPU build runs the plugin too, for the tests' CPU tier. It isn't a serving target.
- For the server: the core library, `strands_decider.core` (the `serve` extra).

## Install

Into the environment where vLLM 0.33 is installed:

```
pip install '.[serve]'
```

## Serving: `strands-decider-vllm serve`

```
CUDA_VISIBLE_DEVICES=0 strands-decider-vllm serve /path/to/checkpoint --port 8000
```

The server renders each request with the core, decodes and resizes its images, and sends vLLM one request per question. The first question runs alone and the others then together, so they reuse the state's cached blocks. Run it with the vLLM environment's `bin` directory on `PATH`, because vLLM's JIT kernel builds need its `ninja`.

The server has no authentication and binds 127.0.0.1 by default; vLLM's internal sockets stay on loopback whatever `--host` is. Reach it from another machine through an SSH port forward. To expose it, put the deployment's own authentication in front of it: SageMaker's IAM-signed endpoint (the server answers `/invocations` and `/ping`) or an authenticating proxy. A non-loopback `--host` starts with a warning. Behind SageMaker, keep `--limit-concurrency` at least twice the instance's peak concurrent invocations, and not below the default 128: past it uvicorn answers 503 to `/ping` too, and SageMaker takes the instance as unhealthy.

The server refuses at startup a checkpoint whose `decision_config.json` doesn't meet the core's schema (serve an intermediate checkpoint with `vllm serve` instead), and refuses image requests if `processor_config.json` sizes images differently from how the core counts them.

| Flag | Default | |
|---|---|---|
| `MODEL` | | the checkpoint directory or Hub repo id; `--revision` for the Hub |
| `--host`, `--port` | `127.0.0.1`, 8000 | |
| `--served-model-name` | `strands-decider` | the `model` of each response |
| `--max-length` | `max_length_validated` | longer prompts are refused (HTTP 422), never truncated |
| `--max-model-len` | `--max-length` + 1 | the engine's context; vLLM refuses a prompt that fills it, so it must be above `--max-length` |
| `--max-num-batched-tokens`, `--max-num-seqs` | `--max-model-len`, vLLM's | the step's budget (at least `--max-model-len`) and request limit |
| `--gpu-memory-utilization`, `--tensor-parallel-size` | 0.90, 1 | the 31B at 8,192 tokens fits one H100 at 0.95; 26B-A4B wants 0.92 |
| `--no-enable-prefix-caching` | caching on | |
| `--mm-processor-cache-gb` | 4 | vLLM's cache of preprocessed images; 0 turns it off |
| `--block-size` | vLLM's | |
| `--dtype` | `bfloat16` | `float32` on a CPU build of vLLM only, for tests |
| `--enforce-eager` | off | no CUDA graphs |
| `--max-body-mb` | 93 (8 for a text-only checkpoint) | larger bodies get HTTP 413; the default holds four 16 MiB image files in base64 and 8 MiB of text |
| `--max-buffered-mb` | 256, or `--max-body-mb` if larger | the most body bytes that requests over 1 MiB hold at once, granted in arrival order and given back once a request's images are decoded |
| `--body-read-timeout` | 30 s | a body with no chunk for this long gets HTTP 408, and one that waits this long for the budget 503, holding no budget afterwards |
| `--limit-concurrency` | 128 | uvicorn's limit on open connections; past it uvicorn answers 503 |
| `--preprocess-workers` | 4, or the CPU count if lower | threads that parse and render requests, and as many that decode images, off the event loop |

### Requests

`POST /v1/systemone` (and `POST /invocations`) takes the System One request:

```json
{
  "state": "The customer attached two photos of the parcel.",
  "images": ["iVBORw0KGgoAAAANSUhEUgAA...", "data:image/jpeg;base64,/9j/4AAQSkZJRg..."],
  "questions": {
    "damaged": {"type": "noul", "instructions": "Is the parcel damaged?"},
    "carrier": {"type": "choice", "instructions": "Which carrier label is shown?",
                "criteria": {"dhl": null, "ups": null, "fedex": null, "none": null}},
    "severity": {"type": "score", "instructions": "How bad is the damage?",
                 "criteria": ["none", "cosmetic", "contents at risk"]}
  }
}
```

`images` is optional: base64 PNG, JPEG, WebP or GIF files (a `data:image/...;base64,` prefix is allowed), each at most 16 MiB and 4096 × 4096 pixels. Their number isn't limited: the prompt's length bounds how many fit. The response adds `latency_ms`, and `cached_tokens` in `usage`, the prompt tokens served from vLLM's prefix cache:

```json
{"model": "strands-decider",
 "answers": {"damaged": {"type": "noul", "noul": 0.9312},
             "carrier": {"type": "choice", "choice": "ups", "probabilities": {"dhl": 0.0412, "ups": 0.9105, "fedex": 0.0398, "none": 0.0085}, "confidence": 0.8807},
             "severity": {"type": "score", "score": 1.2154, "legend": {"0": "none", "1": "cosmetic", "2": "contents at risk"}, "probabilities": {"0": 0.0521, "1": 0.6804, "2": 0.2675}, "confidence": 0.4521}},
 "usage": {"input_tokens": 1243, "output_tokens": 3, "cached_tokens": 1312},
 "latency_ms": 212.4}
```

A body that isn't a valid request, an image that isn't valid or can't be decoded, images for a text-only checkpoint, a prompt over `--max-length` and any refusal of the engine's request checks get HTTP 422 with `{"detail": "..."}`, and the server keeps serving. Anything else that fails is logged and gets HTTP 500 with only an error id. `GET /health` and `GET /ping` answer 200 with the model's name, inputs and limits while the engine is healthy, and 503 when it isn't.

## Serving token ids: `vllm serve`

Clients that render prompts themselves with the core (`strands_decider.core.render`) can send text requests to vLLM's own server. One GPU, the multimodal layout:

```
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
vllm serve /path/to/checkpoint --host 127.0.0.1 --port 8000 --served-model-name m --dtype bfloat16 --seed 0 \
  --hf-overrides '{"architectures": ["StrandsDeciderForConditionalGeneration"], "text_config": {"use_bidirectional_attention": null}}' \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' \
  --max-model-len 32769 --max-num-batched-tokens 32769 --no-enable-chunked-prefill --enable-prefix-caching \
  --enable-prompt-tokens-details --prefix-cache-retention-interval None --generation-config vllm \
  --logprobs-mode processed_logprobs --max-logprobs 255 --gpu-memory-utilization 0.90
```

The architecture is `StrandsDeciderUnifiedForConditionalGeneration` for the 12B, and `StrandsDeciderForCausalLM` with `"use_bidirectional_attention": null` at the top level of `--hf-overrides` (and no `--limit-mm-per-prompt`) for an intermediate decoder checkpoint. This server refuses image requests; use `strands-decider-vllm serve` for them. 26B-A4B wants `--gpu-memory-utilization 0.95`. The 31B fits one GPU up to 12,288 tokens (`--max-model-len 12289 --max-num-batched-tokens 12289 --gpu-memory-utilization 0.95`) and needs two above that (`--tensor-parallel-size 2 --disable-custom-all-reduce --gpu-memory-utilization 0.92`).

| Flag | Why |
|---|---|
| `VLLM_USE_V2_MODEL_RUNNER=1` | The mask is a Model Runner V2 model state. Required. |
| `--hf-overrides` | Selects the plugin's architecture and turns the base's "vision" mode off. Required. |
| `--no-enable-chunked-prefill`, `--max-num-batched-tokens` >= `--max-model-len` | A prompt must be computed in one step. Required. |
| `--max-model-len` one above the longest prompt | vLLM refuses a prompt that fills the context, so 32,769 serves prompts of up to 32,768 tokens. |
| `--logprobs-mode processed_logprobs`, `--max-logprobs 255` | The codes' probabilities are the logprobs after `allowed_token_ids`. Required. |
| `--enable-prefix-caching`, `--prefix-cache-retention-interval None` | Optional; `None` keeps cache checkpoints at every block of the sliding-window KV groups. |
| `--host 127.0.0.1` and the three loopback variables | Keep the server and vLLM's internal sockets on loopback; vLLM's server has no authentication by default. |

Each question is one request to `/v1/completions`, with the rendered prompt's ids, S and the question's code token ids:

```json
{
  "model": "m",
  "prompt": [2, 105, 9731, 107, 3041, 236787, 1001, 1002, 1003, 106, 107, 105, 2364, 107, 9998, 236787, 1004],
  "max_tokens": 1,
  "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "seed": 0,
  "logprobs": 4,
  "allowed_token_ids": [236776, 236799, 236780, 236796],
  "return_tokens_as_token_ids": true,
  "vllm_xargs": {"prefix_end": 9}
}
```

`prompt` is the question's ids (state, then question; the ids above are placeholders), `prefix_end` is S with 1 <= S < L, and `allowed_token_ids` the question's k code ids, with `logprobs` k. The probabilities are `exp` of `choices[0].logprobs.top_logprobs[0]` (keyed `"token_id:<id>"`), renormalized over the k ids. `max_tokens` must be 1 and `temperature` 1.0, because the model already divides by the checkpoint temperature. Several questions about one state are separate requests with the same state and S, and with prefix caching the later ones reuse the state's blocks.

## Limits

- A prompt is computed whole in one step, so the step budget is at least `--max-model-len`. Tested budgets are 32,768 tokens for text and 8,192 for images; at 32K one prompt fills a step.
- No sliding windows: every layer attends over every token. Latency was measured at about 1.2 times stock causal Gemma 4 at 1K tokens and about twice at 32K.
- The 31B on one 80 GB H100 serves contexts up to 12,288 tokens as a decoder and 8,192 with the vision encoder; 16K and 32K need TP=2. KV capacity on one GPU at the flags above: E2B 3,348,843 tokens, E4B 927,147, 12B 134,173, 26B-A4B 100,705.
- Answers were checked against the reference up to 8,192 tokens, the reference's limit.
- The first requests after startup compile FlashAttention 4's non-causal kernels, which vLLM's warm-up doesn't. Send a few requests before measuring latency.
- Streaming input, audio and video aren't served.
- The plugin relies on vLLM 0.33 internals that aren't a public API (the Model Runner V2 model state, the FlashAttention backend and metadata, the Gemma 4 classes and multimodal processor, the default loader's secondary weights, and wrappers around the block-hash extra keys, `EngineCore.preprocess_add_request` and the scheduler), which is why it accepts vLLM 0.33.x only.
- KV-cache events, if enabled, report vLLM's own extra keys without the plugin's digests.

`tools/causal_control/` is a separate diagnostic plugin, not part of the package, that reads Gemma 4's own causal, sliding-window layers with the trained readout: what stock vLLM would compute. It measured the agreement without the mask quoted under Parity (`pip install --no-deps ./tools/causal_control`; serve an intermediate decoder with `"architectures": ["StrandsDeciderCausalControlForCausalLM"]`).

## Tests

- **CPU tier.** `pip install '.[test,serve]'`, then `pytest` from this directory. Without vLLM it runs the request checks, segment plan, cache keys, `decision_config.json`, the scheduler and engine wrappers on stand-ins, and the golden-line reader. With vLLM 0.33 installed it also runs the model classes, the model state, the attention backend, the image processing, vLLM's own block hasher and scheduler, and the server's request path against a stand-in engine. With `STRANDS_DECIDER_TEST_CHECKPOINT` set to a checkpoint directory, the startup checks run on configs vLLM builds. `tests/cpu/ci.sh OUT_DIR` runs the plugin end to end on vLLM's x86 CPU build (`tests/gpu/ci.sh` with `DEVICE=cpu`), in float32, with exact checks. About 14 minutes on 4 cores with warm compile caches.

  ```
  uv pip install "https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
  sudo apt-get install -y libnuma1
  pip install --no-deps '.[test,serve]'
  ```

- **GPU tier.** `tests/gpu/ci.sh OUT_DIR` runs the plugin end to end on one GPU of 24 GB or more, in bf16, on tiny random-weight checkpoints of every layout with the real tokenizer and image processor. It serves them with `strands-decider-vllm serve` and `vllm serve`, checks every answer, the refusals, the cache reuse on exact token counts and the conformance check against a transformers reference, and runs the server's request path in process on a real engine (`tests/gpu/engine_checks.py`). `HETERO_HEADS=1` gives the tiny models Gemma 4's two head sizes, so they run on FlashAttention 4 as the real checkpoints do. Each tolerance is a measured difference with headroom, documented beside it in `ci.sh`. About 30 minutes on an H100.
- **Conformance tier.** `tests/gpu/conformance.sh OUT_DIR CKPT_E2B CKPT_E4B CKPT_12B CKPT_26B_A4B CKPT_31B` serves every size on one 8-GPU node and checks each against its `golden_vectors.jsonl`: the rendering exactly, then at least 98% top-1 agreement and a mean |Δp| of at most 0.02 on at least 1,000 questions. It reports STUB until the released checkpoints publish their goldens.
- **Node checks.** `tests/gpu/run.sh MODEL_DIR ROWS OUT_DIR` checks a server on one node with 6 to 8 GPUs: refusals, cached and batched answers, a server without prefix caching and one from a Hub cache, preemption, the startup refusals, and with `MM_MODEL` and `MM_TEXT_MODEL` the image path and the text-only refusal. About 20 minutes on an 8-GPU H100 node.

The GPU tier, the conformance tier and the node checks are the release gate.

## Parity

### Text

The reference runs the model offline in PyTorch, bf16 with SDPA, one request per forward pass, prompts up to 8,192 tokens. The parity set has 1,200 prompts per model: 300 each of about 256, 1K and 4K tokens, and 300 real states of 6,144 to 8,000 tokens. Top-1 agreement compares the argmax over a question's options; |Δp| is the largest probability difference over its options.

| Model | Top-1 agreement | Mean \|Δp\| | Max \|Δp\| | Top-1 by prompt length: 256 / 1K / 4K / 6K to 8K |
|---|---|---|---|---|
| E2B | 99.33% | 0.0049 | 0.069 | 98.67% / 100.00% / 99.67% / 99.00% |
| E4B | 99.67% | 0.0035 | 0.049 | 99.67% / 99.67% / 99.67% / 99.67% |
| 12B | 98.67% | 0.0081 | 0.120 | 99.00% / 99.00% / 98.67% / 98.00% |
| 26B-A4B | 98.42% | 0.0119 | 0.195 | 97.67% / 97.00% / 100.00% / 99.00% |
| 31B, one GPU | 99.67% | 0.0048 | 0.088 | 99.00% / 99.67% / 100.00% / 100.00% |
| 31B, TP=2 | 99.50% | 0.0050 | 0.106 | 99.00% / 99.33% / 100.00% / 99.67% |

Measured with version 0.2.1 on the intermediate decoder checkpoints; 0.3.0 gives the same answers within run-to-run noise. Every size meets the conformance bounds (98% top-1, mean |Δp| at most 0.02). Most disagreements are near-ties in the reference. Without the mask, with Gemma 4's own causal, windowed attention and the same readout, agreement is 57.5% (E2B) to 91.3% (31B).

### Images

No checkpoint has been trained on images yet, so the test checkpoints are the Gemma 4 bases with their language model replaced by the intermediate decoders, and only their agreement with a reference is meaningful. The reference is `tests/gpu/reference.py`, transformers' own model in bf16 with the mask and the images prepared as the core prepares them. The set has 300 requests of 1 to 4 real photos (900 questions), answered by `strands-decider-vllm serve` on one H100.

| Model | Top-1 agreement | Mean \|Δp\| | Max \|Δp\| | Top-1 by images: 1 / 2 / 3 / 4 |
|---|---|---|---|---|
| E2B | 99.33% | 0.0049 | 0.061 | 98.67% / 100.00% / 100.00% / 99.26% |
| E4B | 98.33% | 0.0048 | 0.062 | 98.67% / 97.99% / 99.29% / 97.04% |
| 12B | 98.00% | 0.0138 | 0.170 | 97.60% / 97.19% / 99.29% / 99.26% |
| 26B-A4B | 96.11% | 0.0346 | 0.527 | 96.27% / 97.99% / 95.74% / 92.59% |
| 31B | 97.00% | 0.0243 | 0.669 | 98.13% / 96.79% / 95.04% / 96.30% |

The image conformance bound will be checked on image-trained checkpoints and their goldens. Against this image-untrained reference, 26B-A4B and the 31B fall short, which points to bf16 rounding rather than the plugin: the same reference in float32 agrees with the bf16 reference on only 95.9% (26B-A4B) and 97.0% (31B), and where the reference is confident (top probability above 0.9) vLLM gives the same answer on 588 of 588 questions at 26B-A4B and 509 of 510 at the 31B. The pixels vLLM's encoder receives match the reference's bit for bit on all 612 images of the set, on Pillow 12 and 13.

### Performance

E2B on one H100, `strands-decider-vllm serve` with prefix caching, one noul question per request, kernels warmed:

| Request | p50, concurrency 1 | Requests/s, 1 | Requests/s, 16 |
|---|---|---|---|
| text, about 330 tokens, new each time | 27 ms | 36.0 | 199 |
| one new 640 × 480 photo (266 tokens) and a short text | 88 ms | 11.3 | 43 |
| a photo the encoder cache holds, with a new text | 41 ms | 24.5 | 143 |
| a photo and text already cached, a new question | 30 ms | 33.4 | 152 |

The vision encoder costs about 47 ms per photo; vLLM's encoder cache skips it for a photo sent again.
