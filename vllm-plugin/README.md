# strands-decider-vllm

A vLLM plugin that serves Strands Decider checkpoints (E2B, E4B, 12B, 26B-A4B, 31B) on stock vLLM 0.34, with text and image input. vLLM loads it through the `vllm.general_plugins` entry point; it patches no vLLM file. For the plugin's architectures only, it replaces the LM head with the trained readout, runs every layer under the `prefix_bidir` mask with no sliding window, keys the prefix cache by what the mask makes each block depend on, and refuses requests and settings that would give wrong answers. Other models in the same environment keep vLLM's behavior.

Its server, `strands-decider-vllm serve`, answers the System One request (`POST /v1/systemone`), images included. Clients that render prompts themselves with the core can use `vllm serve` instead, for text.

## Install

vLLM 0.34.x on Python 3.11 or later. The server also needs the core library, `strands_decider.core`, and Pillow:

```
pip install './vllm-plugin[serve]'
```

Tested with vLLM 0.34.0, torch 2.14.0, transformers 5.22.2 and Pillow 13.0.0 on H100s, where vLLM runs Gemma 4's two head sizes on FlashAttention 4. On Ampere or Ada vLLM picks Triton attention, which the plugin refuses.

## Serving: `strands-decider-vllm serve`

```
CUDA_VISIBLE_DEVICES=0 strands-decider-vllm serve /path/to/checkpoint --port 8000
```

The checkpoint is a directory or a Hugging Face Hub repo id: the base Gemma 4 model's multimodal checkpoint with its fine-tuning merged in, `readout.safetensors` and `decision_config.json`. The server refuses one whose `decision_config.json` doesn't meet the core's schema, or whose `processor_config.json` sizes images other than as the core counts them. Run it with the vLLM environment's `bin` on `PATH`: vLLM's JIT kernel builds need its `ninja`.

The server has no authentication. It binds 127.0.0.1 by default, and vLLM's internal sockets stay on loopback whatever `--host` is; reach it from another machine through an SSH port forward. To expose it, put the deployment's own authentication in front of it: SageMaker's IAM-signed endpoint (the server answers `/invocations` and `/ping`) or an authenticating proxy. A non-loopback `--host` starts with a warning. Behind SageMaker, keep `--limit-concurrency` at least twice the instance's peak concurrent invocations: past it uvicorn answers 503 to `/ping` too, and SageMaker takes the instance as unhealthy.

| Flag | Default | |
|---|---|---|
| `MODEL`, `--revision` | | a checkpoint directory, or a Hub repo id and revision |
| `--host`, `--port` | 127.0.0.1, 8000 | |
| `--served-model-name` | `strands-decider` | each response's `model` |
| `--max-length` | `max_length_validated` | longer prompts get HTTP 422, never truncated |
| `--max-image-token-budget` | the checkpoint's `max_soft_tokens` | the largest token budget a request may give an image (70, 140, 280, 560 or 1120); a larger one gets HTTP 422, and vLLM plans its encoder cache for it |
| `--max-num-batched-tokens`, `--max-num-seqs` | `--max-length` + 1, vLLM's | the step's budget, at least `--max-length` + 1 |
| `--gpu-memory-utilization`, `--tensor-parallel-size` | 0.90, 1 | the 31B at 8,192 tokens fits one H100 at 0.95; 26B-A4B wants 0.92 |
| `--max-body-mb` | 93 (8 without images) | larger bodies get HTTP 413; 93 holds four 16 MiB images in base64 and 8 MiB of text |
| `--max-buffered-mb` | 256 | the body bytes that requests over 1 MiB hold at once, granted in arrival order and given back once a request's images are decoded |
| `--body-read-timeout` | 30 s | a body with no chunk for this long gets HTTP 408, and one that waits this long for the budget 503 |
| `--limit-concurrency` | 128 | uvicorn's limit on open connections |
| `--preprocess-workers` | 4, or the CPU count if lower | threads that render requests, and as many that decode images |

`POST /v1/systemone` (and `/invocations`) takes the System One request:

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

`images` are base64 PNG, JPEG, WebP or GIF files (a `data:image/...;base64,` prefix is allowed), each at most 16 MiB and 4096 x 4096 pixels; the prompt's length bounds how many fit. The response adds `latency_ms`, and `cached_tokens`, the prompt tokens served from vLLM's prefix cache, to `usage`:

```json
{"model": "strands-decider",
 "answers": {"damaged": {"type": "noul", "noul": 0.9312},
             "carrier": {"type": "choice", "choice": "ups", "probabilities": {"dhl": 0.0412, "ups": 0.9105, "fedex": 0.0398, "none": 0.0085}, "confidence": 0.8807},
             "severity": {"type": "score", "score": 1.2154, "legend": {"0": "none", "1": "cosmetic", "2": "contents at risk"}, "probabilities": {"0": 0.0521, "1": 0.6804, "2": 0.2675}, "confidence": 0.4521}},
 "usage": {"input_tokens": 1243, "output_tokens": 3, "cached_tokens": 1312},
 "latency_ms": 212.4}
```

An image's token budget is its own (`{"data": "<base64>", "token_budget": 560}` in `images`), else the request's `image_token_budget`, else the checkpoint's. An invalid body or image, images for a text-only checkpoint, a prompt over `--max-length`, a budget over `--max-image-token-budget` and the engine's refusals get HTTP 422 with `{"detail": "..."}`. Anything else that fails is logged and gets HTTP 500 with only an error id. `GET /health` and `GET /ping` answer 200 with the model's name, inputs and limits while the engine is healthy, and 503 when it isn't.

## Serving token ids: `vllm serve`

vLLM's own server takes text requests whose prompts the client rendered with the core. It also serves the intermediate checkpoints, a bare `Gemma4TextModel` decoder whose `decision_config.json` predates the core's schema. One GPU, the multimodal layout:

```
CUDA_VISIBLE_DEVICES=0 VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
vllm serve /path/to/checkpoint --host 127.0.0.1 --port 8000 --served-model-name m --dtype bfloat16 --seed 0 \
  --hf-overrides '{"architectures": ["StrandsDeciderForConditionalGeneration"], "text_config": {"use_bidirectional_attention": null}}' \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' \
  --max-model-len 32769 --max-num-batched-tokens 32769 --no-enable-chunked-prefill --enable-prefix-caching \
  --enable-prompt-tokens-details --generation-config vllm \
  --logprobs-mode processed_logprobs --max-logprobs 255 --gpu-memory-utilization 0.90
```

The 12B's architecture is `StrandsDeciderUnifiedForConditionalGeneration`. An intermediate decoder checkpoint takes `StrandsDeciderForCausalLM`, with `"use_bidirectional_attention": null` at the top level of `--hf-overrides` and no `--limit-mm-per-prompt`. 26B-A4B wants `--gpu-memory-utilization 0.95`. The 31B fits one GPU up to 12,288 tokens (`--max-model-len 12289 --max-num-batched-tokens 12289 --gpu-memory-utilization 0.95`) and needs two above that (`--tensor-parallel-size 2 --disable-custom-all-reduce --gpu-memory-utilization 0.92`). This server refuses images: vLLM's endpoints let a client choose an image's cache key.

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

`prompt` is the question's ids (state, then question; those above are placeholders), `prefix_end` is S with 1 <= S < L, `allowed_token_ids` the question's k code ids, and `logprobs` k. The probabilities are `exp` of `choices[0].logprobs.top_logprobs[0]` (keyed `"token_id:<id>"`), renormalized over the k ids. Several questions about one state are separate requests with the same state and S; the later ones reuse the state's cached blocks. A request with another `max_tokens`, sampling that would change the answer (a temperature other than 1.0, truncation, penalties, logit bias, bad words, structured outputs), or a missing or invalid `prefix_end` gets HTTP 400, and the server keeps serving.

## Limits

- A prompt is computed whole in one step, so chunked prefill is off and the step's budget is at least the context. Tested budgets: 32,768 tokens for text, 8,192 for images.
- At startup the plugin refuses the settings under which it would serve wrong answers: chunked prefill, `--max-num-scheduled-tokens` below `--max-model-len`, Gemma 4's "vision" mode left on (the 12B, 26B-A4B and 31B bases set it; `--hf-overrides` turns it off, and the plugin's server does), micro-batching under data parallelism, `cudagraph_mode` FULL, a logprobs mode other than the processed ones, any dtype but bf16, quantization, a quantized KV cache, prompt embeddings, speculative decoding, KV or encoder-cache connectors, `--kv-sharing-fast-prefill`, pipeline or context parallelism, a prefix-cache hash other than sha256, and attention backends other than FlashAttention.
- No sliding windows: every layer attends over every token. Latency is about 1.2 times stock causal Gemma 4 at 1K tokens and about twice at 32K.
- The 31B on one 80 GB H100 serves up to 12,288 tokens as a decoder and 8,192 with the vision encoder; 16K and 32K need TP=2.
- Answers were checked against the reference up to 8,192 tokens, the reference's limit: the same top-1 answer on 98.4% to 99.7% of 1,200 text prompts per size.
- The first requests after startup compile FlashAttention 4's non-causal kernels, which vLLM's warm-up doesn't. Send a few before measuring latency.
- Streaming input, audio and video aren't served.
- The plugin relies on vLLM 0.34 internals that aren't a public API: the Model Runner V2 model state, the FlashAttention backend, the Gemma 4 classes (whose sliding-window layers it builds without a window) and multimodal processor, the default loader's secondary weights, and wrappers around `EngineCore.preprocess_add_request`, the block-hash extra keys and the scheduler's encoder inputs. Hence the pin to 0.34.x: on any other vLLM the plugin logs a warning and registers nothing.
- KV-cache events, if enabled, report vLLM's own extra keys without the plugin's digests.

## Tests

`pip install './vllm-plugin[serve,test]'`, then `pytest` in `vllm-plugin/`. The unit tests need vLLM but no GPU, so vLLM's CPU wheel runs them, as CI does.

Before a release, run the GPU tests on one CUDA GPU (an H100 for Gemma 4's head sizes), with Hub access to `google/gemma-4-E2B-it` and `google/gemma-4-12B-it`:

```
pytest -m gpu tests/test_gpu.py
```

They build tiny random-weight checkpoints of every layout, serve them with both servers, and check every answer against transformers' own model under the mask, for text and images; the prefix cache, on exact counts of cached tokens; a prompt of exactly `--max-length`; the startup refusals; and that every socket listens on loopback. Each test runs with equal head sizes (FlashAttention 3) and with Gemma 4's two (FlashAttention 4): about 20 minutes on an H100.
