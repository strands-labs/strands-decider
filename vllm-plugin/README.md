# strands-decider-v2-vllm

A vLLM plugin that serves Strands Decider v2 checkpoints (E2B, E4B, 12B, 26B-A4B, 31B) on vLLM 0.33. It patches no vLLM file; vLLM loads it through the `vllm.general_plugins` entry point.

A v2 checkpoint directory holds:

- a merged bf16 Gemma 4 text decoder (`Gemma4TextModel`, `model_type: gemma4_text`, bare `layers.*` / `embed_tokens.*` keys), its `config.json` and the tokenizer;
- `readout.safetensors`, one tensor `weight` of shape [255, hidden];
- `decision_config.json`: `format_version: 1`, `attention_mode: prefix_bidir`, `pooling: last`, `temperature`, the 255 answer codes (`codes`: `A`, `B`, `C`, ...) and their token ids (`token_ids`).

## What the plugin does

A v2 decision is read at the last prompt token of a prompt split into a state block `[0, S)` and a question block `[S, L)`. Stock vLLM 0.33 cannot load the bare decoder checkpoint. With a repackaged checkpoint it would read the model through the language-model head with Gemma 4's causal, sliding-window attention, which gives different answers (see Parity). The plugin adds three things.

**Readout at the 255 code ids.** `StrandsDeciderV2ForCausalLM` subclasses vLLM's `Gemma4ForCausalLM`, loads the bare decoder (its keys renamed to vLLM's `model.*`) and loads `readout.safetensors`. Its `compute_logits` applies the readout (bf16) to the final-norm hidden state, then in float32 Gemma's soft-cap `cap * tanh(z / cap)` (`final_logit_softcapping`, 30 for Gemma 4), then divides by the checkpoint temperature. The 255 values are placed at the codes' token ids and every other vocabulary id gets -inf. The language-model head is not used. A request that restricts sampling to a question's codes therefore gets the log-softmax over exactly those codes.

**The `prefix_bidir` mask on every layer, with no sliding window.** State tokens attend bidirectionally to the state `[0, S)` and never to the question. Question tokens attend bidirectionally to the whole prompt `[0, L)`. A request sends S as `vllm_xargs.prefix_end`. A Model Runner V2 model state keeps S per request. In each step that computes prompt tokens of such a request, it attaches a split to the attention metadata: the scheduled state tokens become one non-causal FlashAttention sequence over keys `[0, S)`, and the scheduled question tokens one over keys `[0, L)`. Every attention layer runs the split as one non-causal FlashAttention call on the paged KV cache, plus one causal call for any other tokens in the step. Gemma 4's sliding-window layers get the same mask with no window cut: the model class builds them with a window of `--max-model-len` tokens, so they cut nothing, and FlashAttention gets window `(-1, -1)` on every layer. Requests without `prefix_end` are served causally. So is a step that computes only a prompt's last token (an exact repeat served from the cache), because causal attention over `[0, L)` is that token's mask.

**Content-keyed prefix-cache hashes.** Under this mask a state block's keys and values depend on every state token, and a question block's on the whole prompt, so vLLM's own block hashes would let a request reuse blocks that are not valid for it. In the engine core the plugin wraps `generate_block_hash_extra_keys`. Block 0 of every `prefix_bidir` request gets a SHA-256 of the state tokens as an extra key. Every block that ends past S gets a SHA-256 of the question tokens. A further question on a cached state then reuses the state's full blocks. An exact repeat reuses all of its full blocks. Different states, different boundaries and causal requests share nothing.

At startup the plugin refuses configurations that would serve the mask incorrectly: chunked prefill, `--max-num-batched-tokens` below `--max-model-len`, Model Runner V1, speculative decoding, KV connectors, `--kv-sharing-fast-prefill`, pipeline or context parallelism, a prefix-cache hash other than `sha256` / `sha256_cbor`, and vLLM's multimodal PrefixLM mode.

## Requirements

- vLLM 0.33.x. The plugin checks the version when vLLM loads it and refuses every other release, because it uses vLLM internals that change between minor releases (see Limits). It was tested on vLLM 0.33.0 with torch 2.14.0+cu130 and transformers 5.20.2.
- Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`).
- NVIDIA H100 80 GB is the only GPU tested. vLLM selects FlashAttention 4 for Gemma 4's head sizes on Hopper.
- Python 3.10 or later.

## Install

Into the environment where vLLM 0.33 is installed:

```
pip install .
```

The tests need no GPU: `pip install '.[test]'`, then run `pytest` from this directory. With vLLM installed, the tests also cover the modules that import vLLM.

## Serving

For one GPU:

```
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 \
VLLM_HOST_IP=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo \
vllm serve /path/to/checkpoint --host 127.0.0.1 --port 8000 --served-model-name m --dtype bfloat16 --seed 0 \
  --hf-overrides '{"architectures": ["StrandsDeciderV2ForCausalLM"], "use_bidirectional_attention": null}' \
  --max-model-len 32768 --max-num-batched-tokens 32768 --no-enable-chunked-prefill --enable-prefix-caching \
  --enable-prompt-tokens-details --prefix-cache-retention-interval None --generation-config vllm \
  --logprobs-mode processed_logprobs --max-logprobs 255 --gpu-memory-utilization 0.90
```

Run it with the vLLM environment's `bin` directory on `PATH`, because vLLM's JIT kernel builds need its `ninja`. Per size:

- 26B-A4B: `--gpu-memory-utilization 0.95`.
- 31B on one GPU: `--max-model-len 12288 --max-num-batched-tokens 12288 --gpu-memory-utilization 0.95`.
- 31B on two GPUs (needed for contexts above 12,288 tokens): `CUDA_VISIBLE_DEVICES=0,1` and `--tensor-parallel-size 2 --disable-custom-all-reduce --gpu-memory-utilization 0.92`.
- Without prefix caching: `--no-enable-prefix-caching` instead of `--enable-prefix-caching`.

The flags:

| Flag | Why |
|---|---|
| `VLLM_USE_V2_MODEL_RUNNER=1` | The mask is implemented as a Model Runner V2 model state. Required. |
| `--hf-overrides` `architectures` | Selects `StrandsDeciderV2ForCausalLM`. Required. |
| `--hf-overrides` `"use_bidirectional_attention": null` | The 12B, 26B-A4B and 31B configs carry `"vision"` here from their base models. That turns on vLLM's image PrefixLM mode, which selects a backend the plugin refuses. v2 is text-only. Required for those sizes, harmless for the others. |
| `--no-enable-chunked-prefill`, `--max-num-batched-tokens` >= `--max-model-len` | Every prompt token belongs to a bidirectional block, so a prompt must be computed in one step. Required. |
| `--logprobs-mode processed_logprobs`, `--max-logprobs 255` | Logprobs are taken after `allowed_token_ids` is applied, for up to all 255 codes. |
| `--generation-config vllm` | Keeps the checkpoint's sampling defaults (`top_k=64`, `top_p=0.95`) out of the logprobs. |
| `--enable-prefix-caching`, `--prefix-cache-retention-interval None` | Optional. The plugin's sliding layers still form vLLM sliding-window KV groups (with a window of `--max-model-len`). By default vLLM keeps prefix-cache checkpoints for such groups only at some blocks; `None` keeps them at every block. All measured runs used both flags. |
| `--enable-prompt-tokens-details` | Optional. Reports cached prompt tokens in each response's `usage`. |
| `--host 127.0.0.1` and the three loopback variables | Keep the HTTP server and vLLM's internal sockets on loopback. vLLM's server has no authentication by default; reach it from another machine through an SSH port forward, not a wider bind. |

## Requests

The plugin does not render prompts. The client renders the v2 prompt format with the checkpoint's tokenizer and sends token ids to `/v1/completions`, one request per question:

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

The prompt ids above are placeholders; a real prompt has hundreds to thousands of tokens. The four allowed ids are the token ids of codes `A` to `D`.

- `prompt`: the token ids of the whole prompt, L tokens.
- `vllm_xargs.prefix_end`: S, the number of tokens in the state block, which is the index of the first question token. The state block is the system turn, `State:` and the state, up to the end of the state; the question block is the rest. It must be an integer with 0 < S < L; other values are rejected. Without it the request is served causally, which is not a v2 decision.
- `allowed_token_ids`: the question's code ids, the first n entries of `decision_config.json` `token_ids` for a question with n options.
- `logprobs`: n, so that every code is returned. With `return_tokens_as_token_ids`, `choices[0].logprobs.top_logprobs[0]` is keyed `"token_id:<id>"`. The probabilities are `exp` of these logprobs, renormalized over the n ids.
- `temperature`: 1.0. The model already divides by the checkpoint temperature (1.0 for every v2 checkpoint). `top_p`, `top_k` and `min_p` must not truncate the codes.

A minimal client:

```python
import math
import requests


def decide(url, token_ids, prefix_end, code_ids):
    body = {"model": "m", "prompt": token_ids, "max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
            "min_p": 0.0, "seed": 0, "logprobs": len(code_ids), "allowed_token_ids": code_ids,
            "return_tokens_as_token_ids": True, "vllm_xargs": {"prefix_end": prefix_end}}
    resp = requests.post(f"{url}/v1/completions", json=body, timeout=600)
    resp.raise_for_status()
    top = resp.json()["choices"][0]["logprobs"]["top_logprobs"][0]
    logprobs = [top.get(f"token_id:{t}", -math.inf) for t in code_ids]
    m = max(logprobs)
    weights = [math.exp(x - m) for x in logprobs]
    total = sum(weights)
    return [w / total for w in weights]
```

Several questions about one state are separate requests with the same state tokens and the same S. With prefix caching on, the later ones reuse the state's full KV blocks.

## Limits

- Chunked prefill must be off and the step budget (`--max-num-batched-tokens`) at least `--max-model-len`, because a prompt is computed whole in one step. The tested budget is 32,768 tokens. At 32K tokens one prompt fills a step, so throughput no longer grows with concurrency.
- No sliding windows: every layer keeps and attends over every token. This costs memory and time. Latency is at most about 1.2 times that of stock causal Gemma 4 at 1K tokens, and about twice at 32K tokens.
- 31B on one 80 GB H100: the bf16 weights take 57.9 GiB. At about 0.9 MB of KV cache per token (60 layers, no windows), the cache holds 14,952 tokens, so one GPU serves contexts up to 12,288 tokens. 16K and 32K contexts need TP=2, which holds 90,865 tokens. The KV capacities of the other sizes on one GPU, at the flags above: E2B 3,348,843 tokens, E4B 927,147, 12B 134,173, 26B-A4B 100,705.
- Answers were checked against the reference only up to 8,192 tokens, the reference's limit. Longer prompts run the same code path but were not compared.
- The plugin relies on vLLM internals that are not a public API and can change in any release: `get_model_state_cls` and `DefaultModelState.prepare_attn` (Model Runner V2); the `InputBatch` and `NewRequestData` fields; `FlashAttentionImpl.forward` and the `FlashAttentionMetadata` dataclass; `register_backend` over `FLASH_ATTN`; `compilation_config.static_forward_context`; `MODELS_CONFIG_MAP`; a temporary patch of `gemma4.Attention` while the model is built; and a wrapper around `vllm.v1.core.kv_cache_utils.generate_block_hash_extra_keys`. This is why it accepts vLLM 0.33.x only.
- KV-cache events (`--kv-events-config`), if enabled, report vLLM's own extra keys for each stored block, without the plugin's digests. vLLM's block pool imports that function by name, so the wrapper does not reach it.
- As a general plugin it loads in every vLLM process of the environment it is installed in. For other models the attention path and cache keys stay vLLM's own, except that a request carrying `vllm_xargs.prefix_end` gets the extra cache keys.
- Not supported (refused at startup): chunked prefill, speculative decoding, KV connectors, pipeline and context parallelism, `--kv-sharing-fast-prefill`.

**Stock-attention control.** `tools/causal_control/` holds a separate diagnostic plugin that is not part of the package. It registers the architecture `StrandsDeciderV2CausalControlForCausalLM`, which keeps Gemma 4's own causal, sliding-window layers and reads them with the v2 readout. That is what stock vLLM would compute from a repackaged v2 checkpoint, and it measured the 57.5% to 91.3% agreement without the mask quoted under Parity. To run it, install it into the same environment (`pip install --no-deps ./tools/causal_control`). Serve with `"architectures": ["StrandsDeciderV2CausalControlForCausalLM"]` in `--hf-overrides` and the other flags above, and send requests without `prefix_end`; the control ignores a `prefix_end` and logs a warning. Installed, it only adds that architecture name; `StrandsDeciderV2ForCausalLM` is unaffected. Remove it with `pip uninstall strands-decider-v2-causal-control`.

## Parity

The reference is the v2 reference implementation run offline in PyTorch: bf16, SDPA attention, one request per forward pass, prompts up to 8,192 tokens. The parity set has 1,200 prompts per model: 300 each of about 256, 1K and 4K tokens, and 300 real states of 6,144 to 8,000 tokens. Every server answered all 1,200 at client concurrency 8 with the flags above, on vLLM 0.33.0 and one H100 per server unless TP=2. Top-1 agreement compares the argmax over the question's options. |Δp| is the largest absolute probability difference over a question's options, averaged or maximized over questions. The gate is at least 98% top-1.

| Model | Top-1 agreement | Mean \|Δp\| | Max \|Δp\| |
|---|---|---|---|
| E2B | 99.33% | 0.0050 | 0.069 |
| E4B | 99.67% | 0.0035 | 0.038 |
| 12B | 98.67% | 0.0081 | 0.120 |
| 26B-A4B | 98.42% | 0.0119 | 0.195 |
| 31B, one GPU | 99.67% | 0.0048 | 0.088 |
| 31B, TP=2 | 99.50% | 0.0050 | 0.106 |

Most disagreements are near-ties in the reference: at E2B every one has a reference top-2 gap of at most 0.025, at 12B at most 0.056. 26B-A4B is the weakest size, at 97.0% on its 1K bucket and 97.7% on its 256-token bucket. That is bf16 mixture-of-experts noise, not the mask: the same server gives the same agreement at concurrency 1. The mask cannot be skipped. With Gemma 4's own causal, windowed attention and the same readout, agreement is 57.5% (E2B) to 91.3% (31B). With prefix caching on, a further question on a cached state and an exact repeat matched a cold run on every top label, within a mean |Δp| of 0.0040.

The table was measured with version 0.1.0. Version 0.2.0 calls FlashAttention differently and was re-checked on vLLM 0.33.0 with the same 1,200 prompts and flags: E2B 99.33% top-1 (mean |Δp| 0.0049, max 0.069), E4B 99.67% (0.0035, 0.049), 12B 98.67% (0.0081, 0.120). The further-question and exact-repeat checks again matched a cold run on every top label.
