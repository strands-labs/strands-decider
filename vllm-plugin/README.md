# strands-decider-v2-vllm

A vLLM plugin that serves Strands Decider v2 checkpoints (E2B, E4B, 12B, 26B-A4B, 31B) on vLLM 0.33. It patches no vLLM file; vLLM loads it through the `vllm.general_plugins` entry point.

A v2 checkpoint holds:

- a merged bf16 Gemma 4 text decoder (`Gemma4TextModel`, `model_type: gemma4_text`, bare `layers.*` / `embed_tokens.*` keys), its `config.json` and the tokenizer;
- `readout.safetensors`, one tensor `weight` of shape [255, hidden];
- `decision_config.json`: `interface: strands-decider-v2`, `format_version: 1`, `attention_mode: prefix_bidir`, `pooling: last`, `temperature`, `final_logit_softcapping`, the 255 answer codes (`codes`: `A`, `B`, `C`, ...) and their token ids (`token_ids`).

The model can be a local directory or a Hugging Face Hub repo id: the plugin reads `decision_config.json` the way vLLM reads a model's other config files, and `readout.safetensors` through vLLM's weight loader. It refuses, at startup, a `decision_config.json` whose `interface` is anything other than `strands-decider-v2`, and `token_ids` that are not 255 distinct ids of the vocabulary. Checkpoints made before the `interface` field load with a one-line warning. The soft-cap comes from `decision_config.json`, or from the decoder's `config.json` when the file has none; a checkpoint with neither, or with two different values, is refused, so the cap is never skipped.

## What the plugin does

A v2 decision is read at the last prompt token of a prompt split into a state part `[0, S)` and a question part `[S, L)`. Stock vLLM 0.33 cannot load the bare decoder checkpoint. With a repackaged checkpoint it would read the model through the language-model head with Gemma 4's causal, sliding-window attention, which gives different answers (see Parity). The plugin adds four things.

**Readout at the 255 code ids.** `StrandsDeciderV2ForCausalLM` subclasses vLLM's `Gemma4ForCausalLM`, loads the bare decoder (its keys renamed to vLLM's `model.*`) and loads `readout.safetensors`. Its `compute_logits` applies the readout as a bf16 GEMM to the final-norm hidden state, then in float32 the soft-cap `cap * tanh(z / cap)` (30 for every Gemma 4 size), then divides by the checkpoint temperature. The 255 values are placed at the codes' token ids and every other vocabulary id gets -inf. The language-model head is not used. A request that restricts sampling to a question's codes therefore gets the log-softmax over exactly those codes.

**The `prefix_bidir` mask on every layer, with no sliding window.** State tokens attend bidirectionally to the state `[0, S)` and never to the question. Question tokens attend bidirectionally to the whole prompt `[0, L)`. A request sends S as `vllm_xargs.prefix_end`. A Model Runner V2 model state keeps S per request. In each step that computes prompt tokens of such a request, it attaches a split to the attention metadata: the scheduled state tokens become one non-causal FlashAttention sequence over keys `[0, S)`, and the scheduled question tokens one over keys `[0, L)`. Every attention layer runs the split as one non-causal FlashAttention call on the paged KV cache, plus one causal call for any other tokens in the step. Gemma 4's sliding-window layers get the same mask with no window cut: the model class builds them with a window of `--max-model-len` tokens, so they cut nothing, and FlashAttention gets window `(-1, -1)` on every layer. A step that computes only a prompt's last token (an exact repeat served from the cache) runs causally, because causal attention over `[0, L)` is that token's mask.

The split needs every prompt computed to its end in one step, and turning chunked prefill off is not enough for that: `--long-prefill-token-threshold` and micro-batching under data parallelism can still split a prompt. The plugin refuses those settings at startup (below). A v2 request generates one token (`max_tokens` 1), so vLLM never resumes one with output tokens after a preemption. If one were resumed, its output tokens would run causally, as in a decode. And if a prompt were ever scheduled in parts, its tokens would attend only to keys already computed and the server would log an error. Neither case raises.

**Content-keyed prefix-cache hashes.** Under this mask a state block's keys and values depend on every state token, and a question block's on the whole prompt, so vLLM's own block hashes would let a request reuse blocks that are not valid for it. In the engine core the plugin wraps `generate_block_hash_extra_keys`. Block 0 of every v2 request gets a SHA-256 of the state tokens as an extra key. Every block that ends past S gets a SHA-256 of the question tokens. A further question on a cached state then reuses the state's full blocks. An exact repeat reuses all of its full blocks. Different states, different boundaries and requests to other models share nothing.

**Request checks before scheduling.** The engine core checks every request to a v2 model before it is scheduled, with prefix caching on or off. A request it refuses gets HTTP 400 with a message that names the parameter, and the server keeps serving every other request. These are refused:

- a missing `vllm_xargs.prefix_end`, or one that is not an integer (a string, a float, a list, NaN, a Python `bool`), or one outside `1 <= S < L`;
- a prompt given as embeddings rather than token ids;
- a streaming-input request (an `AsyncLLM` session whose prompt grows chunk by chunk; such requests do not exist over HTTP);
- `max_tokens` other than 1, including an omitted `max_tokens`, which `/v1/completions` turns into 16;
- sampling that would change the answer: `temperature` other than 1.0, `top_k` between 1 and 254, `top_p` below 1, `min_p` above 0, any repetition, presence or frequency penalty, a `logit_bias`, `bad_words` or structured outputs;
- a request without `allowed_token_ids` whose `logprobs` is below 255: its top-n would rank all 255 codes and could leave out some of the question's own.

**Refused at startup.** The plugin refuses to start with settings under which it would serve the mask incorrectly, or under which one request could stop the engine, and names each setting:

- chunked prefill; `--max-num-batched-tokens` or `--max-num-scheduled-tokens` below `--max-model-len`; `--long-prefill-token-threshold` between 1 and `--max-model-len` minus 1;
- data parallelism together with `--enable-dbo` or `--ubatch-size` above 1;
- `cudagraph_mode` FULL, which captures prefill steps as graphs that replay their captured attention. The default, FULL_AND_PIECEWISE, is accepted, as are FULL_DECODE_ONLY, PIECEWISE and NONE: under them only uniform one-token batches run as full graphs, and those never need the split;
- a `--logprobs-mode` other than `processed_logprobs` or `processed_logits`;
- a `--dtype` other than bfloat16, `--quantization`, a `--kv-cache-dtype` other than `auto` or `bfloat16` (fp8 and other quantized KV caches), and `--enable-prompt-embeds`;
- Model Runner V1, speculative decoding, KV connectors, `--kv-sharing-fast-prefill`, pipeline and context parallelism, a prefix-cache hash other than `sha256` / `sha256_cbor`, and vLLM's multimodal PrefixLM mode.

**Other models in the same environment.** As a general plugin it loads in every vLLM process of the environment it is installed in. On vLLM 0.33.x it registers the `StrandsDeciderV2ForCausalLM` architecture and its Gemma 4 config hook. It also replaces the `FLASH_ATTN` backend with a subclass whose `forward` is vLLM's own unless the plugin's model state has attached a split, which only a v2 model does. And it wraps two engine-core functions, the block-hash extra keys and `EngineCore.preprocess_add_request`, which act only on requests to a v2 model. Other models therefore keep vLLM's attention, cache keys and request handling. On any other vLLM release the plugin logs a warning and registers nothing, so the environment's other models are unaffected and the v2 architecture is unavailable.

## Requirements

- vLLM 0.33.x. The plugin uses vLLM internals that change between minor releases (see Limits). It was tested on vLLM 0.33.0 with torch 2.14.0+cu130 and transformers 5.20.2.
- Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`).
- NVIDIA H100 80 GB is the only GPU tested. vLLM selects FlashAttention 4 for Gemma 4's head sizes on Hopper.
- Python 3.10 or later.

## Install

Into the environment where vLLM 0.33 is installed:

```
pip install .
```

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
- From the Hub: the repo id in place of `/path/to/checkpoint`.

The flags:

| Flag | Why |
|---|---|
| `VLLM_USE_V2_MODEL_RUNNER=1` | The mask is implemented as a Model Runner V2 model state. Required. |
| `--hf-overrides` `architectures` | Selects `StrandsDeciderV2ForCausalLM`. Required. |
| `--hf-overrides` `"use_bidirectional_attention": null` | The 12B, 26B-A4B and 31B configs carry `"vision"` here from their base models. That turns on vLLM's image PrefixLM mode, which selects a backend the plugin refuses. v2 is text-only. Required for those sizes, harmless for the others. |
| `--no-enable-chunked-prefill`, `--max-num-batched-tokens` >= `--max-model-len` | A prompt's state and question tokens attend to keys across their whole part, so a prompt must be computed in one step. Required. |
| `--logprobs-mode processed_logprobs`, `--max-logprobs 255` | The codes' probabilities are the logprobs taken after `allowed_token_ids` is applied, for up to all 255 codes. Required (`processed_logits` also works). The default, `raw_logprobs`, is taken before `allowed_token_ids`, so a question's codes can be missing from the top-n. |
| `--generation-config vllm` | Optional. No v2 checkpoint ships a `generation_config.json`, so it changes nothing for them. It keeps vLLM from applying one if a checkpoint ever does; requests whose defaulted sampling would drop codes would then be refused. |
| `--enable-prefix-caching`, `--prefix-cache-retention-interval None` | Optional. The plugin's sliding layers still form vLLM sliding-window KV groups (with a window of `--max-model-len`). By default vLLM keeps prefix-cache checkpoints for such groups only at some blocks; `None` keeps them at every block. All measured runs used both flags. |
| `--enable-prompt-tokens-details` | Optional. Reports cached prompt tokens in each response's `usage`. |
| `--host 127.0.0.1` and the three loopback variables | Keep the HTTP server and vLLM's internal sockets on loopback. vLLM's server has no authentication by default; reach it from another machine through an SSH port forward, not a wider bind. |

## Requests

The plugin does not render prompts. The strands-decider core library renders them (`strands_decider.v2`, added in PR #129): `strands_decider.v2.render(request, tokenizer, variant)` turns a v2 request into one prompt per question, with the prompt's token ids (`ids`), S (`state_len`, the number of tokens in the state part) and the question's code token ids (`code_token_ids`). Each question is one request to `/v1/completions`:

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

- `prompt`: the question's token ids, L tokens: the state part's ids followed by the question part's.
- `vllm_xargs.prefix_end`: S, a JSON integer with 1 <= S < L. Required; anything else gets HTTP 400. vLLM's request model turns JSON `true` and `false` into 1 and 0 before the plugin sees them, so `false` is refused as S = 0, and `true` is served as S = 1.
- `max_tokens`: 1. Required.
- `allowed_token_ids`: the question's code token ids, the first k entries of `decision_config.json` `token_ids` for a question with k options. Without it, `logprobs` must be 255: the response then has every code's logprob, which renormalizes over the question's k codes to the same probabilities.
- `logprobs`: k, so that every code is returned. With `return_tokens_as_token_ids`, `choices[0].logprobs.top_logprobs[0]` is keyed `"token_id:<id>"`. The probabilities are `exp` of these logprobs, renormalized over the k ids.
- `temperature`: 1.0, because the model already divides by the checkpoint temperature. `top_p` 1.0, `top_k` -1 (or 0), `min_p` 0.0, and no penalties, `logit_bias`, `bad_words` or structured outputs. Other values get HTTP 400.

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

- Chunked prefill must be off and the step budget at least `--max-model-len`, because a prompt is computed whole in one step. The tested budget is 32,768 tokens. At 32K tokens one prompt fills a step, so throughput no longer grows with concurrency.
- No sliding windows: every layer keeps and attends over every token. This costs memory and time. Measured with version 0.1.0, latency was at most about 1.2 times that of stock causal Gemma 4 at 1K tokens, and about twice at 32K tokens.
- 31B on one 80 GB H100: the bf16 weights take 57.9 GiB. At about 0.9 MB of KV cache per token (60 layers, no windows), the cache holds 14,952 tokens, so one GPU serves contexts up to 12,288 tokens. 16K and 32K contexts need TP=2, which holds 90,865 tokens. The KV capacities of the other sizes on one GPU, at the flags above: E2B 3,348,843 tokens, E4B 927,147, 12B 134,173, 26B-A4B 100,705.
- Answers were checked against the reference only up to 8,192 tokens, the reference's limit. Longer prompts run the same code path but were not compared.
- The first v2 requests after startup compile FlashAttention 4's non-causal kernels inside an engine step, because vLLM's own warm-up compiles causal kernels only. Send a few requests before measuring latency.
- Streaming input is not supported: the engine core refuses a streaming-input (resumable) request to a v2 model. A later chunk would extend the prompt after its boundary and cache digests were fixed.
- The plugin relies on vLLM internals that are not a public API and can change in any release: `get_model_state_cls` and `DefaultModelState.prepare_attn` (Model Runner V2); the `InputBatch` and `NewRequestData` fields; `FlashAttentionImpl.forward` and the `FlashAttentionMetadata` dataclass; `register_backend` over `FLASH_ATTN`; `compilation_config.static_forward_context`; `MODELS_CONFIG_MAP`; a temporary patch of `gemma4.Attention` while the model is built; the default loader's `secondary_weights`; and wrappers around `vllm.v1.core.kv_cache_utils.generate_block_hash_extra_keys` and `EngineCore.preprocess_add_request`. This is why it accepts vLLM 0.33.x only.
- KV-cache events (`--kv-events-config`), if enabled, report vLLM's own extra keys for each stored block, without the plugin's digests. vLLM's block pool imports that function by name, so the wrapper does not reach it.

**Stock-attention control.** `tools/causal_control/` holds a separate diagnostic plugin that is not part of the package. It registers the architecture `StrandsDeciderV2CausalControlForCausalLM`, which keeps Gemma 4's own causal, sliding-window layers and reads them with the v2 readout. That is what stock vLLM would compute from a repackaged v2 checkpoint, and it measured the 57.5% to 91.3% agreement without the mask quoted under Parity. To run it, install it into the same environment (`pip install --no-deps ./tools/causal_control`). Serve with `"architectures": ["StrandsDeciderV2CausalControlForCausalLM"]` in `--hf-overrides` and the other flags above, and send requests without `prefix_end`; the control ignores a `prefix_end` and logs a warning, and the plugin's request checks do not apply to it. Installed, it only adds that architecture name; `StrandsDeciderV2ForCausalLM` is unaffected. Remove it with `pip uninstall strands-decider-v2-causal-control`.

## Tests

`pip install '.[test]'`, then run `pytest` from this directory. Without vLLM, the tests of the request checks, the segment plan, the cache keys, `decision_config.json` and the version gate run on any machine. With vLLM 0.33 installed, the tests of the model class, the model state, the attention backend, the engine-core wrapper and vLLM's own block hasher run too, on CPU. The startup checks are tested on configs that vLLM itself builds from a checkpoint's `config.json`; set `SDV2_TEST_CHECKPOINT` to a v2 checkpoint directory to run them.

`tests/gpu/run.sh MODEL_DIR ROWS OUT_DIR` checks a server on one node with 6 to 8 GPUs, with vLLM and the plugin installed. `ROWS` is a JSON Lines file of rendered requests (`token_ids`, `prefix_end`, `label_token_ids`, and optionally a reference's `probabilities`) of about 1K tokens each. The script starts one loopback-bound server per check:

- every malformed request gets HTTP 400, the server stays healthy, and the next request is answered as before;
- cached and mixed-batch answers match cold ones;
- a server without prefix caching, and one serving the checkpoint as a Hub repo id from a local cache, give the same answers;
- concurrent requests succeed on a small KV cache;
- v2 requests that are preempted and resumed mid-generation, with the `max_tokens` check bypassed, do not stop the engine;
- each refused setting stops the server at startup with the plugin's message.

## Parity

The reference is the v2 reference implementation run offline in PyTorch: bf16, SDPA attention, one request per forward pass, prompts up to 8,192 tokens. The parity set has 1,200 prompts per model: 300 each of about 256, 1K and 4K tokens, and 300 real states of 6,144 to 8,000 tokens. Every server answered all 1,200 at client concurrency 8 with the flags above, on vLLM 0.33.0 and one H100 per server unless TP=2. Top-1 agreement compares the argmax over the question's options. |Δp| is the largest absolute probability difference over a question's options, averaged or maximized over questions. The gate is at least 98% top-1.

| Model | Top-1 agreement | Mean \|Δp\| | Max \|Δp\| | Top-1 by prompt length: 256 / 1K / 4K / 6K to 8K |
|---|---|---|---|---|
| E2B | 99.33% | 0.0049 | 0.069 | 98.67% / 100.00% / 99.67% / 99.00% |
| E4B | 99.67% | 0.0035 | 0.049 | 99.67% / 99.67% / 99.67% / 99.67% |
| 12B | 98.67% | 0.0081 | 0.120 | 99.00% / 99.00% / 98.67% / 98.00% |
| 26B-A4B | 98.42% | 0.0119 | 0.195 | 97.67% / 97.00% / 100.00% / 99.00% |
| 31B, one GPU | 99.67% | 0.0048 | 0.088 | 99.00% / 99.67% / 100.00% / 100.00% |
| 31B, TP=2 | 99.50% | 0.0050 | 0.106 | 99.00% / 99.33% / 100.00% / 99.67% |

The table was measured with version 0.2.1 on vLLM 0.33.0. Every size passes the gate with the same agreement as version 0.1.0 on the same prompts and flags, and as version 0.2.0 at E2B, E4B and 12B, the sizes 0.2.0 was checked at. There, 0.2.1's probabilities differ from 0.2.0's by a mean of 0.00005 and at most 0.023 on any question, which is run-to-run noise.

Most disagreements are near-ties in the reference: every one has a reference top-2 gap of at most 0.025 at E2B, 0.056 at 12B and 0.062 at 31B. 26B-A4B is the weakest size, at 97.0% on its 1K bucket and 97.7% on its 256-token bucket, and 16 of its 19 disagreements have gaps of at most 0.1. That is bf16 mixture-of-experts noise, not the mask: with version 0.1.0 the same server gave the same agreement at concurrency 1. The mask cannot be skipped. With Gemma 4's own causal, windowed attention and the same readout, agreement is 57.5% (E2B) to 91.3% (31B). With prefix caching on, a further question on a cached state and an exact repeat matched a cold run on every top label at every size, within a mean |Δp| of 0.0040.
