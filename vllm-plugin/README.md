# strands-decider-vllm

A vLLM plugin that serves Strands Decider checkpoints (E2B, E4B, 12B, 26B-A4B, 31B) on vLLM 0.33, with text and image input. It patches no vLLM file; vLLM loads it through the `vllm.general_plugins` entry point. It also ships a server, `strands-decider-vllm serve`, which answers the System One request (`POST /v1/systemone`, images included) with vLLM's engine in process.

## Checkpoints

A checkpoint is the base Gemma 4 model's own multimodal checkpoint, in the base's layout, with its fine-tuning merged in (specification, `docs/spec.md`, section 9, added in PR #129):

- `config.json` and `processor_config.json`: the base's, unchanged. `Gemma4ForConditionalGeneration` for E2B, E4B, 26B-A4B and 31B, `Gemma4UnifiedForConditionalGeneration` for the 12B.
- The weights under the base's names: the decoder under `model.language_model.`, the vision encoder and its projection (and the audio encoder of E2B and E4B) under theirs. There is no LM-head tensor.
- The base's tokenizer, with no chat template.
- `readout.safetensors`: one tensor `weight` of shape [255, hidden].
- `decision_config.json` (section 8): `interface: strands-decider`, `inputs` (`["text"]` or `["text", "image"]`), `final_logit_softcapping`, `provenance.base_model` and `provenance.base_revision`, the 255 code token ids and the rest.

The plugin also serves the intermediate checkpoints made before this layout: a bare `Gemma4TextModel` decoder (`model_type: gemma4_text`, bare `layers.*` / `embed_tokens.*` keys), its readout and a `decision_config.json` without `interface`, `inputs` or `provenance`. Each missing field is a one-line warning, and such a checkpoint is served as text-only. A decoder checkpoint whose `inputs` name `"image"` is refused, because it has no vision encoder.

The model can be a local directory or a Hugging Face Hub repo id: the plugin reads `decision_config.json` the way vLLM reads a model's other config files, and `readout.safetensors` through vLLM's weight loader. At startup it refuses:

- an `interface` other than `strands-decider`, checked before any other field;
- for the multimodal layout, a missing `interface`, `inputs`, `final_logit_softcapping` or provenance;
- malformed `inputs` or provenance;
- `token_ids` that are not 255 distinct ids of the vocabulary.

The soft-cap comes from `decision_config.json`; an intermediate checkpoint without it takes its decoder config's. A checkpoint with neither, or with two different values, is refused, so the cap is never skipped.

## What the plugin does

A decision, as the specification defines it, is read at the last prompt token of a prompt split into a state part `[0, S)` and a question part `[S, L)`. Images go first in the state, each as `<|image>`, N `<|image|>` placeholders and `<image|>`, with N at most 280 (section 3.2); S counts them. Stock vLLM 0.33 would read the model through the language-model head with Gemma 4's causal, sliding-window attention, which gives different answers (see Parity). The plugin adds the following.

**Three architectures, one per layout.** Each subclasses vLLM's own Gemma 4 class for its layout and replaces its LM head with the readout:

| Layout | Plugin architecture | vLLM class |
|---|---|---|
| multimodal, E2B, E4B, 26B-A4B, 31B | `StrandsDeciderForConditionalGeneration` | `Gemma4ForConditionalGeneration` |
| multimodal, 12B | `StrandsDeciderUnifiedForConditionalGeneration` | `Gemma4UnifiedForConditionalGeneration` |
| intermediate bare decoder | `StrandsDeciderForCausalLM` | `Gemma4ForCausalLM` |

The multimodal classes use vLLM's Gemma 4 image path as it is: its processor expands each `<|image|>` into the image's placeholders, its vision encoder (or the 12B's patch embedder) fills them, and its encoder cache keeps the outputs.

**Readout at the 255 code ids.** `compute_logits` applies the readout as a bf16 GEMM to the final-norm hidden state. Then, in float32, it applies the soft-cap `cap * tanh(z / cap)` (30 for every Gemma 4 size) and divides by the checkpoint temperature. The 255 values are placed at the codes' token ids, and every other vocabulary id gets -inf. A request that restricts sampling to a question's codes therefore gets the log-softmax over exactly those codes.

**The `prefix_bidir` mask on every layer of the language model, with no sliding window.** State tokens, image placeholders included, attend bidirectionally to the state `[0, S)` and never to the question. Question tokens attend bidirectionally to the whole prompt `[0, L)`. A request sends S as `vllm_xargs.prefix_end`. A Model Runner V2 model state keeps S per request and, in each step that computes prompt tokens, attaches a split to the attention metadata: the scheduled state tokens become one non-causal FlashAttention sequence over keys `[0, S)`, and the scheduled question tokens one over keys `[0, L)`. Every attention layer of the language model runs the split as one non-causal FlashAttention call on the paged KV cache, plus one causal call for any other tokens in the step. The vision encoder is transformers' own module and never sees the split. Gemma 4's sliding-window layers get the same mask with no window cut: the model classes build them with a window of `--max-model-len` tokens, so they cut nothing, and FlashAttention gets window `(-1, -1)` on every layer. A step that computes only a prompt's last token (an exact repeat served from the cache) runs causally, because causal attention over `[0, L)` is that token's mask.

The split needs every prompt computed to its end in one step. Turning chunked prefill off is not enough for that: `--long-prefill-token-threshold` and micro-batching under data parallelism can still split a prompt, and the plugin refuses those settings at startup (below). With images, vLLM's scheduler also splits a prompt when the encoder's budget or cache cannot take one of its images in this step: it would schedule the tokens in front of the image and the rest later. The plugin wraps that scheduling step (`Scheduler._try_schedule_encoder_inputs`). A decision prompt it would cut is not scheduled in that step at all, the encoder-cache references the attempt took are released, and the request waits for the next step. A decision request generates one token (`max_tokens` 1), so vLLM never resumes one with output tokens after a preemption. If one were resumed, its output tokens would run causally, as in a decode. If a prompt were ever scheduled in parts, its tokens would attend only to keys already computed, and the server would log an error. Neither case raises.

**The base's "vision" mode is turned off.** The 12B, 26B-A4B and 31B `config.json` carry `"use_bidirectional_attention": "vision"`. In vLLM this turns on its multimodal PrefixLM mode: each image's tokens attend to each other in both directions on the sliding-window layers (clamped to the window), the rest stays causal, and the image masks go to a Triton composite backend. Under `prefix_bidir` every state token already attends to the whole state in both directions on every layer, so the mode adds nothing and must be off. The checkpoint keeps the field, because its `config.json` is the base's unchanged (section 9) and transformers and the other runtimes read it there. The checkpoint's `decision_config.json` names the mask that governs (`attention_mode: prefix_bidir`), as it does for the sliding windows that `config.json` also keeps. The runtime turns the base's mode off with `--hf-overrides '{"text_config": {"use_bidirectional_attention": null}}'` (top level for a bare decoder). `strands-decider-vllm serve` sets this override itself, and the plugin refuses to start if the mode is still on.

**The specification's pixels.** Section 11.1 resizes each decoded image to its W × H with Pillow's bicubic resampling, then runs the checkpoint's image processor with its own resize off. The server does the resize and sends `do_resize=False` with each image request. Two parts of vLLM 0.33 would not follow it, and the plugin registers its own processor for its two multimodal classes (vLLM's classes keep theirs):

- vLLM counts an image's placeholders with its own formula, which assumes the processor resizes the image. On an image already at W × H that formula is not N for 1,787 of the 3,200 sizes of the core's soft-token table, and the encoder's output would not fit the placeholders. The plugin counts one placeholder per 48 × 48 pixels when a request turns the resize off, which is what the processor produces.
- transformers resolves `Gemma4ImageProcessor` to its torchvision backend, which converts the pixels to floating point differently even with the resize off. The plugin gives vLLM's processor `Gemma4ImageProcessorPil`, the reference's, for E2B, E4B, 26B-A4B and 31B. The 12B's `Gemma4UnifiedImageProcessor` has only a torchvision backend, and the reference uses it too.

The pixels vLLM encodes were checked to be the reference's, bit for bit (see Parity).

**Content-keyed prefix-cache hashes.** Under this mask a state block's keys and values depend on every state token, images included, and a question block's on the whole prompt. vLLM's own block hashes would let a request reuse blocks that are not valid for it. An image's placeholder ids are the same for every image of the same size, and vLLM's multimodal key covers only the blocks that hold an image's tokens, not the state tokens in front of it, which attend to it under the mask. In the engine core the plugin wraps `generate_block_hash_extra_keys`:

- Block 0 of every request to a Strands Decider model gets a SHA-256 of the state's token ids and of each of its images: vLLM's multimodal hash of the image and its placeholder span.
- Every block that ends past S gets a SHA-256 of that digest and the question's tokens.

The server passes each image file's sha256 as vLLM's per-item id, so vLLM's multimodal hash of an image is the specification's cache key, the file's content (section 6). A further question on a cached state then reuses the state's full blocks, and an exact repeat reuses all of its full blocks. Different states, different images, different boundaries and requests to other models share nothing.

**Request checks before scheduling.** The engine core checks every request to a Strands Decider model before it is scheduled, with prefix caching on or off. A refused request gets HTTP 400 with a message that names the parameter, and the server keeps serving every other request. These are refused:

- a missing `vllm_xargs.prefix_end`, or one that is not an integer (a string, a float, a list, NaN, a Python `bool`), or one outside `1 <= S < L`;
- a prompt given as embeddings rather than token ids;
- a streaming-input request (an `AsyncLLM` session whose prompt grows chunk by chunk; such requests do not exist over HTTP);
- `max_tokens` other than 1, including an omitted `max_tokens`, which `/v1/completions` turns into 16;
- sampling that would change the answer: `temperature` other than 1.0, `top_k` between 1 and 254, `top_p` below 1, `min_p` above 0, any repetition, presence or frequency penalty, a `logit_bias`, `bad_words` or structured outputs;
- a request without `allowed_token_ids` whose `logprobs` is below 255: its top-n would rank all 255 codes and could leave out some of the question's own;
- images for a checkpoint whose `inputs` lack `"image"`;
- more than 4 images, media other than images, and an image whose tokens do not lie inside the state `[1, S)`;
- images on an engine that `strands-decider-vllm serve` did not start. vLLM's own endpoints let a client name an image by an id of its choosing (`multi_modal_uuids`, or the pre-hashed `features` of `/inference/v1/generate`), which vLLM then uses as the image's cache key. With it, two different images could share cached state. The server marks its engines in vLLM's `additional_config` under `strands_decider_server`. The mark is trust at the operator's level, not a security boundary: it says that every request reaching the engine names each image by its file's sha256. An operator who sets it for `vllm serve` (`--additional-config`) lets that server's clients make two images share cached state.

**Refused at startup.** The plugin refuses to start with settings under which it would serve the mask incorrectly, or under which one request could stop the engine, and names each setting:

- chunked prefill; `--max-num-batched-tokens` or `--max-num-scheduled-tokens` below `--max-model-len`; `--long-prefill-token-threshold` between 1 and `--max-model-len` minus 1;
- data parallelism together with `--enable-dbo` or `--ubatch-size` above 1;
- `cudagraph_mode` FULL, which captures prefill steps as graphs that replay their captured attention. The default, FULL_AND_PIECEWISE, is accepted, as are FULL_DECODE_ONLY, PIECEWISE and NONE: under them only uniform one-token batches run as full graphs, and those never need the split;
- a `--logprobs-mode` other than `processed_logprobs` or `processed_logits`;
- a `--dtype` other than bfloat16, `--quantization`, a `--kv-cache-dtype` other than `auto` or `bfloat16` (fp8 and other quantized KV caches), and `--enable-prompt-embeds`;
- Model Runner V1, speculative decoding, KV connectors, encoder-cache connectors, `--kv-sharing-fast-prefill`, pipeline and context parallelism, a prefix-cache hash other than `sha256` / `sha256_cbor`, and vLLM's multimodal PrefixLM mode.

**Other models in the same environment.** As a general plugin it loads in every vLLM process of the environment it is installed in. On vLLM 0.33.x it:

- registers its three architectures and Gemma 4's config hooks for them;
- replaces the `FLASH_ATTN` backend, and on a CPU build of vLLM the `CPU_ATTN` backend, with a subclass whose `forward` is vLLM's own unless the plugin's model state has attached a split, which only a Strands Decider model does;
- wraps engine-core functions: the block-hash extra keys, `EngineCore.preprocess_add_request`, and the scheduler's `__init__` (to read the checkpoint's `decision_config.json` at startup), `schedule` and encoder-input step. Each acts only on engines or requests of a Strands Decider model.

Other models therefore keep vLLM's attention, cache keys, scheduling and request handling. On any other vLLM release the plugin logs a warning and registers nothing, so the environment's other models are unaffected and the plugin's architectures are unavailable.

## Requirements

- vLLM 0.33.x. The plugin uses vLLM internals that change between minor releases (see Limits). It was tested on vLLM 0.33.0 with torch 2.14.0+cu130, transformers 5.20.2, torchvision 0.29.0 and Pillow 13.0.0.
- Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`; the server sets it).
- NVIDIA H100 80 GB is the only GPU tested. vLLM selects FlashAttention 4 for Gemma 4's two head sizes on Hopper. On Ampere or Ada GPUs vLLM's Gemma 4 hook picks Triton attention for the real checkpoints instead, which the plugin refuses; FlashAttention 2 there is untested.
- vLLM's x86 CPU build runs the plugin too, on vLLM's CPU attention, for the tests' CPU tier (below). It is not a serving target: the checkpoints are bf16, and the tier runs tiny checkpoints in float32, which a CPU build of vLLM also accepts.
- Python 3.10 or later.
- For the server: the core library, `strands_decider.core` (the `strands-decider` distribution, from PR #129 on: `strands-decider>=0.3.1.dev0,<0.5`, the `serve` extra), with `tokenizers`. A checkout installs as a `0.3.1.dev` version only when git can see its tags (`fetch-depth: 0` in GitHub Actions). The server checks the core's functions at startup, before the engine starts, and names what is missing.

## Install

Into the environment where vLLM 0.33 is installed:

```
pip install '.[serve]'
```

## Serving with images: `strands-decider-vllm serve`

vLLM 0.33 has no endpoint that takes a rendered prompt with images and keys its cache on the images' content:

- `/v1/completions` takes token ids but no images.
- The chat endpoints render a chat template, which the specification forbids for these prompts (section 3.5).
- `/inference/v1/generate` takes token ids with images, but it lets a client name an image by an id of its choosing, and the client would also have to collapse each image's placeholders the way vLLM's Gemma 4 processor expects.

So the plugin ships a server that takes v1's request and does the rest in process. It renders the request with the core library and the checkpoint's prompt template. It decodes and resizes each image as the specification says, with the core's `check_decoded_size` and `resize_image` (Pillow's bicubic resize, the width's pass and then the height's, which gives the specification's pixels on every Pillow version). For each question it sends vLLM the prompt with each image's line collapsed to the one `<|image|>` that vLLM expands, the images, their sha256 and S. It checks that vLLM's expanded prompt is the rendered one, token for token, and answers in v1's response format. The first question of a request runs alone and the others then together, so that they reuse the state's cached blocks.

```
CUDA_VISIBLE_DEVICES=0 strands-decider-vllm serve /path/to/checkpoint --port 8000
```

`python -m strands_decider_vllm serve ...` does the same. Run it with the vLLM environment's `bin` directory on `PATH`, because vLLM's JIT kernel builds need its `ninja`. The server binds 127.0.0.1 by default. vLLM's internal sockets stay on loopback whatever `--host` is: it sets `VLLM_HOST_IP`, `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` unless they are set. It has no authentication. Reach it from another machine through an SSH port forward. To expose it beyond the machine, put the deployment's own authentication in front of it: SageMaker's IAM-signed endpoint (the server answers SageMaker's `/invocations` and `/ping`), or an authenticating proxy. A `--host` other than loopback starts with a warning that says so.

It starts vLLM with the flags of the next section for the checkpoint's layout: the plugin's architecture, the override that turns the base's "vision" mode off, at most 4 images and no video or audio per prompt (no images for an `inputs: ["text"]` checkpoint), `sha256` prefix-cache hashes, and a step budget of `--max-model-len` tokens. For a Hub repo it resolves the revision to a commit once and reads every file, and the weights, at that commit. Its flags:

| Flag | Default | |
|---|---|---|
| `MODEL` | | the checkpoint directory or Hub repo id; `--revision` for the Hub |
| `--host`, `--port` | `127.0.0.1`, 8000 | |
| `--served-model-name` | `strands-decider` | the `model` of each response |
| `--max-length` | `max_length_validated` | longer prompts are refused (HTTP 400), never truncated; raising it past the validated length is the operator's decision |
| `--max-model-len` | `--max-length` + 1 | the engine's context. vLLM 0.33 refuses a prompt that fills the context, because the answer token needs one more position, so it must be above `--max-length`; the server refuses to start otherwise |
| `--max-num-batched-tokens`, `--max-num-seqs` | `--max-model-len`, vLLM's | the step's budget (at least `--max-model-len`) and its request limit |
| `--gpu-memory-utilization`, `--tensor-parallel-size` | 0.90, 1 | the 31B with 8,192-token contexts fits one H100 at 0.95; 26B-A4B wants 0.92 |
| `--no-enable-prefix-caching` | caching on | |
| `--mm-processor-cache-gb` | 4 | vLLM's cache of preprocessed images; 0 turns it off |
| `--block-size` | vLLM's | the KV-cache block size |
| `--dtype` | `bfloat16` | `float32` is accepted on a CPU build of vLLM only, for tests |
| `--enforce-eager` | off | no CUDA graphs |
| `--max-body-mb` | 187 (16 for a text-only checkpoint) | larger request bodies get HTTP 413, by `Content-Length` before reading and by size while reading. The default holds four 32 MiB image files in base64 (more than a 4,096 × 4,096 photo takes as JPEG, WebP or compressed PNG) and 16 MiB of text |
| `--limit-concurrency` | 128 | uvicorn's limit on open connections and requests; past it uvicorn answers HTTP 503. `/ping` and `/health` count toward it, so set it above the clients' concurrency |
| `--preprocess-workers` | 4, or the CPU count if lower | threads that parse, render and decode requests, and so the most requests decoded at once; the others wait without holding a thread |

The server reads `decision_config.json` with the core's strict loader and the tokenizer with the core's `check_tokenizer`, and refuses at startup a checkpoint that does not meet the specification's schema. An intermediate checkpoint has no `prompt` field and is served with `vllm serve` and pre-rendered token ids (next section). The server refuses image requests if the checkpoint's `processor_config.json` sets other values than the specification's `patch_size` 16, `pooling_kernel_size` 3 and `max_soft_tokens` 280 (the core's `check_image_processor`).

Parsing, rendering and image decoding run on the `--preprocess-workers` threads, never on the event loop, so `/ping` and `/health` answer while images are decoded. vLLM runs its own image preprocessing on its renderer's thread.

### Requests

`POST /v1/systemone` (and `POST /invocations`) takes v1's request (section 1):

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

`images` is optional: at most 4 base64 PNG, JPEG, WebP or GIF files (a `data:image/...;base64,` prefix is allowed), each at most 4096 × 4096 pixels. The response is v1's, plus `latency_ms` and, in `usage`, `cached_tokens`, the prompt tokens served from vLLM's prefix cache over all questions. Its shape (the numbers are illustrative):

```json
{"model": "strands-decider",
 "answers": {"damaged": {"type": "noul", "noul": 0.9312},
             "carrier": {"type": "choice", "choice": "ups", "probabilities": {"dhl": 0.0412, "ups": 0.9105, "fedex": 0.0398, "none": 0.0085}, "confidence": 0.8807},
             "severity": {"type": "score", "score": 1.2154, "legend": {"0": "none", "1": "cosmetic", "2": "contents at risk"}, "probabilities": {"0": 0.0521, "1": 0.6804, "2": 0.2675}, "confidence": 0.4521}},
 "usage": {"input_tokens": 1243, "output_tokens": 3, "cached_tokens": 1312},
 "latency_ms": 212.4}
```

These are refused with HTTP 400 and `{"detail": "..."}`, with the messages of v1 and the specification, and the server keeps serving:

- a body that is not a JSON object;
- a malformed request or question;
- more than 4 images;
- an image that is not base64, not one of the four formats, too large, or not decodable, or whose decoded size is neither its header's nor its transpose;
- images for a text-only checkpoint (`this checkpoint does not take images: its decision_config.json inputs are ["text"]`), before any image is decoded;
- a prompt longer than `--max-length`;
- any refusal of the engine's request checks (`VLLMValidationError`, which `AsyncLLM.generate` raises for a request vLLM's input processing or the plugin's engine-core checks refuse).

A body over `--max-body-mb` gets HTTP 413. Anything else that fails, a prompt whose expansion by vLLM differs from the rendered one included, is logged with its traceback and gets HTTP 500 with `{"detail": "internal server error", "error_id": "..."}`, the id in the log line, never an answer and never the failure's own text. If one question of a request fails, its other questions' engine requests are cancelled. `GET /health` and `GET /ping` answer 200 with the model's name, inputs, prompt and limits while the engine is healthy, and 503 when it is not.

## Serving token ids: `vllm serve`

Clients that render prompts themselves with the core library (`strands_decider.core.render(request, tokenizer, variant)`) can send text requests to vLLM's own server. One GPU, the multimodal layout:

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

The architecture is `StrandsDeciderUnifiedForConditionalGeneration` for the 12B, and `StrandsDeciderForCausalLM` with `"use_bidirectional_attention": null` at the top level of `--hf-overrides` for an intermediate decoder checkpoint (without `--limit-mm-per-prompt`). With `--limit-mm-per-prompt` at 0, vLLM builds no vision or audio encoder. Image requests are refused on this server (see the request checks); use `strands-decider-vllm serve` for them. Per size:

- 26B-A4B: `--gpu-memory-utilization 0.95`.
- 31B on one GPU: `--max-model-len 12289 --max-num-batched-tokens 12289 --gpu-memory-utilization 0.95`.
- 31B on two GPUs (needed for contexts above 12,288 tokens): `CUDA_VISIBLE_DEVICES=0,1` and `--tensor-parallel-size 2 --disable-custom-all-reduce --gpu-memory-utilization 0.92`.
- Without prefix caching: `--no-enable-prefix-caching` instead of `--enable-prefix-caching`.
- From the Hub: the repo id in place of `/path/to/checkpoint`.

The flags:

| Flag | Why |
|---|---|
| `VLLM_USE_V2_MODEL_RUNNER=1` | The mask is implemented as a Model Runner V2 model state. Required. |
| `--hf-overrides` `architectures` | Selects the plugin's architecture for the layout. Required. |
| `--hf-overrides` `use_bidirectional_attention: null` | Turns off the base's "vision" mode, which `prefix_bidir` replaces (above). Required for the 12B, 26B-A4B and 31B, harmless for the others. |
| `--no-enable-chunked-prefill`, `--max-num-batched-tokens` >= `--max-model-len` | A prompt's state and question tokens attend to keys across their whole part, so a prompt must be computed in one step. Required. |
| `--max-model-len` one above the longest prompt | vLLM 0.33 refuses a prompt of exactly `--max-model-len` tokens (the answer token needs a position), so 32,769 serves prompts of up to 32,768. |
| `--logprobs-mode processed_logprobs`, `--max-logprobs 255` | The codes' probabilities are the logprobs taken after `allowed_token_ids` is applied, for up to all 255 codes. Required (`processed_logits` also works). The default, `raw_logprobs`, is taken before `allowed_token_ids`, so a question's codes can be missing from the top-n. |
| `--generation-config vllm` | Optional. No Strands Decider checkpoint ships a `generation_config.json`, so it changes nothing for them. It keeps vLLM from applying one if a checkpoint ever does; requests whose defaulted sampling would drop codes would then be refused. |
| `--enable-prefix-caching`, `--prefix-cache-retention-interval None` | Optional. The plugin's sliding layers still form vLLM sliding-window KV groups (with a window of `--max-model-len`). By default vLLM keeps prefix-cache checkpoints for such groups only at some blocks; `None` keeps them at every block. All measured runs used both flags. |
| `--enable-prompt-tokens-details` | Optional. Reports cached prompt tokens in each response's `usage`. |
| `--host 127.0.0.1` and the three loopback variables | Keep the HTTP server and vLLM's internal sockets on loopback. vLLM's server has no authentication by default; reach it from another machine through an SSH port forward, not a wider bind. |

Each question is one request to `/v1/completions`, with the rendered prompt's ids (`ids`), S (`state_len`) and the question's code token ids (`code_token_ids`):

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

- Chunked prefill must be off and the step budget at least `--max-model-len`, because a prompt is computed whole in one step. The tested budget is 32,768 tokens for text and 8,192 for images. At 32K tokens one prompt fills a step, so throughput no longer grows with concurrency.
- No sliding windows: every layer keeps and attends over every token. This costs memory and time. Measured with version 0.1.0, latency was at most about 1.2 times that of stock causal Gemma 4 at 1K tokens, and about twice at 32K tokens.
- 31B on one 80 GB H100: the bf16 weights take 57.9 GiB as a decoder and 59.0 GiB with the vision encoder. At about 0.9 MB of KV cache per token (60 layers, no windows), one GPU serves contexts up to 12,288 tokens as a decoder and, with the encoder, 8,192 tokens (16,662 tokens of KV cache at `--gpu-memory-utilization 0.95`). 16K and 32K contexts need TP=2, which holds 90,865 tokens. The KV capacities of the other sizes on one GPU, at the flags above: E2B 3,348,843 tokens, E4B 927,147, 12B 134,173, 26B-A4B 100,705.
- Answers were checked against the reference only up to 8,192 tokens, the reference's limit. Longer prompts run the same code path but were not compared.
- The first requests after startup compile FlashAttention 4's non-causal kernels inside an engine step, because vLLM's own warm-up compiles causal kernels only. Send a few requests before measuring latency.
- A prompt with images is scheduled whole or not at all. When the encoder's budget or cache cannot take one of its images this step, vLLM would compute the part in front of that image; the plugin holds the whole prompt back instead, and for the rest of that step also holds back every later request with an image to compute, so that smaller image requests cannot keep taking the room it needs. It is scheduled as soon as the steps in flight have been processed (one step with asynchronous scheduling). `tests/test_scheduling.py` drives vLLM's own scheduler to show both: a 4-image request behind a stream of 1-image requests is scheduled within two steps, and without the rule never.
- Streaming input is not supported: the engine core refuses a streaming-input (resumable) request to a Strands Decider model. A later chunk would extend the prompt after its boundary and cache digests were fixed.
- Audio and video are not served: the engine core refuses them, and the server starts vLLM with neither.
- The plugin relies on vLLM internals that are not a public API and can change in any release: `get_model_state_cls` and `DefaultModelState.prepare_attn` (Model Runner V2); the `InputBatch` and `NewRequestData` fields; `FlashAttentionImpl.forward` and the `FlashAttentionMetadata` dataclass; `register_backend` over `FLASH_ATTN`; `compilation_config.static_forward_context`; `MODELS_CONFIG_MAP`; a temporary patch of `gemma4.Attention` while the model is built; the default loader's `secondary_weights`; the Gemma 4 multimodal classes' constructors, weight mappers and processing info, and `Gemma4MultiModalProcessor._get_prompt_updates`; and wrappers around `vllm.v1.core.kv_cache_utils.generate_block_hash_extra_keys`, `EngineCore.preprocess_add_request`, and the `Scheduler`'s `__init__`, `schedule` and `_try_schedule_encoder_inputs` (whose return value it rebuilds from vLLM's own). This is why it accepts vLLM 0.33.x only.
- KV-cache events (`--kv-events-config`), if enabled, report vLLM's own extra keys for each stored block, without the plugin's digests. vLLM's block pool imports that function by name, so the wrapper does not reach it.

**Stock-attention control.** `tools/causal_control/` holds a separate diagnostic plugin that is not part of the package. It registers the architecture `StrandsDeciderCausalControlForCausalLM`, which keeps Gemma 4's own causal, sliding-window layers and reads them with the trained readout. That is what stock vLLM would compute from a repackaged checkpoint, and it measured the 57.5% to 91.3% agreement without the mask quoted under Parity. To run it, install it into the same environment (`pip install --no-deps ./tools/causal_control`). Serve an intermediate decoder checkpoint with `"architectures": ["StrandsDeciderCausalControlForCausalLM"]` in `--hf-overrides` and the other flags above, and send requests without `prefix_end`. The control ignores a `prefix_end` and logs a warning, and the plugin's request checks do not apply to it. Installed, it only adds that architecture name; the plugin's architectures are unaffected. Remove it with `pip uninstall strands-decider-causal-control`.

## Tests

Three tiers. The CPU tier runs on any machine, in CI. The GPU tier and the conformance tier run on a GPU cluster, and together with `tests/gpu/run.sh` they are the release gate: a release passes all three first. No tier's tolerance is wider than a measured difference with headroom, and `tests/gpu/ci.sh` gives the measurement behind each one beside it. Checks of the prefix cache are exact counts of cached tokens.

**CPU tier.** `pip install '.[test,serve]'`, then `pytest` from this directory. It needs no GPU, and a test that does need one is marked `requires_cuda` and skips itself.

- Without vLLM, these run: the request and image checks, the segment plan, the cache keys and their digests, `decision_config.json` for both layouts, the scheduler and engine-core wrappers on stand-ins, the reading of golden lines, and the version gate.
- With vLLM 0.33 installed (its CUDA wheel or its CPU build), these run too: the model classes and the readout's arithmetic, the model state, the attention backend, the image processing, vLLM's own block hasher with images, vLLM's own scheduler driven step by step with a full encoder cache (a prompt cut before an image is held back whole, and a held-back 4-image request is not starved), and the server's whole request path against a stand-in engine. That path covers rendering, the collapsed prompt, the resize, the v1 response, the HTTP refusals, the body limit, the generic 500, `/ping` answering while an image is decoded, and the cancellation of a failed request's other questions; it needs the core library and Pillow.
- With `STRANDS_DECIDER_TEST_CHECKPOINT` set to a checkpoint directory (only its `config.json` and tokenizer are read, in either layout), the startup checks also run, on configs vLLM itself builds.
- With `STRANDS_DECIDER_GEMMA4_TOKENIZER` set to Gemma 4's `tokenizer.json`, the tests that need the real tokenizer run.

On 4 cores with the CPU build of vLLM 0.33.0 and the core installed, 454 tests pass and 42 skip (those that need a checkpoint) in 51 s; with the CUDA wheel on 8 cores of a GPU node (no GPU used) and both variables set, 495 pass and 1 skips in under 2 minutes.

The CPU tier's second half runs the plugin end to end with no GPU: `tests/cpu/ci.sh OUT_DIR` on a CPU build of vLLM, which is `tests/gpu/ci.sh` (next) with `DEVICE=cpu`. On vLLM's CPU attention it serves tiny random-weight checkpoints of every layout and checks the answers against the transformers reference, in float32. The CPU build's attention (`CPU_ATTN`) runs paged KV for causal and non-causal sequences alike, and the plugin runs its split there as on FlashAttention (`attention_cpu.py`). Its KV blocks are 128 tokens by default; the tier serves with 32-token blocks, so that the cache checks see reuse. The CPU build of vLLM 0.33.0 is a separate wheel, for glibc 2.39 or later (Ubuntu 24.04), with torch's CPU build:

```
uv pip install "https://wheels.vllm.ai/4ba787ec9959e566d3b587a796dff07673324da5/vllm-0.33.0%2Bcpu-cp38-abi3-manylinux_2_39_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
sudo apt-get install -y libnuma1   # vLLM's CPU kernels; g++ for torch's compiler
pip install --no-deps '.[test,serve]'
```

The wheel is 153 MiB and the environment 1.7 GB; the install took 97 s from a cold cache. Its `vllm.__version__` is `0.33.0`, which the plugin's version gate accepts. On 4 cores of an x86 server CPU with AVX2 and no AVX-512, in a `python:3.12-slim-trixie` container with 16 GB, the tier took CPU_TIER_TIME with vLLM's compile caches warm, most of it starting its three servers (190 to 310 s each, mostly compilation). Keep `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` between runs (about 30 MB); a cold server start takes about 500 s. In float32 the checks are exact: a cached, repeated or batched answer equals the cold one bit for bit (tolerance 1e-6), and the answers agree with the float32 reference on every top-1 code with a mean |Δp| of at most 0.0001 (gates: top-1 1.0, mean 0.001, max 0.005).

**GPU tier.** `tests/gpu/ci.sh OUT_DIR` runs the plugin end to end on one GPU of 24 GB or more, in bf16 as it serves, with no dataset and no real weights. It generates tiny random-weight checkpoints in the real layouts, with the real tokenizer and image processor (`tests/gpu/tiny.py`): multimodal, multimodal text-only, unified (the 12B's class) and a bare decoder. Their attention's q/k norms are scaled at initialization so that a random model's attention is smooth: with Gemma 4's norms at their initial weight, a random model's bf16 answers differ from its own float32 answers on 15% of questions, which would make it useless as a test. It writes requests with synthetic photos (`tests/gpu/make_requests.py`) and computes the transformers reference on each checkpoint (`tests/gpu/reference.py`). Then:

- `strands-decider-vllm serve` on the multimodal and unified checkpoints: text and image requests, the refusals with the server still serving, the image-content cache test and state reuse on exact counts of cached tokens (`tests/gpu/server_client.py`), every answer against the reference, and the conformance check (`tests/gpu/conformance.py`) on goldens written from that reference (`tests/gpu/goldens.py`);
- the server's request path in process on a real engine (`tests/gpu/engine_checks.py`): a prompt of exactly `--max-length` tokens is answered, one token more is refused, vLLM refuses a prompt that fills the context, and a request the engine core refuses raises `VLLMValidationError` from `AsyncLLM.generate`, which the server answers with HTTP 400;
- `vllm serve` on the bare decoder: the `/v1/completions` checks of `tests/gpu/client.py`, with exact cache counts, against the reference and against a second server without prefix caching.

It fetches the four base files it needs (`config.json`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`) from the Hub at pinned revisions, or takes them from `STRANDS_DECIDER_GEMMA4_BASE` and `STRANDS_DECIDER_GEMMA4_UNIFIED_BASE`. With `STRANDS_DECIDER_SMOKE_CHECKPOINT` it also smoke-tests a real checkpoint. In bf16 a tiny model's answers carry rounding noise of their own (a mean |Δp| of 0.003 to 0.009 between its bf16 and float32 answers), while the wrong mask moves them by a mean of 0.15 to 0.36, so the tier gates on 95% top-1 and a mean |Δp| of at most 0.015. A cached, repeated or batched answer runs other attention kernels than a cold one, which on a tiny model in bf16 moved it by at most 0.0149 over four runs (0.007 on the released E2B), and two servers by at most 0.0273; the tolerances are 0.02 and 0.035. On one H100 the multimodal and unified checkpoints agreed with the reference on 90 of 90 top-1 codes, within a mean |Δp| of 0.0065 and 0.0040, and the tier took GPU_TIER_TIME, most of it starting seven engines (150 to 225 s each). FlashAttention 2, on Ampere and Ada GPUs, is untested: the measurements are on H100s. With one head size (the tiny models') vLLM picks FlashAttention 3 on Hopper and 2 elsewhere, which the plugin runs; the real checkpoints' two head sizes need FlashAttention 4, which only Hopper has.

**Conformance tier.** `tests/gpu/conformance.sh OUT_DIR CKPT_E2B CKPT_E4B CKPT_12B CKPT_26B_A4B CKPT_31B` serves every size on one 8-GPU node (the 31B on two GPUs) and checks each against its own `golden_vectors.jsonl`, by section 10's fields and criteria: the rendering exactly (variant, state ids, image hashes and spans, each question's ids and codes), then the core's `agreement` against the reference, at least 98% top-1 and a mean |Δp| of at most 0.02 in bf16 on at least 1,000 questions (`Agreement.conforms`). It is meant for nightly and release runs. `tests/gpu/conformance.py` exits with STUB (code 3) on a checkpoint without goldens or with lines it cannot read, so the tier reports STUB, never PASS, until the released checkpoints publish their goldens. The GPU and CPU tiers run the same check on tiny goldens, with the minimum lowered to their 90 questions.

**Node checks.** `tests/gpu/run.sh MODEL_DIR ROWS OUT_DIR` checks a server on one node with 6 to 8 GPUs, with vLLM and the plugin installed. `ROWS` is a JSON Lines file of rendered requests (`token_ids`, `prefix_end`, `label_token_ids`, and optionally a reference's `probabilities`) of about 1K tokens each. The script starts one loopback-bound server per check:

- every malformed request gets HTTP 400, the server stays healthy, and the next request is answered as before;
- cached and mixed-batch answers match cold ones;
- a server without prefix caching, and one serving the checkpoint as a Hub repo id from a local cache, give the same answers;
- concurrent requests succeed on a small KV cache;
- requests that are preempted and resumed mid-generation, with the `max_tokens` check bypassed, do not stop the engine;
- each refused setting stops the server at startup with the plugin's message;
- with `MM_MODEL` (the checkpoint in the multimodal layout) and `MM_TEXT_MODEL` (one with `inputs: ["text"]`), a second phase: `strands-decider-vllm serve` with text, one, two and four images, the refusals, and the image-content cache test; the text-only server's refusal of images; and `vllm serve` on the multimodal layout answering `MODEL_DIR`'s text rows alike.

On one 8-GPU H100 node, with E2B as the decoder checkpoint and in both multimodal checkpoints, `run.sh` passed every check in 20.5 minutes.

`tests/gpu/reference.py` is also the image parity runs' reference (see Parity). `tests/gpu/dump_pixels/` is a test-only hook that records the pixel values vLLM's encoder receives, to compare them with the reference's.

## Parity

### Text

The reference is the specification's reference implementation, run offline in PyTorch: bf16, SDPA attention, one request per forward pass, prompts up to 8,192 tokens. The parity set has 1,200 prompts per model: 300 each of about 256, 1K and 4K tokens, and 300 real states of 6,144 to 8,000 tokens. Every server answered all 1,200 at client concurrency 8 with the flags above, on vLLM 0.33.0 and one H100 per server unless TP=2. Top-1 agreement compares the argmax over the question's options. |Δp| is the largest absolute probability difference over a question's options, averaged or maximized over questions. Section 10's bounds for bf16 are at least 98% top-1 and a mean |Δp| of at most 0.02; every size meets both.

| Model | Top-1 agreement | Mean \|Δp\| | Max \|Δp\| | Top-1 by prompt length: 256 / 1K / 4K / 6K to 8K |
|---|---|---|---|---|
| E2B | 99.33% | 0.0049 | 0.069 | 98.67% / 100.00% / 99.67% / 99.00% |
| E4B | 99.67% | 0.0035 | 0.049 | 99.67% / 99.67% / 99.67% / 99.67% |
| 12B | 98.67% | 0.0081 | 0.120 | 99.00% / 99.00% / 98.67% / 98.00% |
| 26B-A4B | 98.42% | 0.0119 | 0.195 | 97.67% / 97.00% / 100.00% / 99.00% |
| 31B, one GPU | 99.67% | 0.0048 | 0.088 | 99.00% / 99.67% / 100.00% / 100.00% |
| 31B, TP=2 | 99.50% | 0.0050 | 0.106 | 99.00% / 99.33% / 100.00% / 99.67% |

The table was measured with version 0.2.1 on vLLM 0.33.0, on the intermediate decoder checkpoints. Version 0.3.0 was run again on the same 1,200 prompts at E2B and the 31B (one GPU): every answer is 0.2.1's within run-to-run noise, with no top-1 change and a largest difference of 0.0085 (E2B) and 0.0011 (31B) on any question. In the multimodal layout, E2B served with `vllm serve` and token ids agrees with the reference on 99.50% of the prompts (mean |Δp| 0.0052, max 0.066). Its answers differ from the decoder layout's by a mean of 0.0041 and at most 0.049, with two top-1 changes, both at reference near-ties. That is more than the run-to-run noise between two starts of one layout (at most 0.023). The decoder weights are the same, but vLLM's multimodal class computes the token and per-layer embeddings outside the compiled decoder, so its bf16 rounding differs.

Most disagreements are near-ties in the reference: every one has a reference top-2 gap of at most 0.025 at E2B, 0.056 at 12B and 0.062 at 31B. 26B-A4B is the weakest size, at 97.0% on its 1K bucket and 97.7% on its 256-token bucket, and 16 of its 19 disagreements have gaps of at most 0.1. That is bf16 mixture-of-experts noise, not the mask: with version 0.1.0 the same server gave the same agreement at concurrency 1. The mask cannot be skipped. With Gemma 4's own causal, windowed attention and the same readout, agreement is 57.5% (E2B) to 91.3% (31B). With prefix caching on, a further question on a cached state and an exact repeat matched a cold run on every top label at every size, within a mean |Δp| of 0.0040.

### Images

No checkpoint has been trained on images yet. The test checkpoints are the Gemma 4 bases (`-it`, at the revisions their provenance records) with their language model replaced by the intermediate decoders, their vision encoders unchanged, so their answers about images are not meaningful as decisions; agreement with a reference is. The reference is `tests/gpu/reference.py`, transformers' own `Gemma4ForConditionalGeneration` (`Gemma4UnifiedForConditionalGeneration` for the 12B) in bf16 with SDPA, written from the specification: the prefix_bidir mask on every layer as a dict keyed by layer type, the images prepared as section 11.1 says (Pillow's bicubic resize, then `Gemma4ImageProcessorPil` or the 12B's processor with its resize off), and the readout in bf16 with the float32 cap. On 60 stored text rows of the parity set (15 per length) it matches the stored reference at 100% top-1 at E2B, E4B and 31B, and 98.3% at the 12B and 26B-A4B.

The image set has 300 requests of 1 to 4 real photographs each (125, 83, 47 and 45 requests), 900 questions: 160 COCO val2017 photos and 40 crops of them resized to between 64 × 48 and 4,000 × 1,000 pixels, from 1:4 to 4:1, in JPEG, PNG, WebP and GIF (N from 252 to 280), with a choice of 2 to 8 options, a noul and a score of 2 to 4 levels, and short, empty or long (up to about 4,500 tokens) state text. Every server answered all 300 at concurrency 8 (`strands-decider-vllm serve`, one H100, 8,192-token context).

| Model | Top-1 agreement | Mean \|Δp\| | Max \|Δp\| | Top-1 by images: 1 / 2 / 3 / 4 | Within section 10's bf16 bounds |
|---|---|---|---|---|---|
| E2B | 99.33% | 0.0049 | 0.061 | 98.67% / 100.00% / 100.00% / 99.26% | yes |
| E4B | 98.33% | 0.0048 | 0.062 | 98.67% / 97.99% / 99.29% / 97.04% | yes |
| 12B | 98.00% | 0.0138 | 0.170 | 97.60% / 97.19% / 99.29% / 99.26% | yes, at the top-1 bound |
| 26B-A4B | 96.11% | 0.0346 | 0.527 | 96.27% / 97.99% / 95.74% / 92.59% | no |
| 31B | 97.00% | 0.0243 | 0.669 | 98.13% / 96.79% / 95.04% / 96.30% | no |

**Image serving ships at all five sizes, but the image conformance bound is not set yet.** Section 10's bounds (98% top-1 and a mean |Δp| of at most 0.02 in bf16) will be checked for images on image-trained checkpoints and their image goldens; none exists yet. Against this image-untrained reference, 26B-A4B and the 31B currently fall short on both counts, and E2B, E4B and the 12B meet them, the 12B with no margin. The tiers' tolerances were not widened for this.

The measurements point to bf16 rounding, not the plugin. The same reference in float32 (`--dtype float32`) agrees with the bf16 reference on only 95.9% of the questions at 26B-A4B and 97.0% at the 31B. vLLM agrees with the float32 answers on 96.8% and 97.0%, as closely as the bf16 reference does or more so (at E2B, 98.9% for the bf16 reference and 99.3% for vLLM). vLLM also differs from itself when only the batching changes: 26B-A4B's answers at client concurrency 1 and 8 agree on 98.3% (mean |Δp| 0.014, max 0.36). Where the reference is confident (top probability above 0.9), vLLM gives the same answer on all 588 such questions at 26B-A4B and 509 of 510 at the 31B, and every disagreement at E2B, E4B and the 12B is a near-tie in the reference (top-2 gap at most 0.022, 0.048 and 0.14). These checkpoints were never trained on images, and their answers about images are this sensitive to bf16 rounding.

The pixels vLLM's encoder receives are the reference's, bit for bit: at every size the bf16 pixel values of all 612 images of the set, as the reference's processor prepares them, are among those vLLM encodes (the test hook `tests/gpu/dump_pixels`). Both sides resize with the core's `resize_image`, and on Pillow 12.2.0 and 13.0.0 alike the 612 images' pixels are the same, at E2B's processor and the 12B's. A single `resize((W, H))` call on Pillow 13 would change 12 of them (the 600 × 2,400 crops) by up to 2 levels. vLLM casts the pixels to bf16 before the encoder, as the reference does. Resizing with transformers' torchvision backend instead of Pillow, which is what stock vLLM would do, moves pixels by up to 2/255; measured on an earlier run of this set, the reference's own answers then move by about as much as vLLM's differ from it (top-1 99.22%, 97.78%, 95.78% and 97.33% at E2B, the 12B, 26B-A4B and the 31B), more than the bound allows at the 12B.

The cache keys hold on real photos too (`tests/gpu/server_client.py images`, at E2B): after a request with one photo, a request with a different photo of the same size and the same text reused no cached token, while a further question on the first photo reused its state's full blocks.

### Performance

E2B on one H100, `strands-decider-vllm serve` with prefix caching, one noul question per request, kernels warmed, median latency at concurrency 1 and throughput at concurrency 1 and 16:

| Request | p50, concurrency 1 | Requests/s, 1 | Requests/s, 16 |
|---|---|---|---|
| text, about 330 tokens, new each time | 27 ms | 36.0 | 199 |
| one new 640 × 480 photo (266 tokens) and a short text | 88 ms | 11.3 | 43 |
| a photo the encoder cache holds, with a new text | 41 ms | 24.5 | 143 |
| a photo and text already cached, a new question | 30 ms | 33.4 | 152 |

The vision encoder costs about 47 ms per photo, and the photo's 266 tokens through the decoder about 11 ms. vLLM's encoder cache keeps an image's encoder output while the cache has room, so a photo sent again skips the encoder even when its state's text, and so its KV, is new; the prefix cache then also skips the state. Without prefix caching the last row costs what the one above it does (40 ms). At concurrency 16 cached image requests reach about 150 per second: the server decodes and resizes images on its `--preprocess-workers` threads, off the event loop, which doubled this figure from the 72 per second of a server that did it on the loop. Requests with a new photo reach about 43 per second; what bounds them (the encoder, or vLLM's one renderer thread that prepares each image) was not measured.
