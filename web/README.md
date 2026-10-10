# Strands Decider in the browser

The newest Strands Decider release, running in the browser on the GPU through WebGPU with
[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/): current Chrome or Edge, no flags, no
inference server (Safari and Firefox have WebGPU too but are untested). The int4 weights (1.1 GB) download once, stay in the browser's private storage (OPFS), and every
decision runs locally: about 90 ms for a short question on an Apple M4 Pro, 0.9 s to load from the cache.

Nothing here needs attention when a release is published: `.github/workflows/web.yml` notices the new
`StrandsAgents/strands-decider-2B-*` repo, converts it, checks it against this package, uploads it and redeploys
the site (see [Publishing](#publishing)).

## Layout

| Path | What |
|---|---|
| `convert/convert.py` | release → `manifest.json`, `model/` (tokenizer, config, pointer head), `onnx/` (graph, weight shards) |
| `convert/check.py` | the converted graph on ONNX Runtime (CPU) against `strands_decider` (PyTorch); writes `reference.json` |
| `convert/publish.py` | picks the release to serve; uploads converted weights to the Hub and deletes what nothing serves |
| `site/` | the page; `js/worker.js` (session, cache), `js/prompt.js` and `js/head.js` (ports of `prompting.py` and the head) |
| `build.mjs` | `site/` plus ONNX Runtime Web and the tokenizer library → a static site that points at one converted release |
| `test/` | `questions.json` (the shared test set), `prompt.test.mjs` (Node), `site.spec.mjs` (Playwright) |

## Using it from a page

```js
import { Decider } from "./js/decider-client.js";
const decider = new Decider({ weights: "https://huggingface.co/<repo>/resolve/main/<folder>/" });
await decider.ready;  // rejects with err.unsupported (and downloads nothing) without WebGPU and shader-f16
const { answer } = await decider.decide("Help! My payouts have been failing for 3 days!",
  { type: "choice", instructions: "Which team should handle this?", criteria: { billing: null, sales: null, retail: null } });
const { answers } = await decider.decideMany(state, { team: {...}, urgent: { type: "noul", instructions: "Is this urgent?" } });
```

Questions and answers have the shapes of `strands_decider/schema.py` (`ChoiceQuestion`, `ChoiceAnswer`, ...), so a
`strands-decider serve` request's `questions` work unchanged. `options: [names]` is accepted as a short form of
`criteria`. `decideMany` encodes the state once for all its questions. One difference from Python: JavaScript
objects list integer-like keys first, so choice criteria named `"2024"`, `"10"` and so on come out in another
order (and so give other answers); use the `options` array for those.

## Run locally

```bash
python -m venv web/.venv && source web/.venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e . -r web/convert/requirements.txt pytest
python web/convert/convert.py StrandsAgents/strands-decider-2B-hobson-v21 --out dist/weights   # ~10 GB RAM
python web/convert/check.py dist/weights                                                         # ~3 min on CPU
pytest web/convert
cd web && npm ci && npx playwright install chromium firefox webkit
node server.mjs --weights ../dist/weights                        # http://127.0.0.1:8787/
WEIGHTS=$PWD/../dist/weights npm test         # Node tests, then the site on WASM in Chromium, Firefox and WebKit
WEIGHTS=$PWD/../dist/weights CHROME=1 DEVICE=webgpu npx playwright test --project chromium   # WebGPU, installed Chrome
```

`?device=wasm` runs on the CPU instead (several seconds per decision); `?nocache=1` skips the cache.

## How it fits together

- **Conversion.** `convert.py` folds the LoRA adapter into the base model the release pins, with PEFT; exports the
  text torso with the ONNX Runtime GenAI model builder (int4 weights, fp16 activations, no LM head, output
  `hidden_states`); splits the weights into shards of at most 512 MB (a browser worker cannot allocate one 1 GB
  buffer); and rewrites the int4 embedding lookup into standard ops, so the CPU (WASM) build runs the same graph.
  torch, transformers and peft are whatever this package installs; the ONNX toolchain is pinned in
  `convert/requirements.txt`. Conversion is deterministic on one platform (torch 2.7 and 2.14 give byte-identical
  weights); Linux x86 and macOS arm64 differ in 0.04% of bytes (rounding), with the same graph.
- **Check.** `check.py` runs `test/questions.json` (47 questions: the official examples, a 27-item benchmark,
  structured states and options, criteria, truncated states and questions, multi-question requests) through both,
  and fails on a different answer where the reference is not a near-tie (top-two margin under 0.05) or a mean
  error above 0.05. hobson-v21: mean error 0.026 (int4 quantization), and the one changed answer is a near-tie. The token ids, option positions and
  answers it records are what `prompt.test.mjs` holds the JavaScript port to, exactly.
- **Browser.** The worker checks for WebGPU with `shader-f16` before downloading anything. It reads `manifest.json`
  (sizes and sha256 of every file, the shapes of the empty recurrent and KV state), downloads what the cache lacks,
  verifies each download's hash, and stores files under their hash in its own OPFS directory. After the model
  loads, cached files the manifest does not list are deleted: a new release replaces the old one on disk, and a
  file that did not change (a re-conversion is byte-identical) is not downloaded again.

## Publishing

`web.yml` runs on every push to `main`, daily, and on demand (`gh workflow run web.yml`; a release can be named).
A Hub release creates no GitHub event, so the push and the schedule are what notice it. Jobs:

1. **resolve**: the release to serve: the newest `strands-decider-2B-*` repo tagged `strands-decider` that
   convert.py can handle (complete upload, pointer head, LoRA, Qwen3.5 base; others are skipped with the reason
   logged), or the `DECIDER_RELEASE` variable, or the dispatch input. Its folder is
   `<release>/<inputs id>/<converter id>/`: a hash of the release files convert.py reads (so a model-card edit
   changes nothing) and a hash of `convert.py` and `requirements.txt`. When the live site serves that folder already
   and the push changed nothing under `web/` or `src/strands_decider/`, the run stops here (seconds).
2. **convert**, only when the folder does not exist yet (a new release, or a changed converter): `convert.py` and
   `check.py` on a standard runner, about 20 minutes.
3. **test**: refreshes `reference.json` from the checked-out `strands_decider` (`check.py --prompts-only`, which
   needs no model), then `npm test` against the new conversion or the published one. CI has no GPU (and its software WebGPU,
   SwiftShader, lacks fp16), so the browser test runs on WASM in Chromium, Firefox and WebKit: the same graph and
   JavaScript with other kernels. WebGPU is tested by hand (the command above).
4. **publish**, from `main` only: uploads a new conversion, builds and deploys the site, then deletes every folder
   but the new one and the one served until now, and squashes the weights repo's history to free the storage.

Pull requests that touch `web/` or `src/strands_decider/` run resolve, convert (if needed) and test. A change to
`prompting.py` or the engine's tokenization that the JavaScript port does not follow fails `prompt.test.mjs` there.

A release `convert.py` cannot handle (a base model other than Qwen3.5, a non-pointer head) fails the convert job,
and the site keeps serving the last good release.

One-time setup is in [MAINTAINING.md](../MAINTAINING.md#the-browser-demo).

## Versions

Nothing here needs regular updates:

- **Actions** are pinned by commit, like every workflow in this repo, and Dependabot's github-actions updates
  cover `web.yml` with the others.
- **torch, transformers, peft, huggingface_hub** are not pinned: the jobs install what `pip install -e .` resolves.
- **The ONNX toolchain** (`convert/requirements.txt`) and **ONNX Runtime Web** (`package.json`) are pinned exactly,
  because they decide the graph and must agree with each other. Nothing watches them, so they produce no update
  PRs; change them only on purpose (a change to `requirements.txt` converts the release again, and the tests check
  the result in three browsers).
- **Node** is the current LTS and **Python** 3.12, as in the other workflows.

## Credits and license

The converted weights keep the release's license (Apache-2.0; base model Qwen/Qwen3.5-2B-Base, Apache-2.0).
ONNX Runtime Web (MIT) and @huggingface/tokenizers (Apache-2.0) are copied into the built site.
