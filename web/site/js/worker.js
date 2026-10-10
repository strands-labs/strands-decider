// Inference worker: owns the ONNX Runtime Web session, the tokenizer and the pointer head, and answers
// {state, question} requests. Query parameters (set by decider-client.js):
//   weights=<url>  directory holding manifest.json, model/ and onnx/ (convert.py output), on any origin with CORS
//   device=wasm    run on the CPU (WASM) instead of WebGPU: several seconds per decision, used by the tests in CI
//   nocache=1      do not read or write the on-disk cache
import { Tokenizer } from "../tokenizers/tokenizers.min.mjs";
import { buildInputs, readAnswer } from "./prompt.js";
import { loadHead, pointerLogits } from "./head.js";

const QS = new URL(self.location).searchParams;
const WEIGHTS = new URL(QS.get("weights") || "../weights/", self.location);
const ORT_DIR = new URL("../ort/", self.location).href;
// OPFS directory, files named by their sha256. Only this directory is touched: every GitHub Pages site of an org
// shares one origin, and so one OPFS.
const CACHE_DIR = "strands-decider-weights";
let ort, session, tokenizer, head, cfg, manifest, runtime, adapter;
const post = (type, data) => self.postMessage({ type, ...data });

async function cacheDir() {
  if (QS.get("nocache")) return null;
  try {
    return await (await self.navigator.storage.getDirectory()).getDirectoryHandle(CACHE_DIR, { create: true });
  } catch { return null; }
}

const cachedSize = (dir, name) => dir.getFileHandle(name).then((h) => h.getFile()).then((f) => f.size, () => -1);

/** Delete cached files the current manifest does not list (an older release or converter, a partial download). */
async function prune(dir) {
  const keep = new Set(Object.values(manifest.files).map((f) => f.sha256));
  for await (const name of dir.keys()) if (!keep.has(name)) await dir.removeEntry(name).catch(() => {});
}

const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");

/** A file listed in the manifest, from the cache or downloaded (then cached). Checked against its sha256. */
async function load(dir, path, label = path) {
  const { sha256, bytes } = manifest.files[path];
  if (dir) {
    try {
      const file = await (await dir.getFileHandle(sha256)).getFile();
      if (file.size === bytes) {
        post("progress", { label, got: bytes, total: bytes, cached: true });
        return { buf: new Uint8Array(await file.arrayBuffer()), hit: true };
      }
    } catch {}
  }
  const res = await fetch(new URL(path, WEIGHTS));
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const buf = new Uint8Array(bytes); let got = 0, last = 0;
  const reader = res.body.getReader();
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    if (got + value.length > bytes) throw new Error(`${path}: more than the ${bytes} bytes in the manifest`);
    buf.set(value, got); got += value.length;
    if (got - last > 32 << 20) { last = got; post("progress", { label, got, total: bytes }); }
  }
  if (got !== bytes || hex(await crypto.subtle.digest("SHA-256", buf)) !== sha256) throw new Error(`${path}: download does not match the manifest`);
  post("progress", { label, got, total: bytes, cached: false });
  if (dir) {
    try {
      const tmp = await dir.getFileHandle(`${sha256}.tmp`, { create: true });
      const w = await tmp.createWritable(); await w.write(buf); await w.close();
      await tmp.move(dir, sha256);  // a file under its final name is complete. move(name) alone fails in WebKit
    } catch (e) { post("status", { msg: `not cached (${e.name}): ${label}` }); }
  }
  return { buf, hit: false };
}

/** Why this browser cannot run the model on WebGPU, before anything is downloaded; null if it can. */
async function webgpuProblem() {
  if (!self.navigator.gpu) return "this browser has no WebGPU";
  adapter = await self.navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return "WebGPU found no usable GPU";
  // The graph computes in fp16; ONNX Runtime fails at the first run on an adapter without it.
  if (!adapter.features.has("shader-f16")) return "this GPU has no fp16 shader support (shader-f16)";
  return null;
}

async function init({ device }) {
  const t0 = performance.now(), timings = {};
  runtime = device === "wasm" ? "wasm" : "webgpu";
  if (runtime === "webgpu") {
    const problem = await webgpuProblem();
    if (problem) { post("unsupported", { reason: problem }); return; }
  }
  manifest = await fetch(new URL("manifest.json", WEIGHTS)).then((r) => { if (!r.ok) throw new Error(`manifest.json: HTTP ${r.status}`); return r.json(); });
  post("status", { msg: `runtime ${runtime}`, runtime });

  let dir = await cacheDir();
  if (dir) {
    // Files of an older release or converter go first: they would otherwise use the space the new ones need.
    await prune(dir);
    // Cache only when the missing files fit; otherwise every visit downloads them again, as without OPFS.
    let missing = 0;
    for (const f of Object.values(manifest.files)) missing += (await cachedSize(dir, f.sha256)) === f.bytes ? 0 : f.bytes;
    const { quota = 0, usage = 0 } = await self.navigator.storage.estimate();
    if (quota - usage < missing) { post("status", { msg: "not enough storage to cache the model" }); dir = null; }
  }
  const files = Object.keys(manifest.files);
  const results = await Promise.all(files.map((f) => load(dir, f, f.split("/").pop())));
  const got = Object.fromEntries(files.map((f, i) => [f, results[i]]));
  const text = (f) => JSON.parse(new TextDecoder().decode(got[f].buf));
  timings.fetch_ms = performance.now() - t0;

  tokenizer = new Tokenizer(text("model/tokenizer.json"), text("model/tokenizer_config.json"));
  cfg = text("model/decider_config.json");
  head = loadHead(got["model/head.safetensors"].buf.buffer);

  let t = performance.now();
  ort = await import(`${ORT_DIR}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_DIR;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, self.navigator.hardwareConcurrency || 4) : 1;
  ort.env.logLevel = "warning";
  ort.env.webgpu.powerPreference = "high-performance";  // the discrete GPU on a laptop that has two
  session = await ort.InferenceSession.create(got[manifest.graph].buf, {
    executionProviders: [runtime],
    externalData: manifest.shards.map((s) => ({ path: s.split("/").pop(), data: got[s].buf })),
    graphOptimizationLevel: "all",
    logSeverityLevel: 3,  // errors only: ORT warns on every load that it places shape ops on the CPU, which is expected
    // Recurrent and KV state stay on the GPU, so a shared state prefix can feed later question suffixes.
    preferredOutputLocation: runtime === "webgpu" ? "gpu-buffer" : "cpu",
  });
  timings.session_ms = performance.now() - t;
  // One short forward compiles the GPU kernels now, so the first decision is not slow and a GPU problem shows at load.
  t = performance.now(); release((await forward(tokenizer.encode("<state>\n", { add_special_tokens: false }).ids)).cache);
  timings.warmup_ms = performance.now() - t;
  timings.total_load_ms = performance.now() - t0;
  const gpu = adapter && [adapter.info.vendor, adapter.info.architecture].filter(Boolean).join(" ");
  post("ready", { runtime, timings, cached: results.every((r) => r.hit), gpu, source: manifest.source });
}

/** Empty recurrent, conv and KV-cache inputs for a forward with no prefix (shapes from the manifest). */
function zeros(name) {
  const { dtype, shape } = manifest.state_inputs[name], n = shape.reduce((a, b) => a * b, 1);
  return new ort.Tensor(dtype, dtype === "float16" ? new Uint16Array(n) : new Float32Array(n), shape);
}

const pastName = (o) => (o.endsWith(".key") || o.endsWith(".value") ? o.replace("present.", "past_key_values.") : o.replace("present.", "past."));
const i64 = (arr, dims) => new ort.Tensor("int64", BigInt64Array.from(arr, BigInt), dims);
const release = (cache) => Object.values(cache || {}).forEach((t) => t.dispose?.());

/** One forward over `ids`, continuing from `past` (a prefix cache) or from empty state. */
async function forward(ids, past = null, start = 0) {
  const L = ids.length, feeds = {};
  for (const name of session.inputNames) {
    if (name === "input_ids") feeds[name] = i64(ids, [1, L]);
    else if (name === "attention_mask") feeds[name] = new ort.Tensor("int64", new BigInt64Array(start + L).fill(1n), [1, start + L]);
    // Qwen3.5's M-RoPE takes [3, batch, seq] positions; for text all three rows are the same.
    else if (name === "position_ids") feeds[name] = i64(Array.from({ length: 3 * L }, (_, i) => start + (i % L)), [3, 1, L]);
    else feeds[name] = past ? past[name] : zeros(name);
  }
  const out = await session.run(feeds);
  const hs = out.hidden_states;
  const hidden = hs.location === "cpu" ? hs.data : await hs.getData(true);
  const cache = {};
  for (const [k, v] of Object.entries(out)) if (k !== "hidden_states") cache[pastName(k)] = v;
  return { hidden, d: hs.dims[2], cache };
}

function answer(rq, hidden, d, answerPos, optIdx) {
  const temp = cfg.temperature_by_kind[rq.kind] ?? cfg.temperature;
  return readAnswer(rq, pointerLogits(head, hidden, d, answerPos, optIdx, temp).probs, cfg.ordinal_smoothing);
}

async function decide({ id, state, question }) {
  const t0 = performance.now();
  const { stateIds, questions: [q] } = buildInputs(tokenizer, state, [question], cfg.max_length);
  const ids = [...stateIds, ...q.ids], t1 = performance.now();
  const { hidden, d, cache } = await forward(ids);
  release(cache);
  const t2 = performance.now();
  post("result", { id, answer: answer(q.rq, hidden, d, ids.length - 1, q.optIdx), tokens: ids.length, timings: { prep_ms: t1 - t0, forward_ms: t2 - t1, total_ms: performance.now() - t0 }, runtime });
}

/** Several named questions about one state: the state is encoded once, then each question from its cached state. */
async function decideMany({ id, state, questions }) {
  const t0 = performance.now(), names = Object.keys(questions);
  if (!names.length) throw new Error("decideMany needs at least one question");
  const { stateIds, questions: built } = buildInputs(tokenizer, state, Object.values(questions), cfg.max_length);
  const sLen = stateIds.length, prefix = await forward(stateIds);
  const t1 = performance.now(), answers = {}, per = {};
  try {
    for (const [i, q] of built.entries()) {
      const ts = performance.now();
      const { hidden, d, cache } = await forward(q.ids, prefix.cache, sLen);
      release(cache);
      answers[names[i]] = answer(q.rq, hidden, d, q.ids.length - 1, q.optIdx.map((j) => j - sLen));
      per[names[i]] = performance.now() - ts;
    }
  } finally { release(prefix.cache); }  // GPU buffers on WebGPU
  post("result", { id, answers, tokens: sLen + built.reduce((n, q) => n + q.ids.length, 0), timings: { prefix_ms: t1 - t0, per_question_ms: per, total_ms: performance.now() - t0 }, runtime });
}

// One request at a time: ORT Web hangs on overlapping session.run calls.
let queue = Promise.resolve();
self.onmessage = ({ data }) => { queue = queue.then(() => handle(data)); };
async function handle(data) {
  try {
    if (data.type === "init") await init(data);
    else if (data.type === "decide") await decide(data);
    else if (data.type === "decideMany") await decideMany(data);
  } catch (e) { post("error", { id: data.id, error: String(e?.stack || e) }); }
}
