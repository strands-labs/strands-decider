// Static build of the demo (about 26 MB, mostly ONNX Runtime's WASM): the page, its scripts, ONNX Runtime Web and the
// tokenizer library. The model is not part of it; --weights is the URL of a convert.py output directory, e.g.
// https://huggingface.co/<weights repo>/resolve/main/<folder>/ as publish.py resolves it (a folder never changes).
// Usage: node build.mjs --out dist/site --weights <url>
import fs from "node:fs";
import path from "node:path";

const arg = (k) => { const i = process.argv.indexOf(`--${k}`); return i > 0 ? process.argv[i + 1] : null; };
const out = path.resolve(arg("out") || "dist/site"), weights = arg("weights");
if (!weights) throw new Error("--weights <url of the converted release> is required");
const here = import.meta.dirname, modules = path.join(here, "node_modules");

fs.rmSync(out, { recursive: true, force: true });
fs.cpSync(path.join(here, "site"), out, { recursive: true });
const index = path.join(out, "index.html"), tag = '<meta name="decider-weights" content="weights/">';
const html = fs.readFileSync(index, "utf8");
if (!html.includes(tag)) throw new Error("index.html: decider-weights meta tag not found");
fs.writeFileSync(index, html.replace(tag, `<meta name="decider-weights" content="${weights.replace(/"/g, "&quot;")}">`));
// ort.webgpu.min.mjs loads the asyncify build (WebGPU and WASM in one binary); nothing else from dist/ is needed.
for (const f of ["ort.webgpu.min.mjs", "ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"]) {
  fs.mkdirSync(path.join(out, "ort"), { recursive: true });
  fs.copyFileSync(path.join(modules, "onnxruntime-web/dist", f), path.join(out, "ort", f));
}
fs.cpSync(path.join(modules, "@huggingface/tokenizers/dist"), path.join(out, "tokenizers"), { recursive: true });
fs.writeFileSync(path.join(out, ".nojekyll"), "");

const size = fs.readdirSync(out, { recursive: true }).reduce((n, f) => n + (fs.statSync(path.join(out, f)).isFile() ? fs.statSync(path.join(out, f)).size : 0), 0);
console.log(`site ${(size / 2 ** 20).toFixed(1)} MiB in ${out}, weights from ${weights}`);
