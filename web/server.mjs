// Local demo straight from site/: node server.mjs --weights <convert.py output directory>  ->  http://127.0.0.1:8787/
import { startStatic } from "./test/static.mjs";

const i = process.argv.indexOf("--weights"), weights = i > 0 ? process.argv[i + 1] : process.env.WEIGHTS;
if (!weights) throw new Error("--weights <convert.py output directory> is required");
const port = Number(process.env.PORT || 8787), here = import.meta.dirname;
await startStatic({
  port, isolate: true,
  routes: [["/ort/", `${here}/node_modules/onnxruntime-web/dist`], ["/tokenizers/", `${here}/node_modules/@huggingface/tokenizers/dist`], ["/weights/", weights], ["/", `${here}/site`]],
});
console.log(`http://127.0.0.1:${port}/  weights from ${weights}`);
