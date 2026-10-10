// Plain static file server for the tests and the local server. routes: [[url prefix, directory], ...], longest first.
// cors: send Access-Control-Allow-Origin (as Hugging Face does). isolate: COOP/COEP, which lets ONNX Runtime use threads.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".wasm": "application/wasm" };

export function startStatic({ routes, port, cors = false, isolate = false }) {
  const server = http.createServer((req, res) => {
    let url;
    try { url = decodeURIComponent(req.url.split("?")[0]); } catch { res.writeHead(400).end(); return; }
    const route = routes.find(([prefix]) => url.startsWith(prefix));
    if (!route) { res.writeHead(404).end(); return; }
    const [prefix, root] = route.map((p, i) => (i ? path.resolve(p) : p));
    let file = path.join(root, url.slice(prefix.length));
    if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, "index.html");
    if (!file.startsWith(root) || !fs.existsSync(file)) { res.writeHead(404).end(); return; }
    const headers = { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream", "Content-Length": fs.statSync(file).size, "Cache-Control": "no-cache" };
    if (cors) headers["Access-Control-Allow-Origin"] = "*";
    if (isolate) Object.assign(headers, { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" });
    res.writeHead(200, headers);
    if (req.method === "HEAD") res.end(); else fs.createReadStream(file).pipe(res);
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve(server)));
}
