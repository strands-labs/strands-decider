// Main-thread handle to the inference worker: one warm session, monotonic request ids.
// `weights` is the directory convert.py wrote (manifest.json, model/, onnx/), on any origin that sends CORS headers.
// `ready` rejects with `err.unsupported` set when the browser cannot run the model on WebGPU; nothing is downloaded then.
export class Decider {
  constructor({ weights, device = "webgpu", onEvent = () => {} } = {}) {
    const url = new URL("./worker.js", import.meta.url);
    for (const [k, v] of new URLSearchParams(location.search)) url.searchParams.set(k, v);  // ?device=wasm, ?nocache=1
    if (weights) url.searchParams.set("weights", new URL(weights, location.href).href);
    // Ask the browser not to evict the ~1 GB cache under storage pressure (granted silently, or not, by Chrome).
    navigator.storage?.persist?.().catch(() => {});
    this.worker = new Worker(url, { type: "module" });
    this.pending = new Map(); this.nextId = 1; this.onEvent = onEvent;
    this.ready = new Promise((resolve, reject) => { this._ready = { resolve, reject }; });
    this.worker.onmessage = ({ data }) => {
      if (data.type === "ready") { this.info = data; this._ready.resolve(data); }
      else if (data.type === "unsupported") this._ready.reject(Object.assign(new Error(data.reason), { unsupported: true }));
      else if (data.type === "result") { this.pending.get(data.id)?.resolve(data); this.pending.delete(data.id); }
      else if (data.type === "error") {
        if (data.id && this.pending.has(data.id)) { this.pending.get(data.id).reject(new Error(data.error)); this.pending.delete(data.id); }
        else this._ready.reject(new Error(data.error));
      }
      this.onEvent(data);
    };
    // A crashed worker answers nothing more: fail the load and every request still waiting.
    this.worker.onerror = (e) => {
      const err = new Error(e.message || "the inference worker crashed");
      this._ready.reject(err); for (const p of this.pending.values()) p.reject(err); this.pending.clear();
    };
    this.worker.postMessage({ type: "init", device: url.searchParams.get("device") || device });
  }
  #request(msg) {
    return this.ready.then(() => new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, { resolve, reject }); this.worker.postMessage({ ...msg, id });
    }));
  }
  /** question: {type: "choice" | "noul" | "score", instructions, options?} */
  decide(state, question) { return this.#request({ type: "decide", state, question }); }
  /** Several named questions about one state; the state is encoded once. */
  decideMany(state, questions) { return this.#request({ type: "decideMany", state, questions }); }
}
