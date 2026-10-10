// Pointer head: LayerNorm -> q (answer token) / k (option tokens) projections -> scaled dot product.
const f16 = (h) => {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  return e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
};

export function loadHead(buf) {
  const n = Number(new DataView(buf).getBigUint64(0, true));
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const t = {};
  for (const [k, v] of Object.entries(meta)) {
    if (k === "__metadata__") continue;
    if (v.dtype !== "F32") throw new Error(`head tensor ${k} is ${v.dtype}`);
    t[k] = { data: new Float32Array(buf.slice(8 + n + v.data_offsets[0], 8 + n + v.data_offsets[1])), shape: v.shape };
  }
  return t;
}

function row(hidden, d, i) {
  const x = new Float32Array(d);
  if (hidden instanceof Uint16Array) for (let j = 0; j < d; j++) x[j] = f16(hidden[i * d + j]);
  else for (let j = 0; j < d; j++) x[j] = hidden[i * d + j];
  return x;
}

function project(H, name, x) {
  const d = x.length, m = x.reduce((a, b) => a + b, 0) / d;
  const v = x.reduce((a, b) => a + (b - m) ** 2, 0) / d, r = 1 / Math.sqrt(v + 1e-5);
  const nw = H["norm.weight"].data, nb = H["norm.bias"].data, W = H[`${name}.weight`], b = H[`${name}.bias`].data;
  const xn = new Float32Array(d); for (let j = 0; j < d; j++) xn[j] = (x[j] - m) * r * nw[j] + nb[j];
  const out = new Float32Array(W.shape[0]);
  for (let o = 0; o < W.shape[0]; o++) { let s = b[o]; const off = o * d; for (let j = 0; j < d; j++) s += W.data[off + j] * xn[j]; out[o] = s; }
  return out;
}

/** hidden: flat [L, d] (Uint16 fp16 bits or Float32); answerPos = L-1. */
export function pointerLogits(H, hidden, d, answerPos, optIdx, temperature) {
  const q = project(H, "q", row(hidden, d, answerPos)), scale = q.length ** -0.5;
  const logits = optIdx.map((i) => { const k = project(H, "k", row(hidden, d, i)); let s = 0; for (let j = 0; j < k.length; j++) s += k[j] * q[j]; return (s * scale) / temperature; });
  const mx = Math.max(...logits), e = logits.map((l) => Math.exp(l - mx)), z = e.reduce((a, b) => a + b, 0);
  return { logits, probs: e.map((x) => x / z) };
}
