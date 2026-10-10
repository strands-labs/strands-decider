// The demo page: examples, live re-decision while typing, probability bars. All inference is in worker.js.
import { Decider } from "./decider-client.js";
const weights = document.querySelector("meta[name=decider-weights]").content;
const $ = (id) => document.getElementById(id);

const EXAMPLES = [
  { name: "Route a ticket", state: "Help! My payouts have been failing for 3 days!", type: "choice", instructions: "Which team should handle this?", options: "billing, sales, retail" },
  { name: "Urgent?", state: "Help! My payouts have been failing for 3 days!", type: "noul", instructions: "Does this convey urgency?", options: "" },
  { name: "Frustration", state: "Help! My payouts have been failing for 3 days!", type: "score", instructions: "How frustrated is the writer?", options: "calm, frustrated, furious" },
  { name: "Language ID", state: "sihamba ngokushesha", type: "choice", instructions: "What language is this phrase in?", options: "English, Zulu, Dutch" },
  { name: "Agent: next action", type: "choice", instructions: "Which action should the agent take next?",
    state: "Goal: submit a refund request for order #4417.\nPage: Refund form\n  [input] Order number: \"4417\"\n  [select] Reason: (not selected) options: damaged, late, wrong item\n  [textarea] Details: empty\n  [button] Submit (disabled until reason is selected)",
    options: "click Submit, select a reason, type order number, ask the user for details" },
  { name: "Phishing?", state: "URGENT: your account has been suspended. Verify your password at http://secure-login.example-bank.ru now.", type: "choice", instructions: "Is this email legitimate or phishing?", options: "legitimate, phishing" },
  { name: "Sentiment", state: "Absolutely love the new design, checkout is so much faster now!", type: "choice", instructions: "What is the sentiment of this review?", options: "positive, neutral, negative" },
];

const hints = { choice: "(comma separated)", noul: "(none: answers true / false)", score: "(comma separated, low → high)" };
function question() {
  const type = $("qtype").value, options = $("options").value.split(",").map((s) => s.trim()).filter(Boolean);
  return type === "noul" ? { type, instructions: $("instructions").value } : { type, instructions: $("instructions").value, options };
}
function load(ex, chip) {
  $("state").value = ex.state; $("instructions").value = ex.instructions; $("qtype").value = ex.type; $("options").value = ex.options;
  document.querySelectorAll(".chip.on").forEach((c) => c.classList.remove("on")); chip?.classList.add("on");
  syncType(); schedule(0);
}
function syncType() { $("opt-hint").textContent = hints[$("qtype").value]; $("options").disabled = $("qtype").value === "noul"; }
EXAMPLES.forEach((ex) => { const b = document.createElement("button"); b.className = "chip"; b.textContent = ex.name; b.onclick = () => load(ex, b); $("examples").append(b); });

const loaded = {};
const decider = new Decider({ weights, onEvent: (e) => {
  if (e.type === "progress" && e.total) {
    loaded[e.label] = [e.got, e.total];
    const [g, t] = Object.values(loaded).reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]);
    $("loadbar").firstElementChild.style.width = `${(100 * g / t).toFixed(0)}%`;
    $("p-status").innerHTML = `${e.cached ? "reading cache" : "downloading"} <b>${(g / 2 ** 20).toFixed(0)} MiB</b>`;
  }
  if (e.type === "status") console.info(`[decider] ${e.msg}`);
  if (e.type === "status" && e.runtime) $("p-backend").innerHTML = `ONNX Runtime Web <b>${e.runtime === "webgpu" ? "WebGPU" : "WASM (CPU)"}</b>`;
}});
window.decider = decider;

decider.ready.then((r) => {
  $("p-status").className = "pill ok"; $("p-status").innerHTML = "status <b>ready</b>";
  $("p-load").innerHTML = `loaded in <b>${(r.timings.total_load_ms / 1000).toFixed(2)} s</b>${r.cached ? " from cache" : ""}`;
  const src = r.source, name = src.repo.split("/").pop();
  $("f-source").href = `https://huggingface.co/${src.repo}/tree/${src.revision}`; $("f-source").textContent = `${name} @ ${src.revision.slice(0, 7)}`;
  if (r.gpu) $("p-gpu").innerHTML = `GPU <b>${r.gpu}</b>`;
  $("loadbar").style.visibility = "hidden"; document.body.dataset.ready = "1";
  schedule(0);
}, (e) => {
  $("p-status").innerHTML = e.unsupported
    ? `<b>needs WebGPU</b>: ${e.message}. Use a current Chrome or Edge on a computer, or <a href="?device=wasm">run on the CPU</a> (a 1.1 GB download, then several seconds per decision).`
    : `<b>load failed</b>: ${e.message}`;
  $("loadbar").style.visibility = "hidden"; document.body.dataset.ready = e.unsupported ? "unsupported" : "error";
});

// One request in flight at a time; edits made while it runs collapse into a single follow-up decision.
let busy = false, dirty = false, timer = 0, n = 0;
const history_ = [];
function schedule(ms = 40) { clearTimeout(timer); timer = setTimeout(run, ms); }
async function run() {
  if (document.body.dataset.ready !== "1") return;
  if (busy) { dirty = true; return; }
  busy = true; dirty = false; $("out").classList.add("busy");
  try { render(await decider.decide($("state").value, question())); }
  catch (e) { $("answer").innerHTML = `<small>${e.message.split("\n")[0]}</small>`; $("bars").innerHTML = ""; }
  busy = false; $("out").classList.remove("busy");
  if (dirty) run();
}

function render(r) {
  const a = r.answer, probs = a.type === "noul" ? [["false", 1 - a.noul], ["true", a.noul]] : Object.entries(a.probabilities);
  const top = a.type === "choice" ? a.choice : a.type === "noul" ? (a.noul >= 0.5 ? "true" : "false") : probs.reduce((m, p) => (p[1] > m[1] ? p : m))[0];
  const labels = a.type === "score" ? a.legend : null;
  $("answer").innerHTML = a.type === "choice" ? `${a.choice} <small>confidence ${(a.confidence * 100).toFixed(0)}%</small>`
    : a.type === "noul" ? `${a.noul >= 0.5 ? "true" : "false"} <small>P(true) = ${a.noul.toFixed(2)}</small>`
    : `${a.score.toFixed(2)} <small>on 0–${probs.length - 1} · nearest “${labels[Math.round(a.score)] ?? ""}” · confidence ${(a.confidence * 100).toFixed(0)}%</small>`;
  const rows = probs.map(([k, p]) => ({ k, p, name: labels ? `${k} · ${labels[+k]}` : k }));
  const bars = $("bars");
  if (bars.children.length !== rows.length || [...bars.children].some((el, i) => el.dataset.k !== rows[i].k)) {
    bars.innerHTML = rows.map((r) => `<div class="bar" data-k="${r.k}"><span class="name"></span><div class="track"><div class="fill"></div></div><span class="pct"></span></div>`).join("");
  }
  rows.forEach((row, i) => {
    const el = bars.children[i];
    el.classList.toggle("top", row.k === top);
    el.querySelector(".name").textContent = row.name;
    el.querySelector(".fill").style.width = `${(row.p * 100).toFixed(1)}%`;
    el.querySelector(".pct").textContent = `${(row.p * 100).toFixed(1)}%`;
  });
  $("s-fwd").textContent = `${r.timings.forward_ms.toFixed(0)} ms`;
  $("s-tok").textContent = r.tokens;
  $("s-n").textContent = ++n;
  history_.push(r.timings.forward_ms); if (history_.length > 40) history_.shift();
  const mx = Math.max(...history_, 1);
  $("spark").innerHTML = history_.map((ms) => `<div style="height:${Math.max(2, (36 * ms) / mx).toFixed(0)}px" title="${ms.toFixed(0)} ms"></div>`).join("");
  document.body.dataset.inferences = String(n);
}

for (const id of ["state", "instructions", "options"]) $(id).addEventListener("input", () => { if ($("live").checked) schedule(); });
$("qtype").onchange = () => { syncType(); schedule(0); };
$("go").onclick = () => schedule(0);
load(EXAMPLES[0], $("examples").firstElementChild);
