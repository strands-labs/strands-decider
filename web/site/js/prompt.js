// Port of strands_decider/prompting.py and the engine's tokenization (infer.py `_fit`, `_option_idx`, `_answer`).
// Questions and answers have the shapes of strands_decider/schema.py, so a request body for `strands-decider serve`
// works here unchanged. test/prompt.test.mjs checks token ids and option positions against the Python engine.
const NOUL_SLOT_LABELS = ["false", "true"];
const NOUL_DEFAULT_CRITERIA = { false: "the statement does not hold for this state", true: "the statement holds for this state" };
const HEADERS = {
  noul: "Decide whether the statement is true of the state.",
  choice: "Select exactly one option.",
  score: "Rate the state against the ordered levels below (lowest first).",
};
const MAX_QUESTION_FRACTION = 0.75;  // EngineConfig.max_question_fraction

/** render_content: strings stripped, structured data as 2-space JSON (Python writes 1.0 where JS writes 1). */
export function renderContent(c) {
  if (c === null || c === undefined) return "";
  return typeof c === "string" ? c.trim() : JSON.stringify(c, null, 2);
}

// The tokenizer NFC-normalizes its input, and the byte offsets in tokenByteOffsets() are offsets into that normalized
// text, so the prompt is normalized first (Python maps offsets back to the raw text instead; the ids are the same).
const nfc = (s) => s.normalize("NFC");

export const renderState = (state) => nfc(`<state>\n${renderContent(state)}\n</state>\n`);

/**
 * A question as schema.py defines it: {type: "noul", instructions, criteria?: {true?, false?}},
 * {type: "choice", instructions, criteria: {name: description | null}}, {type: "score", instructions, criteria: [levels]}.
 * `options: [names]` is accepted for choice (no descriptions) and `options: [levels]` for score.
 */
export function renderQuestion(q) {
  const criteria = q.criteria ?? q.options;
  let pairs;
  if (q.type === "noul") {
    const crit = { ...NOUL_DEFAULT_CRITERIA, ...(criteria || {}) };
    pairs = NOUL_SLOT_LABELS.map((l) => [l, renderContent(crit[l])]);
  } else if (q.type === "choice") {
    pairs = Array.isArray(criteria) ? criteria.map((o) => [o, ""]) : Object.entries(criteria).map(([n, d]) => [n, renderContent(d)]);
    if (pairs.length < 2) throw new Error("choice requires at least 2 options");
  } else if (q.type === "score") {
    pairs = criteria.map((d, i) => [String(i), d]);
    if (pairs.length < 2 || pairs.length > 10) throw new Error("score requires between 2 and 10 levels");
  } else throw new Error(`unknown question type ${q.type}`);
  const prefix = nfc(`<question type="${q.type}">\n${HEADERS[q.type]}\n${renderContent(q.instructions)}\n<options>\n`);
  const lines = [], spans = [];
  let cursor = prefix.length;
  pairs.forEach(([name, desc], i) => {
    desc = (desc || "").split(/\s+/).filter(Boolean).join(" ");
    const line = nfc(`${i + 1}. ${name}` + (desc ? ` — ${desc}` : ""));
    lines.push(line); spans.push([cursor, cursor + line.length]); cursor += line.length + 1;
  });
  const text = prefix + lines.join("\n") + "\n</options>\n</question>\n<answer>";
  return { text, kind: q.type, labels: pairs.map((p) => p[0]), descriptions: pairs.map((p) => (p[1] || "").trim()), spans };
}

const utf8 = new TextEncoder();
const byteLen = (s) => utf8.encode(s).length;

/** Byte-level BPE: every character of a token string stands for one byte of the input. */
function tokenByteOffsets(tokens) {
  const out = []; let pos = 0;
  for (const t of tokens) { out.push([pos, pos + t.length]); pos += t.length; }
  return out;
}

/**
 * Tokenize the state and each question separately, as `_fit` does: the longest question decides how much of the window
 * questions may take (front-truncated past that), and the state gets the rest, so every question shares one state
 * prefix. Returns the state's ids and, per question, its ids, option positions (in the full prompt) and rendering.
 */
export function buildInputs(tokenizer, state, questions, maxLength) {
  const rendered = questions.map(renderQuestion);
  const encoded = rendered.map((rq) => tokenizer.encode(rq.text, { add_special_tokens: false }));
  const longest = Math.max(...encoded.map((e) => e.ids.length));
  const reserve = Math.min(longest, Math.max(1, Math.floor(maxLength * MAX_QUESTION_FRACTION)));
  const stateIds = tokenizer.encode(renderState(state), { add_special_tokens: false }).ids.slice(0, Math.max(1, maxLength - reserve));
  const parts = rendered.map((rq, i) => {
    const cut = Math.max(0, encoded[i].ids.length - reserve);
    const ids = encoded[i].ids.slice(cut), offs = tokenByteOffsets(encoded[i].tokens).slice(cut);
    const optIdx = rq.spans.map(([a, b]) => {
      const A = byteLen(rq.text.slice(0, a)), B = byteLen(rq.text.slice(0, b));
      let last = -1;
      offs.forEach(([lo, hi], j) => { if (hi > lo && lo >= A && hi <= B) last = j; });
      if (last < 0) throw new Error("the prompt was truncated through its option list");
      return stateIds.length + last;
    });
    return { ids, optIdx, rq };
  });
  return { stateIds, questions: parts };
}

const r4 = (x) => Math.round(x * 1e4) / 1e4;
const clamp01 = (x) => Math.max(0, Math.min(1, x));

/** The answer for slot probabilities `probs`, as infer.py builds NoulAnswer / ChoiceAnswer / ScoreAnswer. */
export function readAnswer(rq, probs, ordinalSmoothing = 0) {
  const by = Object.fromEntries(rq.labels.map((l, i) => [l, probs[i]]));
  if (rq.kind === "noul") return { type: "noul", noul: r4(by.true) };
  if (rq.kind === "choice") {
    const n = probs.length, pmax = Math.max(...probs);
    return {
      type: "choice", choice: rq.labels[probs.indexOf(pmax)],
      probabilities: Object.fromEntries(Object.entries(by).map(([k, p]) => [k, r4(p)])),
      confidence: r4(n <= 1 ? 1 : clamp01((n * pmax - 1) / (n - 1))),
    };
  }
  const levels = rq.labels.map(Number).sort((a, b) => a - b).map(String), ordered = levels.map((l) => by[l]);
  return {
    type: "score", score: r4(ordered.reduce((s, p, i) => s + i * p, 0)),
    legend: Object.fromEntries(levels.map((l) => [l, rq.descriptions[rq.labels.indexOf(l)]])),
    probabilities: Object.fromEntries(levels.map((l, i) => [l, r4(ordered[i])])),
    confidence: r4(scoreConfidence(ordered, ordinalSmoothing)),
  };
}

function scoreConfidence(probs, eps) {
  const n = probs.length; if (n <= 1) return 1;
  const total = probs.reduce((a, b) => a + b, 0); if (total <= 0) return 0;
  const p = probs.map((x) => x / total);
  const mean = p.reduce((s, pi, i) => s + i * pi, 0);
  const sigma = Math.sqrt(p.reduce((s, pi, i) => s + pi * (i - mean) ** 2, 0));
  const sMax = (n - 1) / 2, sFloor = eps > 0 ? Math.sqrt(eps) : 0;
  if (sMax <= sFloor) return sigma <= sFloor ? 1 : 0;
  return clamp01((sMax - sigma) / (sMax - sFloor));
}
