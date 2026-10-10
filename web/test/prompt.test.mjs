// prompt.js (the browser's port of prompting.py and the engine's tokenization and answer readout) against the
// Python engine's token ids, option positions and answers, which check.py recorded in reference.json.
// Needs WEIGHTS=<convert.py output directory, after check.py>.
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { Tokenizer } from "@huggingface/tokenizers";
import { buildInputs, readAnswer } from "../site/js/prompt.js";

const W = process.env.WEIGHTS, skip = !W && "set WEIGHTS";
const read = (f) => JSON.parse(fs.readFileSync(f, "utf8"));
function* cases() {
  const tok = new Tokenizer(read(path.join(W, "model/tokenizer.json")), read(path.join(W, "model/tokenizer_config.json")));
  const cfg = read(path.join(W, "model/decider_config.json")), questions = read(new URL("questions.json", import.meta.url));
  for (const ref of read(path.join(W, "reference.json")).results) {
    const item = questions[ref.set].find((x) => x.name === ref.name);
    const built = buildInputs(tok, item.state, Object.values(item.questions), cfg.max_length);
    yield { ref, built, cfg, where: `${ref.set}/${ref.name}`, keys: Object.keys(item.questions) };
  }
}

test("token ids and option positions match strands_decider", { skip }, () => {
  for (const { ref, built, where, keys } of cases()) {
    assert.deepEqual(built.stateIds, ref.state_ids, `${where}: state token ids`);
    keys.forEach((k, i) => {
      assert.deepEqual(built.questions[i].ids, ref.questions[k].ids, `${where}/${k}: question token ids`);
      assert.deepEqual(built.questions[i].optIdx, ref.questions[k].opt_idx, `${where}/${k}: option positions`);
    });
  }
});

test("answers have the shape and values strands_decider gives for the same probabilities", { skip }, () => {
  const close = (a, b, where) => {
    if (typeof a === "number") return assert.ok(Math.abs(a - b) < 2e-3, `${where}: ${a} vs ${b}`);
    if (a && typeof a === "object") {
      assert.deepEqual(Object.keys(a), Object.keys(b), `${where}: keys`);
      for (const k of Object.keys(a)) close(a[k], b[k], `${where}.${k}`);
    } else assert.equal(a, b, where);
  };
  for (const { ref, built, cfg, where, keys } of cases()) {
    keys.forEach((k, i) => {
      if (!ref.questions[k].probabilities) return;  // added since the last full check: ids are compared, answers not yet
      close(readAnswer(built.questions[i].rq, ref.questions[k].probabilities, cfg.ordinal_smoothing), ref.questions[k].answer, `${where}/${k}`);
    });
  }
});
