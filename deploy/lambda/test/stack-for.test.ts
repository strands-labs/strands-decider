// Which stack a model gets, from what its checkpoint declares on the Hub (README.md, Which stack a
// model gets), with the Hub stood in for. Run: npm test.
import { strict as assert } from "node:assert";
import { test } from "node:test";
import { DeciderFunctionStack } from "../lib/decider-function-stack";
import { DecisionFunctionStack } from "../lib/decision-function-stack";
import { stackFor } from "../lib/stack-for";

const DECISION = { interface: "strands-decider", format_version: 1 };

/** A Hub whose `model` holds `files` at revision `rev`, decision_config.json reading `config`. */
function hub(files: string[], config: unknown = DECISION, rev = "main"): typeof fetch {
  return (async (url: string) => {
    if (url === `https://huggingface.co/api/models/org/m/revision/${rev}`) {
      return Response.json({ siblings: files.map((rfilename) => ({ rfilename })) });
    }
    if (url === `https://huggingface.co/org/m/resolve/${rev}/decision_config.json`) return Response.json(config);
    if (url.includes("/org/m/")) return new Response("", { status: 404 });
    return new Response("", { status: 401 });
  }) as typeof fetch;
}

test("the checkpoint's own config picks the stack", async () => {
  assert.equal(await stackFor("org/m", "main", hub(["strands_decider_config.json", "lora/adapter_config.json"])), DeciderFunctionStack);
  assert.equal(await stackFor("org/m", "main", hub(["config.json", "decision_config.json", "readout.safetensors"])), DecisionFunctionStack);
  assert.equal(await stackFor("org/m", "261009", hub(["decision_config.json"], DECISION, "261009")), DecisionFunctionStack);
  assert.equal(await stackFor("s3://bucket/e2b/", "main", hub([])), DecisionFunctionStack);
});

test("anything the app can't name is refused, never given a default", async () => {
  const refused: [typeof fetch, RegExp][] = [
    [hub(["config.json"]), /has neither strands_decider_config.json nor decision_config.json/],
    [hub(["strands_decider_config.json", "decision_config.json"]), /has both/],
    [hub(["decision_config.json"], { interface: "pplx-decider", format_version: 1 }), /another model's/],
    [hub(["decision_config.json"], { ...DECISION, format_version: 2 }), /format_version 2, and this app builds format_version 1/],
    [hub(["decision_config.json"], DECISION, "other"), /no such revision/],
  ];
  for (const [get, message] of refused) await assert.rejects(stackFor("org/m", "main", get), message);
  await assert.rejects(stackFor("org/missing", "main", hub([])), /no public model of that name/);
  const offline = (async () => {
    throw new TypeError("fetch failed");
  }) as typeof fetch;
  await assert.rejects(stackFor("org/m", "main", offline), /could not reach the Hugging Face Hub to read org\/m at main: fetch failed/);
  await assert.rejects(stackFor("", "main", hub([])), /set the model/);
});
