import { DeciderFunctionStack } from "./decider-function-stack";
import { DecisionFunctionStack } from "./decision-function-stack";

const HUB = "https://huggingface.co";
// What each format's checkpoint declares itself with (README.md, Model checkpoints).
const V1_CONFIG = "strands_decider_config.json";
const DECISION_CONFIG = "decision_config.json";
const INTERFACE = "strands-decider";
const FORMAT_VERSION = 1;

export type FunctionStack = typeof DeciderFunctionStack | typeof DecisionFunctionStack;

/**
 * The stack that deploys `model` at `revision`, from what the checkpoint declares itself to be:
 * v1's stack for a checkpoint with a strands_decider_config.json, and DecisionFunctionStack for one
 * whose decision_config.json names this interface at a format_version it builds. Anything else,
 * including a checkpoint the Hub can't show, is refused. An s3:// checkpoint takes
 * DecisionFunctionStack: v1's build reads only the Hub, and the decision build checks its
 * decision_config.json before converting anything.
 */
export async function stackFor(model: string, revision: string, get: typeof fetch = fetch): Promise<FunctionStack> {
  if (!model) throw new Error("set the model: -c model=<Hub id, or s3://bucket/prefix/>");
  if (model.startsWith("s3://")) return DecisionFunctionStack;
  const where = `${model} at ${revision}`;
  const listing = await hubJson(get, `${HUB}/api/models/${model}/revision/${encodeURIComponent(revision)}`, where);
  const files = new Set(((listing.siblings ?? []) as { rfilename: string }[]).map((s) => s.rfilename));
  const v1 = files.has(V1_CONFIG);
  if (v1 === files.has(DECISION_CONFIG)) {
    throw new Error(
      `${where} has ${v1 ? `both ${V1_CONFIG} and` : `neither ${V1_CONFIG} nor`} ${DECISION_CONFIG}, so it ` +
        "is not a Strands Decider checkpoint this app can deploy",
    );
  }
  if (v1) return DeciderFunctionStack;
  const config = await hubJson(get, `${HUB}/${model}/resolve/${encodeURIComponent(revision)}/${DECISION_CONFIG}`, where);
  if (config.interface !== INTERFACE) {
    throw new Error(`${where}: its ${DECISION_CONFIG} is another model's (interface ${JSON.stringify(config.interface)})`);
  }
  if (config.format_version !== FORMAT_VERSION) {
    throw new Error(
      `${where}: format_version ${JSON.stringify(config.format_version)}, and this app builds ` +
        `format_version ${FORMAT_VERSION}; a newer strands-decider may deploy it`,
    );
  }
  return DecisionFunctionStack;
}

async function hubJson(get: typeof fetch, url: string, where: string): Promise<any> {
  let response: Response;
  try {
    response = await get(url);
  } catch (e) {
    throw new Error(`could not reach the Hugging Face Hub to read ${where}: ${(e as Error).message}`);
  }
  // The Hub answers 401 for a repository that doesn't exist, as for a private one.
  if (response.status === 401) throw new Error(`${where}: no public model of that name on the Hugging Face Hub`);
  if (response.status === 404) throw new Error(`${where}: no such revision on the Hugging Face Hub`);
  if (!response.ok) throw new Error(`${where}: the Hugging Face Hub answered ${response.status} for ${url}`);
  return response.json();
}
