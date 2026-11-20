/**
 * Download a strands-decider model, laid out for SageMaker.
 *
 *     npx tsx scripts/fetch-model.ts MODEL REVISION OUT_DIR
 *
 * MODEL is a Hugging Face Hub id. The script writes:
 *
 *     OUT_DIR/<model name>/  the model, at REVISION
 *     OUT_DIR/base           (v1 only) the base model, at the revision the model records
 *
 * SageMaker copies OUT_DIR to /opt/ml/model (the stack's ModelDataSource), so the container loads
 * everything without network access. Repos download to the Hugging Face cache (HF_HUB_CACHE).
 *
 * v1 checkpoints (strands_decider_config.json / hobson_config.json) ship only the fine-tuned
 * parts, so the script also downloads the base model and rewrites the config to point at
 * BASE_DIR. OUT_DIR/base symlinks the base's cached snapshot, which `aws s3 sync` follows, so
 * its weights (up to about 52 GB) are not copied a second time on the build machine.
 *
 * v2 checkpoints (decision_config.json) are self-contained — they ship every weight the runtime
 * loads — so the base is metadata only and the script neither downloads it nor rewrites the
 * config.
 *
 * For v1, it repeats the base-revision rule of strands_decider.modeling.base_revision, so it
 * needs only @huggingface/hub.
 */
import { existsSync } from "node:fs";
import { cp, mkdir, readFile, symlink, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { snapshotDownload } from "@huggingface/hub";

const BASE_DIR = "/opt/ml/model/base";
const DECISION_CONFIG = "decision_config.json";
const V1_CONFIG_NAMES = ["strands_decider_config.json", "hobson_config.json"];
const CONFIG_NAMES = [DECISION_CONFIG, ...V1_CONFIG_NAMES];

interface V1Config {
  base_model: string;
  base_revision?: string | null;
  [key: string]: unknown;
}

interface V2Config {
  interface: "strands-decider";
  provenance: { base_model: string; base_revision: string };
  [key: string]: unknown;
}

/** The config's base revision, else the one provenance.json records for the same base. */
async function baseRevision(checkpointDir: string, config: V1Config): Promise<string | undefined> {
  if (config.base_revision) return config.base_revision;
  const prov = path.join(checkpointDir, "provenance.json");
  if (!existsSync(prov)) return undefined;
  const meta = JSON.parse(await readFile(prov, "utf8"));
  return meta.base_model === config.base_model ? meta.base_model_revision : undefined;
}

async function main(model: string, revision: string, outDir: string): Promise<void> {
  // snapshotDownload returns <cache>/models--<org>--<name>/snapshots/<commit>.
  const snapshot = await snapshotDownload({ repo: model, revision });
  const checkpointDir = path.join(outDir, model.split("/").pop()!);
  await cp(snapshot, checkpointDir, { recursive: true, dereference: true });

  const configPath = CONFIG_NAMES.map((n) => path.join(checkpointDir, n)).find((p) => existsSync(p));
  if (!configPath) throw new Error(`${model}: no ${CONFIG_NAMES.join(" or ")}`);

  if (path.basename(configPath) === DECISION_CONFIG) {
    const config: V2Config = JSON.parse(await readFile(configPath, "utf8"));
    const { base_model, base_revision } = config.provenance;
    console.error(`v2 model ${model}@${path.basename(snapshot)}; base ${base_model}@${base_revision}`);
    return;
  }

  const config: V1Config = JSON.parse(await readFile(configPath, "utf8"));
  const base = config.base_model;
  const baseRev = await baseRevision(checkpointDir, config);
  console.error(`model ${model}@${path.basename(snapshot)}; base model ${base}@${baseRev ?? "main"}`);
  const baseSnapshot = await snapshotDownload({ repo: base, revision: baseRev });
  await mkdir(outDir, { recursive: true });
  await symlink(baseSnapshot, path.join(outDir, "base"), "dir");

  // The loader reads the base model from BASE_DIR. With base_model no longer the Hub id in
  // provenance.json, base_revision() returns None, so nothing asks for a Hub revision offline.
  config.base_model = BASE_DIR;
  config.base_revision = null;
  await writeFile(configPath, JSON.stringify(config, null, 2));
}

const [model, revision, outDir] = process.argv.slice(2);
main(model, revision, outDir).catch((err) => {
  console.error(err);
  process.exit(1);
});
