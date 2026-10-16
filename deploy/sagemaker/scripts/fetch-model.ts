/**
 * Download a strands-decider model and the base model it loads, laid out for SageMaker.
 *
 *     npx tsx scripts/fetch-model.ts MODEL REVISION OUT_DIR
 *
 * MODEL is a Hugging Face Hub id. The script writes:
 *
 *     OUT_DIR/<model name>/  the model, at REVISION, with its config pointing at BASE_DIR
 *     OUT_DIR/base           the base model, at the revision the model records
 *
 * SageMaker copies OUT_DIR to /opt/ml/model (the stack's ModelDataSource), so the container loads
 * both without network access. Both repos download to the Hugging Face cache (HF_HUB_CACHE).
 * OUT_DIR/base links to the base model's cached snapshot, which `aws s3 sync` follows, so its
 * weights (up to about 52 GB) are not copied a second time on the build machine.
 *
 * It repeats the base-revision rule of strands_decider.modeling.base_revision, so it needs only
 * @huggingface/hub.
 */
import { existsSync } from "node:fs";
import { cp, mkdir, readFile, symlink, writeFile } from "node:fs/promises";
import * as path from "node:path";
import { snapshotDownload } from "@huggingface/hub";

const BASE_DIR = "/opt/ml/model/base";
const CONFIG_NAMES = ["strands_decider_config.json", "hobson_config.json"];

interface DeciderConfig {
  base_model: string;
  base_revision?: string | null;
  [key: string]: unknown;
}

/** The config's base revision, else the one provenance.json records for the same base. */
async function baseRevision(checkpoint: string, config: DeciderConfig): Promise<string | undefined> {
  if (config.base_revision) return config.base_revision;
  const prov = path.join(checkpoint, "provenance.json");
  if (!existsSync(prov)) return undefined;
  const meta = JSON.parse(await readFile(prov, "utf8"));
  return meta.base_model === config.base_model ? meta.base_model_revision : undefined;
}

async function main(model: string, revision: string, outDir: string): Promise<void> {
  // snapshotDownload returns <cache>/models--<org>--<name>/snapshots/<commit>.
  const snapshot = await snapshotDownload({ repo: model, revision });
  const checkpoint = path.join(outDir, model.split("/").pop()!);
  await cp(snapshot, checkpoint, { recursive: true, dereference: true });

  const configPath = CONFIG_NAMES.map((n) => path.join(checkpoint, n)).find((p) => existsSync(p));
  if (!configPath) throw new Error(`${model}: no ${CONFIG_NAMES.join(" or ")}`);
  const config: DeciderConfig = JSON.parse(await readFile(configPath, "utf8"));
  const base = config.base_model;
  const baseRev = await baseRevision(checkpoint, config);
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
