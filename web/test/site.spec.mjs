// The deployed setup: the built site under a subpath with no COOP/COEP headers (GitHub Pages), the weights from a
// second origin with CORS (Hugging Face). Answers must match the Python reference (check.py's reference.json), a
// reload must read the OPFS cache, and files the manifest does not list must be deleted.
//   WEIGHTS=<convert.py output>   required
//   DEVICE=wasm|webgpu            default wasm: CI runners have no GPU, and the only WebGPU there (SwiftShader) has no fp16
//   CHROME=1                      installed Google Chrome for the chromium project (needed for WebGPU on a real GPU)
import { test, expect, chromium, firefox, webkit } from "@playwright/test";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { startStatic } from "./static.mjs";

const W = process.env.WEIGHTS, DEVICE = process.env.DEVICE || "wasm", SITE = 8789, WEIGHTS = 8790, PREFIX = "/strands-decider/";
const NEAR_TIE = 0.05, MAX_ERROR = 0.15;
let servers = [], out, profile;

test.skip(!W, "set WEIGHTS");
test.beforeAll(async () => {
  out = fs.mkdtempSync(path.join(os.tmpdir(), "decider-site-")); profile = path.join(out, "profile");
  execFileSync("node", ["build.mjs", "--out", path.join(out, "site"), "--weights", `http://127.0.0.1:${WEIGHTS}/`], { cwd: path.join(import.meta.dirname, ".."), stdio: "inherit" });
  servers = [await startStatic({ port: SITE, routes: [[PREFIX, path.join(out, "site")]] }), await startStatic({ port: WEIGHTS, cors: true, routes: [["/", W]] })];
});
test.afterAll(() => { servers.forEach((s) => s.close()); if (out) fs.rmSync(out, { recursive: true, force: true }); });

async function open(ctx) {
  const page = await ctx.newPage(), logs = [];
  page.on("console", (m) => logs.push(m.text()));
  page.on("pageerror", (e) => logs.push(e.message));
  await page.goto(`http://127.0.0.1:${SITE}${PREFIX}?device=${DEVICE}`);
  await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 10 * 60_000 });
  expect(await page.evaluate(() => document.body.dataset.ready), await page.textContent("#p-status")).toBe("1");
  const info = await page.evaluate(() => window.decider.info);
  console.log(`load: ${Object.entries(info.timings).map(([k, v]) => `${k} ${v.toFixed(0)}`).join(", ")}${info.cached ? " (cached)" : ""}`);
  return { page, logs, info };
}

test("site: answers like Python, caches, prunes stale files", async ({ browserName }) => {
  test.skip(DEVICE === "webgpu" && browserName !== "chromium", "WebGPU is tested in Chrome");
  const engine = { chromium, firefox, webkit }[browserName];
  const ctx = await engine.launchPersistentContext(path.join(profile, browserName), { headless: !process.env.HEADED, ...(process.env.CHROME && browserName === "chromium" ? { channel: "chrome" } : {}) });
  const hosts = new Set(), files = new Set();
  ctx.on("request", (r) => { const u = new URL(r.url()); if (u.protocol.startsWith("http")) { hosts.add(u.host); files.add(u.pathname); } });
  try {
    // Start from an empty cache (WebKit keeps OPFS outside the profile directory).
    const blank = await ctx.newPage();
    await blank.goto(`http://127.0.0.1:${SITE}${PREFIX}js/prompt.js`);
    await blank.evaluate(() => navigator.storage.getDirectory().then((d) => d.removeEntry("strands-decider-weights", { recursive: true })).catch(() => {}));
    await blank.close();
    let { page, logs, info } = await open(ctx);
    expect(info.runtime).toBe(DEVICE);
    expect(info.cached).toBe(false);
    await expect(page.getByTestId("answer")).toContainText("billing", { timeout: 60_000 });

    // Every question of the fixtures (one per item) through decide(), and the multi-question items through decideMany().
    const questions = JSON.parse(fs.readFileSync(path.join(import.meta.dirname, "questions.json"), "utf8"));
    const refs = JSON.parse(fs.readFileSync(path.join(W, "reference.json"), "utf8")).results
      // Long states are left to prompt.test.mjs and check.py: on single-threaded WASM they take minutes.
      .filter((r) => (r.set === "fixtures" || r.set === "many") && r.state_ids.length < 512)
      .filter((r) => Object.values(r.questions).every((q) => q.probabilities));  // none until the next full check
    const check = (got, ref, where) => {
      const p = got.type === "noul" ? [1 - got.noul, got.noul] : Object.values(got.probabilities), want = ref.probabilities;
      const err = Math.max(...p.map((x, i) => Math.abs(x - want[i]))), top = (v) => v.indexOf(Math.max(...v)), sorted = [...want].sort((x, y) => y - x);
      expect(Object.keys(got), where).toEqual(Object.keys(ref.answer));
      expect(err, where).toBeLessThan(MAX_ERROR);
      if (sorted[0] - sorted[1] >= NEAR_TIE) expect(top(p), where).toBe(top(want));
      return err;
    };
    for (const ref of refs) {
      const item = questions[ref.set].find((x) => x.name === ref.name);
      if (ref.set === "fixtures") {
        const got = await page.evaluate(([s, q]) => window.decider.decide(s, q), [item.state, item.questions.q]);
        console.log(`${ref.name.padEnd(26)} ${got.tokens} tok  err ${check(got.answer, ref.questions.q, ref.name).toFixed(3)}  forward ${got.timings.forward_ms.toFixed(0)} ms`);
      } else {
        const got = await page.evaluate(([s, q]) => window.decider.decideMany(s, q), [item.state, item.questions]);
        const errs = Object.keys(item.questions).map((k) => check(got.answers[k], ref.questions[k], `${ref.name}/${k}`).toFixed(3));
        console.log(`${ref.name.padEnd(26)} ${got.tokens} tok  err ${errs.join(" ")}  total ${got.timings.total_ms.toFixed(0)} ms`);
      }
    }

    // A file from an older release: the next load must delete it.
    await page.evaluate(async () => {
      const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("strands-decider-weights");
      const w = await (await dir.getFileHandle("0".repeat(64), { create: true })).createWritable(); await w.write("stale"); await w.close();
    });
    await page.close();

    ({ page, logs, info } = await open(ctx));
    // Playwright's WebKit on Linux grants an origin less storage than the model needs; the page must then say so and
    // run uncached (Safari's WebKit on macOS grants about 20 GB, and caches). Everywhere else the reload is cached.
    const noRoom = browserName === "webkit" && process.platform === "linux" && logs.includes("[decider] not enough storage to cache the model");
    expect(info.cached, logs.filter((l) => l.startsWith("[decider]")).join("\n")).toBe(!noRoom);
    await expect(page.getByTestId("answer")).toContainText("billing", { timeout: 60_000 });
    const cached = await page.evaluate(async () => {
      const dir = await (await navigator.storage.getDirectory()).getDirectoryHandle("strands-decider-weights"), names = [];
      for await (const n of dir.keys()) names.push(n);
      return names.sort();
    });
    const manifest = JSON.parse(fs.readFileSync(path.join(W, "manifest.json"), "utf8"));
    // The stale file is gone either way; without room, nothing else was written.
    expect(cached).toEqual(noRoom ? [] : [...new Set(Object.values(manifest.files).map((f) => f.sha256))].sort());
    expect([...hosts].sort()).toEqual([`127.0.0.1:${SITE}`, `127.0.0.1:${WEIGHTS}`]);
    console.log("ort files:", [...files].filter((f) => f.includes("/ort/")).join(" "));
    expect(logs.filter((l) => /error|failed|404/i.test(l))).toEqual([]);
  } finally { await ctx.close(); }
});

// Without a usable WebGPU adapter the page says so and downloads nothing. Headless Chromium has no adapter by default,
// and with --enable-unsafe-webgpu it gets SwiftShader, which lacks shader-f16.
for (const [what, args, reason] of [["no adapter", [], /no usable GPU/], ["no shader-f16", ["--enable-unsafe-webgpu"], /shader-f16/]]) {
  test(`site: ${what} -> needs WebGPU, nothing downloaded`, async ({ browserName }) => {
    test.skip(browserName !== "chromium", "the adapter cases are Chromium's");
    const browser = await chromium.launch({ args });
    try {
      const page = await browser.newPage(), weights = [];
      page.on("request", (r) => { if (new URL(r.url()).port === String(WEIGHTS)) weights.push(r.url()); });
      await page.goto(`http://127.0.0.1:${SITE}${PREFIX}`);
      await page.waitForFunction(() => document.body.dataset.ready, null, { timeout: 60_000 });
      expect(await page.evaluate(() => document.body.dataset.ready)).toBe("unsupported");
      await expect(page.locator("#p-status")).toContainText(reason);
      await expect(page.locator("#p-status a")).toHaveAttribute("href", "?device=wasm");
      expect(weights).toEqual([]);
    } finally { await browser.close(); }
  });
}
