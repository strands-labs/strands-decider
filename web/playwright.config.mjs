import { defineConfig } from "@playwright/test";
// One project per engine: the WASM path runs in all three (WebKit is Safari's engine). `--project chromium` runs one.
export default defineConfig({
  testDir: "test", testMatch: /.*\.spec\.mjs/, timeout: 20 * 60_000, workers: 1, reporter: [["list"]],
  projects: ["chromium", "firefox", "webkit"].map((browserName) => ({ name: browserName, use: { browserName } })),
});
