// Runs in: cloud session. Logged-out browser check for scripts/check_space_public.py.
//   node scripts/space_browser_check.mjs <url> <space-host> [screenshot-dir]
// Fresh Chromium context per viewport (no cookies, no storage). Prints {"problems": [...]} last.
import fs from "node:fs";
import { createRequire } from "node:module";

// The session's pre-installed Playwright is resolved the CommonJS way (ESM ignores NODE_PATH).
const require = createRequire(import.meta.url);
let pw;
try { pw = require("playwright"); }
catch { pw = require(require.resolve("playwright", { paths: [process.env.PLAYWRIGHT_DIR || "/opt/node-tools/node_modules"] })); }
const { chromium } = pw;

const [url, host, shots] = process.argv.slice(2);
const problems = [];
const browser = await chromium.launch();
try {
  for (const [label, viewport] of [["desktop", { width: 1366, height: 900 }],
                                   ["phone", { width: 390, height: 844 }]]) {
    const ctx = await browser.newContext({ viewport });
    const page = await ctx.newPage();
    page.on("console", (m) => { if (m.type() === "error") problems.push(`[${label}] console error: ${m.text()}`); });
    page.on("pageerror", (e) => problems.push(`[${label}] page error: ${e.message}`));
    page.on("requestfailed", (r) => problems.push(`[${label}] request failed: ${r.url()} (${r.failure()?.errorText})`));
    page.on("response", (r) => {
      const u = r.url();
      if (u.startsWith("data:") || u.startsWith("blob:")) return;
      if (r.status() >= 400) problems.push(`[${label}] HTTP ${r.status()}: ${u}`);
      if (new URL(u).hostname.toLowerCase() !== host)
        problems.push(`[${label}] resource from another host (data must be bundled): ${u}`);
    });
    await page.goto(url, { waitUntil: "networkidle", timeout: 90000 });
    if ((await page.locator("[data-testid=episode]").count()) === 0)
      problems.push(`[${label}] no episode rendered on load`);
    const controls = page.locator("[data-check]");
    const n = await controls.count();
    for (let i = 0; i < n; i++) {
      const c = controls.nth(i);
      if (!(await c.isVisible())) continue;
      const name = await c.getAttribute("data-check");
      await c.click();
      // waitForLoadState("networkidle") returns at once on a loaded page, before the task file
      // fetched by this click arrives, so wait for the case panel to show the selected case.
      const synced = await page.waitForFunction(() => {
        const cur = document.querySelector("#tasks button[aria-current=true] .tid")?.textContent;
        const head = (document.querySelector(".case-head h2")?.textContent || "").split(" ")[0];
        return !cur || cur === head;
      }, null, { timeout: 3000 }).then(() => true, () => false);
      if (!synced) problems.push(`[${label}] control "${name}": case panel does not show the selected case after 3 s`);
      if ((await page.locator("[data-testid=episode]").count()) === 0)
        problems.push(`[${label}] control "${name}": no episode rendered`);
    }
    if (shots) {
      fs.mkdirSync(shots, { recursive: true });
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: `${shots}/${label}.png`, fullPage: true });
    }
    await ctx.close();
  }
} catch (e) {
  problems.push(`browser check error: ${e.message}`);
} finally {
  await browser.close();
}
console.log(JSON.stringify({ problems }));
