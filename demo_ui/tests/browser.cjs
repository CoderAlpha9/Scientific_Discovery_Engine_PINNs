const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const checks = [];
const errors = [];

async function check(name, action) {
  await action();
  checks.push(name);
  console.log(`Passed: ${name}`);
}

async function main() {
  const server = spawn(process.env.DISCOVERY_PYTHON || "python", [
    "-u",
    path.join(root, "server.py"),
    "--port",
    "0",
  ]);
  server.stderr.pipe(process.stderr);
  const address = await new Promise((resolve, reject) => {
    server.stdout.once("data", (chunk) =>
      resolve(chunk.toString().trim().split(" ").at(-1)),
    );
    server.once("error", reject);
    server.once("exit", (code) =>
      reject(new Error(`Server exited with ${code}`)),
    );
  });
  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      executablePath: process.env.BROWSER_EXECUTABLE || undefined,
    });
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") errors.push(message.text());
    });
    await page.goto(address);
    await page.waitForSelector("#heatmap");

    await check("Field controls and future prediction", async () => {
      await page.locator("#time-slider").fill("60");
      assert.match(await page.locator("#time-tag").innerText(), /FUTURE/);
      await page.locator('[data-heat="error"]').click();
      await page.locator("#show-sensors").uncheck();
      assert.equal(await page.locator("#show-sensors").isChecked(), false);
      await page.locator('[data-heat="prediction"]').click();
      await page.locator("[data-probe]").first().click();
      assert.match(
        await page.locator("#notice").innerText(),
        /Suggested observation/,
      );
    });

    await check("Candidate diagnostics and independent rejection", async () => {
      await page.locator('[data-view="models"]').click();
      assert.equal(await page.locator("[data-candidate]").count(), 5);
      await page.locator('[data-candidate="4"]').click();
      assert.match(
        await page.locator("#candidate-detail").innerText(),
        /independent/i,
      );
      await page.locator('[data-candidate="0"]').focus();
      await page.keyboard.press("Enter");
      assert.equal(await page.locator("#training-chart").count(), 1);
    });

    await check("Graph paths and source filtering", async () => {
      await page.locator(".citation-link").first().click();
      await page.waitForSelector("#evidence-graph");
      assert.ok(await page.locator("#evidence-filter").inputValue());
      await page.locator("#evidence-filter").fill("");
      assert.equal(await page.locator(".evidence-card").count(), 28);
      assert.ok(
        (await page.locator("#evidence-graph [data-source]").count()) > 1,
      );
      await page.locator("#evidence-filter").fill("no matching evidence xyz");
      assert.equal(await page.locator(".evidence-card").count(), 0);
    });

    await check("All benchmark arms and adversarial verdicts", async () => {
      await page.locator('[data-view="evaluation"]').click();
      await page.waitForSelector("#question-category");
      assert.equal(await page.locator("[data-question]").count(), 95);
      await page.locator('[data-arm="rag"]').click();
      assert.ok((await page.locator(".answer-text").innerText()).length > 10);
      await page.locator('[data-arm="rag_kg"]').click();
      await page.locator("#question-category").selectOption("adversarial");
      assert.equal(await page.locator("[data-question]").count(), 20);
      assert.match(
        await page.locator("#answer-detail").innerText(),
        /Validator decision/,
      );
      await page.locator("#question-filter").fill("A20");
      assert.equal(await page.locator("[data-question]").count(), 1);
    });

    await check("Artifact verification through the UI", async () => {
      await page.locator("#verify").click();
      await page.waitForFunction(
        () =>
          document
            .querySelector("#notice")
            .textContent.includes("verification passed"),
        null,
        { timeout: 60000 },
      );
    });

    await check("Alternate domain and artifact download", async () => {
      await page.locator('[data-view="experiment"]').click();
      await page.locator("#study-select").selectOption("transport");
      await page.waitForFunction(() =>
        document.querySelector("#export-equation").href.includes("/transport/"),
      );
      const response = await page.request.get(
        `${address}/api/studies/transport/field`,
      );
      assert.equal((await response.json()).length, 2);
      const [download] = await Promise.all([
        page.waitForEvent("download"),
        page.locator("#export-equation").click(),
      ]);
      const file = path.join(
        os.tmpdir(),
        `discovery-equation-${process.pid}.json`,
      );
      await download.saveAs(file);
      assert.deepEqual(JSON.parse(fs.readFileSync(file)).terms, [
        "u_xx",
        "u_x",
      ]);
      fs.unlinkSync(file);
    });

    await check("Mobile layout without page overflow", async () => {
      await page.setViewportSize({ width: 390, height: 844 });
      for (const view of ["experiment", "models", "evidence", "evaluation"]) {
        await page.locator(`[data-view="${view}"]`).click();
        const overflow = await page.evaluate(
          () => document.documentElement.scrollWidth - window.innerWidth,
        );
        assert.ok(overflow <= 1, `${view}: ${overflow}px overflow`);
      }
      if (process.env.SCREENSHOT_DIR) {
        await page.screenshot({
          path: path.join(process.env.SCREENSHOT_DIR, "ui-mobile.png"),
          fullPage: true,
        });
      }
      await page.setViewportSize({ width: 1440, height: 1000 });
    });

    if (process.argv.includes("--live")) {
      await check("Full live discovery launched from the UI", async () => {
        await page.locator("#open-run").click();
        await page.locator("#example").selectOption("transport_reaction");
        await page.locator("#sensors").selectOption("16");
        await page.locator("#noise").selectOption("0.01");
        await page.locator("#seed").fill("11");
        await page.locator("#start-run").click();
        await page.waitForFunction(
          () =>
            ["Completed", "Failed", "Inconclusive"].includes(
              document.querySelector("#job-state").textContent,
            ),
          null,
          { timeout: 600000 },
        );
        assert.equal(await page.locator("#job-state").innerText(), "Completed");
        await page.locator("#close-run").click();
        await page.waitForFunction(() =>
          /[a-f0-9]{12}/.test(document.querySelector("#study-select").value),
        );
        assert.equal(await page.locator(".equation-bar").count(), 1);
      });
      await check("Live job cancellation", async () => {
        await page.locator("#open-run").click();
        await page.locator("#start-run").click();
        await page.waitForFunction(
          () => document.querySelector("#job-state").textContent === "Running",
        );
        await page.locator("#cancel-run").click();
        await page.waitForFunction(
          () =>
            document.querySelector("#job-state").textContent === "Cancelled",
        );
        await page.locator("#close-run").click();
      });
    }
    await page.locator('[data-view="experiment"]').click();
    await page.locator("#study-select").selectOption("reference");
    await page.waitForFunction(() =>
      document.querySelector("#export-equation").href.includes("/reference/"),
    );
    if (process.env.SCREENSHOT_DIR) {
      await page.screenshot({
        path: path.join(process.env.SCREENSHOT_DIR, "ui-desktop.png"),
        fullPage: true,
      });
      for (const view of ["models", "evidence", "evaluation"]) {
        await page.locator(`[data-view="${view}"]`).click();
        await page.screenshot({
          path: path.join(process.env.SCREENSHOT_DIR, `ui-${view}.png`),
          fullPage: true,
        });
      }
    }
    assert.deepEqual(errors, [], "Browser errors");
    const result = {
      status: "passed",
      browser: browser.version(),
      checked_at: new Date().toISOString(),
      live_training: process.argv.includes("--live"),
      checks,
      errors,
    };
    fs.writeFileSync(
      path.join(__dirname, "browser_results.json"),
      JSON.stringify(result, null, 2) + "\n",
    );
  } finally {
    if (browser) await browser.close();
    server.kill();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
