# Discovery workbench

An optional local interface for the sibling `discovery_submission` research backend. It uses plain HTML, CSS and JavaScript, with a small Python HTTP server. It loads actual research artifacts and starts real backend experiments. The backend remains usable without this folder.

## Start

Install the dependencies using the backend README and activate that Python environment. From the extracted archive's top-level folder:

```bash
python demo_ui/server.py
```

Open **http://127.0.0.1:8765**. To change the port or backend location:

```bash
python demo_ui/server.py --port 9000 --backend /path/to/discovery_submission
```

The server binds to the local computer only. It needs no Node installation, frontend build, external assets or API key. Stop it with Ctrl+C; a running experiment is stopped with it. On Windows, keep the activated environment's terminal open while using the browser.

## What to inspect

The **Experiment** view shows field and future-prediction errors, the discovered equation, predicted and reference fields, observation locations, calibrated intervals and independent physical checks. Move the time slider beyond the observed interval to inspect frozen-model extrapolation. The error toggle shows absolute field error. Click a proposed measurement to inspect a future time at which competing laws disagree.

**Candidate laws** presents generated hypotheses and the fitted shortlist. Select a row to inspect loss histories, score components, conditioning and rejection evidence. Source links open the corresponding scientific evidence inside the interface.

**Evidence** displays the actual evidence-concept-evidence paths used to ground the hypotheses. Search the 28 records by ID, text, source or concept. Publication links open the original scientific sources.

**Evaluation** contains the four-stage comparison and all 95 benchmark records. Switch between RAG and RAG+KG answers, inspect failed responses, and filter the adversarial cases. The verification button reloads the saved checkpoints and recomputes the artifact consistency checks.

The experiment selector includes the main transport-reaction problem, transport on a longer domain and nonlinear transport. Export the selected equation or open the complete technical report.

## Run a new experiment

Choose an example, sensor count, noise level and seed. The server invokes the backend's `experiment --offline` command with the full configured training budgets. It generates observations, proposes and searches structures, fits the candidate PINNs, confirms rejected laws independently, predicts future dynamics and trains the uncertainty ensemble. The progress panel reports the current stage and exposes the execution log. Only one training job runs at a time.

Completed experiments are stored in `demo_ui/runs/` and appear in the experiment selector, including after a server restart. You can remove this generated folder when no job is running. The bundled research results are not overwritten. Cancelled or inconclusive runs retain their logs but do not appear as completed studies. Very sparse or noisy data can produce an inconclusive result or an inaccurate law; ground-truth errors remain visible in completed controlled experiments.

The shipped pretrained request cache supports these examples without downloads. Numerical training still runs locally on CPU. Runtime depends on the machine. Low error is measured for the supplied experiments, not guaranteed for every setting.

## Files and checks

`server.py` serves a fixed set of local assets and JSON endpoints, restricts artifact downloads to an allowlist, validates experiment settings and manages backend subprocesses. `index.html` supplies the page structure; `styles.css` supplies the light theme and responsive layout; `app.js` renders the actual arrays, metrics and evidence and controls jobs. No scientific fitting logic is duplicated in the interface.

Run the API tests from this folder:

```bash
python -m pytest -q tests/test_server.py
```

The optional browser test requires Node.js and Playwright only for development:

```bash
npm install --no-save --package-lock=false playwright
npx playwright install chromium
node tests/browser.cjs
```

Add `--live` to train a fresh experiment through the interface and test cancellation. The script starts and stops its own server. `DISCOVERY_PYTHON` can select a specific Python executable; otherwise it uses `python`. `tests/browser_results.json` records the executed browser checks, and `tests/results.json` records the API test result. Runtime users do not need these JavaScript test dependencies.
