"use strict";

const state = {
  view: "experiment",
  studyId: "reference",
  study: null,
  field: null,
  evidence: null,
  research: null,
  time: 40,
  heat: "prediction",
  sensors: true,
  candidate: 0,
  question: "Q001",
  arm: "rag_kg",
  job: null,
};
const titles = {
  experiment: [
    "Experiment",
    "Explore the experiment.",
    "A measured field, competing explanations, and a prediction to test.",
  ],
  models: [
    "Candidate laws",
    "Compare the explanations.",
    "Inspect generated hypotheses, fitted equations, and the reasons for rejection.",
  ],
  evidence: [
    "Evidence",
    "Follow the scientific evidence.",
    "Trace each proposed mechanism to its sources and the graph connecting them.",
  ],
  evaluation: [
    "Evaluation",
    "Examine the research results.",
    "Compare baselines and inspect every recorded benchmark response.",
  ],
};
const $ = (selector) => document.querySelector(selector);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const number = (value, digits = 3) =>
  Number.isFinite(value) ? Number(value).toPrecision(digits) : "n/a";
const percent = (value) =>
  Number.isFinite(value) ? `${(100 * value).toFixed(2)}%` : "n/a";
const tag = (value) =>
  `<span class="tag ${esc(value)}">${esc(value.replaceAll("_", " "))}</span>`;
const citations = (ids = []) =>
  ids
    .map(
      (id) =>
        `<button class="citation-link" data-source="${esc(id)}">${esc(id)}</button>`,
    )
    .join("");

async function api(path, options) {
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok)
    throw new Error(value.error || `Request failed (${response.status})`);
  return value;
}

function notice(message, error = false) {
  const node = $("#notice");
  node.textContent = message;
  node.className = error ? "error" : "";
  node.hidden = !message;
}

function equation(value, coefficients = true) {
  if (!value?.terms) return "No equation selected";
  const operators = {
    u_xx: "∂²u/∂x²",
    u_x: "∂u/∂x",
    u: "u",
    u2: "u²",
    u3: "u³",
    "u*u_x": "u ∂u/∂x",
    1: "1",
  };
  return (
    "∂u/∂t = " +
    value.terms
      .map((term, i) => {
        const coefficient = coefficients ? value.coefficients[i] : 1;
        const sign = i
          ? coefficient < 0
            ? " − "
            : " + "
          : coefficient < 0
            ? "−"
            : "";
        return (
          sign +
          (coefficients
            ? `${Math.abs(coefficient).toPrecision(5)} `
            : `a${"₁₂₃₄"[i]} `) +
          operators[term]
        );
      })
      .join("")
  );
}

function metric(label, value, note) {
  return `<div class="metric"><div class="metric-label">${label}</div><div class="metric-value">${value}</div><div class="metric-note">${note}</div></div>`;
}

async function refreshStudies() {
  const result = await api("/api/studies");
  $("#study-select").innerHTML = result.studies
    .map(
      (study) =>
        `<option value="${esc(study.id)}">${esc(study.title)}</option>`,
    )
    .join("");
  $("#study-select").value = state.studyId;
}

async function loadStudy(id) {
  const requestId = ++loadStudy.requestId;
  notice("");
  $("#content").innerHTML =
    '<div class="loading">Reading experiment results…</div>';
  try {
    const [study, field] = await Promise.all([
      api(`/api/studies/${id}`),
      api(`/api/studies/${id}/field`),
    ]);
    if (requestId !== loadStudy.requestId) return;
    state.studyId = id;
    state.study = study;
    state.field = field;
    state.time = field.t.reduce(
      (best, time, i) =>
        Math.abs(time - field.observed_end) <
        Math.abs(field.t[best] - field.observed_end)
          ? i
          : best,
      0,
    );
    state.candidate = study.candidates.selected_index ?? 0;
    $("#study-select").value = id;
    $("#study-description").textContent =
      `${study.dataset.training_points} training measurements · ${100 * study.dataset.noise_rms_fraction}% noise`;
    $("#export-equation").href = `/api/studies/${id}/download/equation`;
    await render();
  } catch (error) {
    notice(error.message, true);
    $("#content").innerHTML =
      '<div class="empty">The experiment could not be loaded. Check that the backend artifacts are complete.</div>';
  }
}
loadStudy.requestId = 0;

async function render() {
  const [name, heading, subtitle] = titles[state.view];
  $("#breadcrumb").textContent = name;
  $("#page-title").textContent = heading;
  $("#page-subtitle").textContent = subtitle;
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === state.view);
    button.setAttribute(
      "aria-current",
      button.dataset.view === state.view ? "page" : "false",
    );
  });
  if (!state.study) return;
  if (state.view === "experiment") renderExperiment();
  if (state.view === "models") renderModels();
  if (state.view === "evidence") {
    state.evidence ||= await api("/api/evidence");
    if (state.view === "evidence") renderEvidence();
  }
  if (state.view === "evaluation") {
    state.research ||= await api("/api/research");
    if (state.view === "evaluation") renderEvaluation();
  }
}

function renderExperiment() {
  const { study, field } = state;
  const m = study.metrics.inverse || {};
  const probes = study.design.probes || [];
  $("#content").innerHTML = `
    <section class="metric-strip" aria-label="Measured performance">
      ${metric("Field error", percent(m.prediction_relative_error), "Within the observation interval")}
      ${metric("Future prediction error", percent(m.ood_relative_error), "Frozen model beyond observed time")}
      ${metric("Parameter error", percent(m.parameter_relative_error), "Joint relative coefficient error")}
      ${metric("Candidate supports", study.search.evaluated_supports ?? "n/a", "Structures screened against data")}
    </section>
    <section class="equation-bar"><div class="eyebrow">SELECTED LAW<br><br>${tag("selected")}</div><div class="equation">${esc(equation(study.equation))}</div></section>
    <div class="charts-grid">
      <section class="panel"><div class="panel-head"><div><h2>Field over space and time</h2><p>The dashed line marks the end of observations.</p></div><div class="segmented" aria-label="Field display"><button data-heat="prediction" class="${state.heat === "prediction" ? "selected" : ""}">Prediction</button><button data-heat="error" class="${state.heat === "error" ? "selected" : ""}">Error</button></div></div><div class="panel-body chart-wrap"><canvas id="heatmap" aria-label="Heatmap of the predicted field across space and time" role="img"></canvas></div><div class="chart-foot"><label><input type="checkbox" id="show-sensors" ${state.sensors ? "checked" : ""}>Observed sensors</label><span>Low <i class="color-scale"></i> High</span></div></section>
      <section class="panel"><div class="panel-head"><div><h2>Spatial profile</h2><p id="profile-caption"></p></div><span id="time-tag" class="tag"></span></div><div class="panel-body chart-wrap"><svg id="profile" role="img" aria-label="Prediction, reference and uncertainty at the selected time"></svg></div><div class="chart-foot"><div class="legend"><span><i></i>Prediction</span><span><i class="truth"></i>Reference</span>${field.lower ? '<span><i class="band"></i>90% interval</span>' : ""}</div></div><div class="time-control"><label for="time-slider">Time</label><input id="time-slider" type="range" min="0" max="${field.t.length - 1}" step="1" value="${state.time}"><output id="time-value" for="time-slider"></output></div></section>
    </div>
    <div class="lower-grid"><section class="panel"><div class="panel-head"><div><h2>Independent physical checks</h2><p>Evaluated away from the training collocation points.</p></div></div><div class="panel-body"><ul class="checks">${[
      ["PDE residual MSE", m.pde_residual_mse],
      ["Periodic boundary loss", m.boundary_loss],
      ["Integrated balance RMS", m.balance_rms],
      ["Inferred initial profile MSE", m.initial_loss],
    ]
      .map(
        ([label, value]) =>
          `<li><span>${label}</span><strong>${number(value, 4)}</strong></li>`,
      )
      .join(
        "",
      )}</ul><p class="small-note">Initial values are inferred. The periodic boundary type is known experimental metadata.</p></div></section>
    <section class="panel"><div class="panel-head"><div><h2>Where to measure next</h2><p>Future locations where competitive laws disagree.</p></div></div><div class="panel-body">${probes.length ? probes.map((probe, i) => `<button class="probe" data-probe="${i}" aria-label="Inspect suggested measurement ${i + 1}"><span class="number">0${i + 1}</span><span>x = ${probe.x.toFixed(3)}</span><span>t = ${probe.t.toFixed(3)}</span><span class="strength">${probe.disagreement_to_noise.toFixed(2)} × noise</span></button>`).join("") : '<p class="small-note">No distinct alternative laws are available for a disagreement-based recommendation.</p>'}<p class="small-note">These are proposed measurements, derived from separate numerical rollouts. They are not additional observations.</p></div></section></div>`;
  drawHeatmap();
  drawProfile();
}

function drawHeatmap() {
  const canvas = $("#heatmap");
  if (!canvas) return;
  const width = canvas.clientWidth,
    height = canvas.clientHeight,
    ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  const f = state.field,
    margin = { left: 43, right: 9, top: 12, bottom: 33 };
  const w = width - margin.left - margin.right,
    h = height - margin.top - margin.bottom;
  const values = f.prediction.map((row, i) =>
    row.map((v, j) =>
      state.heat === "error" ? Math.abs(v - f.reference[i][j]) : v,
    ),
  );
  const flat = values.flat(),
    low = Math.min(...flat),
    high = Math.max(...flat);
  context.clearRect(0, 0, width, height);
  const xAt = (x) => margin.left + (x / f.length) * w;
  const tAt = (t) => margin.top + h * (1 - t / f.t.at(-1));
  values.forEach((row, i) =>
    row.forEach((value, j) => {
      const q = Math.max(0, Math.min(1, (value - low) / (high - low || 1)));
      context.fillStyle = `rgb(${Math.round(238 - q * 204)},${Math.round(244 - q * 147)},${Math.round(248 - q * 104)})`;
      const top =
        i === values.length - 1
          ? margin.top
          : (tAt(f.t[i]) + tAt(f.t[i + 1])) / 2;
      const bottom =
        i === 0 ? margin.top + h : (tAt(f.t[i - 1]) + tAt(f.t[i])) / 2;
      context.fillRect(
        margin.left + (j / row.length) * w,
        top,
        w / row.length + 0.5,
        bottom - top + 0.5,
      );
    }),
  );
  if (state.sensors) {
    context.fillStyle = "rgba(255,255,255,.72)";
    f.observations.x_train.forEach((x, i) => {
      context.fillRect(
        xAt(x) - 1.2,
        tAt(f.observations.t_train[i]) - 1.2,
        2.4,
        2.4,
      );
    });
  }
  context.strokeStyle = "rgba(33,51,56,.7)";
  context.lineWidth = 1;
  context.setLineDash([4, 3]);
  context.beginPath();
  context.moveTo(margin.left, tAt(f.observed_end));
  context.lineTo(margin.left + w, tAt(f.observed_end));
  context.stroke();
  context.setLineDash([]);
  context.strokeStyle = "rgba(255,255,255,.9)";
  context.beginPath();
  context.moveTo(margin.left, tAt(f.t[state.time]));
  context.lineTo(margin.left + w, tAt(f.t[state.time]));
  context.stroke();
  context.font = "9px Arial";
  context.fillStyle = "#74827e";
  context.textAlign = "center";
  [0, 0.25, 0.5, 0.75, 1].forEach((value) =>
    context.fillText(
      (value * f.length).toFixed(2),
      margin.left + value * w,
      height - 17,
    ),
  );
  context.fillText("Position", margin.left + w / 2, height - 2);
  context.textAlign = "right";
  [0, 0.4, 0.8, 1.2]
    .filter((value) => value <= f.t.at(-1) + 1e-9)
    .forEach((value) =>
      context.fillText(value.toFixed(1), margin.left - 8, tAt(value) + 3),
    );
  context.save();
  context.translate(10, margin.top + h / 2);
  context.rotate(-Math.PI / 2);
  context.textAlign = "center";
  context.fillText("Time", 0, 0);
  context.restore();
}

function linePlot(
  target,
  xs,
  series,
  band = null,
  labels = ["Position", "Concentration"],
) {
  const svg = $(target);
  if (!svg) return;
  const W = 460,
    H = 280,
    p = { l: 47, r: 12, t: 15, b: 35 };
  const values = series
    .flatMap((s) => s.values)
    .concat(band ? [...band.lower, ...band.upper] : [])
    .filter(Number.isFinite);
  if (!values.length || !xs.length) return;
  const lo = Math.min(...values),
    hi = Math.max(...values),
    padding = (hi - lo || 1) * 0.1;
  const min = lo - padding,
    max = hi + padding;
  const X = (x) =>
    p.l + ((x - xs[0]) / (xs.at(-1) - xs[0] || 1)) * (W - p.l - p.r);
  const Y = (y) => H - p.b - ((y - min) / (max - min)) * (H - p.t - p.b);
  const path = (ys) =>
    ys
      .map(
        (value, i) =>
          `${i ? "L" : "M"}${X(xs[i]).toFixed(2)},${Y(value).toFixed(2)}`,
      )
      .join(" ");
  let markup = "";
  for (let i = 0; i < 5; i++) {
    const value = min + ((max - min) * i) / 4,
      y = Y(value);
    markup += `<line x1="${p.l}" y1="${y}" x2="${W - p.r}" y2="${y}" stroke="#e8ede7"/><text x="${p.l - 9}" y="${y + 3}" text-anchor="end">${number(value, 2)}</text>`;
    const x = xs[0] + ((xs.at(-1) - xs[0]) * i) / 4;
    markup += `<text x="${X(x)}" y="${H - p.b + 18}" text-anchor="middle">${number(x, 2)}</text>`;
  }
  if (band) {
    const lower = band.lower.map((value, i) => [X(xs[i]), Y(value)]);
    const upper = band.upper.map((value, i) => [X(xs[i]), Y(value)]).reverse();
    markup += `<path d="M${[...lower, ...upper].map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" L")} Z" fill="#e1ebf2"/>`;
  }
  series.forEach((s) => {
    markup += `<path d="${path(s.values)}" fill="none" stroke="${s.color}" stroke-width="${s.dashed ? 1.3 : 1.8}" ${s.dashed ? 'stroke-dasharray="4 3"' : ""}/>`;
  });
  markup += `<text x="${(p.l + W - p.r) / 2}" y="${H - 2}" text-anchor="middle">${esc(labels[0])}</text><text transform="translate(12 ${(p.t + H - p.b) / 2}) rotate(-90)" text-anchor="middle">${esc(labels[1])}</text>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = `<g font-family="Arial, sans-serif" font-size="9" fill="#7d8b83">${markup}</g>`;
}

function drawProfile() {
  const f = state.field,
    i = state.time,
    time = f.t[i];
  const ood = time > f.observed_end + 1e-9;
  $("#time-tag").className = `tag ${ood ? "ood" : ""}`;
  $("#time-tag").textContent = ood ? "Unseen future" : "Observed interval";
  $("#profile-caption").textContent = `Field slice at time ${time.toFixed(2)}`;
  $("#time-value").textContent = time.toFixed(2);
  linePlot(
    "#profile",
    f.x,
    [
      { values: f.reference[i], color: "#7a8985", dashed: true },
      { values: f.prediction[i], color: "#205e9b" },
    ],
    f.lower ? { lower: f.lower[i], upper: f.upper[i] } : null,
  );
}

function renderModels() {
  const hypotheses = state.study.hypotheses.hypotheses || [],
    trials = state.study.candidates.pinn_trials || [];
  $("#content").innerHTML =
    `<section class="hypothesis-grid">${hypotheses.map((h, i) => `<article class="hypothesis"><span class="number">HYPOTHESIS 0${i + 1}</span><code>${esc(equation(h, false))}</code><p>${esc(h.reason)}</p>${citations(h.evidence)}</article>`).join("")}</section><p class="small-note">The a coefficients are unknown and fitted from observations.</p>
    <div class="section-head"><h2>PINN validation and selection</h2><p>Choose a candidate to inspect its diagnostics.</p></div><div class="table-wrap"><table><thead><tr><th>Candidate</th><th>Active terms</th><th>Status</th><th class="right">Selection score</th><th class="right">Validation MSE</th><th class="right">PDE residual</th></tr></thead><tbody>${trials.map((trial) => `<tr class="interactive ${trial.index === state.candidate ? "current" : ""}" data-candidate="${trial.index}" tabindex="0" role="button" aria-label="Inspect candidate ${trial.index + 1}"><td class="mono">0${trial.index + 1}</td><td class="mono">${esc(trial.equation.terms.join(" + "))}</td><td>${tag(trial.status)}</td><td class="right mono">${number(trial.selection_score, 4)}</td><td class="right mono">${number(trial.metrics?.validation_data_loss, 4)}</td><td class="right mono">${number(trial.metrics?.pde_residual_mse, 4)}</td></tr>`).join("")}</tbody></table></div><section class="panel candidate-detail"><div class="panel-head"><div><h2>Candidate diagnostics</h2><p id="candidate-equation"></p></div><span>Lower selection scores are preferred.</span></div><div class="panel-body" id="candidate-detail"></div></section>`;
  renderCandidate();
}

function renderCandidate() {
  const trial = state.study.candidates.pinn_trials.find(
    (value) => value.index === state.candidate,
  );
  if (!trial) return;
  $("#candidate-equation").textContent = equation(trial.equation);
  const components = trial.score_components || {},
    total =
      Object.values(components).reduce((sum, value) => sum + value, 0) || 1;
  const diagnostic = trial.identifiability || {};
  $("#candidate-detail").innerHTML =
    `<div class="candidate-grid"><div><div class="chart-wrap"><svg id="training-chart" role="img" aria-label="Training losses across optimization steps"></svg></div><div class="legend"><span><i></i>Total</span><span><i class="truth"></i>Data</span><span><i class="band"></i>Physics</span></div><p class="small-note">Losses are shown on a log scale. Physics activates after the observation warm-up.</p></div><div>${Object.entries(
      components,
    )
      .map(
        ([label, value]) =>
          `<div class="score-row"><span>${esc(label)}</span><div class="score-track"><div class="score-fill" data-width="${(100 * value) / total}"></div></div><code>${number(value, 3)}</code></div>`,
      )
      .join(
        "",
      )}<ul class="checks"><li><span>Weak-design rank</span><strong>${diagnostic.rank ?? "n/a"} / ${diagnostic.unknowns ?? "n/a"}</strong></li><li><span>Scaled condition number</span><strong>${number(diagnostic.scaled_condition_number, 4)}</strong></li><li><span>Term overlap</span><strong>${number(diagnostic.maximum_term_overlap, 4)}</strong></li></ul><div class="diagnostic-note">${esc(trial.rejection_reason || "This candidate is compatible with the held-out measurements. Full rank alone does not prove unique structural identification.")}${trial.independent_confirmation ? `<br>Independent numerical validation MSE: <code>${number(trial.independent_confirmation.validation_mse, 4)}</code>.` : ""}</div></div></div>`;
  document.querySelectorAll("[data-width]").forEach((node) => {
    node.style.width = `${node.dataset.width}%`;
  });
  const history = trial.training?.history || [];
  linePlot(
    "#training-chart",
    history.map((record) => record.step),
    [
      {
        values: history.map((r) => Math.log10(Math.max(r.total, 1e-10))),
        color: "#205e9b",
      },
      {
        values: history.map((r) => Math.log10(Math.max(r.data, 1e-10))),
        color: "#8a968c",
        dashed: true,
      },
      {
        values: history.map((r) => Math.log10(Math.max(r.pde, 1e-10))),
        color: "#b2c5ca",
      },
    ],
    null,
    ["Adam step", "log₁₀ normalized loss"],
  );
}

function renderEvidence(filter = "") {
  const records = state.evidence.records;
  $("#content").innerHTML =
    `<section class="panel graph-panel"><div class="panel-head"><div><h2>Grounding the proposed mechanisms</h2><p>Direct term evidence connects through a physical concept to an independent source.</p></div><span class="tag">MULTI-SOURCE</span></div><div class="panel-body"><svg id="evidence-graph" viewBox="0 0 760 220" role="img" aria-label="Scientific evidence and concept paths for the generated hypotheses"></svg><p class="graph-note">The language model proposes structures. The graph supplies auditable support paths; numerical validation decides whether a proposal fits the observations.</p></div></section><div class="toolbar"><input id="evidence-filter" type="search" value="${esc(filter)}" placeholder="Find evidence by term, concept, source or ID…" aria-label="Filter evidence"><span>${records.length} evidence records · ${new Set(records.map((r) => r.source)).size} source documents</span></div><div id="evidence-cards" class="evidence-grid"></div>`;
  drawEvidenceGraph();
  filterEvidence(filter);
}

function drawEvidenceGraph() {
  const paths =
    state.study.hypotheses.hypotheses?.flatMap((h) => h.support_paths || []) ||
    [];
  const unique = [
    ...new Map(
      paths.map((entry) => [entry.path.join("/"), entry.path]),
    ).values(),
  ];
  if (!unique.length) {
    $("#evidence-graph").innerHTML =
      '<text x="30" y="80" fill="#7a8780" font-size="13">Source records are available below. This archived run has no recorded support paths.</text>';
    return;
  }
  const columns = [0, 1, 2].map((index) => [
    ...new Set(unique.map((path) => path[index])),
  ]);
  const pos = new Map();
  columns.forEach((items, column) =>
    items.forEach((id, row) =>
      pos.set(`${column}:${id}`, [
        80 + column * 300,
        40 + ((row + 0.5) * 150) / items.length,
      ]),
    ),
  );
  let markup = "";
  unique.forEach((path) => {
    for (let i = 0; i < 2; i++) {
      const [x1, y1] = pos.get(`${i}:${path[i]}`),
        [x2, y2] = pos.get(`${i + 1}:${path[i + 1]}`);
      markup += `<path d="M${x1 + 55} ${y1} C${x1 + 170} ${y1},${x2 - 170} ${y2},${x2 - 55} ${y2}" fill="none" stroke="#bfd0d3"/>`;
    }
  });
  ["TERM EVIDENCE", "SHARED CONCEPT", "INDEPENDENT SOURCE"].forEach(
    (label, i) => {
      markup += `<text x="${80 + 300 * i}" y="17" text-anchor="middle" font-size="8" fill="#86958c" letter-spacing="1">${label}</text>`;
    },
  );
  columns.forEach((items, column) =>
    items.forEach((id) => {
      const [x, y] = pos.get(`${column}:${id}`);
      markup += `<g class="${column === 1 ? "" : "graph-node"}" ${column !== 1 ? `data-source="${esc(id)}" role="button" tabindex="0" aria-label="Inspect evidence ${esc(id)}"` : ""}><rect x="${x - 55}" y="${y - 18}" width="110" height="36" fill="${column === 1 ? "#f0f4ed" : "#f2f6fa"}" stroke="#c7d6cf"/><text x="${x}" y="${y + 4}" text-anchor="middle" font-size="11" fill="#37556b">${esc(id)}</text></g>`;
    }),
  );
  $("#evidence-graph").innerHTML = markup;
}

function filterEvidence(query) {
  const words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const records = state.evidence.records.filter((record) =>
    words.every((word) =>
      `${record.id} ${record.text} ${record.keywords} ${record.title}`
        .toLowerCase()
        .includes(word),
    ),
  );
  $("#evidence-cards").innerHTML = records.length
    ? records
        .map(
          (r) =>
            `<article class="evidence-card"><header><span>${esc(r.id)}</span><span>${esc(r.location)}</span></header><p>${esc(r.text)}</p><a href="${esc(r.source)}" target="_blank" rel="noopener noreferrer">${esc(r.title)} ↗</a><small>${esc(r.concepts.join(" · "))}</small></article>`,
        )
        .join("")
    : '<div class="empty">No matching evidence records.</div>';
}

function renderEvaluation() {
  const research = state.research,
    summary = research.benchmark.summary;
  $("#content").innerHTML =
    `<section class="metric-strip">${metric("Scientific questions", summary.scientific.count, "Concept and citation checks")}${metric("Multi-hop questions", summary.multi_hop.count, "Evidence-combination questions")}${metric("Adversarial cases", `${summary.adversarial.accuracy * summary.adversarial.count}/${summary.adversarial.count}`, "Physical validator verdicts")}${metric("Backend tests", research.tests.passed ?? "n/a", "Recorded automated checks")}</section><div class="section-head"><h2>The four-stage comparison</h2><button class="secondary" id="verify">Verify saved artifacts</button></div><div class="table-wrap"><table><thead><tr><th>System</th><th class="right">Field error</th><th class="right">Future error</th><th class="right">Parameter error</th><th>Exact support</th></tr></thead><tbody>${[
      ["llm_rag", "LLM / RAG"],
      ["rag_kg", "RAG + knowledge graph"],
      ["predefined_pinn", "Fixed-support PINN"],
      ["generative_pinn_discovery", "Generative discovery"],
    ]
      .map(([key, label]) => {
        const r = research.ablation[key];
        return `<tr><td>${label}</td><td class="right mono">${percent(r.prediction_relative_error)}</td><td class="right mono">${percent(r.ood_relative_error)}</td><td class="right mono">${percent(r.parameter_relative_error)}</td><td>${r.support_exact ? "Recovered" : "Not recovered"}</td></tr>`;
      })
      .join(
        "",
      )}</tbody></table></div><p class="evaluation-note">All arms use the same observation split. Retrieval arms supply structures to a common numerical evaluator; the full engine searches structures and refines coefficients with PINNs. This comparison refers to the main research run.</p><div class="section-head"><h2>Inspect the 95-item benchmark</h2><p>Recorded responses with retrieval context.</p></div><div class="toolbar"><input id="question-filter" type="search" aria-label="Search benchmark questions" placeholder="Search the benchmark…"><select id="question-category" aria-label="Question category"><option value="all">All questions</option><option value="scientific">Scientific</option><option value="multi_hop">Multi-hop</option><option value="adversarial">Adversarial</option></select></div><div class="benchmark-grid"><div id="question-list" class="question-list"></div><div id="answer-detail" class="answer-detail"></div></div><p class="evaluation-note">Answer scoring uses synonym-group coverage and citation resolution. It is a reproducible heuristic, not an expert semantic grade. Failed responses are retained.</p>`;
  filterQuestions();
}

function filterQuestions() {
  const query = $("#question-filter").value.toLowerCase(),
    category = $("#question-category").value;
  const items = state.research.benchmark.items.filter(
    (q) =>
      (category === "all" || q.category === category) &&
      `${q.id} ${q.question}`.toLowerCase().includes(query),
  );
  if (!items.some((q) => q.id === state.question))
    state.question = items[0]?.id || null;
  $("#question-list").innerHTML = items.length
    ? items
        .map(
          (q) =>
            `<button class="question ${q.id === state.question ? "current" : ""}" data-question="${esc(q.id)}"><small>${esc(q.id)} / ${esc(q.category.replaceAll("_", " "))}</small>${esc(q.question)}</button>`,
        )
        .join("")
    : '<div class="empty">No matching questions.</div>';
  renderAnswer();
}

function renderAnswer() {
  const q = state.research.benchmark.items.find(
    (item) => item.id === state.question,
  );
  if (!q) {
    $("#answer-detail").innerHTML =
      '<p class="empty">Choose another filter.</p>';
    return;
  }
  if (q.category === "adversarial") {
    $("#answer-detail").innerHTML =
      `${tag(q.correct ? "passed" : "failed")}<h3>${esc(q.question)}</h3><div class="answer-text">${esc(q.reason)}</div><div class="answer-meta">Validator decision: <strong>${esc(q.status)}</strong></div>`;
    return;
  }
  const answer = q.systems[state.arm];
  $("#answer-detail").innerHTML =
    `<div class="segmented"><button data-arm="rag" class="${state.arm === "rag" ? "selected" : ""}">RAG</button><button data-arm="rag_kg" class="${state.arm === "rag_kg" ? "selected" : ""}">RAG + KG</button></div><h3>${esc(q.question)}</h3><div class="answer-text">${esc(answer.answer)}</div><div class="answer-meta"><span>Concept match ${percent(answer.concept_fraction)}</span><span>Citations ${answer.valid_citations ? "resolved" : "missing / invalid"}</span></div><div class="eyebrow">RETRIEVED EVIDENCE</div>${citations(answer.evidence)}${answer.paths.length ? `<details><summary>Graph retrieval paths</summary><pre>${esc(answer.paths.map((path) => path.join(" → ")).join("\n"))}</pre></details>` : ""}`;
}

async function handleClick(event) {
  const button = event.target.closest(
    "[data-view],[data-heat],[data-probe],[data-candidate],[data-source],[data-question],[data-arm],#verify",
  );
  if (!button) return;
  try {
    if (button.dataset.view) {
      state.view = button.dataset.view;
      await render();
    }
    if (button.dataset.heat) {
      state.heat = button.dataset.heat;
      renderExperiment();
    }
    if (button.dataset.probe !== undefined) {
      const probe = state.study.design.probes[Number(button.dataset.probe)];
      state.time = state.field.t.reduce(
        (best, t, i) =>
          Math.abs(t - probe.t) < Math.abs(state.field.t[best] - probe.t)
            ? i
            : best,
        0,
      );
      $("#time-slider").value = state.time;
      drawProfile();
      drawHeatmap();
      notice(
        `Suggested observation: x = ${probe.x.toFixed(3)}, t = ${probe.t.toFixed(3)}. Candidate disagreement is ${probe.disagreement_to_noise.toFixed(2)} times the observation noise.`,
      );
    }
    if (button.dataset.candidate !== undefined) {
      state.candidate = Number(button.dataset.candidate);
      renderModels();
    }
    if (button.dataset.source) {
      state.view = "evidence";
      state.evidence ||= await api("/api/evidence");
      await render();
      renderEvidence(button.dataset.source);
      $("#evidence-filter").focus();
    }
    if (button.dataset.question) {
      state.question = button.dataset.question;
      filterQuestions();
    }
    if (button.dataset.arm) {
      state.arm = button.dataset.arm;
      renderAnswer();
    }
    if (button.id === "verify") {
      button.disabled = true;
      button.textContent = "Verifying…";
      try {
        const result = await api("/api/verify", { method: "POST" });
        notice(result.message, !result.passed);
      } finally {
        button.disabled = false;
        button.textContent = "Verify saved artifacts";
      }
    }
  } catch (error) {
    notice(error.message, true);
  }
}

document.addEventListener("click", handleClick);
document.addEventListener("keydown", (event) => {
  const node = event.target.closest("tr[data-candidate],g[data-source]");
  if (node && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    node.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  }
});
document.addEventListener("input", (event) => {
  if (event.target.id === "time-slider") {
    state.time = Number(event.target.value);
    drawProfile();
    drawHeatmap();
  }
  if (event.target.id === "evidence-filter") filterEvidence(event.target.value);
  if (event.target.id === "question-filter") filterQuestions();
});
document.addEventListener("change", (event) => {
  if (event.target.id === "study-select") loadStudy(event.target.value);
  if (event.target.id === "show-sensors") {
    state.sensors = event.target.checked;
    drawHeatmap();
  }
  if (event.target.id === "question-category") filterQuestions();
});
window.addEventListener("resize", () => {
  if (state.view === "experiment") drawHeatmap();
});
$("#open-run").addEventListener("click", () => $("#run-dialog").showModal());
$("#close-run").addEventListener("click", () => $("#run-dialog").close());

function updateJob(job) {
  state.job = job;
  $("#job-panel").hidden = false;
  $("#job-state").textContent = job.state[0].toUpperCase() + job.state.slice(1);
  $("#job-time").textContent = `${Math.floor(job.seconds)}s`;
  const progress = job.progress || {};
  if (job.state === "running") {
    $("#job-progress").removeAttribute("value");
    $("#job-stage").textContent = progress.stage
      ? `${progress.stage} · ${progress.completed} / ${progress.total}`
      : "Preparing observations and scientific evidence…";
  } else {
    $("#job-progress").value = job.state === "completed" ? 1 : 0;
    $("#job-stage").textContent =
      job.state === "completed"
        ? "The fitted model, predictions and uncertainty estimates are ready."
        : job.log.at(-1) || "Run stopped.";
  }
  $("#job-log").textContent = job.log.join("\n");
  $("#cancel-run").hidden = job.state !== "running";
  $("#start-run").disabled = job.state === "running";
  $("#open-run").textContent =
    job.state === "running" ? "View running experiment" : "Run experiment ↗";
}

async function pollJob(identifier) {
  try {
    const job = await api(`/api/runs/${identifier}`);
    updateJob(job);
    if (job.state === "running") {
      setTimeout(() => pollJob(identifier), 1200);
      return;
    }
    if (job.state === "completed") {
      state.studyId = job.study_id;
      state.view = "experiment";
      await refreshStudies();
      await loadStudy(job.study_id);
      notice(
        "Live discovery completed. You are viewing the newly trained model and its independent evaluation.",
      );
    }
  } catch (error) {
    notice(`Could not refresh the run: ${error.message}`, true);
    if (state.job?.state === "running")
      setTimeout(() => pollJob(identifier), 3000);
  }
}

$("#run-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  $("#start-run").disabled = true;
  try {
    const payload = {
      example: $("#example").value,
      sensors: Number($("#sensors").value),
      noise: Number($("#noise").value),
      seed: Number($("#seed").value),
    };
    const job = await api("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    updateJob(job);
    pollJob(job.id);
  } catch (error) {
    $("#start-run").disabled = false;
    $("#job-panel").hidden = false;
    $("#job-state").textContent = "Could not start";
    $("#job-stage").textContent = error.message;
    $("#cancel-run").hidden = true;
  }
});
$("#cancel-run").addEventListener("click", async () => {
  if (!state.job) return;
  try {
    updateJob(
      await api(`/api/runs/${state.job.id}/cancel`, { method: "POST" }),
    );
  } catch (error) {
    notice(error.message, true);
  }
});

(async () => {
  try {
    await refreshStudies();
    await loadStudy("reference");
  } catch (error) {
    notice(error.message, true);
    $("#content").innerHTML =
      '<div class="empty">Start this interface with server.py from an environment containing the backend dependencies.</div>';
  }
})();
