const $ = (id) => document.getElementById(id);
const POLL_MS = 1000;   // how often to ask the server which step it's on
let timer = null;       // next status check
let clock = null;       // ticks the elapsed time every second
let startedAt = 0;
let currentStep = "";   // what the status line says the app is doing
const keepCols = new Set();   // columns the user chose to keep even though they were flagged as leakage

const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
  (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

function setStatus(state, text) {
  $("status").dataset.state = state;
  $("status-text").textContent = text;
}

// 83 seconds -> "01:23"; over an hour -> "1:02:03"
function formatElapsed(ms) {
  const total = Math.floor(ms / 1000);
  const h = Math.floor(total / 3600), m = Math.floor((total % 3600) / 60), s = total % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}

// The clock runs on its own, so it ticks every second even between status checks
function startClock() {
  startedAt = Date.now();
  clearInterval(clock);
  clock = setInterval(showRunning, 1000);
  showRunning();
}

function showRunning() {
  setStatus("running", `${currentStep} · ${formatElapsed(Date.now() - startedAt)}`);
}

// Checklist of the steps: done ones ticked, the current one highlighted, the rest greyed out
function stepsChecklist(steps, current) {
  const budget = Number($("time_budget").value) || 60;
  return `<ol class="progress-steps">${steps.map((name, i) => {
    const state = i < current ? "done" : i === current ? "current" : "todo";
    const label = name === "Searching for the best model" ? `${name} (up to ${budget} seconds)` : name;
    return `<li class="${state}"${state === "current" ? ' aria-current="step"' : ""}>${escapeHtml(label)}</li>`;
  }).join("")}</ol>`;
}

// Guess the separator: the one that splits the first lines into the same number of pieces most often.
function detectDelimiter(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim()).slice(0, 10);
  let best = ",", bestScore = 0;
  for (const d of [",", ";", "\t", "|"]) {
    const counts = lines.map((l) => l.split(d).length - 1);
    if (!counts[0]) continue;
    const score = counts.filter((n) => n === counts[0]).length * 1000 + counts[0];
    if (score > bestScore) { best = d; bestScore = score; }
  }
  return best;
}

// Small CSV reader for the preview only (handles quoted fields). The server does the real reading.
function parseCSV(text, delimiter = ",", maxRows = 6) {
  const rows = []; let row = [], field = "", inQuotes = false;
  for (let i = 0; i < text.length && rows.length < maxRows; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') inQuotes = false;
      else field += c;
    } else if (c === '"') inQuotes = true;
    else if (c === delimiter) { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += c;
  }
  if (rows.length < maxRows && (field || row.length)) { row.push(field); rows.push(row); }
  return rows;
}

function previewTable(rows) {
  const [header, ...body] = rows;
  return `<div class="table-wrap"><table>
    <thead><tr>${header.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
    <tbody>${body.map((r) => `<tr>${r.map((v) => `<td>${escapeHtml(v)}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div>`;
}

// When a file is chosen: preview it and fill the target dropdown (alphabetical, ignoring case).
$("file").addEventListener("change", async (e) => {
  $("file-error").textContent = "";
  const file = e.target.files[0];
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    $("file-error").textContent = "Choose a .csv file.";
    return;
  }
  const text = await file.slice(0, 200000).text();
  const rows = parseCSV(text, detectDelimiter(text));
  if (rows.length < 2) {
    $("file-error").textContent = "This file needs a header row and at least one data row.";
    return;
  }
  // Blank headers get the same name pandas gives them on the server ("Unnamed: 0")
  rows[0] = rows[0].map((c, i) => c.trim() || `Unnamed: ${i}`);
  // Skip those columns in the dropdown: they're dropped before training anyway
  const columns = rows[0].filter((c) => !c.startsWith("Unnamed:")).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));
  $("target").innerHTML = '<option value="">Choose a column</option>' +
    columns.map((c) => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join("");
  $("target").disabled = false;
  keepCols.clear();
  $("column-list").innerHTML = rows[0].filter((c) => !c.startsWith("Unnamed:")).map((c) =>
    `<label class="choice"><input type="checkbox" value="${escapeHtml(c)}" checked> ${escapeHtml(c)}</label>`).join("");
  $("columns-field").hidden = false;
  updateColumnList();
  $("results").innerHTML = previewTable(rows);
  setStatus("idle", `Loaded ${file.name}. Showing the first rows.`);
});

$("target").addEventListener("change", () => { $("target-error").textContent = ""; updateColumnList(); });

const columnBoxes = () => [...$("column-list").querySelectorAll("input[type=checkbox]")];

// Hide the target's own checkbox (it's never an input) and show how many columns will be used
function updateColumnList() {
  const target = $("target").value;
  columnBoxes().forEach((box) => { box.parentElement.hidden = box.value === target; });
  const usable = columnBoxes().filter((box) => box.value !== target);
  $("columns-count").textContent = `(${usable.filter((box) => box.checked).length} of ${usable.length})`;
}
$("column-list").addEventListener("change", updateColumnList);

// Submit: send the file to Python, get a job id back, then check on the job every second.
$("train-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("file").files[0];
  let valid = true;
  if (!file) { $("file-error").textContent = "Choose a CSV file first."; valid = false; }
  if (!$("target").value) { $("target-error").textContent = "Choose the column to predict."; valid = false; }
  if (!valid) return;

  $("submit-btn").disabled = true;
  currentStep = "Uploading your file";
  startClock();
  $("results").innerHTML = `<ol class="progress-steps"><li class="current" aria-current="step">Uploading your file</li></ol>`;
  try {
    const form = new FormData($("train-form"));
    columnBoxes().filter((box) => !box.checked && box.value !== $("target").value)
      .forEach((box) => form.append("exclude", box.value));
    keepCols.forEach((col) => form.append("keep", col));
    const response = await fetch("/api/train", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Upload failed (HTTP ${response.status}).`);
    pollJob(data.job_id);
  } catch (err) {
    finish("error", err instanceof TypeError ? "Couldn't reach the server. Is uvicorn still running?" : err.message);
  }
});

async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || `Status check failed (HTTP ${response.status}).`);

    if (job.status === "done") {
      showResult(job.result, jobId);
      finish("done", `Finished in ${formatElapsed(Date.now() - startedAt)}`);
    } else if (job.status === "error") {
      finish("error", job.message || "Training failed.");
    } else {
      const steps = job.steps || [];
      const n = job.step || 0;
      currentStep = steps.length ? `Step ${n + 1} of ${steps.length}: ${steps[n]}` : "Working";
      showRunning();
      if (steps.length) $("results").innerHTML = stepsChecklist(steps, n);
      timer = setTimeout(() => pollJob(jobId), POLL_MS);
    }
  } catch (err) {
    finish("error", err instanceof TypeError ? "Lost contact with the server." : err.message);
  }
}

function finish(state, text) {
  clearTimeout(timer);
  clearInterval(clock);
  setStatus(state, text);
  $("submit-btn").disabled = false;
}

function stepsList(steps) {
  const entries = Object.entries(steps || {});
  if (!entries.length) return `<p class="hint">No changes were needed.</p>`;
  return `<dl class="cleaning">${entries.map(([step, cols]) =>
    `<dt>${escapeHtml(step)}</dt><dd>${cols.map(escapeHtml).join(", ")}</dd>`).join("")}</dl>`;
}

function importanceList(items) {
  if (!items || !items.length) return `<p class="hint">No single column made a clear difference.</p>`;
  const top = items.slice(0, 10), max = top[0].share;
  const pct = (share) => share < 0.005 ? "<1%" : `${Math.round(share * 100)}%`;
  const rest = items.length - top.length;
  return `<ol class="importance">${top.map((i) => `
    <li title="${escapeHtml(i.column)}">
      <span class="imp-name">${escapeHtml(i.column)}</span>
      <span class="imp-bar"><span style="width:${(i.share / max * 100).toFixed(1)}%"></span></span>
      <span class="imp-value">${pct(i.share)}</span>
    </li>`).join("")}</ol>` +
    (rest > 0 ? `<p class="hint">${rest} other column${rest > 1 ? "s" : ""} had a smaller effect.</p>` : "");
}

// "Leakage check" box: columns that give away the answer, plus "too good to be true" warnings
function leakageBox(r) {
  const items = r.leakage || [], warnings = r.warnings || [];
  if (!items.length && !warnings.length) {
    return `<div class="notice notice-ok">Leakage check: no column gives away the answer on its own.</div>`;
  }
  const button = (action, col, label) =>
    `<button type="button" class="link-btn" data-action="${action}" data-col="${escapeHtml(col)}">${label}</button>`;
  const lines = items.map((i) => {
    const name = `<strong>${escapeHtml(i.column)}</strong>`;
    if (i.excluded) return `<li>${name} was left out because it gives away the answer. ${escapeHtml(i.detail)}
      ${button("keep", i.column, "Keep it and retrain")}</li>`;
    if (i.kept_by_you) return `<li>${name} gives away the answer, but you chose to keep it. ${escapeHtml(i.detail)}
      ${button("exclude", i.column, "Leave it out and retrain")}</li>`;
    return `<li>${name} almost predicts the answer on its own. ${escapeHtml(i.detail)} If it's only known after the outcome, leave it out.
      ${button("exclude", i.column, "Leave it out and retrain")}</li>`;
  }).concat(warnings.map((w) => `<li>${escapeHtml(w)}</li>`));
  return `<div class="notice notice-warn"><h3>Check before trusting this score</h3><ul>${lines.join("")}</ul></div>`;
}

// The buttons in the leakage box change which columns are used, then train again
$("results").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-action]");
  if (!btn || $("submit-btn").disabled) return;
  const col = btn.dataset.col;
  const box = columnBoxes().find((b) => b.value === col);
  if (btn.dataset.action === "keep") { keepCols.add(col); if (box) box.checked = true; }
  else { keepCols.delete(col); if (box) box.checked = false; }
  updateColumnList();
  $("train-form").requestSubmit();
});

function showResult(r, jobId) {
  const task = r.task === "classification" ? "Category" : "Number";
  $("results").innerHTML = `
    <p class="headline">${escapeHtml(r.summary)}</p>
    ${leakageBox(r)}
    <ul class="facts">
      <li>Predicting: <strong>${escapeHtml(r.target)}</strong></li>
      <li>Type: <strong>${task}</strong></li>
      <li>Best model: <strong>${escapeHtml(r.best_model)}</strong></li>
      <li>Rows used: <strong>${Number(r.rows_used).toLocaleString()}</strong></li>
    </ul>
    <a class="button-link" href="/api/jobs/${encodeURIComponent(jobId)}/report.pdf" download>Download PDF report</a>
    <h3>What mattered most:</h3>
    <p class="hint">How much the predictions relied on each column. Measured by mixing up one column's values at a time and seeing how much worse the predictions got.</p>
    ${importanceList(r.importance)}
    <h3>How your data was cleaned:</h3>
    ${stepsList(r.cleaning)}
    <h3>How your data was prepared for the model:</h3>
    ${stepsList(r.engineering)}`;
}
