const $ = (id) => document.getElementById(id);
const POLL_MS = 2000;
let timer = null;

const escapeHtml = (s) => String(s).replace(/[&<>"']/g,
  (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

function setStatus(state, text) {
  $("status").dataset.state = state;
  $("status-text").textContent = text;
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
  $("results").innerHTML = previewTable(rows);
  setStatus("idle", `Loaded ${file.name}. Showing the first rows.`);
});

$("target").addEventListener("change", () => { $("target-error").textContent = ""; });

// Submit: send the file to Python, get a job id back, then check on the job every few seconds.
$("train-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = $("file").files[0];
  let valid = true;
  if (!file) { $("file-error").textContent = "Choose a CSV file first."; valid = false; }
  if (!$("target").value) { $("target-error").textContent = "Choose the column to predict."; valid = false; }
  if (!valid) return;

  $("submit-btn").disabled = true;
  setStatus("running", "Uploading...");
  try {
    const response = await fetch("/api/train", { method: "POST", body: new FormData($("train-form")) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `Upload failed (HTTP ${response.status}).`);
    pollJob(data.job_id, Date.now());
  } catch (err) {
    finish("error", err instanceof TypeError ? "Couldn't reach the server. Is uvicorn still running?" : err.message);
  }
});

async function pollJob(jobId, startedAt) {
  try {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || `Status check failed (HTTP ${response.status}).`);

    if (job.status === "done") {
      showResult(job.result);
      finish("done", "Training finished");
    } else if (job.status === "error") {
      finish("error", job.message || "Training failed.");
    } else {
      const seconds = Math.round((Date.now() - startedAt) / 1000);
      setStatus("running", `Training... ${seconds}s`);
      timer = setTimeout(() => pollJob(jobId, startedAt), POLL_MS);
    }
  } catch (err) {
    finish("error", err instanceof TypeError ? "Lost contact with the server." : err.message);
  }
}

function finish(state, text) {
  clearTimeout(timer);
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

function showResult(r) {
  const task = r.task === "classification" ? "Category" : "Number";
  $("results").innerHTML = `
    <p class="headline">${escapeHtml(r.summary)}</p>
    <ul class="facts">
      <li>Predicting: <strong>${escapeHtml(r.target)}</strong></li>
      <li>Type: <strong>${task}</strong></li>
      <li>Best model: <strong>${escapeHtml(r.best_model)}</strong></li>
      <li>Rows used: <strong>${Number(r.rows_used).toLocaleString()}</strong></li>
    </ul>
    <h3>How your data was cleaned:</h3>
    ${stepsList(r.cleaning)}
    <h3>How your data was prepared for the model:</h3>
    ${stepsList(r.engineering)}
    <h3>What mattered most:</h3>
    <p class="hint">How much the predictions relied on each column. Measured by mixing up one column's values at a time and seeing how much worse the predictions got.</p>
    ${importanceList(r.importance)}`;
}