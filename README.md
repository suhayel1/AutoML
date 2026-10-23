# AutoML Web App (prototype)

Upload a spreadsheet (CSV), pick the column you want to predict, and the app cleans the data, finds the best model automatically, and explains the result in plain language. No coding or data-science knowledge needed.

> **Status: work in progress.** This is an early prototype for internal review. It runs on one machine, has no login, and should not be used with confidential data yet.

## What it does

1. **Upload a CSV.** Commas, semicolons, tabs and `|` separators are detected automatically, and a preview of the first rows is shown.
2. **Choose the column to predict**, e.g. *Amount*, *Churned*, *Popularity*. The app works out whether it's a **number** (regression) or a **category** (classification), or the user can choose.
3. **Set a time budget** (5–600 seconds) for how long to search for the best model.
4. **Get results:**
   - **Accuracy compared with a simple baseline**, e.g. *"Off by 41.23 on average (always guessing the average: 218.49)"*. The baseline shows whether the model is actually learning anything.
   - **What mattered most:** which columns the predictions relied on.
   - **A plain-language report** of every cleaning and preparation step applied to the data.

## How it works

```
Browser (static/)  ──upload CSV──▶  FastAPI server (server.py)  ──▶  pipeline.py  ──▶  FLAML AutoML
      ▲                                     │
      └──── checks progress every 2 s ◀─────┘   (training runs in the background)
```

| File | Role |
|---|---|
| `pipeline.py` | All the data and ML logic: reading, cleaning, preparing, training, scoring and column importance. No web code, so it can be reused by other interfaces. |
| `server.py` | FastAPI backend. Receives the upload, starts training in a background thread, and reports progress and results as JSON. |
| `static/` | The web page (`index.html`, `style.css`, `script.js`). |

**Automatic cleaning:** removes duplicate rows, blank-header columns, empty columns, columns with a single value and ID-like text columns. Converts numbers stored as text (`"1,234"`), dates and yes/no / true/false columns. Fills blank text with `Missing`.

**Preparation:** splits dates into year, month, day and weekday, and turns text into categories.

**Training:** 80% of rows train the model and 20% are held back to test it. [FLAML](https://github.com/microsoft/FLAML) tries several model types (LightGBM, XGBoost, random forest, extra trees…) and their settings within the time budget, and keeps the best one.

**Column importance:** each column is shuffled in turn on the test rows. The more the predictions get worse, the more the model relied on that column. This works with any model.

## Running it locally

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn server:app --port 8000
```

Then open <http://localhost:8000>.

To share a temporary public link for a demo, install [cloudflared](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/) and run `cloudflared tunnel --url http://localhost:8000` in a second terminal. The link only works while this computer is on and both commands are running.

## Tested with

Kaggle datasets: Spotify Most Streamed Songs 2024, Amazon Sale Report (129k rows), Top 100 Trending Books, and a pipe-separated grocery bestseller file. Testing these found and fixed several issues: pipe separators, numbers with commas, True/False columns, file encoding and duplicate rows.

## Known issues / still to do

- [ ] **Random train/test split can overstate accuracy** when the same item appears many times, e.g. the same product recorded daily. Needs a time-based or grouped split.
- [ ] **No leakage warnings yet.** A column that gives away the answer (e.g. `fulfilled-by` when predicting fulfilment) shows up in *What mattered most*, but the app doesn't warn the user.
- [ ] **ID columns with a few repeats** (order IDs, song titles) are only removed if *every* value is different. Change this to "different in most rows" (> 75%).
- [ ] **Row-number columns with a name** (e.g. `index`) are not detected.
- [ ] **No way to exclude columns** before training.
- [ ] **Imbalanced categories** (e.g. 99% "No") make accuracy look good when it isn't. Needs better measures (precision/recall).
- [ ] **Jobs are kept in memory**, so restarting the server loses running and finished jobs.
- [ ] **No login, upload size limit or rate limiting**, so it's not ready for public hosting.
- [ ] Error messages could be clearer, e.g. when removing duplicates leaves too few rows.

## Possible future features

- **Predict on new data:** upload a new CSV and download it with a predictions column (needs the trained model and its cleaning rules saved).
- **Exclude columns** with checkboxes, plus **automatic leakage warnings**.
- **Friendlier summary**, e.g. *"81% better than guessing"*, and charts such as predicted vs. actual.
- **Per-prediction explanations**: why *this* row got *this* prediction.
- **Excel (.xlsx) uploads.**
- **Switchable engine:** AutoGluon as an alternative to FLAML for higher accuracy when there's more time.
- **Save and reuse models**, with a history of past runs.
- **Hosting and user accounts** so clients can use it without installing anything.
- **Industry templates:** preset targets and cleaning rules for common SMB data (sales, churn, inventory).
