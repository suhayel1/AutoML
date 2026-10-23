"""
server.py - The backend. Serves the webpage and runs AutoML jobs.

Run with:   uvicorn server:app --reload
Then open:  http://localhost:8000
"""
import threading
import uuid

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from pipeline import read_csv_bytes, run_automl

app = FastAPI()
jobs = {}   # job id -> {"status": ..., "result": ..., "message": ...}. Kept in memory, lost on restart.


def work(job_id, df, target, time_budget, task):
    """Runs in a background thread so the upload request can return immediately."""
    try:
        result = run_automl(df, target, time_budget=time_budget, task=task)
        result.pop("model")   # the trained model can't be sent as JSON; save it to disk here later
        jobs[job_id] = {"status": "done", "result": result}
    except ValueError as e:
        jobs[job_id] = {"status": "error", "message": str(e)}
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": f"Training failed: {e}"}


@app.post("/api/train")
async def train(file: UploadFile,
                target: str = Form(...),
                task: str = Form("auto"),
                time_budget: int = Form(60)):
    try:
        df = read_csv_bytes(await file.read())
    except Exception:
        raise HTTPException(400, "Couldn't read that file as a CSV.")
    df.columns = [str(c).strip() for c in df.columns]

    if target not in df.columns:
        raise HTTPException(400, f"Column '{target}' not found in the file.")
    if task not in ("auto", "classification", "regression"):
        raise HTTPException(400, "Prediction type must be auto, classification or regression.")
    time_budget = max(5, min(time_budget, 600))   # keep runs between 5 seconds and 10 minutes

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "running"}
    threading.Thread(target=work, daemon=True,
                     args=(job_id, df, target, time_budget, None if task == "auto" else task)).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job. The server may have restarted.")
    return jobs[job_id]


# Serves index.html (and any CSS/JS files) from the "static" folder.
# Keep this last: it matches every URL, so routes defined after it would never be reached.
app.mount("/", StaticFiles(directory="static", html=True), name="static")