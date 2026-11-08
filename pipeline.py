"""
pipeline.py - The AutoML logic, with no printing, no input() and no sys.argv.
Every interface (command line, Streamlit, a web API) calls these same functions.
"""
import csv
import io
import numpy as np
import pandas as pd
from flaml import AutoML

from eda import explore
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

TRUE_TERMS = {"yes", "y", "true", "t"}
FALSE_TERMS = {"no", "n", "false", "f"}
YES_NO_MAP = {**{t: 1 for t in TRUE_TERMS}, **{f: 0 for f in FALSE_TERMS}}

NUMBER_PATTERN = r"-?\d{1,3}(,\d{3})+(\.\d+)?|-?\d+(\.\d+)?"   # 1,234,567.89 or 1234.5

MIN_ROWS = 20

# The steps run_automl() goes through, in order. The web page shows them as a checklist while it works.
STEPS = [
    "Removing duplicate rows",
    "Cleaning the data",
    "Preparing the columns for the model",
    "Checking for columns that give away the answer",
    "Searching for the best model",
    "Testing the model on rows it hasn't seen",
    "Measuring which columns mattered most",
    "Analysing the data for the report",
]

# Leakage check: how much of the gap between "always guessing" and a perfect score one column closes on its own
LEAK_EXCLUDE_AT = 0.99   # left out automatically (unless the user chose to keep it)
LEAK_WARN_AT = 0.95      # kept, but flagged

def read_csv_bytes(raw):
    """Read an uploaded CSV, detecting whether columns are separated by , ; tab or |."""
    sample = raw[:50000].decode("utf-8", errors="replace")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    try:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding="latin-1", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def try_numeric(col):
    """Turn text columns that are really numbers ("390,470,936") into numbers."""
    if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_datetime64_any_dtype(col):
        return col
    text = col.dropna().astype(str).str.strip()
    if text.empty or not text.str.fullmatch(NUMBER_PATTERN).all():
        return col
    return pd.to_numeric(col.astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")

def try_datetime(col):
    text = col.dropna().astype(str).str.strip()
    if text.empty or text.str.len().median() < 6:   # too short to be dates ("May", "Q1", "12")
        return col
    # For dates written with numbers like 05-02-2010, check whether the day comes first:
    # if the first number is ever above 12 (and the second never is), it must be the day.
    parts = text.str.extract(r"^(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}\b")
    dayfirst = False
    if parts.notna().all(axis=None):
        first, second = parts[0].astype(int), parts[1].astype(int)
        dayfirst = bool((first > 12).any() and not (second > 12).any())
    try:
        return pd.to_datetime(col, format="mixed", dayfirst=dayfirst)
    except (ValueError, TypeError, OverflowError):
        return col


def is_yes_no(col):
    values = set(col.dropna().astype(str).str.strip().str.lower())
    return values <= (TRUE_TERMS | FALSE_TERMS) and bool(values & TRUE_TERMS) and bool(values & FALSE_TERMS)


def clean_features(X):
    """Your cleaning steps. Returns the cleaned data and a dict of what happened, instead of printing."""
    report = {}

    # Columns with no header, usually a row number saved by mistake (pandas names them "Unnamed: 0")
    unnamed_cols = [c for c in X.columns if str(c).startswith("Unnamed:")]
    X = X.drop(columns=unnamed_cols)
    report["Removed columns with no header name"] = unnamed_cols

    text_cols = X.select_dtypes(exclude=["number", "bool"]).columns
    X[text_cols] = X[text_cols].apply(try_numeric)
    report["Converted text columns that hold numbers (like '1,234,567') to numbers"] = list(text_cols[[pd.api.types.is_numeric_dtype(X[c]) for c in text_cols]])

    empty_cols = X.columns[X.isnull().all()]
    X = X.drop(columns=empty_cols)
    report["Removed columns with no values"] = list(empty_cols)

    single_value_cols = X.columns[X.nunique(dropna=False) == 1]
    X = X.drop(columns=single_value_cols)
    report["Removed columns with only 1 unique value"] = list(single_value_cols)

    non_num_cols = X.select_dtypes(exclude="number").columns
    X[non_num_cols] = X[non_num_cols].apply(try_datetime)
    report["Converted columns to dates"] = list(X.select_dtypes(include="datetime").columns)

    text_cols = X.select_dtypes(exclude=["number", "datetime"]).columns
    binary_cols = text_cols[X[text_cols].apply(is_yes_no).astype(bool)]
    X[binary_cols] = X[binary_cols].apply(lambda c: c.astype(str).str.strip().str.lower().map(YES_NO_MAP))
    report["Converted yes/no and true/false columns to 1/0"] = list(binary_cols)

    text_cols = X.select_dtypes(exclude=["number", "datetime"]).columns
    unique_text_cols = text_cols[X[text_cols].nunique() == X[text_cols].notna().sum()]
    X = X.drop(columns=unique_text_cols)
    report["Removed text columns with unique value in every non-empty row"] = list(unique_text_cols)

    text_cols = X.select_dtypes(exclude=["number", "datetime"]).columns
    cols_with_blanks = text_cols[X[text_cols].isna().any()]
    X[cols_with_blanks] = X[cols_with_blanks].fillna("Missing")
    report["Filled blanks with 'Missing'"] = list(cols_with_blanks)

    return X, report

def detect_task(y):
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15:
        return "regression"
    return "classification"

def engineer_features(X):
    """Convert any existing features into their compatible form for learning"""
    report = {}

    date_cols = X.select_dtypes(include="datetime").columns
    for col in date_cols:
        X[f"{col}_year"] = X[col].dt.year
        X[f"{col}_month"] = X[col].dt.month
        X[f"{col}_day"] = X[col].dt.day
        X[f"{col}_weekday"] = X[col].dt.dayofweek
    X = X.drop(columns=date_cols)
    report["Converted datetime columns to separate year, month, day and weekday columns"] = list(date_cols)

    text_cols = X.select_dtypes(exclude=["number", "datetime"]).columns
    X[text_cols] = X[text_cols].astype("category")
    report["Converted remaining text columns to category columns"] = list(text_cols)

    return X, report

def prediction_error(predictions, y_true, task):
    """One number where lower is better: share of wrong answers, or average distance from the truth."""
    if task == "classification":
        return float((np.asarray(predictions) != y_true.to_numpy()).mean())
    return float(np.abs(np.asarray(predictions) - y_true.to_numpy()).mean())

def column_groups(columns, date_cols):
    """Map each original column to the model columns made from it (a date becomes year/month/day/weekday)."""
    groups = {}
    for col in columns:
        source = next((d for d in date_cols if col.startswith(f"{d}_")), col)
        groups.setdefault(source, []).append(col)
    return groups


def single_column_check(X_train, y_train, task, groups, max_rows=5000):
    """Train a tiny model on each column ALONE. A column that predicts the target almost perfectly by itself
    is usually the answer in disguise (leakage). Uses training rows only, so the test rows stay untouched.
    Blanks count too: they stay as "Missing" or NaN, so a column whose blanks line up with the answer is caught."""
    if len(X_train) > max_rows:
        X_train = X_train.sample(max_rows, random_state=42)
        y_train = y_train.loc[X_train.index]
    if len(X_train) < 100:   # too few rows for a reliable check
        return {}
    stratify = y_train if task == "classification" and y_train.value_counts().min() >= 2 else None
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=stratify)

    guess = y_fit.mode()[0] if task == "classification" else y_fit.mean()
    guess_error = prediction_error(np.full(len(y_val), guess, dtype=object if task == "classification" else float),
                                   y_val, task)
    if guess_error == 0:
        return {}

    results = {}
    for source, cols in groups.items():
        if any(isinstance(X_fit[c].dtype, pd.CategoricalDtype) for c in cols):
            # Text column: look up the most common answer (or the average) for each value
            key_fit, key_val = X_fit[cols[0]].astype(str), X_val[cols[0]].astype(str)
            lookup = y_fit.groupby(key_fit).agg(lambda v: v.mode()[0]) if task == "classification" \
                else y_fit.groupby(key_fit).mean()
            predictions = key_val.map(lookup).fillna(guess).to_numpy()
        else:
            # Number/date column: a decision tree splits on exact values, so near-copies of the target show up
            model = DecisionTreeClassifier(min_samples_leaf=5, random_state=0) if task == "classification" \
                else DecisionTreeRegressor(min_samples_leaf=5, random_state=0)
            model.fit(X_fit[cols], y_fit)
            predictions = model.predict(X_val[cols])
        error = prediction_error(predictions, y_val, task)
        results[source] = {"strength": (guess_error - error) / guess_error, "error": error, "guess_error": guess_error}
    return results


def describe_alone(task, error, guess_error):
    if task == "classification":
        return f"On its own it gets {1 - error:.0%} right (always guessing the most common answer: {1 - guess_error:.0%})."
    return f"On its own it is off by {error:,.2f} on average (always guessing the average: {guess_error:,.2f})."


def column_importance(model, X_test, y_test, task, date_cols, repeats=3, max_rows=2000):
    """How much the model relies on each column: shuffle that column's values in the test rows
    and measure how much worse the predictions get. Works with any model that has .predict()."""
    if len(X_test) > max_rows:
        X_test = X_test.sample(max_rows, random_state=42)
        y_test = y_test.loc[X_test.index]

    # The year/month/day/weekday parts of a date are shuffled together and reported as the original date column
    groups = column_groups(X_test.columns, date_cols)

    base_error = prediction_error(model.predict(X_test), y_test, task)
    rng = np.random.default_rng(42)
    rise = {}
    for source, cols in groups.items():
        total = 0.0
        for _ in range(repeats):
            order = rng.permutation(len(X_test))
            shuffled = X_test.copy()
            for col in cols:
                values = X_test[col].iloc[order]   # keeps the column's type (category stays category)
                values.index = X_test.index
                shuffled[col] = values
            total += prediction_error(model.predict(shuffled), y_test, task) - base_error
        rise[source] = total / repeats

    helpful = {col: r for col, r in rise.items() if r > 0}
    total = sum(helpful.values())
    if total == 0:
        return []
    ranked = sorted(helpful.items(), key=lambda item: item[1], reverse=True)
    return [{"column": col, "share": r / total} for col, r in ranked]

def describe_test_results(predictions, y_test, task, max_points=500):
    """Predicted vs actual for the test rows, in a small JSON-friendly form for charts.
    Numbers: up to 500 (actual, predicted) pairs. Categories: how often each answer was predicted correctly."""
    predictions, actual = np.asarray(predictions), y_test.to_numpy()
    if task == "regression":
        keep = np.random.default_rng(42).permutation(len(actual))[:max_points]
        return {"actual": [float(v) for v in actual[keep]], "predicted": [float(v) for v in predictions[keep]]}
    answers = pd.Series(actual).value_counts().index[:8]   # the 8 most common answers
    return [{"answer": str(a), "count": int((actual == a).sum()),
             "correct": int(((actual == a) & (predictions == a)).sum())} for a in answers]


def run_automl(df, target, time_budget=60, task=None, exclude=(), keep=(), progress=None):
    """The whole job: clean, prepare, train, score. Returns a plain dict any interface can display.
    task can be "classification", "regression", or None to detect it automatically.
    exclude: columns the user chose to leave out. keep: columns the user wants kept even if they look like leakage.
    progress: optional function, called with the number of each step in STEPS as it starts."""
    def step(number):
        if progress:
            progress(number)

    if target not in df.columns:
        raise ValueError(f"Column '{target}' not found.")

    step(0)
    # Identical rows could land in both training and testing, which makes the score look better than it is.
    # Row-number columns ("Unnamed: 0") are ignored here, otherwise they make every row look different.
    real_cols = [c for c in df.columns if not str(c).startswith("Unnamed:")]
    duplicates = df.duplicated(subset=real_cols)
    df = df[~duplicates]
    
    step(1)
    y = try_numeric(df[target])
    user_excluded = [c for c in exclude if c in df.columns and c != target]
    X = df.drop(columns=[target, *user_excluded])
    X, cleaning_report = clean_features(X)
    cleaning_report = {"Removed duplicate rows": [f"{duplicates.sum():,} rows"] if duplicates.any() else [],
                       "Left out by you": user_excluded,
                       **cleaning_report}
    if X.shape[1] == 0:
        raise ValueError("No usable columns left to learn from after cleaning.")

    task = task or detect_task(y.dropna())
    if task == "regression":
        y = pd.to_numeric(y, errors="coerce")
    has_target = y.notna()
    X, y = X[has_target], y[has_target]
    if len(y) < MIN_ROWS:
        raise ValueError(f"Only {len(y)} rows have a usable value in '{target}'. At least {MIN_ROWS} are needed.")
    if task == "classification":
        y = y.astype(str)
        counts = y.value_counts()
        if counts.min() < 2:
            rare = ", ".join(map(str, counts[counts < 2].index[:5]))
            raise ValueError(f"Some values of '{target}' appear only once ({rare}). "
                             "Each category needs at least 2 rows. Is this really a category?")

    step(2)
    X_clean = X.copy()   # cleaned but not yet prepared for the model: used for the data analysis (EDA)
    date_cols = list(X.select_dtypes(include="datetime").columns)
    X, engineering_report = engineer_features(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task == "classification" else None)

    step(3)
    # Leakage check: leave out columns that give away the answer on their own, flag ones that come close
    groups = column_groups(X.columns, date_cols)
    checks = single_column_check(X_train, y_train, task, groups)
    leakage = []
    for source, c in sorted(checks.items(), key=lambda item: item[1]["strength"], reverse=True):
        if c["strength"] < LEAK_WARN_AT:
            continue
        excluded = c["strength"] >= LEAK_EXCLUDE_AT and source not in keep
        leakage.append({"column": source, "excluded": excluded, "kept_by_you": source in keep,
                        "detail": describe_alone(task, c["error"], c["guess_error"])})
    leaked_cols = [col for item in leakage if item["excluded"] for col in groups[item["column"]]]
    if len(leaked_cols) == X_train.shape[1]:
        raise ValueError("Every column predicts the answer almost perfectly on its own, which usually means they "
                         "all contain the answer. Check the data, or choose to keep the columns you trust.")
    X_train, X_test = X_train.drop(columns=leaked_cols), X_test.drop(columns=leaked_cols)

    step(4)
    automl = AutoML()
    automl.fit(X_train=X_train, y_train=y_train, task=task, time_budget=time_budget, verbose=0)
    step(5)
    predictions = automl.predict(X_test)

    if task == "classification":
        score = float((predictions == y_test.to_numpy()).mean())
        baseline = float((y_test == y_train.mode()[0]).mean())
        summary = (f"Model Accuracy: {score:.0%}\n"
                   f"Baseline (always guessing most common answer): {baseline:.0%}")
    else:
        score = float(abs(predictions - y_test.to_numpy()).mean())
        baseline = float(abs(y_test - y_train.mean()).mean())
        summary = (f"Model's Average Error: {score:,.2f}\n"
                   f"Baseline (always guessing average): {baseline:,.2f}")

    step(6)
    importance = column_importance(automl, X_test, y_test, task, date_cols)
    test_results = describe_test_results(predictions, y_test, task)
    step(7)
    eda = explore(df, X_clean, y, task, target, cleaning_report, importance)
    warnings = []
    error, guess_error = (1 - score, 1 - baseline) if task == "classification" else (score, baseline)
    if guess_error > 0 and error <= 0.05 * guess_error:
        top = ", ".join(f"'{i['column']}'" for i in importance[:3])
        relied_on = f" The model relied mostly on {top}." if top else ""
        warnings.append("This score looks too good to be true." + relied_on +
                        " Check whether these are only known after the thing you're predicting has happened, "
                        "or whether the same item appears in many rows.")
    flagged = {item["column"] for item in leakage}
    if importance and importance[0]["share"] >= 0.8 and importance[0]["column"] not in flagged:
        warnings.append(f"The predictions rely almost entirely on '{importance[0]['column']}' "
                        f"({importance[0]['share']:.0%}). Make sure it's known before the answer is.")

    return {
        "target": target,
        "task": task,
        "best_model": str(automl.best_estimator),
        "summary": summary,
        "cleaning": {k: v for k, v in cleaning_report.items() if v},
        "engineering": {k: v for k, v in engineering_report.items() if v},
        "importance": importance,
        "leakage": leakage,
        "warnings": warnings,
        "rows_used": int(len(X)),
        "score": score,
        "baseline": baseline,
        "test_rows": int(len(y_test)),
        "most_common": str(y_train.mode()[0]) if task == "classification" else None,
        "test_results": test_results,
        "eda": eda,
        "model": automl,
    }
