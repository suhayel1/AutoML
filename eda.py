"""
eda.py - Exploratory data analysis: describes the data itself in plain language, for the first part of the report.

explore() returns a JSON-friendly dict (no NaN, only plain numbers and text) with:
  overview      what's in the file: rows, columns, date range, and a line per column
  missing       which columns have blank cells
  target        what the values of the predicted column look like
  trend         how the predicted column changes over time (when there is a date column)
  pattern       a repeating pattern: by month of the year, or by day of the week
  drivers       how the predicted column differs across the most important columns
  correlations  which number columns rise and fall together with the predicted column
Each section carries a "text" with the main finding, written from the numbers.
"""
import math

import numpy as np
import pandas as pd

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Short reasons for columns that were not used, keyed by the cleaning step's name
REMOVAL_REASONS = {
    "Left out by you": "left out by you",
    "Removed columns with no header name": "no column name (usually a row number)",
    "Removed columns with no values": "completely empty",
    "Removed columns with only 1 unique value": "same value in every row",
}


# ---------- formatting ----------

def clean(value):
    """Plain float for JSON, with NaN/inf turned into None."""
    value = float(value)
    return None if math.isnan(value) or math.isinf(value) else value


def compact(value):
    """1234567 -> 1.2M, 4500 -> 4.5K, 12.345 -> 12.35"""
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:,.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"


def whole_numbers(s):
    s = s.dropna()
    return pd.api.types.is_numeric_dtype(s) and len(s) > 0 and bool((s == s.round()).all())


def number_label(value, whole=False, year_like=False):
    if year_like:
        return str(int(value))
    if whole and abs(value) < 10000:
        return f"{int(value):,}"
    return compact(value)


def day(ts, with_year=False):
    return f"{ts:%b} {ts.day}" + (f", {ts.year}" if with_year else "")


def flat(values, info):
    """True when the highest and lowest values are too close to be worth describing."""
    gap = max(values) - min(values)
    if info["kind"] == "share":
        return gap < 0.03
    middle = abs(np.mean(values)) or 1
    return gap / middle < 0.05


def label_like(s, m, min_rows):
    """A number column that is really a label (store numbers, region codes): grouping by each value explains
    the target far better than splitting the numbers into ranges."""
    if not whole_numbers(s) or not 12 < s.nunique() <= 100:
        return False
    total = m.var()
    if not total > 0:
        return False
    def explained(keys):
        g = m.groupby(keys, observed=True)
        means, counts = g.transform("mean"), g.transform("count")
        keep = counts >= min_rows
        return ((means[keep] - m[keep].mean()) ** 2).mean() / total if keep.any() else 0
    by_value = explained(s)
    by_range = explained(pd.qcut(s, q=5, duplicates="drop"))
    return by_value >= 0.1 and by_value >= 1.5 * by_range


def label_of(value, yes_no=False):
    """How a group's value is shown: 1.0 -> '1', yes/no columns -> 'Yes'/'No'."""
    if yes_no and value in (0, 1):
        return "Yes" if value == 1 else "No"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def make_metric(y, task, target):
    """The number every chart in the EDA shows for a group of rows:
    the average of the target (numbers), or the share of rows with one particular answer (categories)."""
    if task == "regression":
        info = {"kind": "average", "label": f"Average {target}", "short": f"the average {target}",
                "having": f"an average {target} of {{}}", "higher": f"tend to have a higher {target}",
                "lower": f"tend to have a lower {target}"}
        return y.astype(float), info
    counts = y.value_counts()
    focus = counts.index[-1] if len(counts) == 2 else counts.index[0]   # the rarer of two answers, else the most common
    info = {"kind": "share", "label": f"Share where {target} is '{focus}'",
            "short": f"the share of rows where {target} is '{focus}'", "focus": str(focus),
            "having": f"{target} '{focus}' in {{}} of cases", "higher": f"are more likely to have {target} '{focus}'",
            "lower": f"are less likely to have {target} '{focus}'"}
    return (y == focus).astype(float), info


def fmt(value, info):
    return f"{value:.0%}" if info["kind"] == "share" else compact(value)


def difference(new, old, info):
    """'12% higher' for averages, '5 percentage points higher' for shares."""
    if info["kind"] == "share":
        points = (new - old) * 100
        return f"{abs(points):.0f} percentage points {'higher' if points >= 0 else 'lower'}"
    if old == 0:
        return "different"
    change = (new - old) / abs(old)
    return f"{abs(change):.0%} {'higher' if change >= 0 else 'lower'}"


# ---------- sections ----------

def main_date_column(X):
    dates = X.select_dtypes(include="datetime")
    return dates.nunique().idxmax() if dates.shape[1] else None


def overview(raw, X, target, cleaning_report, max_rows=30):
    removed = {}
    for step, cols in cleaning_report.items():
        for col in cols:
            if step in REMOVAL_REASONS:
                removed[col] = REMOVAL_REASONS[step]
            elif step.startswith("Removed"):
                removed[col] = "a different value in almost every row (like an ID)"
    yes_no = set(cleaning_report.get("Converted yes/no and true/false columns to 1/0", []))

    table = []
    for col in raw.columns:
        filled = clean(raw[col].notna().mean())
        if col == target:
            kind, summary = "Predicted column", describe_values(raw[col])
        elif col in X.columns:
            s = X[col]
            if pd.api.types.is_datetime64_any_dtype(s):
                kind = "Date"
                span = (s.max() - s.min()).days
                f = "%b %d, %Y" if span < 400 else "%b %Y"
                summary = f"{s.min():{f}} to {s.max():{f}}"
            elif col in yes_no:
                kind, summary = "Yes / no", f"{s.mean():.0%} yes"
            else:
                kind, summary = ("Number" if pd.api.types.is_numeric_dtype(s) else "Category"), describe_values(s)
        else:
            kind, summary = "Not used", removed.get(col, "not used")
        table.append({"column": str(col), "kind": kind, "filled": filled, "summary": summary})

    date_col = main_date_column(X)
    date_range = None
    if date_col is not None:
        d = X[date_col]
        date_range = {"column": str(date_col), "start": d.min().strftime("%B %d, %Y").replace(" 0", " "),
                      "end": d.max().strftime("%B %d, %Y").replace(" 0", " ")}
    text = f"The file has {len(raw):,} rows and {raw.shape[1]} columns."
    if date_range:
        text += f" Based on '{date_range['column']}', it covers {date_range['start']} to {date_range['end']}."
    unused = sum(1 for row in table if row["kind"] == "Not used")
    if unused:
        text += f" {unused} column{'s were' if unused > 1 else ' was'} not used; the table shows why."
    return {"rows": len(raw), "columns": raw.shape[1], "date_range": date_range,
            "table": table[:max_rows], "more": max(0, len(table) - max_rows), "text": text}


def describe_values(s):
    s = s.dropna()
    if s.empty:
        return "no values"
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return f"{compact(s.min())} to {compact(s.max())}, average {compact(s.mean())}"
    counts = s.astype(str).value_counts()
    return f"{len(counts):,} different values; most common '{str(counts.index[0])[:30]}' ({counts.iloc[0] / len(s):.0%})"


def missing(raw):
    share = raw.isna().mean()
    gaps = share[share > 0].sort_values(ascending=False)
    if gaps.empty:
        return {"items": [], "text": "Every column is complete: there are no blank cells."}
    items = [{"column": str(c), "share": clean(v)} for c, v in gaps.head(10).items()]
    text = (f"{len(gaps)} of {raw.shape[1]} columns {'has' if len(gaps) == 1 else 'have'} blank cells. The most gaps are in "
            f"'{gaps.index[0]}' ({gaps.iloc[0]:.0%} blank).")
    full = gaps[gaps >= 0.999]
    if len(full):
        text += f" {len(full)} column{'s are' if len(full) > 1 else ' is'} completely empty."
    return {"items": items, "text": text}


def target_summary(y, task, target):
    if task == "classification":
        counts = y.value_counts()
        shares = counts / len(y)
        items = [{"label": str(k), "count": int(counts[k]), "share": clean(shares[k])} for k in counts.index[:8]]
        if len(counts) > 8:
            items.append({"label": f"{len(counts) - 8} other answers", "count": int(counts.iloc[8:].sum()),
                          "share": clean(shares.iloc[8:].sum())})
        text = (f"'{target}' has {len(counts)} different answers. The most common is '{counts.index[0]}' "
                f"({shares.iloc[0]:.0%} of rows)")
        text += f", and '{counts.index[1]}' makes up the other {shares.iloc[1]:.0%}." if len(counts) == 2 else "."
        if shares.iloc[0] >= 0.85:
            text += (f" Because one answer is so common, a prediction model has to be right more than "
                     f"{shares.iloc[0]:.0%} of the time to beat simply guessing it.")
        return {"kind": "category", "items": items, "text": text}

    y = y.astype(float)
    q = y.quantile([0, 0.25, 0.5, 0.75, 0.99, 1]).tolist()
    stats = {"min": q[0], "p25": q[1], "median": q[2], "p75": q[3], "max": q[5], "mean": float(y.mean())}
    # Leave extreme values out of the chart so they don't squash everything else into one bar
    upper = q[5]
    iqr = q[3] - q[1]
    clipped = iqr > 0 and q[5] > q[3] + 5 * iqr
    if clipped:
        upper = q[4]
    counts, edges = np.histogram(y[y <= upper], bins=min(30, max(5, int(np.sqrt(len(y))))))
    text = (f"'{target}' ranges from {compact(stats['min'])} to {compact(stats['max'])}. Half of the rows fall "
            f"between {compact(stats['p25'])} and {compact(stats['p75'])}, and the average is {compact(stats['mean'])}.")
    if stats["median"] > 0 and stats["mean"] > stats["median"] * 1.15:
        text += (f" The average is above the typical (middle) value of {compact(stats['median'])} because "
                 f"a smaller number of rows have very high values.")
    elif stats["median"] > 0 and stats["mean"] < stats["median"] * 0.85:
        text += (f" The average is below the typical (middle) value of {compact(stats['median'])} because "
                 f"a smaller number of rows have very low values.")
    return {"kind": "number", "stats": {k: clean(v) for k, v in stats.items()},
            "hist": {"edges": [clean(e) for e in edges], "counts": [int(c) for c in counts]},
            "clipped_above": clean(upper) if clipped else None, "text": text}


def trend(X, m, info):
    col = main_date_column(X)
    if col is None:
        return None
    d = X[col]
    span = (d.max() - d.min()).days
    if span < 14:
        return None
    if span <= 92:
        freq, name, show = "D", "day", lambda t: day(t)
    elif span <= 730:
        freq, name, show = "W", "week", lambda t: "week of " + day(t, with_year=True)
    elif span <= 5500:
        freq, name, show = "M", "month", lambda t: f"{t:%b %Y}"
    else:
        freq, name, show = "Y", "year", lambda t: f"{t:%Y}"
    valid = d.notna()
    grouped = m[valid].groupby(d[valid].dt.to_period(freq)).agg(["mean", "count"])
    # Leave out periods with very few rows: their averages jump around and mislead
    grouped = grouped[grouped["count"] >= max(5, 0.25 * grouped["count"].median())]
    if len(grouped) < 3:
        return None
    points = [{"date": p.start_time.strftime("%Y-%m-%d"), "label": show(p.start_time),
               "value": clean(r["mean"]), "rows": int(r["count"])} for p, r in grouped.iterrows()]
    if flat(grouped["mean"].tolist(), info):
        text = (f"Looking at each {name}, {info['short']} stayed about the same throughout "
                f"(around {fmt(grouped['mean'].mean(), info)}).")
        return {"column": str(col), "period": name, "points": points, "text": text}
    best, worst = grouped["mean"].idxmax(), grouped["mean"].idxmin()
    edge = max(1, len(grouped) // 4)
    early, late = grouped["mean"].iloc[:edge].mean(), grouped["mean"].iloc[-edge:].mean()
    when = "on" if name == "day" else "in the" if name == "week" else "in"
    text = (f"Looking at each {name}, {info['short']} was highest {when} {show(best.start_time)} "
            f"({fmt(grouped.loc[best, 'mean'], info)}) and lowest {when} {show(worst.start_time)} "
            f"({fmt(grouped.loc[worst, 'mean'], info)}). The most recent {name}s are "
            f"{difference(late, early, info)} than the earliest ones.")
    return {"column": str(col), "period": name, "points": points, "text": text}


def pattern(X, m, info):
    col = main_date_column(X)
    if col is None:
        return None
    d = X[col]
    span = (d.max() - d.min()).days
    if span >= 730 and d.dt.month.nunique() == 12:
        key, names, unit = d.dt.month, MONTHS, "month"
    elif span >= 28 and d.dt.dayofweek.nunique() >= 5:
        key, names, unit = d.dt.dayofweek, WEEKDAYS, "day of the week"
    else:
        return None
    grouped = m.groupby(key).agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= 5]
    if len(grouped) < 5:
        return None
    overall = m.mean()
    if flat(grouped["mean"].tolist(), info):
        return {"unit": unit, "overall": clean(overall), "points": [],
                "text": f"There is no clear pattern by {unit}: {info['short']} is about the same throughout."}
    best, worst = grouped["mean"].idxmax(), grouped["mean"].idxmin()
    names_of = (lambda k: names[k - 1]) if unit == "month" else (lambda k: names[k])
    text = (f"On average, {names_of(best)} is the strongest {unit} ({fmt(grouped.loc[best, 'mean'], info)}, "
            f"{difference(grouped.loc[best, 'mean'], overall, info)} than the overall average) and "
            f"{names_of(worst)} the weakest ({fmt(grouped.loc[worst, 'mean'], info)}, "
            f"{difference(grouped.loc[worst, 'mean'], overall, info)}).")
    return {"unit": unit, "overall": clean(overall), "text": text,
            "points": [{"label": names_of(k), "value": clean(r["mean"]), "rows": int(r["count"])}
                       for k, r in grouped.iterrows()]}


def drivers(X, m, info, importance, yes_no, max_items=4):
    """For the columns the model relied on most: how the metric differs between groups of rows."""
    candidates = [i["column"] for i in importance if i["column"] in X.columns
                  and not pd.api.types.is_datetime64_any_dtype(X[i["column"]])]
    if not candidates:   # no importance available: fall back to the columns in file order
        candidates = [c for c in X.columns if not pd.api.types.is_datetime64_any_dtype(X[c])]
    results = []
    min_rows = max(5, int(0.005 * len(X)))
    for col in candidates:
        if len(results) == max_items:
            break
        s = X[col]
        if (pd.api.types.is_numeric_dtype(s) and s.nunique() > 12 and col not in yes_no
                and not label_like(s, m, min_rows)):
            # Number column: split into 5 equal-sized ranges, from low to high
            ranges = pd.qcut(s, q=5, duplicates="drop")
            grouped = m.groupby(ranges, observed=True).agg(["mean", "count"])
            whole = whole_numbers(s)
            year_like = whole and s.min() >= 1800 and s.max() <= 2100
            items = []
            for i, (iv, r) in enumerate(grouped.iterrows()):
                left = s.min() if i == 0 else (np.floor(iv.left) + 1 if whole else iv.left)
                items.append({"label": f"{number_label(left, whole, year_like)} to {number_label(iv.right, whole, year_like)}",
                              "value": clean(r["mean"]), "count": int(r["count"])})
            if len(items) < 2:
                continue
            low, high = items[0], items[-1]
            text = (f"Rows with the highest '{col}' ({high['label']}) have "
                    f"{info['having'].format(fmt(high['value'], info))}, compared with {fmt(low['value'], info)} "
                    f"for the lowest ({low['label']}).")
            results.append({"column": str(col), "kind": "ranges", "items": items, "note": None, "text": text})
            continue
        # Category (or a number with only a few values): one group per value
        grouped = m.groupby(s.astype(object).where(s.notna(), "Missing")).agg(["mean", "count"])
        grouped = grouped[grouped["count"] >= min_rows].sort_values("mean", ascending=False)
        if len(grouped) < 2:
            continue
        note = None
        if len(grouped) > 10:
            note = (f"Showing the 5 highest and 5 lowest of {len(grouped)} groups "
                    f"(groups with at least {min_rows:,} rows).")
            grouped = pd.concat([grouped.head(5), grouped.tail(5)])
        items = [{"label": label_of(k, col in yes_no)[:40], "value": clean(r["mean"]), "count": int(r["count"])}
                 for k, r in grouped.iterrows()]
        top, bottom = items[0], items[-1]
        text = (f"{info['short'][0].upper() + info['short'][1:]} is highest when '{col}' is '{top['label']}' "
                f"({fmt(top['value'], info)}) and lowest when it is '{bottom['label']}' ({fmt(bottom['value'], info)}).")
        results.append({"column": str(col), "kind": "groups", "items": items, "note": note, "text": text})
    return results


def correlations(X, m, info, task, max_items=8):
    if task == "classification" and info["kind"] == "share" and m.nunique() < 2:
        return None
    min_rows = max(5, int(0.005 * len(X)))
    numbers = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and X[c].nunique() > 1
               and not label_like(X[c], m, min_rows)]
    scores = {c: X[c].corr(m, method="spearman") for c in numbers}
    scores = {c: r for c, r in scores.items() if r == r}   # drop NaN
    if len(scores) < 2:
        return None
    ranked = sorted(scores.items(), key=lambda item: abs(item[1]), reverse=True)[:max_items]
    items = [{"column": str(c), "r": clean(r)} for c, r in ranked]
    c, r = ranked[0]
    if abs(r) < 0.1:
        text = "None of the number columns has a clear link with the predicted column on its own."
    else:
        strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
        text = (f"The clearest link is with '{c}' ({strength}, {r:+.2f}): rows with a higher '{c}' "
                f"{info['higher'] if r > 0 else info['lower']}.")
    return {"items": items, "text": text}


def explore(raw, X, y, task, target, cleaning_report, importance):
    """raw: the uploaded data (after removing duplicate rows). X, y: the cleaned columns and the predicted column,
    for the rows that have a value to predict. Returns everything the report's EDA part needs."""
    m, info = make_metric(y, task, target)
    yes_no = set(cleaning_report.get("Converted yes/no and true/false columns to 1/0", []))
    return {
        "metric": info,
        "overview": overview(raw, X, target, cleaning_report),
        "missing": missing(raw),
        "target": target_summary(y, task, target),
        "trend": trend(X, m, info),
        "pattern": pattern(X, m, info),
        "drivers": drivers(X, m, info, importance, yes_no),
        "correlations": correlations(X, m, info, task),
    }
