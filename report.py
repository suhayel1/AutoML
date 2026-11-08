"""
report.py - Turns a finished result from run_automl() into a PDF that a non-technical client can read.

  At a glance  the main findings in a few bullet points
  Part 1       Exploring your data (EDA, from eda.py): what's in the file, blank cells, the predicted column,
               trends over time, what goes with higher/lower values, and which number columns move with it
  Part 2       The prediction model: how good it is, what to check, what it relied on, how close it was
  Appendix     How the data was prepared, and technical details
"""
import io
from datetime import date, datetime
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")   # draw charts to images, no window (also safe inside the server's threads)
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

# Colours: one blue for the model's data, a neutral grey for the "just guessing" comparison,
# dark/medium grey for text, and a light hairline grey for gridlines.
BLUE = "#2a78d6"
GUESS_GREY = "#8f8e88"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
HAIRLINE = "#e4e3de"
RED = "#e34948"          # negative side of the correlation chart (blue is positive)
PANEL_BG = "#f4f3ef"
WARN_BG = "#fff8e5"
WARN_EDGE = "#c98500"

PAGE_WIDTH = LETTER[0] - 1.5 * inch   # printable width with 0.75" margins

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": INK_SOFT,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
})

STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=10.5, leading=14, textColor=INK_SOFT),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=INK,
                         spaceBefore=16, spaceAfter=6),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=9, leading=12.5, textColor=INK_SOFT),
    "verdict": ParagraphStyle("verdict", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=INK),
    "part_label": ParagraphStyle("part_label", fontName="Helvetica-Bold", fontSize=9, leading=12,
                                 textColor=colors.HexColor(BLUE), spaceBefore=4),
    "part": ParagraphStyle("part", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=INK, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=INK,
                         spaceBefore=10, spaceAfter=3),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=11, textColor=INK),
    "cell_head": ParagraphStyle("cell_head", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=INK_SOFT),
}


# ---------- helpers ----------

def compact(value):
    """1234567 -> 1.2M, 4500 -> 4.5K, 12.345 -> 12.35"""
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:,.1f}".rstrip("0").rstrip(".") + suffix
    return f"{value:,.2f}".rstrip("0").rstrip(".") if value % 1 else f"{value:,.0f}"


def number(value):
    """Full number for sentences: 41.23, 55,580"""
    return f"{value:,.0f}" if abs(value) >= 1000 else f"{value:,.2f}"


def shorten(text, limit=30):
    text = str(text)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def value_formatter(info):
    if info["kind"] == "share":
        return PercentFormatter(1.0, decimals=0)
    return FuncFormatter(lambda v, _: compact(v))


def value_text(v, info):
    return f"{v:.0%}" if info["kind"] == "share" else compact(v)


def chart_image(fig):
    """Render a matplotlib figure into a ReportLab image at its real size (so 9pt text prints as 9pt)."""
    width, height = fig.get_size_inches()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor="white")
    buf.seek(0)
    return Image(buf, width=width * inch, height=height * inch)


def new_figure(height, width=PAGE_WIDTH / inch):
    return Figure(figsize=(width, height), layout="constrained", facecolor="white")


def tidy_axes(ax, value_axis="x"):
    """Recessive styling: no box, hairline grid along the value axis only."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(length=0)
    ax.grid(axis=value_axis, color=HAIRLINE, linewidth=0.8)
    ax.set_axisbelow(True)


def strength_of(result):
    """Share of the gap between 'just guessing' and a perfect score that the model closes (1 = perfect)."""
    if result["task"] == "classification":
        error, guess_error = 1 - result["score"], 1 - result["baseline"]
    else:
        error, guess_error = result["score"], result["baseline"]
    return (guess_error - error) / guess_error if guess_error > 0 else 0.0


def verdict(result):
    if any("too good to be true" in w for w in result.get("warnings", [])):
        return "Check first: this result looks too good to be true. See \"Before you rely on this\" below."
    s = strength_of(result)
    if s >= 0.5:
        return "Strong: the model is much better than guessing."
    if s >= 0.2:
        return "Moderate: the model is clearly better than guessing, but its predictions will often be off."
    if s > 0:
        return "Weak: the model is only slightly better than guessing. Use its predictions with caution."
    return "Not useful yet: the model is no better than guessing."


# ---------- charts ----------

def score_chart(result):
    """Two bars: the model and 'always guessing'."""
    classification = result["task"] == "classification"
    labels = ["Our model", "Always guessing the most common answer" if classification
              else "Always guessing the average"]
    values = [result["score"], result["baseline"]]

    fig = new_figure(1.55)
    ax = fig.add_subplot()
    bars = ax.barh([1, 0], values, height=0.5, color=[BLUE, GUESS_GREY])
    ax.set_yticks([1, 0], labels, fontsize=9.5, color=INK)
    tidy_axes(ax)
    top = max(values) * 1.18 if not classification else 1.12
    ax.set_xlim(0, top)
    if classification:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: compact(v)))
    for bar, value in zip(bars, values):
        text = f"{value:.0%}" if classification else number(value)
        ax.text(bar.get_width() + top * 0.012, bar.get_y() + bar.get_height() / 2, text,
                va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax.set_xlabel("Share of answers that were right (longer is better)" if classification
                  else "Average error (shorter is better)", fontsize=8.5)
    return chart_image(fig)


def importance_chart(importance, max_items=8):
    items = importance[:max_items]
    fig = new_figure(0.36 * len(items) + 0.6)
    ax = fig.add_subplot()
    positions = list(range(len(items)))[::-1]
    shares = [i["share"] for i in items]
    ax.barh(positions, shares, height=0.55, color=BLUE)
    ax.set_yticks(positions, [shorten(i["column"]) for i in items], fontsize=9, color=INK)
    tidy_axes(ax)
    ax.set_xlim(0, max(shares) * 1.15)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for y, share in zip(positions, shares):
        ax.text(share + max(shares) * 0.012, y, "<1%" if share < 0.005 else f"{share:.0%}",
                va="center", fontsize=9, color=INK)
    ax.set_xlabel("Share of the model's reliance", fontsize=8.5)
    return chart_image(fig)


def predicted_vs_actual_chart(test_results, target):
    actual, predicted = test_results["actual"], test_results["predicted"]
    low, high = min(actual + predicted), max(actual + predicted)
    pad = (high - low) * 0.04 or 1
    fig = new_figure(3.0, width=3.4)
    ax = fig.add_subplot()
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=GUESS_GREY, linewidth=1.2, zorder=1)
    ax.scatter(actual, predicted, s=22, color=BLUE, alpha=0.55, edgecolors="white", linewidths=0.6, zorder=2)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(HAIRLINE)
    ax.tick_params(length=0)
    ax.grid(color=HAIRLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    fmt = FuncFormatter(lambda v, _: compact(v))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel(f"Actual {shorten(target, 40)}", fontsize=8.5)
    ax.set_ylabel(f"Predicted {shorten(target, 40)}", fontsize=8.5)
    return chart_image(fig)


def per_answer_chart(test_results):
    items = test_results
    fig = new_figure(0.36 * len(items) + 0.6)
    ax = fig.add_subplot()
    positions = list(range(len(items)))[::-1]
    rates = [i["correct"] / i["count"] for i in items]
    ax.barh(positions, rates, height=0.55, color=BLUE)
    ax.set_yticks(positions, [f"{shorten(i['answer'], 26)}  ({i['count']:,} rows)" for i in items],
                  fontsize=9, color=INK)
    tidy_axes(ax)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for y, rate in zip(positions, rates):
        ax.text(rate + 0.012, y, f"{rate:.0%}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Share predicted correctly", fontsize=8.5)
    return chart_image(fig)


# ---------- EDA charts ----------

def hbar_chart(labels, values, info=None, gaps_after=None, xlabel=None, percent=False):
    """Horizontal bars, top to bottom in the given order. gaps_after: index after which to leave a gap
    (for 'top 5 ... bottom 5' charts)."""
    as_share = percent or (info is not None and info["kind"] == "share")
    positions, y = [], 0
    for i in range(len(values)):
        positions.append(y)
        y -= 1.7 if gaps_after is not None and i == gaps_after else 1
    fig = new_figure(0.33 * (len(values) + (0.7 if gaps_after is not None else 0)) + 0.6)
    ax = fig.add_subplot()
    ax.barh(positions, values, height=0.6, color=BLUE)
    ax.set_yticks(positions, [shorten(l, 34) for l in labels], fontsize=8.5, color=INK)
    tidy_axes(ax)
    top = max(max(values), 0) or 1
    ax.set_xlim(0, top * 1.16)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0) if as_share
                                 else FuncFormatter(lambda v, _: compact(v)))
    for pos, value in zip(positions, values):
        text = ("<1%" if 0 < value < 0.005 else f"{value:.0%}") if as_share else compact(value)
        ax.text(value + top * 0.012, pos, text, va="center", fontsize=8.5, color=INK)
    if gaps_after is not None:
        ax.text(top * 0.01, positions[gaps_after] - 0.85, "· · ·", va="center", fontsize=9, color=INK_SOFT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5)
    return chart_image(fig)


def histogram_chart(hist, target):
    edges, counts = hist["edges"], hist["counts"]
    fig = new_figure(2.1)
    ax = fig.add_subplot()
    widths = [b - a for a, b in zip(edges[:-1], edges[1:])]
    ax.bar(edges[:-1], counts, width=widths, align="edge", color=BLUE, edgecolor="white", linewidth=0.8)
    tidy_axes(ax, value_axis="y")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: compact(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: compact(v)))
    ax.set_xlabel(shorten(target, 50), fontsize=8.5)
    ax.set_ylabel("Number of rows", fontsize=8.5)
    return chart_image(fig)


def trend_chart(trend, info):
    points = trend["points"]
    dates = [datetime.strptime(pt["date"], "%Y-%m-%d") for pt in points]
    values = [pt["value"] for pt in points]
    fig = new_figure(2.3)
    ax = fig.add_subplot()
    average = sum(v * pt["rows"] for v, pt in zip(values, points)) / sum(pt["rows"] for pt in points)
    ax.axhline(average, color=GUESS_GREY, linewidth=1, zorder=1)
    ax.plot(dates, values, color=BLUE, linewidth=2, solid_joinstyle="round", solid_capstyle="round", zorder=2,
            marker="o" if len(points) <= 40 else None, markersize=4.5, markeredgecolor="white", markeredgewidth=1)
    ax.plot(dates[-1:], values[-1:], "o", color=BLUE, markersize=6, markeredgecolor="white",
            markeredgewidth=1.2, zorder=3)
    tidy_axes(ax, value_axis="y")
    ax.yaxis.set_major_formatter(value_formatter(info))
    locator = mdates.AutoDateLocator(maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_ylabel(shorten(info["label"], 45), fontsize=8.5)
    return chart_image(fig)


def pattern_chart(pattern, info):
    points = pattern["points"]
    labels, values = [pt["label"] for pt in points], [pt["value"] for pt in points]
    fig = new_figure(2.1)
    ax = fig.add_subplot()
    ax.bar(range(len(values)), values, width=0.62, color=BLUE)
    ax.axhline(pattern["overall"], color=GUESS_GREY, linewidth=1)
    ax.set_xticks(range(len(values)), labels, fontsize=8.5, color=INK)
    tidy_axes(ax, value_axis="y")
    ax.yaxis.set_major_formatter(value_formatter(info))
    top = max(values)
    ax.set_ylim(0, top * 1.15)
    for i in {values.index(max(values)), values.index(min(values))}:   # label only the highest and lowest
        ax.text(i, values[i] + top * 0.02, value_text(values[i], info), ha="center", fontsize=8.5, color=INK)
    ax.set_ylabel(shorten(info["label"], 45), fontsize=8.5)
    return chart_image(fig)


def correlation_chart(items):
    labels, values = [i["column"] for i in items], [i["r"] for i in items]
    positions = list(range(len(items)))[::-1]
    fig = new_figure(0.33 * len(items) + 0.7)
    ax = fig.add_subplot()
    ax.barh(positions, values, height=0.6, color=[BLUE if v >= 0 else RED for v in values])
    ax.axvline(0, color=INK_SOFT, linewidth=0.8)
    ax.set_yticks(positions, [shorten(l, 34) for l in labels], fontsize=8.5, color=INK)
    tidy_axes(ax)
    ax.set_xlim(-1.15, 1.15)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ["-1", "-0.5", "0", "+0.5", "+1"])
    for pos, v in zip(positions, values):
        ax.text(v + (0.03 if v >= 0 else -0.03), pos, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    ax.set_xlabel("Link score (-1 to +1)", fontsize=8.5)
    return chart_image(fig)


# ---------- text sections ----------

def summary_text(result):
    target = escape(result["target"])
    tested = f"{result['test_rows']:,}"
    if result["task"] == "classification":
        return (f"We built a model that predicts <b>{target}</b> from the other columns in your data. "
                f"We tested it on {tested} rows it had never seen. It chose the right answer "
                f"<b>{result['score']:.0%}</b> of the time. For comparison, always guessing the most common answer "
                f"(\"{escape(shorten(result['most_common'], 40))}\") would be right {result['baseline']:.0%} of the time.")
    smaller = strength_of(result)
    return (f"We built a model that predicts <b>{target}</b> from the other columns in your data. "
            f"We tested it on {tested} rows it had never seen. On average its predictions were off by "
            f"<b>{number(result['score'])}</b>. Always guessing the average would be off by "
            f"{number(result['baseline'])}, so the model's errors are <b>{max(smaller, 0):.0%} smaller</b> than guessing.")


def checks_box(result):
    """Leakage findings and warnings, in a light yellow box. Returns None when there is nothing to check."""
    lines = []
    for item in result.get("leakage", []):
        col = f"<b>{escape(item['column'])}</b>"
        detail = escape(item["detail"])
        if item["excluded"]:
            lines.append(f"{col} was left out, because it gives away the answer by itself. {detail}")
        elif item["kept_by_you"]:
            lines.append(f"{col} gives away the answer by itself, but was kept on request. {detail}")
        else:
            lines.append(f"{col} almost predicts the answer by itself. {detail} If it is only known after "
                         f"the outcome, the model should be retrained without it.")
    lines += [escape(w) for w in result.get("warnings", [])]
    if not lines:
        return None
    bullets = ListFlowable([ListItem(p(line), leftIndent=12, value="•") for line in lines],
                           bulletType="bullet", start="•", leftIndent=12)
    box = Table([[p("<b>Before you rely on this</b>")], [bullets]], colWidths=[PAGE_WIDTH])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(WARN_BG)),
        ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(WARN_EDGE)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
    ]))
    return box


def steps_list(steps, max_columns=8):
    items = []
    for step, cols in steps.items():
        names = [escape(str(c)) for c in cols]
        shown = ", ".join(names[:max_columns]) + (f" and {len(names) - max_columns} more" if len(names) > max_columns else "")
        items.append(ListItem(p(f"{escape(step)}: <font color='{INK_SOFT}'>{shown}</font>"), leftIndent=12, value="•"))
    return ListFlowable(items, bulletType="bullet", start="•", leftIndent=12) if items else p("No changes were needed.")


def details_table(result):
    rows = [
        ("Predicting", result["target"]),
        ("Type of prediction", "A category" if result["task"] == "classification" else "A number"),
        ("Best model found", result["best_model"]),
        ("Rows used", f"{result['rows_used']:,}"),
        ("Rows held back for testing", f"{result['test_rows']:,} (20%, never seen during training)"),
    ]
    if result.get("time_budget"):
        rows.append(("Time spent searching for a model", f"{result['time_budget']} seconds"))
    if result.get("file_name"):
        rows.append(("Data file", result["file_name"]))
    table = Table([[p(k, "small"), p(escape(str(v)), "small")] for k, v in rows],
                  colWidths=[2.3 * inch, PAGE_WIDTH - 2.3 * inch])
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor(HAIRLINE)),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def overview_table(overview):
    head = [p("Column", "cell_head"), p("Type", "cell_head"), p("Filled in", "cell_head"),
            p("What's in it", "cell_head")]
    rows = [head] + [[p(escape(shorten(r["column"], 32)), "cell"), p(escape(r["kind"]), "cell"),
                      p("" if r["filled"] is None else f"{r['filled']:.0%}", "cell"), p(escape(r["summary"]), "cell")]
                     for r in overview["table"]]
    table = Table(rows, colWidths=[1.85 * inch, 1.05 * inch, 0.7 * inch, PAGE_WIDTH - 3.6 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor(HAIRLINE)),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def part_heading(label, title):
    return [Spacer(1, 6), p(label.upper(), "part_label"), p(title, "part")]


def eda_story(eda, target):
    t = escape(target)
    info = eda["metric"]
    story = part_heading("Part 1", "Exploring your data")
    story.append(p("Before building any model, we looked at the data itself: what it contains, how complete it is, "
                   "and which patterns stand out."))

    ov = eda["overview"]
    story += [p("What's in your data", "h2"), p(escape(ov["text"])), Spacer(1, 6), overview_table(ov)]
    if ov["more"]:
        story.append(p(f"…and {ov['more']} more columns.", "small"))

    miss = eda["missing"]
    block = [p("Blank cells", "h2"), p(escape(miss["text"]))]
    if miss["items"]:
        block += [Spacer(1, 4), hbar_chart([i["column"] for i in miss["items"]], [i["share"] for i in miss["items"]],
                                           percent=True, xlabel="Share of rows left blank")]
    story.append(KeepTogether(block))

    tgt = eda["target"]
    block = [p(f"The values of {t}", "h2"), p(escape(tgt["text"]))]
    if tgt["kind"] == "number":
        note = "Each bar counts the rows whose value falls in that range."
        if tgt["clipped_above"]:
            note += (f" A few very high values (above {compact(tgt['clipped_above'])}) are left out of the chart "
                     f"so the rest stays readable.")
        block += [Spacer(1, 4), histogram_chart(tgt["hist"], target), p(note, "small")]
    else:
        block += [Spacer(1, 4), hbar_chart([f"{i['label']}  ({i['count']:,} rows)" for i in tgt["items"]],
                                           [i["share"] for i in tgt["items"]], percent=True, xlabel="Share of rows")]
    story.append(KeepTogether(block))

    if eda["trend"]:
        tr = eda["trend"]
        story.append(KeepTogether([
            p("Over time", "h2"), p(escape(tr["text"])), Spacer(1, 4), trend_chart(tr, info),
            p(f"Each point is one {tr['period']}, based on '{escape(tr['column'])}'. "
              f"The grey line is the overall average.", "small")]))
        pat = eda["pattern"]
        if pat:
            block = [p(f"By {pat['unit']}", "h3"), p(escape(pat["text"]))]
            if pat["points"]:
                block += [Spacer(1, 4), pattern_chart(pat, info)]
            story.append(KeepTogether(block))

    if eda["drivers"]:
        intro = [p(f"What goes with higher and lower {t}", "h2"),
                 p(f"These are the columns the prediction model found most useful (see Part 2). For each one, "
                   f"the chart compares {escape(info['short'])} between groups of rows.")]
        for n, d in enumerate(eda["drivers"]):
            items = d["items"]
            gap = 4 if d["note"] and len(items) == 10 else None
            block = [p(f"By {escape(d['column'])}", "h3"), p(escape(d["text"])), Spacer(1, 3),
                     hbar_chart([i["label"] for i in items], [i["value"] for i in items], info=info,
                                gaps_after=gap, xlabel=shorten(info["label"], 60))]
            if d["note"]:
                block.append(p(escape(d["note"]), "small"))
            story.append(KeepTogether(intro + block if n == 0 else block))   # keep the heading with the first chart

    corr = eda["correlations"]
    if corr:
        story.append(KeepTogether([
            p(f"Number columns that move with {t}", "h2"),
            p(escape(corr["text"])),
        ] + ([
            p("The link score runs from -1 to +1. Near +1 (blue bars): both tend to rise together. Near -1 (red bars): "
              "one tends to fall as the other rises. Near 0: no clear link. A link does not prove that one causes "
              "the other.", "small"),
            Spacer(1, 4), correlation_chart(corr["items"]),
        ] if abs(corr["items"][0]["r"]) >= 0.1 else [])))   # no chart when every link is close to zero
    return story


def glance_box(result):
    """The main findings in a few bullets, at the top of the report."""
    eda = result.get("eda") or {}
    lines = []
    if eda:
        ov = eda["overview"]
        line = f"{ov['rows']:,} rows and {ov['columns']} columns"
        if ov["date_range"]:
            line += f", covering {ov['date_range']['start']} to {ov['date_range']['end']}"
        lines.append(line + ".")
        lines.append(eda["target"]["text"].split(". ")[0].rstrip(".") + ".")
        pattern, trend = eda.get("pattern"), eda.get("trend")
        if pattern and pattern["points"]:
            lines.append(pattern["text"])
        elif trend:
            lines.append(trend["text"].split(". ")[0].rstrip(".") + ".")
        if eda.get("drivers"):
            lines.append(eda["drivers"][0]["text"])
    lines.append(f"Prediction model: {verdict(result)}")
    bullets = ListFlowable([ListItem(p(escape(line)), leftIndent=12, value="•") for line in lines],
                           bulletType="bullet", start="•", leftIndent=12)
    box = Table([[p("<b>At a glance</b>")], [bullets]], colWidths=[PAGE_WIDTH])
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL_BG)),
        ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, -1), (-1, -1), 9),
    ]))
    return box


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor(INK_SOFT))
    canvas.drawString(0.75 * inch, 0.5 * inch, "Data analysis and prediction report")
    canvas.drawRightString(LETTER[0] - 0.75 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


# ---------- the report ----------

def build_report(result):
    """Returns the PDF as bytes."""
    target = escape(result["target"])
    subtitle = " · ".join(filter(None, [
        f"Predicting {target}",
        escape(result["file_name"]) if result.get("file_name") else None,
        date.today().strftime("%B %d, %Y").replace(" 0", " "),
    ]))
    story = [p("Data analysis and prediction report", "title"), Spacer(1, 4), p(subtitle, "subtitle"),
             Spacer(1, 12), glance_box(result)]

    # Part 1: exploratory data analysis
    if result.get("eda"):
        story += eda_story(result["eda"], result["target"])
        story.append(PageBreak())

    # Part 2: the prediction model
    story += part_heading("Part 2", "The prediction model")
    story += [p("The short version", "h2"), p(verdict(result), "verdict"), Spacer(1, 6),
              p(summary_text(result)), Spacer(1, 8), score_chart(result)]
    box = checks_box(result)
    if box:
        story += [Spacer(1, 12), box]

    if result.get("importance"):
        top = [f"<b>{escape(i['column'])}</b>" for i in result["importance"][:2]]
        story.append(KeepTogether([
            p("What the model paid attention to", "h2"),
            p(f"The model relied most on {' and '.join(top)}. Each bar shows how much worse the predictions got "
              f"when that column's values were mixed up, so longer bars mean the model depended on it more. "
              f"This shows what the model used, not necessarily what causes {target}."),
            Spacer(1, 6), importance_chart(result["importance"]),
        ]))

    tests = result.get("test_results")
    if tests and result["task"] == "regression":
        story.append(KeepTogether([
            p("How close the predictions were", "h2"),
            p("Each dot is one row the model had never seen. The grey line shows perfect predictions: "
              "dots near it were predicted closely, dots far from it were not. A tight band along the line "
              "means reliable predictions."),
            Spacer(1, 6), predicted_vs_actual_chart(tests, result["target"]),
        ]))
    elif tests:
        story.append(KeepTogether([
            p("How often each answer was predicted correctly", "h2"),
            p("For each answer, the share of test rows where the model got it right. "
              "Answers that are rare in your data are usually harder to predict."),
            Spacer(1, 6), per_answer_chart(tests),
        ]))

    # Appendix
    steps = {**result.get("cleaning", {}), **result.get("engineering", {})}
    story += part_heading("Appendix", "How we prepared your data")
    story += [p("Before training, your data was checked and tidied automatically:"), Spacer(1, 4), steps_list(steps)]
    story.append(KeepTogether([p("Details", "h2"), details_table(result)]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                            title=f"Data analysis and prediction report: {result['target']}", author="AutoML")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
