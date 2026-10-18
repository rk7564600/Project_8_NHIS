"""Build Project8_Presentation.pptx from output/metrics.json (native, editable PowerPoint charts)."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
M = json.loads((ROOT / "output" / "metrics.json").read_text(encoding="utf-8"))
INK, ACC, RED, MUTE, LIGHT = RGBColor(0x14, 0x24, 0x1F), RGBColor(0x1F, 0x6F, 0x5B), RGBColor(0xB5, 0x48, 0x3A), RGBColor(0x5B, 0x68, 0x62), RGBColor(0xEE, 0xF3, 0xF0)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(slide, x, y, w, h, s, size=16, bold=False, color=INK, align=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color; p.font.name = "Calibri"
        p.space_after = Pt(6)
        if align: p.alignment = align
    return tb


def new(title, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    text(s, 0.6, 0.35, 12.1, 0.8, title, 28, True)
    if subtitle:
        text(s, 0.6, 1.05, 12.1, 0.5, subtitle, 15, False, MUTE)
    return s


def bullets(slide, x, y, w, h, items, size=16):
    return text(slide, x, y, w, h, ["\u2022  " + i for i in items], size)


def chart(slide, kind, cats, series, x, y, w, h, title=None, colors=None, legend=False, number_format=None, labels=True, font=11):
    cd = CategoryChartData(); cd.categories = cats
    for name, vals in series.items():
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.font.size = Pt(font); ch.font.name = "Calibri"
    ch.has_title = bool(title)
    if title:
        ch.chart_title.text_frame.text = title; ch.chart_title.text_frame.paragraphs[0].font.size = Pt(13); ch.chart_title.text_frame.paragraphs[0].font.bold = True
    ch.has_legend = legend
    if legend:
        ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
    pl = ch.plots[0]
    if labels and kind != XL_CHART_TYPE.LINE_MARKERS:
        pl.has_data_labels = True
        pl.data_labels.font.size = Pt(font - 1)
        if number_format: pl.data_labels.number_format = number_format; pl.data_labels.number_format_is_linked = False
    for i, s in enumerate(pl.series):
        col = (colors or [ACC, RED, MUTE, RGBColor(0xC9, 0x9A, 0x2E)])[i % 4]
        if kind == XL_CHART_TYPE.LINE_MARKERS:
            s.format.line.color.rgb = col; s.marker.format.fill.solid(); s.marker.format.fill.fore_color.rgb = col
        else:
            s.format.fill.solid(); s.format.fill.fore_color.rgb = col
    ch.value_axis.has_major_gridlines = True
    ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xE2, 0xE8, 0xE4)
    return ch


def table(slide, x, y, w, rows, col_w=None, size=12, row_h=0.36, left=False):
    t = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows))).table
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            cell = t.cell(r, c); cell.text = str(v)
            p = cell.text_frame.paragraphs[0]; p.font.size = Pt(size); p.font.name = "Calibri"
            p.font.bold = r == 0; p.font.color.rgb = RGBColor(255, 255, 255) if r == 0 else INK
            if c > 0 and not left: p.alignment = PP_ALIGN.RIGHT
            cell.fill.solid(); cell.fill.fore_color.rgb = ACC if r == 0 else (LIGHT if r % 2 == 0 else RGBColor(255, 255, 255))
    if col_w:
        for i, cw in enumerate(col_w): t.columns[i].width = Inches(cw)
    return t


D, T1, T2, T3, T4, T5, T7, T8 = (M[k] for k in ("dataset", "task1", "task2", "task3", "task4", "task5", "task7", "task8"))

# 1 title
s = prs.slides.add_slide(BLANK)
s.background.fill.solid(); s.background.fill.fore_color.rgb = INK
text(s, 0.8, 2.2, 11.5, 1.4, "Job Market Analysis and Recommendation System", 40, True, RGBColor(255, 255, 255))
text(s, 0.8, 3.7, 11.5, 1.2, [f"{D['rows']:,} Upwork postings (dense coverage 13 Feb - 24 Mar 2024)", "Eight analyses, recommendation API, monthly dashboard, Docker deployment"], 20, False, RGBColor(0xC9, 0xD8, 0xD1))

# 2 data
s = new("What the data supports", "Know the limits before reading any trend")
vals = {"Hourly share": f"{D['hourly_share_pct']:.1f}%", "Countries": f"{D['countries']}", "Postings in analysis window": f"{D['rows_in_window']:,}", "Valid hourly rates": f"{D['hourly_rate_valid']:,}"}
for i, (k, v) in enumerate(vals.items()):
    text(s, 0.6 + i * 3.1, 1.8, 2.9, 0.7, v, 32, True, ACC); text(s, 0.6 + i * 3.1, 2.5, 2.9, 0.5, k, 14, False, MUTE)
bullets(s, 0.6, 3.4, 12.1, 3.6, [
    "Only ~5.5 weeks of dense data (13 Feb - 23 Mar 2024); 283 earlier postings are sparse and excluded from trends",
    "\"Monthly\" tracking is therefore weekly; 15 Feb has a scrape gap (1.8k vs ~6k postings/day)",
    f"Outliers handled: {D['hourly_outliers_removed']} hourly rows (> $200/hr) and {D['budget_outliers_removed']} fixed budgets (raw max $1,000,000)",
    "Categories and remote flags are keyword rules (19.6% of titles stay 'Other'); no description text exists",
])

# 3 task 1
s = new("Task 1: title keywords vs pay", f"Held-out R\u00b2 of all keywords: hourly {T1['hourly']['holdout_r2_all_keywords']:.0%}, fixed {T1['fixed']['holdout_r2_all_keywords']:.0%}")
for j, (kind, unit, nf) in enumerate([("hourly", "$/hr", '"$"0.0'), ("fixed", "$", '"$"0')]):
    rows = T1[kind]["top_positive"][:5] + T1[kind]["top_negative"][:5]
    chart(s, XL_CHART_TYPE.BAR_CLUSTERED, [r["keyword"] for r in rows][::-1], {"Median with keyword": [r["median_with"] for r in rows][::-1]},
          0.5 + j * 6.4, 1.6, 6.2, 4.6, f"{kind.title()}: median with keyword (overall {T1[kind]['top_positive'][0]['median_without']:g})", number_format=nf)
text(s, 0.6, 6.4, 12, 0.8, "Specialist words (consultant, engineer, developer) go with higher pay; assistant / virtual / lead-gen wording with much lower pay. Titles explain only a fifth to a quarter of pay variance.", 14, False, MUTE)

# 4 task 2
s = new("Task 2: emerging and declining categories", T2["window"])
cs = sorted(T2["categories"], key=lambda c: c["growth_pct"])
ch = chart(s, XL_CHART_TYPE.BAR_CLUSTERED, [c["category"] for c in cs], {"Share change, week 1 to week 5 (%)": [round(c["growth_pct"], 1) for c in cs]}, 0.5, 1.5, 7.6, 5.4, "Change in share of postings (%)", number_format='0.0')
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
for i, c in enumerate(cs):
    pt = ch.plots[0].series[0].points[i]; pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = ACC if c["status"] == "Emerging" else RED if c["status"] == "Declining" else RGBColor(0xA7, 0xB2, 0xAD)
bullets(s, 8.4, 1.7, 4.5, 5, [
    "Emerging: Video & Audio (+5.4%, p=0.006)",
    "Declining: Writing & Translation (-12.7%, p=0.0003), Admin/VA (-5.9%, p=0.03)",
    "Rising keywords: crypto +49%, js +35%, facebook +25%, ads +18%",
    "Only the Writing decline survives a Bonferroni correction across 12 categories",
], 15)

# 5 task 3
s = new("Task 3: predicting high-demand roles", f"Train {T3['train_period']}, test {T3['test_period']} (5-day recursive forecast)")
names = ["seasonal_naive", "Ridge", "GradientBoosting", "RandomForest", [n for n in T3["scores"] if n.startswith("Ensemble")][0]]
short = ["Seasonal naive", "Ridge", "Grad. boosting", "Random forest", "Ensemble"]
chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, short, {"MAPE %": [round(T3["scores"][n]["MAPE_pct"], 1) for n in names]}, 0.5, 1.6, 6.0, 4.3, "Held-out MAPE (%), lower is better", number_format='0.0')
rk = T3["ranking"][:6]
table(s, 6.9, 1.7, 6.0, [["Category", "Next 14 d", "vs last 14 d"]] + [[r["category"], f"{r['next14_forecast']:,.0f}", f"{r['change_pct']:+.1f}%"] for r in rk], [3.0, 1.5, 1.5], 13, 0.42)
text(s, 0.6, 6.0, 12.2, 1.2, [f"Category ranking predicted almost perfectly (Spearman {T3['scores'][names[-1]]['rank_spearman']:.2f}, top-3 correct), but the simple seasonal baseline is as accurate as the ML models.",
                              "With 5 weeks of data, weekly seasonality explains nearly everything; the ensemble gives the lowest MAE (30.6 vs 33.1)."], 14, False, MUTE)

# 6 task 4
s = new("Task 4: hourly rates offered by client country", "Median $/hr, countries with 200+ rated jobs; interactive map in output/interactive/")
rows = T4["top10"][:6] + T4["bottom10"]
rows = sorted(rows, key=lambda r: r["median"])
chart(s, XL_CHART_TYPE.BAR_CLUSTERED, [r["country"] for r in rows], {"Median $/hr": [r["median"] for r in rows]}, 0.5, 1.5, 7.4, 5.6, number_format='0.0')
bullets(s, 8.2, 1.7, 4.7, 5, [f"Overall median ${T4['overall_median']:g}/hr, mean ${T4['overall_mean']:.2f}", f"US: median ${T4['us']['median']:g}, mean ${T4['us']['mean']:.2f} (n={T4['us']['n']:,})",
                              f"Country effect significant (Kruskal-Wallis H={T4['kruskal_H']:,.0f}, p<0.001)", "Lowest: South Korea $10, Philippines $11, Japan $15, India/Pakistan $18.5",
                              "Top countries are a near-tie with wide intervals; the Philippines mean is 2x its median"], 14)

# 7 task 5
s = new("Task 5: recommendation engine", "TF-IDF title matching + filters + recency, served by a Flask API and web UI")
e = T5["evaluation"]
chart(s, XL_CHART_TYPE.COLUMN_CLUSTERED, ["Random", "Engine"], {"Precision@10": [round(e["random_baseline_precision"], 2), round(e["precision_at_10"], 2)]}, 0.5, 1.6, 4.6, 4.5, "Precision@10 (same-category hits)", number_format='0.00')
bullets(s, 5.4, 1.7, 7.5, 5, [f"Index: {T5['index_size']:,} postings, {T5['vocab']:,} terms; ~0.1 s per query",
                              f"Offline test on {e['queries']} profiles: hit-rate@10 {e['hit_rate_at_10']:.2f}, nDCG@10 {e['ndcg_at_10']:.2f}",
                              "Filters: contract type, country, category, min $/hr, min fixed budget, explicit remote",
                              "Endpoints: /api/recommend, /api/meta, /api/metrics, /api/health (docs/API.md)",
                              "Limit: titles only, no click data - relevance is topical, not validated with users"], 15)

# 8 task 6
s = new("Task 6: monthly dashboard", "output/dashboard.html, regenerated on the 1st of each month by the scheduler")
mt = M["task6"]["monthly"]
table(s, 0.6, 1.8, 12.1, [["Month", "Days", "Postings/day", "Hourly share", "Mean $/hr", "Median fixed $", "Remote wording", "Non-US clients"]] +
      [[m["month"], m["days_covered"], f"{m['postings_per_day']:,}", f"{m['hourly_share_pct']}%", f"${m['mean_hourly_rate']}", f"${m['median_fixed_budget']:.0f}", f"{m['remote_wording_pct']}%", f"{m['non_us_pct']}%"] for m in mt], None, 14, 0.5)
bullets(s, 0.6, 4.0, 12.1, 3, ["Dashboard: KPI cards (last 7 days vs prior 7), daily volume with range buttons, weekly category mix, pay trends, top countries, month table",
                               "Month-to-month changes are small (+2.2% postings/day, -0.1% mean rate)", "Updating = drop a new data.csv in data/; the scheduler container re-runs the pipeline"], 15)

# 9 task 7
s = new("Task 7: remote-work trends", f"Explicit remote wording in titles: {T7['remote_titles']:,} postings ({T7['remote_share_pct']:.2f}%)")
wk = T7["weekly_remote_share_pct"]; fc = T7["forecast_next4_weeks_pct"]
labels = [f"W{i}" for i in range(1, 10)]
ch = chart(s, XL_CHART_TYPE.LINE_MARKERS, labels, {"Observed": wk + [None] * 4, "Linear forecast": [None] * 4 + [wk[-1]] + fc}, 0.5, 1.6, 7.4, 4.8, "Weekly share of titles with remote wording (%)", legend=True, labels=False)
bullets(s, 8.2, 1.7, 4.7, 5, ["No trend: slope -0.002 pp/week (p=0.83)", f"Forecast next 4 weeks: ~{fc[0]:.2f}% (\u00b1{T7['forecast_interval_halfwidth_pp']:.2f} pp)",
                              "Highest: Customer Support 3.8%, Sales 2.1%, Admin/VA 1.5%", f"Rate premium ${T7['remote_median_rate']:g} vs ${T7['nonremote_median_rate']:g}: not significant (p={T7['rate_mannwhitney_p']:.2f})",
                              "All Upwork work is online; this only measures advertised remote wording"], 14)

# 10 task 8
s = new("Task 8: future scenarios (next 4 weeks)", "Assumption-based projections anchored on observed weeks 2-5")
obs = T8["weekly_volume_observed"]; vs = T8["volume_scenarios_next4w"]
f = lambda v: [round(x / 1000, 1) for x in v]
chart(s, XL_CHART_TYPE.LINE_MARKERS, [f"W{i}" for i in range(1, 10)],
      {"Observed": f(obs) + [None] * 4, "Pessimistic": [None] * 4 + [round(obs[-1] / 1000, 1)] + f(vs["pessimistic"]), "Base": [None] * 4 + [round(obs[-1] / 1000, 1)] + f(vs["base"]),
       "Optimistic": [None] * 4 + [round(obs[-1] / 1000, 1)] + f(vs["optimistic"])}, 0.5, 1.6, 7.6, 4.9, "Weekly postings (thousand)", [INK, RED, MUTE, ACC], legend=True, labels=False)
g = T8["weekly_growth_assumptions"]
bullets(s, 8.4, 1.7, 4.5, 5, [f"Base: observed {g['base']*100:.1f}%/week decline continues", f"Pessimistic: {g['pessimistic']*100:.1f}%/week; optimistic: +{g['optimistic']*100:.1f}%/week",
                              "Hourly rate flat ($27.5-28.5): scenarios differ by < $1", "Mix shifts if trends persist: Video & Audio up, Writing & Translation down", "Short-term model: ~6.4-7.0k postings per weekday"], 14)

# 11 deployment
s = new("Deployment and deliverables", "Docker Compose: three services sharing one output volume")
table(s, 0.6, 1.8, 12.1, [["Service", "Image / file", "Role"], ["web", "nginx/Dockerfile", "Reverse proxy on :8080"], ["app", "Dockerfile.app", "gunicorn + Flask API, UI, dashboard hosting"],
                          ["scheduler", "Dockerfile.scheduler", "Runs pipeline at start and on the 1st of each month"]], [2.2, 4.2, 5.7], 14, 0.5, left=True)
bullets(s, 0.6, 4.2, 12.1, 3, ["docker compose build && docker compose up -d  ->  http://localhost:8080", "Reproducible analysis: python -m jobmarket.pipeline writes figures, tables, metrics.json",
                               "Docker files are written but were not built in the authoring environment - build once to confirm", "Tests: pytest tests/ (API validation and filtering)"], 15)

# 12 limits
s = new("Limitations and next steps")
bullets(s, 0.6, 1.5, 12.1, 5.5, ["Five weeks of data: trends are indicative; collect several months before drawing seasonal conclusions",
                                 "Categories and remote flags are rule-based; job descriptions would allow better skill and remote detection",
                                 "Recommender is title-only and evaluated offline; add user feedback (clicks/applies) to validate and personalise",
                                 "Forecast accuracy measured on a 5-day holdout; seasonal baseline is hard to beat at this horizon",
                                 "Next: monthly refresh will lengthen history, enabling real month-over-month and seasonal models"], 18)

out = ROOT / "Project8_Presentation.pptx"
prs.save(out)
print("saved", out)
