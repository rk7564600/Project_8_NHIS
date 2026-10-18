"""Run every analysis task and write figures / tables / metrics.json / dashboard.html.

    python -m jobmarket.pipeline [--data path/to/data.csv] [--out output]
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import analysis, dashboard, forecast, prep
from .recommender import Recommender


def _json(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    raise TypeError(type(o))


def run(data=None, out=None):
    out = Path(out or prep.OUTPUT_DIR)
    fig, tab, itv = out / "figures", out / "tables", out / "interactive"
    for d in (fig, tab, itv):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = prep.load(data)
    log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    log(f"loaded {len(df):,} rows")
    M = {"dataset": {"rows": int(len(df)), "date_min": str(df.date.min().date()), "date_max": str(df.date.max().date()),
                     "analysis_window": f"{prep.START.date()} .. {prep.LAST_FULL.date()}",
                     "rows_in_window": int(len(prep.window(df))), "rows_before_window": int((df.date < prep.START).sum()),
                     "hourly_share_pct": float(df.is_hourly.mean() * 100), "countries": int(df.country.nunique()),
                     "category_counts": df.category.value_counts().to_dict(),
                     "hourly_rate_valid": int(df.hourly_rate.notna().sum()), "budget_valid": int(df.budget_c.notna().sum()),
                     "hourly_outliers_removed": int((df.is_hourly & df[["hourly_low", "hourly_high"]].notna().any(axis=1) & df.hourly_rate.isna()).sum()),
                     "budget_outliers_removed": int((~df.is_hourly & df.budget.notna() & df.budget_c.isna()).sum())}}
    M["task1"] = analysis.task1_keyword_salary(df, fig, tab); log("task1")
    M["task2"] = analysis.task2_emerging(df, fig, tab); log("task2")
    M["task3"] = forecast.task3_demand(df, fig, tab); log("task3")
    M["task4"] = analysis.task4_country_rates(df, fig, tab, itv); log("task4")
    rec = Recommender(df)
    M["task5"] = {"evaluation": rec.evaluate(), "index_size": int(rec.X.shape[0]), "vocab": int(rec.X.shape[1])}; log("task5")
    M["task6"] = dashboard.build(df, out / "dashboard.html"); log("task6")
    M["task7"] = analysis.task7_remote(df, fig, tab); log("task7")
    M["task8"] = forecast.task8_scenarios(df, M["task3"], fig, tab); log("task8")
    (out / "metrics.json").write_text(json.dumps(M, indent=2, default=_json), encoding="utf-8")
    log(f"done -> {out}")
    return M


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--out")
    a = ap.parse_args()
    run(a.data, a.out)
