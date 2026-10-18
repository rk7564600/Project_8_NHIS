"""Task 3 (high-demand role prediction) and Task 8 (future market scenarios)."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from .prep import START, TREND_END, LAST_FULL

HORIZON_DAYS = 14


def _panel(df):
    w = df[df["date"].between(START, LAST_FULL)]
    return w.groupby(["date", "category"]).size().unstack(fill_value=0).astype(float)


def _row(hist: pd.DataFrame, cat: str, date, cats):
    """Features for predicting hist[cat] on `date` from earlier values only."""
    s = hist[cat]
    base = np.log1p(s.iloc[-7])   # same weekday last week; the model learns the *change* from it
    f = {"base": base, "d1": np.log1p(s.iloc[-1]) - base, "d7": np.log1p(s.iloc[-7:].mean()) - base}
    f.update({f"dow{d}": float(date.dayofweek == d) for d in range(7)})
    f.update({f"cat_{c}": float(c == cat) for c in cats})
    return f


def _train_set(panel, first, last):
    cats = list(panel.columns)
    rows, ys = [], []
    for date in pd.date_range(first, last):
        hist = panel.loc[:date - pd.Timedelta(days=1)]
        for c in cats:
            rows.append(_row(hist, c, date, cats)); ys.append(np.log1p(panel.at[date, c]))
    X = pd.DataFrame(rows)
    base = X.pop("base").values
    return X, np.array(ys) - base


def _recursive(model, panel, start_date, days, cats, kind):
    hist = panel.loc[:start_date - pd.Timedelta(days=1)].copy()
    out = {}
    for date in pd.date_range(start_date, periods=days):
        preds = {}
        for c in cats:
            if kind == "seasonal_naive":
                preds[c] = hist[c].iloc[-7]
            elif kind == "mean7":
                preds[c] = hist[c].iloc[-7:].mean()
            else:
                x = pd.DataFrame([_row(hist, c, date, cats)])
                base = x.pop("base").iat[0]
                preds[c] = max(np.expm1(model.predict(x)[0] + base), 0)
        hist.loc[date] = pd.Series(preds)
        out[date] = preds
    return pd.DataFrame(out).T


def _metrics(actual, pred):
    a, p = actual.values.ravel(), pred.values.ravel()
    mask = a > 0
    tot_a, tot_p = actual.sum(), pred.sum()
    top3 = len(set(tot_a.nlargest(3).index) & set(tot_p.nlargest(3).index))
    return {"MAE": float(mean_absolute_error(a, p)), "MAPE_pct": float(np.mean(np.abs(a[mask] - p[mask]) / a[mask]) * 100),
            "R2": float(r2_score(a, p)), "rank_spearman": float(stats.spearmanr(tot_a, tot_p)[0]), "top3_hit": f"{top3}/3"}


def task3_demand(df, figdir, tabdir):
    panel = _panel(df)
    cats = list(panel.columns)
    first_train = START + pd.Timedelta(days=7)
    test_start = TREND_END + pd.Timedelta(days=1)       # 19 Mar
    test_days = (LAST_FULL - test_start).days + 1        # 5 days held out
    Xtr, ytr = _train_set(panel, first_train, TREND_END)
    actual = panel.loc[test_start:LAST_FULL]
    models = {"Ridge": Ridge(alpha=1.0), "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=.05, random_state=0),
              "RandomForest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0)}
    scores, preds = {}, {}
    for name in ("seasonal_naive", "mean7"):
        preds[name] = _recursive(None, panel, test_start, test_days, cats, name)
        scores[name] = _metrics(actual, preds[name])
    for name, m in models.items():
        m.fit(Xtr, ytr)
        preds[name] = _recursive(m, panel, test_start, test_days, cats, "model")
        scores[name] = _metrics(actual, preds[name])
    best_ml = min(models, key=lambda k: scores[k]["MAPE_pct"])
    ens_name = f"Ensemble({best_ml}+seasonal_naive)"
    preds[ens_name] = (preds[best_ml] + preds["seasonal_naive"]) / 2
    scores[ens_name] = _metrics(actual, preds[ens_name])
    best = ens_name if scores[ens_name]["MAPE_pct"] < scores[best_ml]["MAPE_pct"] else best_ml
    pd.DataFrame(scores).T.round(3).to_csv(tabdir / "task3_model_scores.csv")

    # refit on all complete days and forecast the next 14 days
    Xall, yall = _train_set(panel, first_train, LAST_FULL)
    final = models[best_ml].fit(Xall, yall)
    fc_start = LAST_FULL + pd.Timedelta(days=1)
    fc = _recursive(final, panel, fc_start, HORIZON_DAYS, cats, "model")
    if best == ens_name:
        fc = (fc + _recursive(None, panel, fc_start, HORIZON_DAYS, cats, "seasonal_naive")) / 2
    fc.round(1).to_csv(tabdir / "task3_forecast_daily.csv")
    cur = panel.loc[LAST_FULL - pd.Timedelta(days=13):LAST_FULL].sum()
    nxt = fc.sum()
    rank = pd.DataFrame({"last14_actual": cur, "next14_forecast": nxt})
    rank["change_pct"] = (rank.next14_forecast / rank.last14_actual - 1) * 100
    rank["forecast_share_pct"] = rank.next14_forecast / rank.next14_forecast.sum() * 100
    rank = rank.sort_values("next14_forecast", ascending=False)
    rank.round(2).to_csv(tabdir / "task3_demand_ranking.csv")

    # role level: share of role inside its category x category forecast
    w = df[df["date"].between(START, LAST_FULL) & (df["role"].str.len() > 3)]
    top_roles = w["role"].value_counts().head(15).index
    rl = []
    for r in top_roles:
        sub = w[w.role == r]
        c = sub.category.mode().iat[0]
        share_in_cat = len(sub) / (w.category == c).sum()
        rl.append({"role": r, "category": c, "postings_observed": int(len(sub)), "forecast_next14": round(float(share_in_cat * nxt[c]), 1)})
    roles = pd.DataFrame(rl).sort_values("forecast_next14", ascending=False)
    roles.to_csv(tabdir / "task3_role_forecast.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    top6 = rank.index[:6]
    for c, col in zip(top6, plt.cm.tab10.colors):
        h = panel.loc[LAST_FULL - pd.Timedelta(days=27):LAST_FULL, c]
        axes[0].plot(h.index, h.values, color=col, lw=1.5, label=c)
        axes[0].plot(fc.index, fc[c].values, "--", color=col, lw=1.5)
    axes[0].axvline(fc_start, color="k", ls=":"); axes[0].set_title(f"Daily postings: history and {HORIZON_DAYS}-day forecast ({best})")
    axes[0].legend(fontsize=7); axes[0].tick_params(axis="x", rotation=30)
    names = list(scores); mape = [scores[n]["MAPE_pct"] for n in names]
    axes[1].bar([n.replace("(", "\n(") for n in names], mape, color=["#2a7f62" if n == best else "#999" for n in names])
    axes[1].set_ylabel("MAPE % on held-out 19-23 Mar"); axes[1].set_title("Model comparison (lower is better)"); axes[1].tick_params(axis="x", rotation=30, labelsize=7)
    fig.tight_layout(); fig.savefig(figdir / "task3_demand_forecast.png"); plt.close(fig)
    return {"train_period": f"{first_train:%d %b} - {TREND_END:%d %b %Y}", "test_period": f"{test_start:%d %b} - {LAST_FULL:%d %b %Y}",
            "train_rows": int(len(Xtr)), "scores": scores, "best_model": best, "best_ml_model": best_ml,
            "ranking": rank.round(2).reset_index().rename(columns={"index": "category"}).to_dict("records"),
            "roles": roles.to_dict("records")}


def task8_scenarios(df, task3, figdir, tabdir):
    """Scenario projections for the next 4 weeks: volume, category mix, hourly rate."""
    w = df[df["date"].between(START, TREND_END)]
    days = pd.date_range(START, TREND_END)
    daily = w.groupby("date").size().reindex(days)
    wkvol = daily.groupby((days - START).days // 7).sum()
    # week 1 contains a scrape gap (15 Feb: 1.8k postings vs ~6.5k normal), so trends use weeks 2-5
    g_obs = float((wkvol.iloc[-1] / wkvol.iloc[1]) ** (1 / 3) - 1)
    weeks = np.arange(1, 5)
    last = float(wkvol.iloc[-1])
    growth = {"pessimistic": 2 * g_obs, "base": g_obs, "optimistic": abs(g_obs)}
    vol = {k: last * (1 + g) ** weeks for k, g in growth.items()}
    wr = w[w.hourly_rate.notna()].groupby("day_bin").hourly_rate.mean()
    lr = stats.linregress(np.arange(len(wr)), wr.values)
    rate_last = float(wr.iloc[-1])
    rate = {"pessimistic": rate_last + (lr.slope - lr.stderr) * weeks, "base": rate_last + lr.slope * weeks,
            "optimistic": rate_last + (lr.slope + lr.stderr) * weeks}
    # category-share scenarios: persistence of linear trend (base), 2x acceleration, trend stops (reverts to 5-week mean)
    cs = w.groupby(["day_bin", "category"]).size().unstack(fill_value=0)
    cs = cs.div(cs.sum(1), axis=0) * 100
    sc_rows = []
    for c in cs.columns:
        l = stats.linregress(np.arange(len(cs)), cs[c].values)
        last = float(cs[c].iloc[-1])
        sc_rows.append({"category": c, "share_now_pct": last, "trend_pp_per_week": l.slope, "p_value": l.pvalue,
                        "base_4w": last + l.slope * 4, "accelerated_4w": last + 2 * l.slope * 4, "reversion_4w": float(cs[c].mean())})
    sc = pd.DataFrame(sc_rows).sort_values("share_now_pct", ascending=False).round(3)
    sc.to_csv(tabdir / "task8_category_scenarios.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    labs = [f"W{i}" for i in range(1, 6)]
    ax[0].plot(range(5), wkvol.values / 1000, "o-k", label="observed weekly postings (k)")
    for k, col in zip(vol, ["#b5483a", "#555", "#2a7f62"]):
        ax[0].plot(range(5, 9), vol[k] / 1000, "o--", color=col, label=k)
    ax[0].set_title("Weekly posting volume scenarios (thousand)"); ax[0].legend(fontsize=7); ax[0].set_xticks(range(9)); ax[0].set_xticklabels([f"W{i}" for i in range(1, 10)])
    ax[1].plot(range(5), wr.values, "o-k")
    for k, col in zip(rate, ["#b5483a", "#555", "#2a7f62"]):
        ax[1].plot(range(5, 9), rate[k], "o--", color=col, label=k)
    ax[1].set_title("Mean hourly rate scenarios ($/hr)"); ax[1].legend(fontsize=7)
    s2 = sc.set_index("category")
    x = np.arange(len(s2)); ax[2].bar(x - .27, s2.share_now_pct, .27, label="now"); ax[2].bar(x, s2.base_4w, .27, label="base +4w"); ax[2].bar(x + .27, s2.accelerated_4w, .27, label="accelerated +4w")
    ax[2].set_xticks(x); ax[2].set_xticklabels(s2.index, rotation=70, ha="right", fontsize=7); ax[2].set_title("Category share scenarios (%)"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / "task8_scenarios.png"); plt.close(fig)
    return {"weekly_volume_observed": [int(v) for v in wkvol.values], "weekly_growth_assumptions": growth,
            "volume_scenarios_next4w": {k: [int(v) for v in vol[k]] for k in vol},
            "hourly_rate_weekly_mean": [float(v) for v in wr.values], "rate_slope_per_week": float(lr.slope), "rate_slope_p": float(lr.pvalue), "rate_slope_stderr": float(lr.stderr),
            "rate_scenarios_next4w": {k: [round(float(v), 2) for v in rate[k]] for k in rate},
            "category_scenarios": sc.to_dict("records")}
