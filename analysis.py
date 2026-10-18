"""Tasks 1, 2, 4 and 7: keyword-salary correlation, emerging categories, country rates, remote trends."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from .prep import START, TREND_END, LAST_FULL, window

TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z+#]+\b"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})


def _bin_label(b):
    s = START + pd.Timedelta(days=7 * int(b))
    return f"{s:%d %b}-{s + pd.Timedelta(days=6):%d %b}"


# ---------------------------------------------------------------- Task 1
def task1_keyword_salary(df, figdir, tabdir):
    res, fig, axes = {}, *plt.subplots(1, 2, figsize=(13, 6))
    for ax, (kind, col, unit) in zip(axes, [("hourly", "hourly_rate", "$/hr"), ("fixed", "budget_c", "$")]):
        d = df[df[col].notna()]
        y = np.log10(d[col].values)
        cv = CountVectorizer(binary=True, stop_words="english", min_df=300, token_pattern=TOKEN)
        X = cv.fit_transform(d["title_norm"]).tocsc()
        terms = np.array(cv.get_feature_names_out())
        n = X.shape[0]
        m = np.asarray(X.mean(0)).ravel()
        cov = np.asarray(X.T @ y).ravel() / n - m * y.mean()
        r = cov / (np.sqrt(m * (1 - m)) * y.std())
        t = r * np.sqrt((n - 2) / (1 - r ** 2))
        p = 2 * stats.t.sf(np.abs(t), n - 2)
        tab = pd.DataFrame({"keyword": terms, "n_titles": np.asarray(X.sum(0)).ravel(), "pearson_r_log": r, "p_value": p})
        meds = []
        for j in range(len(terms)):
            has = X[:, j].toarray().ravel() > 0
            meds.append((np.median(d[col].values[has]), np.median(d[col].values[~has])))
        tab["median_with"], tab["median_without"] = zip(*meds)
        tab["median_ratio"] = tab["median_with"] / tab["median_without"]
        tab = tab.sort_values("pearson_r_log", ascending=False)
        tab.to_csv(tabdir / f"task1_keywords_{kind}.csv", index=False)
        sig = tab[tab.p_value < 0.001]
        top, bot = sig.head(10), sig.tail(10)
        show = pd.concat([top, bot]).iloc[::-1]
        ax.barh(show["keyword"], show["pearson_r_log"], color=["#2a7f62" if v > 0 else "#b5483a" for v in show["pearson_r_log"]])
        ax.set_title(f"{kind.title()} ({unit}): keyword vs log-{kind} correlation (n={n:,})")
        ax.set_xlabel("Pearson r (keyword present vs log10 value)")
        # predictive strength of the whole keyword set (held-out R^2)
        cv2 = CountVectorizer(binary=True, stop_words="english", min_df=50, token_pattern=TOKEN)
        X2 = cv2.fit_transform(d["title_norm"])
        Xtr, Xte, ytr, yte = train_test_split(X2, y, test_size=0.2, random_state=0)
        r2 = r2_score(yte, Ridge(alpha=3).fit(Xtr, ytr).predict(Xte))
        cols = ["keyword", "n_titles", "pearson_r_log", "median_with", "median_without", "median_ratio"]
        res[kind] = {"n": int(n), "n_keywords_tested": int(len(tab)), "n_significant_p001": int(len(sig)),
                     "holdout_r2_all_keywords": float(r2),
                     "top_positive": top[cols].round(3).to_dict("records"),
                     "top_negative": bot.iloc[::-1][cols].round(3).to_dict("records")}
    fig.tight_layout()
    fig.savefig(figdir / "task1_keyword_salary.png")
    plt.close(fig)
    return res


# ---------------------------------------------------------------- Task 2
def _trend(share_by_day: pd.Series, bins: pd.Series):
    x = np.arange(len(share_by_day))
    lr = stats.linregress(x, share_by_day.values)
    b = share_by_day.groupby(bins.values).mean()
    return lr.slope * 7, lr.pvalue, float(b.iloc[0]), float(b.iloc[-1])


def task2_emerging(df, figdir, tabdir):
    w = df[df["date"].between(START, TREND_END)].copy()
    days = pd.date_range(START, TREND_END)
    bins = pd.Series((days - START).days // 7, index=days)
    tot = w.groupby("date").size().reindex(days)
    cat = w.groupby(["date", "category"]).size().unstack(fill_value=0).reindex(days, fill_value=0)
    rows = []
    for c in cat.columns:
        s = cat[c] / tot
        slope, p, first, last = _trend(s, bins)
        rows.append({"category": c, "share_first_week": first, "share_last_week": last, "growth_pct": (last / first - 1) * 100,
                     "slope_pp_per_week": slope * 100, "p_value": p, "total_postings": int(cat[c].sum())})
    ct = pd.DataFrame(rows).sort_values("growth_pct", ascending=False)
    ct["status"] = np.where((ct.p_value < 0.05) & (ct.slope_pp_per_week > 0), "Emerging",
                            np.where((ct.p_value < 0.05) & (ct.slope_pp_per_week < 0), "Declining", "Stable"))
    ct.to_csv(tabdir / "task2_category_growth.csv", index=False)

    # sub-niche keywords
    cv = CountVectorizer(binary=True, stop_words="english", min_df=400, token_pattern=TOKEN)
    X = cv.fit_transform(w["title_norm"]).tocsc()
    terms = cv.get_feature_names_out()
    dcode = (w["date"] - START).dt.days.values
    krows = []
    for j, term in enumerate(terms):
        has = X[:, j].toarray().ravel()
        per_day = pd.Series(has).groupby(dcode).sum().reindex(range(len(days)), fill_value=0)
        s = per_day.values / tot.values
        slope, p, first, last = _trend(pd.Series(s, index=days), bins)
        if first > 0:
            krows.append({"keyword": term, "postings": int(has.sum()), "share_first_week": first, "share_last_week": last,
                          "growth_pct": (last / first - 1) * 100, "p_value": p})
    kt = pd.DataFrame(krows)
    em = kt[(kt.p_value < 0.001) & (kt.growth_pct > 0)].sort_values("growth_pct", ascending=False)
    de = kt[(kt.p_value < 0.001) & (kt.growth_pct < 0)].sort_values("growth_pct")
    kt.sort_values("growth_pct", ascending=False).to_csv(tabdir / "task2_keyword_growth.csv", index=False)

    wk = (w.groupby(["day_bin", "category"]).size().unstack(fill_value=0))
    wk = wk.div(wk.sum(1), axis=0) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for c in ct.category:
        a1.plot([_bin_label(b) for b in wk.index], wk[c], marker="o", lw=2 if c in ct.head(3).category.values or c in ct.tail(2).category.values else 1, label=c)
    a1.set_ylabel("share of weekly postings (%)"); a1.set_title("Category share by 7-day period"); a1.legend(fontsize=7, ncol=2)
    a1.tick_params(axis="x", rotation=25)
    cs = ct.sort_values("growth_pct")
    a2.barh(cs.category, cs.growth_pct, color=["#2a7f62" if s == "Emerging" else "#b5483a" if s == "Declining" else "#999" for s in cs.status])
    a2.set_title("Share change, first vs last week (green = significant rise, red = significant fall)")
    a2.set_xlabel("%")
    fig.tight_layout(); fig.savefig(figdir / "task2_emerging.png"); plt.close(fig)
    cols = ["keyword", "postings", "share_first_week", "share_last_week", "growth_pct"]
    return {"window": f"{START:%d %b} - {TREND_END:%d %b %Y} (5 full weeks)",
            "categories": ct.round(4).to_dict("records"),
            "emerging_keywords": em[cols].head(15).round(4).to_dict("records"),
            "declining_keywords": de[cols].head(10).round(4).to_dict("records")}


# ---------------------------------------------------------------- Task 4
def task4_country_rates(df, figdir, tabdir, interactive_dir):
    h = df[df["hourly_rate"].notna() & (df["country"] != "Unknown")]
    g = h.groupby("country")["hourly_rate"].agg(n="count", mean="mean", median="median",
                                                 p25=lambda s: s.quantile(.25), p75=lambda s: s.quantile(.75), std="std").reset_index()
    g["ci95"] = 1.96 * g["std"] / np.sqrt(g["n"])
    g = g[g.n >= 30].sort_values("median", ascending=False)
    g.round(2).to_csv(tabdir / "task4_country_hourly_rates.csv", index=False)
    big = g[g.n >= 200]
    groups = [h.loc[h.country == c, "hourly_rate"].values for c in g[g.n >= 200].country]
    kw = stats.kruskal(*groups)
    fig = px.choropleth(g, locations="country", locationmode="country names", color="median", hover_name="country",
                        hover_data={"mean": ":.2f", "median": ":.2f", "n": True, "country": False}, color_continuous_scale="YlGnBu",
                        labels={"median": "Median $/hr"}, title="Median hourly rate by client country (countries with >=30 rated jobs)")
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(interactive_dir / "task4_hourly_rate_map.html", include_plotlyjs=True)
    bar = px.bar(big.sort_values("median").tail(30), x="median", y="country", orientation="h", error_x="ci95", hover_data=["mean", "n"],
                 title="Median hourly rate - top 30 countries with >=200 rated jobs")
    bar.write_html(interactive_dir / "task4_hourly_rate_bar.html", include_plotlyjs=True)
    sub = pd.concat([big.head(12), big.tail(8)]).drop_duplicates("country").iloc[::-1]
    fig2, ax = plt.subplots(figsize=(9, 7))
    ax.barh(sub.country, sub["median"], xerr=sub["ci95"], color="#2a7f62", alpha=.85)
    ax.set_xlabel("Median hourly rate ($/hr)"); ax.set_title("Highest and lowest paying client countries (n>=200 rated jobs)")
    fig2.tight_layout(); fig2.savefig(figdir / "task4_country_rates.png"); plt.close(fig2)
    us = g.loc[g.country == "United States"].iloc[0]
    return {"countries_with_30plus": int(len(g)), "countries_with_200plus": int(len(big)),
            "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
            "us": {"mean": float(us["mean"]), "median": float(us["median"]), "n": int(us["n"])},
            "top10": big.head(10)[["country", "n", "mean", "median"]].round(2).to_dict("records"),
            "bottom10": big.tail(10)[["country", "n", "mean", "median"]].round(2).to_dict("records"),
            "overall_median": float(h.hourly_rate.median()), "overall_mean": float(h.hourly_rate.mean())}


# ---------------------------------------------------------------- Task 7
def task7_remote(df, figdir, tabdir):
    w = df[df["date"].between(START, TREND_END)]
    days = pd.date_range(START, TREND_END)
    bins = pd.Series((days - START).days // 7, index=days)
    tot = w.groupby("date").size().reindex(days)
    rem = w[w.is_remote].groupby("date").size().reindex(days, fill_value=0)
    share = rem / tot
    slope, p, first, last = _trend(share, bins)
    # forecast: OLS on weekly share, project 4 weeks ahead with 95% prediction interval
    wk = (w.groupby("day_bin").is_remote.mean() * 100)
    x = np.arange(len(wk)); lr = stats.linregress(x, wk.values)
    fx = np.arange(len(wk), len(wk) + 4)
    resid_sd = np.std(wk.values - (lr.intercept + lr.slope * x), ddof=2)
    fc = lr.intercept + lr.slope * fx
    # other market dynamics useful as remote-work proxies
    nonus = (w.country.ne("United States") & w.country.ne("Unknown")).groupby(w.day_bin).mean() * 100
    hourly_share = w.is_hourly.groupby(w.day_bin).mean() * 100
    weekend = (w.date.dt.dayofweek >= 5).groupby(w.day_bin).mean() * 100
    rr = w[w.is_remote & w.hourly_rate.notna()].hourly_rate
    nr = w[~w.is_remote & w.hourly_rate.notna()].hourly_rate
    mw = stats.mannwhitneyu(rr, nr)
    by_cat = (w.groupby("category").is_remote.agg(["mean", "sum"]) .assign(mean=lambda d: d["mean"] * 100)
              .sort_values("mean", ascending=False).round(3).reset_index().rename(columns={"mean": "remote_share_pct", "sum": "remote_titles"}))
    by_cat.to_csv(tabdir / "task7_remote_by_category.csv", index=False)
    by_cty = (w[w.country != "Unknown"].groupby("country").is_remote.agg(["mean", "sum", "size"])
              .query("size>=500").sort_values("mean", ascending=False).head(8).reset_index())
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    labs = [_bin_label(b) for b in wk.index]
    ax[0].plot(labs, wk.values, "o-", label="observed"); ax[0].plot([_bin_label(b) for b in fx], fc, "o--", label="linear forecast")
    ax[0].fill_between([_bin_label(b) for b in fx], fc - 1.96 * resid_sd, fc + 1.96 * resid_sd, alpha=.2)
    ax[0].set_title("Postings with explicit remote wording (%)"); ax[0].tick_params(axis="x", rotation=40); ax[0].legend()
    ax[1].plot(labs, nonus.values, "o-", color="#b5483a"); ax[1].set_title("Share of postings from non-US clients (%)"); ax[1].tick_params(axis="x", rotation=40)
    ax[2].plot(labs, hourly_share.values, "o-", color="#555", label="hourly share"); ax[2].plot(labs, weekend.values, "s-", color="#2a7f62", label="weekend share")
    ax[2].set_title("Hourly-contract share and weekend-posting share (%)"); ax[2].tick_params(axis="x", rotation=40); ax[2].legend()
    fig.tight_layout(); fig.savefig(figdir / "task7_remote.png"); plt.close(fig)
    return {"remote_titles": int(w.is_remote.sum()), "remote_share_pct": float(w.is_remote.mean() * 100),
            "weekly_remote_share_pct": [round(float(v), 3) for v in wk.values],
            "trend_slope_pp_per_week": float(lr.slope), "trend_p_value": float(lr.pvalue),
            "daily_trend_p_value": float(p), "forecast_next4_weeks_pct": [round(float(v), 3) for v in fc],
            "forecast_interval_halfwidth_pp": float(1.96 * resid_sd),
            "remote_median_rate": float(rr.median()), "nonremote_median_rate": float(nr.median()), "rate_mannwhitney_p": float(mw.pvalue),
            "remote_rate_n": int(len(rr)),
            "by_category": by_cat.head(6).to_dict("records"), "by_country": by_cty.round(3).to_dict("records"),
            "nonus_share_weekly": [round(float(v), 2) for v in nonus.values],
            "hourly_share_weekly": [round(float(v), 2) for v in hourly_share.values],
            "weekend_share_weekly": [round(float(v), 2) for v in weekend.values]}
