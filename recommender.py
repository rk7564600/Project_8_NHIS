"""Content-based job recommender: TF-IDF over job titles + hard filters + recency boost."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .prep import START


class Recommender:
    W_SIM, W_RECENCY = 0.85, 0.15

    def __init__(self, df: pd.DataFrame):
        d = df[df["date"] >= START].drop_duplicates("link").reset_index(drop=True)
        self.df = d
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=80_000, stop_words="english", sublinear_tf=True)
        self.X = self.vec.fit_transform(d["title_norm"])
        self.analyzer = self.vec.build_analyzer()
        ts = d["published"].astype("int64")
        self.recency = ((ts - ts.min()) / max(ts.max() - ts.min(), 1)).values
        self.countries = sorted(c for c in d["country"].unique() if c != "Unknown")
        self.categories = sorted(d["category"].unique())

    def recommend(self, query="", job_type=None, country=None, category=None, min_hourly=None, min_budget=None,
                  remote_only=False, top_k=10, max_per_title=2):
        d = self.df
        mask = np.ones(len(d), bool)
        if job_type in ("hourly", "fixed"):
            mask &= (d["is_hourly"] == (job_type == "hourly")).values
        if country:
            mask &= (d["country"].str.lower() == country.lower()).values
        if category:
            mask &= (d["category"].str.lower() == category.lower()).values
        if remote_only:
            mask &= d["is_remote"].values
        if min_hourly is not None:       # keep hourly jobs whose upper bound (or midpoint) meets the target
            ok = d[["hourly_high", "hourly_low"]].max(axis=1)
            mask &= ((~d["is_hourly"]) | (ok >= min_hourly)).values if job_type != "hourly" else (ok >= min_hourly).fillna(False).values
        if min_budget is not None:
            mask &= (d["is_hourly"] | (d["budget"] >= min_budget)).values if job_type != "fixed" else (d["budget"] >= min_budget).fillna(False).values

        q = (query or "").strip()
        if q:
            qv = self.vec.transform([q.lower()])
            if qv.nnz == 0:
                return []
            sim = (self.X @ qv.T).toarray().ravel()
            score = np.where(mask & (sim > 0), self.W_SIM * sim + self.W_RECENCY * self.recency, -1.0)
            qtok = set(self.analyzer(q.lower()))
        else:                             # no skills given -> newest jobs matching filters
            sim = np.zeros(len(d))
            score = np.where(mask, self.recency, -1.0)
            qtok = set()
        n = min(len(d), max(top_k * 40, 200))
        cand = np.argpartition(-score, n - 1)[:n]
        cand = cand[score[cand] >= 0]
        cand = cand[np.argsort(-score[cand])]
        out, seen = [], {}
        for i in cand:
            t = d.at[i, "title_norm"]
            if seen.get(t, 0) >= max_per_title:
                continue
            seen[t] = seen.get(t, 0) + 1
            r = d.iloc[i]
            pay = (f"${r.hourly_low:g}-${r.hourly_high:g}/hr" if pd.notna(r.hourly_low) and pd.notna(r.hourly_high)
                   else f"${r.hourly_low:g}+/hr" if pd.notna(r.hourly_low) else "hourly (rate not stated)") if r.is_hourly else \
                  (f"${r.budget:,.0f} fixed" if pd.notna(r.budget) else "fixed (budget not stated)")
            out.append({"title": r.title, "link": r.link, "published": r.published.strftime("%Y-%m-%d %H:%M"), "type": "hourly" if r.is_hourly else "fixed",
                        "pay": pay, "country": r.country, "category": r.category, "score": round(float(score[i]), 4),
                        "similarity": round(float(sim[i]), 4), "matched_terms": sorted(qtok & set(self.analyzer(t)))})
            if len(out) >= top_k:
                break
        return out

    def evaluate(self, n=400, k=10, seed=0):
        """Offline sanity check: use a random job's title as the 'profile'; a recommendation is relevant
        if it falls in the same category (the engine itself never sees the category)."""
        rng = np.random.default_rng(seed)
        pool = self.df.index[(self.df["category"] != "Other") & (self.df["title_norm"].str.split().str.len() >= 2)]
        picks = rng.choice(pool, n, replace=False)
        prec, hits, ndcg = [], 0, []
        for i in picks:
            r = self.df.loc[i]
            recs = self.recommend(r.title_norm, top_k=k + 1, max_per_title=1)
            recs = [x for x in recs if x["link"] != r.link][:k]
            if not recs:
                continue
            rel = np.array([x["category"] == r.category for x in recs], float)
            prec.append(rel.mean()); hits += rel.any()
            ndcg.append((rel / np.log2(np.arange(2, len(rel) + 2))).sum() / (1 / np.log2(np.arange(2, k + 2))).sum())
        base = (self.df["category"].value_counts(normalize=True) ** 2).sum()   # chance of a random same-category hit
        return {"queries": len(prec), f"precision_at_{k}": float(np.mean(prec)), f"hit_rate_at_{k}": hits / len(prec),
                f"ndcg_at_{k}": float(np.mean(ndcg)), "random_baseline_precision": float(base)}
