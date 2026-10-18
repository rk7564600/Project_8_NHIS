"""Flask API + UI for the recommender (Task 5) and dashboard hosting (Task 6)."""
import json
import os

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from pathlib import Path

from . import prep
from .recommender import Recommender


def create_app(df=None, output_dir=None):
    app = Flask(__name__)
    out = Path(output_dir or prep.OUTPUT_DIR)
    df = prep.load() if df is None else df
    rec = Recommender(df)
    state = {"rec": rec}

    def metrics():
        p = out / "metrics.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", indexed_jobs=int(state["rec"].X.shape[0]))

    @app.get("/api/meta")
    def meta():
        r = state["rec"]
        return jsonify(countries=r.countries, categories=r.categories, job_types=["hourly", "fixed"],
                       date_min=str(r.df.published.min().date()), date_max=str(r.df.published.max().date()))

    def _params():
        src = request.get_json(silent=True) if request.method == "POST" else request.args
        src = src or {}
        def num(k):
            v = src.get(k)
            if v in (None, ""):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                abort(400, description=f"'{k}' must be a number")
        jt = src.get("job_type") or None
        if jt not in (None, "hourly", "fixed"):
            abort(400, description="'job_type' must be 'hourly' or 'fixed'")
        k = int(num("top_k") or 10)
        if not 1 <= k <= 50:
            abort(400, description="'top_k' must be between 1 and 50")
        rem = src.get("remote_only", False)
        return dict(query=str(src.get("skills") or src.get("query") or ""), job_type=jt, country=src.get("country") or None,
                    category=src.get("category") or None, min_hourly=num("min_hourly"), min_budget=num("min_budget"),
                    remote_only=str(rem).lower() in ("1", "true", "yes", "on"), top_k=k)

    @app.route("/api/recommend", methods=["GET", "POST"])
    def recommend():
        p = _params()
        res = state["rec"].recommend(**p)
        return jsonify(count=len(res), request={k: v for k, v in p.items() if v not in (None, "", False)}, results=res)

    @app.get("/api/metrics")
    def all_metrics():
        return jsonify(metrics())

    @app.get("/api/metrics/<task>")
    def one_metric(task):
        m = metrics()
        if task not in m:
            abort(404, description=f"unknown task '{task}'; available: {sorted(m)}")
        return jsonify(m[task])

    @app.get("/dashboard")
    def dash():
        if not (out / "dashboard.html").exists():
            abort(404, description="dashboard not built yet - run `python -m jobmarket.pipeline`")
        return send_from_directory(out, "dashboard.html")

    @app.get("/figures/<path:name>")
    def fig(name):
        return send_from_directory(out / "figures", name)

    @app.get("/interactive/<path:name>")
    def interactive(name):
        return send_from_directory(out / "interactive", name)

    @app.errorhandler(400)
    @app.errorhandler(404)
    def err(e):
        return jsonify(error=e.description), e.code

    return app


app = None
def get_app():           # gunicorn "jobmarket.app:get_app()"
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
