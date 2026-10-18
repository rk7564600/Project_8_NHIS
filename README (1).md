# Job Market Analysis and Recommendation System

Analysis of 244k Upwork job postings (13 Feb - 24 Mar 2024) covering all eight project tasks, a recommendation API + UI, a monthly dashboard and Docker deployment. **Start with `REPORT.md`** (results, caveats) and `docs/API.md` (API).

| Task | Where |
|---|---|
| 1 Keyword-salary correlation | `jobmarket/analysis.py::task1_keyword_salary` -> `output/figures/task1_*.png`, `output/tables/task1_*.csv` |
| 2 Emerging categories | `analysis.py::task2_emerging` |
| 3 High-demand role prediction | `jobmarket/forecast.py::task3_demand` (model comparison, 14-day forecast) |
| 4 Hourly rates by country | `analysis.py::task4_country_rates` -> `output/interactive/task4_hourly_rate_map.html` |
| 5 Recommendation engine | `jobmarket/recommender.py`, `jobmarket/app.py`, `jobmarket/templates/index.html`, `docs/API.md` |
| 6 Monthly dashboard | `jobmarket/dashboard.py` -> `output/dashboard.html`; `jobmarket/scheduler.py` |
| 7 Remote-work trends | `analysis.py::task7_remote` |
| 8 Future scenarios | `forecast.py::task8_scenarios` |

## Run locally (no Docker)
```bash
pip install -r requirements-dev.txt
python -m jobmarket.pipeline          # ~1 min: writes everything under output/ (reads data/data.csv)
python -m jobmarket.app               # http://localhost:8000  (UI at /, dashboard at /dashboard, API under /api)
pytest                                # API tests
python make_presentation.py           # rebuilds Project8_Presentation.pptx from output/metrics.json
```
Environment variables: `DATA_PATH` (default `data/data.csv`), `OUTPUT_DIR` (default `output/`).

## Docker
Services: `app` (gunicorn/Flask API+UI), `scheduler` (runs the pipeline at start-up and on the 1st of each month), `web` (nginx reverse proxy on port 8080).
```bash
docker compose build
docker compose up -d                  # first start: the scheduler builds the outputs (~1-2 min), app index takes ~30 s
open http://localhost:8080            # UI;  /dashboard ;  /api/health
docker compose logs -f scheduler
```
**Monthly update:** replace `data/data.csv` with the new export; the scheduler refreshes on the 1st of the month. To refresh immediately: `docker compose exec scheduler python -m jobmarket.pipeline`, then `docker compose restart app` to reload the recommender index.
Note: the Docker files were written but not built/run in the authoring environment (no Docker available) - run `docker compose build` once to confirm on your machine.

## Notes
* The analysis window constants (`START`, `TREND_END`, `LAST_FULL`) are in `jobmarket/prep.py`; update them when the dataset grows beyond March 2024. The dashboard detects the last complete day automatically.
* Category and remote flags are keyword rules in `prep.py` - edit `CATEGORIES` / `REMOTE_RE` to refine.
