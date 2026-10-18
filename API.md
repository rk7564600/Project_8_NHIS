# Job Recommendation API

Base URL: `http://localhost:8080` (Docker) or `http://localhost:8000` (`python -m jobmarket.app`).
All responses are JSON. Errors have the form `{"error": "..."}` with HTTP 400 (bad input) or 404.

## `GET /api/health`
Liveness check. `{"status":"ok","indexed_jobs":244545}`

## `GET /api/meta`
Values for building filters: `countries`, `categories`, `job_types`, `date_min`, `date_max`.

## `GET|POST /api/recommend`
Rank recent postings for a profile. GET uses query-string parameters, POST accepts the same keys as JSON.

| Parameter | Type | Description |
|---|---|---|
| `skills` (alias `query`) | string | Free text: skills, role, tools. Empty = newest postings matching the filters |
| `job_type` | `hourly` \| `fixed` | Contract type |
| `category` | string | One of `/api/meta` categories |
| `country` | string | Client country (case-insensitive) |
| `min_hourly` | number | Hourly jobs whose upper (or only) bound is at least this; fixed jobs are dropped when `job_type=hourly` |
| `min_budget` | number | Fixed jobs with at least this budget; hourly jobs are dropped when `job_type=fixed` |
| `remote_only` | bool | Only titles that explicitly mention remote / work-from-home |
| `top_k` | int 1-50 | Number of results (default 10) |

Scoring: `0.85 * cosine(TF-IDF(profile), TF-IDF(title)) + 0.15 * recency`. At most 2 postings with an identical title are returned.

```bash
curl -X POST localhost:8080/api/recommend -H 'Content-Type: application/json' \
  -d '{"skills":"video editor youtube shorts","job_type":"hourly","min_hourly":20,"top_k":3}'
```
```json
// example output - values are illustrative
{"count":3,"request":{"query":"video editor youtube shorts","job_type":"hourly","min_hourly":20.0,"top_k":3},
 "results":[{"title":"Video editor for YouTube shorts","link":"https://www.upwork.com/jobs/...","published":"2024-03-20 10:12",
 "type":"hourly","pay":"$10-$30/hr","country":"United States","category":"Video & Audio",
 "score":0.93,"similarity":0.99,"matched_terms":["editor","video","video editor"]}]}
```

## `GET /api/metrics` and `GET /api/metrics/<task>`
Results of the analysis pipeline (`task1` ... `task8`, plus `dataset`). 404 if the task is unknown.

## Other routes
`/` web UI · `/dashboard` monthly dashboard · `/figures/<file>.png` · `/interactive/<file>.html` (country rate map and bar chart).
