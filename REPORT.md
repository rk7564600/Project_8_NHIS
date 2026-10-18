# Job Market Analysis and Recommendation System - Project Report

**Dataset:** `data.csv`, 244,828 Upwork job postings (title, link, publish time, hourly/fixed flag, hourly range, fixed budget, client country).
**Code:** `jobmarket/` package - every number below is produced by `python -m jobmarket.pipeline` and stored in `output/metrics.json`; tables are in `output/tables/`, figures in `output/figures/`.

## 0. Read this first: what the data can and cannot support

* **Time span is short.** Only 283 postings are dated before 13 Feb 2024 (1-30 per day); from 13 Feb to 23 Mar 2024 there are ~242,000 (about 7,000 per weekday). The scrape effectively covers **~5.5 weeks**. All trend work uses the complete window 13 Feb - 18 Mar (five 7-day bins); 19-23 Mar is held out for forecast testing; 24 Mar is a partial day and is excluded.
* **"Monthly" is therefore weekly.** Only February (17 days) and March (23 days) exist. Trend claims are about weeks, not seasons, and a 5-point trend line is weak evidence. Where relevant, p-values are reported and results that would not survive a multiple-comparison correction are flagged.
* **Scrape gap:** 15 Feb has 1,811 postings versus ~6,000+ on neighbouring days. Shares are robust to this; raw volumes in week 1 are not (week 1 is excluded from volume trend estimates).
* **Cleaning:** duplicates by link (0 found); HTML entities decoded; hourly rate = midpoint of the stated range (or the one bound given), valid range $3-$200/hr (386 outlier rows removed, raw max $999); fixed budgets valid $5-$100,000 (77 removed; raw max $1,000,000, which had pushed the raw mean to $911 versus a median of $100). Medians are quoted for pay because ranges sit on a coarse grid ($22.5 = midpoint of $15-$30 etc.).
* **No category, remote or skill columns exist.** Categories (11 + "Other", 19.6% unclassified) were assigned by an ordered keyword rule set in `prep.py`; "remote" can only be detected from explicit wording in titles. Both are approximations.

## 1. Keyword vs salary correlation (Task 1)
Method: titles tokenised (stop words removed, keywords in >=300 titles); for each keyword the Pearson (point-biserial) correlation between "keyword present" and log10(pay), with a two-sided t-test; medians with/without the keyword. Hourly n=102,036, fixed n=103,814. 206 of 289 hourly keywords and 184 of 272 fixed keywords are significant at p<0.001 (large n makes small effects significant, so look at effect size).

![Task 1](output/figures/task1_keyword_salary.png)

| Segment | Higher pay keywords (median with vs without) | Lower pay keywords |
|---|---|---|
| Hourly (overall median $22.5) | consultant $40, tax $50, videographer $42.5, consultation $48.5, engineer $30 | assistant $8.5, virtual $7.5, va $6, lead $10, cold $8 |
| Fixed (overall median $100) | development $300, developer $150, sales $250, build $200, engineer $300 | need $50, logo $50, fix $30, speakers $10, recording $15 |

**Finding:** seniority/specialist words (consultant, engineer, expert, developer) go with higher pay; admin and lead-generation wording goes with much lower pay. Individual correlations are small (|r| <= 0.22), and a ridge model using all title keywords explains only **26% (hourly) and 20% (fixed)** of held-out variance in log pay - titles carry real but limited salary signal. Full tables: `task1_keywords_hourly.csv`, `task1_keywords_fixed.csv`.

## 2. Emerging job categories (Task 2)
Method: daily category share of postings (13 Feb - 18 Mar), linear trend (slope in percentage points/week, p-value) and share change between the first and last week.

![Task 2](output/figures/task2_emerging.png)

| Category | Share wk 1 -> wk 5 | Change | Trend p | Status |
|---|---|---|---|---|
| Video & Audio | 10.8% -> 11.4% | +5.4% | 0.006 | **Emerging** |
| Customer Support | 1.0% -> 1.2% | +16.8% | 0.38 | not significant (small base) |
| AI & Automation | 3.1% -> 3.2% | +4.2% | 0.22 | stable |
| Web & Software Dev | 19.4% -> 19.8% | +2.0% | 0.22 | stable |
| Admin & Virtual Assistant | 5.5% -> 5.2% | -5.9% | 0.033 | Declining |
| Writing & Translation | 9.2% -> 8.1% | -12.7% | 0.0003 | **Declining** |

Sub-niche keywords with a significant (p<0.001) rise in share: **crypto (+49%)**, **js (+35%)**, **facebook (+25%)**, **ads (+18%)**; falling: writers (-51%), entry (-43%), spanish (-40%), writing (-38%), data (-20%).
**Caveat:** with 12 categories, a Bonferroni threshold is p<0.004: only the Writing & Translation decline clearly passes; Video & Audio (p=0.006) and Admin/VA (p=0.033) are suggestive. Growth over five weeks is not evidence of a lasting trend.

## 3. Predicting high-demand roles (Task 3)
Target: daily postings per category (12 series). Features: change vs same weekday last week, change of the previous day, 7-day mean, day-of-week, category. Models predict the *change from last week's same day* in log space, which avoids the downward drift an absolute-level recursive model showed in testing. Train 20 Feb - 18 Mar (336 rows); **test 19-23 Mar (5 days, recursive multi-step)**.

| Model | MAE (postings/day/category) | MAPE | R2 | Spearman rank | Top-3 categories |
|---|---|---|---|---|---|
| seasonal_naive | 33.1 | 8.3% | 0.987 | 0.99 | 3/3 |
| mean7 | 70.6 | 15.1% | 0.942 | 0.99 | 3/3 |
| Ridge | 36.9 | 10.1% | 0.985 | 0.99 | 3/3 |
| GradientBoosting | 31.6 | 11.5% | 0.990 | 0.99 | 3/3 |
| RandomForest | 33.8 | 11.1% | 0.989 | 0.98 | 3/3 |
| Ensemble(Ridge+seasonal_naive) | 30.6 | 8.5% | 0.989 | 0.99 | 3/3 |

The best learned model (Ridge) does **not** beat the simple seasonal-naive baseline on MAPE; averaging the two gives the lowest MAE (30.6 vs 33.1) with MAPE about equal to the baseline (8.5% vs 8.3%). Because the holdout is only 5 days, these differences are not statistically meaningful - the honest reading is "weekly seasonality explains almost everything; ML adds little over 5 weeks of data". The ensemble was used for the forecast. The model ranks categories almost perfectly (Spearman 0.99, top-3 correct).

![Task 3](output/figures/task3_demand_forecast.png)

**Forecast 24 Mar - 6 Apr 2024 (14 days) versus the last 14 observed days:**

| Category | Last 14 d actual | Next 14 d forecast | Change | Share of forecast |
|---|---|---|---|---|
| Other | 16,760 | 17,321 | +3.4% | 20.1% |
| Web & Software Dev | 16,888 | 17,110 | +1.3% | 19.9% |
| Design & Creative | 12,203 | 12,256 | +0.4% | 14.2% |
| Video & Audio | 9,727 | 10,038 | +3.2% | 11.7% |
| Writing & Translation | 6,722 | 6,145 | -8.6% | 7.1% |
| Social Media | 4,726 | 4,647 | -1.7% | 5.4% |
| Marketing & SEO | 4,378 | 4,468 | +2.0% | 5.2% |
| Admin & Virtual Assistant | 4,389 | 4,243 | -3.3% | 4.9% |
| Sales & Lead Gen | 3,632 | 3,670 | +1.0% | 4.3% |
| AI & Automation | 2,713 | 2,699 | -0.5% | 3.1% |
| Data & Analytics | 2,507 | 2,475 | -1.3% | 2.9% |
| Customer Support | 990 | 979 | -1.1% | 1.1% |

High-demand roles (normalised titles; forecast = role's share of its category x category forecast):

| Role | Category | Postings observed | Forecast next 14 d |
|---|---|---|---|
| logo design | Design & Creative | 657 | 235 |
| video editing | Video & Audio | 563 | 212 |
| social media manager | Social Media | 507 | 178 |
| graphic design | Design & Creative | 448 | 160 |
| virtual assistant | Admin & Virtual Assistant | 408 | 135 |
| website development | Web & Software Dev | 252 | 91 |
| youtube video editing | Video & Audio | 214 | 80 |
| full stack development | Web & Software Dev | 217 | 78 |
| website design | Design & Creative | 197 | 71 |
| lead generation | Sales & Lead Gen | 172 | 63 |

## 4. Hourly rates by country (Task 4)
Valid hourly rates from 93 countries with >=30 rated jobs (46 with >=200). Interactive choropleth: `output/interactive/task4_hourly_rate_map.html`; bar chart with 95% CIs: `task4_hourly_rate_bar.html`; table: `task4_country_hourly_rates.csv`.

![Task 4](output/figures/task4_country_rates.png)

* Overall: median $22.5/hr, mean $27.98/hr. United States (n=47,975): median $25, mean $30.70.
* Country differences are real (Kruskal-Wallis H=2,220, p<0.001 over the 46 countries with n>=200).
* Lowest median rates among larger markets: South Korea $10, Philippines $11, Japan $15, India and Pakistan $18.5. Note the Philippines mean ($24.5) is more than double its median - a mix of cheap admin work and a high-priced tail - so the mean alone misleads.
* The top of the ranking (Vietnam, Kenya $26.5-27.5; Poland, Turkey, Denmark, Saudi Arabia, Ukraine, US $25) is a near-tie with wide intervals for the small countries (n 250-600). Do not read the order among them as meaningful.
* This is the rate **clients in each country offer**, not what freelancers in that country earn.

## 5. Recommendation engine (Task 5)
Content-based: TF-IDF (unigrams+bigrams, 35,835 terms) over 244,545 deduplicated posting titles, cosine similarity to the user's skills text, plus a 15% recency weight; hard filters for contract type, country, category, minimum hourly rate, minimum fixed budget, explicit-remote titles; at most 2 identical titles per result list. Query latency ~0.1 s.
Deliverables: `jobmarket/recommender.py`, Flask API (`jobmarket/app.py`, documented in `docs/API.md`), web UI at `/`, unit tests (`tests/test_api.py`).

**Offline evaluation** (400 random postings used as a "profile"; a recommendation counts as relevant when its category matches the source posting's category - the engine never sees categories): precision@10 = **0.64** (random baseline 0.13), hit-rate@10 = 0.96, nDCG@10 = 0.65. This measures topical relevance only; there is no user-interaction data, so personalisation quality (clicks, applications) is untested. The engine matches on titles only because descriptions are not in the dataset.

## 6. Monthly dashboard (Task 6)
`output/dashboard.html` (served at `/dashboard`): KPI cards (last 7 days vs prior 7), daily volume with range buttons, weekly category mix, pay trends, top countries, and a month-by-month table. Observed so far: Feb (17 days) 5,983 postings/day, mean hourly rate $28.03, 58.3% hourly; Mar (23 days) 6,115/day (+2.2%), $28.01 (-0.1%), 57.0% hourly. The month-to-month changes are small. The `scheduler` container re-runs the whole pipeline on the 1st of each month using the latest `data/data.csv`; the dashboard picks the last complete day automatically.

## 7. Remote-work trends (Task 7)
Remote status can only be inferred from explicit title wording ("remote", "work from home", "WFH", ...): **1,182 titles = 0.56%**. Every Upwork job is performed online, so this measures how often clients *advertise* remote work, not how many jobs are remote.

![Task 7](output/figures/task7_remote.png)

* Weekly explicit-remote share: 0.60, 0.55, 0.53, 0.55, 0.58% - no trend (slope -0.002 pp/week, p=0.83). Linear forecast for the next four weeks: ~0.55% (+-0.06 pp).
* Highest explicit-remote shares: Customer Support 3.8%, Sales & Lead Gen 2.1%, Admin/VA 1.5%.
* Remote-worded hourly jobs have a median of $25 versus $22.5 for others, but the difference is not significant (n=695, p=0.42).
* Context indicators are flat as well: non-US clients 56.6-57.7% of postings each week; hourly-contract share drifted from 58.5% to 56.3%; weekend postings ~21% (25% in week 1 because of the scrape gap).
* **Forecast statement:** with five weeks of data, the defensible forecast is "no change"; a claim about a long-run shift towards remote work cannot be made from this dataset.

## 8. Future market scenarios (Task 8)
Scenarios are explicit assumptions anchored on the observed weeks, not predictions of external events.

![Task 8](output/figures/task8_scenarios.png)

**Posting volume** (weeks 2-5 declined by about 1.9% per week; last observed week = 42,993 postings): 

| Scenario | Assumption | Week +1 | Week +4 |
|---|---|---|---|
| Pessimistic | decline doubles to -3.7%/week | 41,394 | 36,948 |
| Base | observed -1.9%/week continues | 42,193 | 39,884 |
| Optimistic | recovery of +1.9%/week | 43,792 | 46,279 |

Note the short-term Task 3 model, which keeps weekday levels flat, forecasts ~6,400-7,000 postings per weekday for the next two weeks; the base scenario's slow decline is the longer-horizon assumption. Weekly mean hourly rate has been flat ($27.5-$28.5; trend +0.08 $/week, p=0.57), so rate scenarios (week +4: $27.72 / $28.22 / $28.72) differ by under a dollar.

**Category share (percent of postings), now vs +4 weeks** under (base) current linear trend persists, (accelerated) trend doubles, (reversion) shares return to the 5-week mean:

| Category | Now | Trend pp/wk | Base +4w | Accelerated +4w | Reversion |
|---|---|---|---|---|---|
| Web & Software Dev | 19.7% | +0.10 | 20.1% | 20.5% | 19.4% |
| Other | 19.3% | +0.08 | 19.6% | 19.9% | 19.5% |
| Design & Creative | 14.4% | +0.10 | 14.7% | 15.1% | 14.1% |
| Video & Audio | 11.3% | +0.19 | 12.0% | 12.8% | 11.0% |
| Writing & Translation | 8.1% | -0.50 | 6.1% | 4.1% | 8.9% |
| Social Media | 5.6% | +0.07 | 5.8% | 6.1% | 5.5% |
| Admin & Virtual Assistant | 5.2% | -0.12 | 4.7% | 4.2% | 5.3% |
| Marketing & SEO | 5.1% | +0.02 | 5.2% | 5.2% | 5.0% |
| Sales & Lead Gen | 4.1% | +0.02 | 4.2% | 4.3% | 4.1% |
| AI & Automation | 3.2% | +0.04 | 3.4% | 3.6% | 3.1% |
| Data & Analytics | 2.9% | -0.03 | 2.7% | 2.6% | 2.9% |
| Customer Support | 1.2% | +0.04 | 1.4% | 1.6% | 1.1% |

**Narrative.** (1) Most likely: a stable market - same top-5 categories (Web/Software Dev, Design, Video, Writing, Social Media) with a weekday-driven weekly cycle and about one-third fewer postings per day on weekends. (2) Share shifts, if they continue, favour Video & Audio (+0.19 pp/week) at the expense of Writing & Translation (-0.50 pp/week, the one clearly significant move). (3) Pay and remote-wording indicators show no movement and should be monitored rather than forecast. Updating the dataset monthly via the scheduler will let these scenarios be tested against reality.

## 9. Docker deployment
Files: `Dockerfile.app` (gunicorn + Flask), `Dockerfile.scheduler` (monthly pipeline), `nginx/Dockerfile` + `nginx.conf` (reverse proxy), `docker-compose.yml` (3 services, shared volume for outputs, healthcheck). Build and run instructions are in `README.md`. **These files could not be built or run in the environment where the project was developed (no Docker daemon available); the Python application itself was run and tested directly.**

## 10. Limitations summary
~5.5 weeks of data; categories and remote flags are rule-based; pay is the advertised midpoint, not the final contract value; the recommender uses titles only and was evaluated offline; model tests cover a 5-day holdout; the technology list in the brief mentions TensorFlow/SQL - scikit-learn and pandas were sufficient at this scale and were used instead.
