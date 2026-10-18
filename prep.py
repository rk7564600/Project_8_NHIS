"""Data loading, cleaning and feature engineering shared by every task."""
import html
import os
import re
from pathlib import Path

import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = Path(os.environ.get("DATA_PATH", ROOT / "data" / "data.csv"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", ROOT / "output"))

# Postings before 13 Feb 2024 are sparse (1-30/day vs ~7000/day) -> scrape had not
# started properly, so trend analysis uses only the complete window below.
START = pd.Timestamp("2024-02-13")
TREND_END = pd.Timestamp("2024-03-18")   # 5 complete 7-day bins: 13 Feb .. 18 Mar
LAST_FULL = pd.Timestamp("2024-03-23")   # 24 Mar is a partial day

# Ordered: first match wins.
CATEGORIES = [
    ("AI & Automation", r"\b(ai|a\.i\.?|chatgpt|gpt\d*|openai|machine learning|ml|llm|midjourney|prompt|artificial intelligence|automation|chatbot|zapier|make\.com|n8n)\b"),
    ("Social Media", r"social media|instagram|\bsmm\b|influencer|tiktok|linkedin|pinterest|community manager|facebook page"),
    ("Video & Audio", r"video|editor|youtube|\breels?\b|podcast|audio|voice|music|motion|film|animat|editing|thumbnail"),
    ("Design & Creative", r"logo|design|graphic|brand|illustrat|ui/?ux|figma|canva|photoshop|packaging|3d|flyer|poster|mockup|architect"),
    ("Web & Software Dev", r"develop|web ?site|wordpress|shopify|react|python|javascript|\bapp\b|software|php|laravel|node|full ?stack|front.?end|back.?end|wix|webflow|squarespace|\bios\b|android|programmer|coding|\bapi\b|bubble|engineer|\bgohighlevel\b|\bghl\b"),
    ("Data & Analytics", r"\bdata\b(?! entry)|analyst|analytics|excel|scrap|power ?bi|tableau|\bsql\b|spreadsheet|google sheets"),
    ("Writing & Translation", r"writ|copy|content|blog|article|translat|proofread|edit|ghost|script|transcri|ebook|book|resume|\bcv\b"),
    ("Sales & Lead Gen", r"sales|lead gen|\blead\b|appointment|setter|closer|cold (call|email)|outreach|\bsdr\b|b2b|telemarket|recruit"),
    ("Marketing & SEO", r"marketing|\bseo\b|\bads?\b|ppc|email|funnel|media buyer|growth|e-?commerce|amazon|affiliate|campaign|klaviyo|clickfunnels"),
    ("Customer Support", r"customer|support|service|chat agent|helpdesk"),
    ("Admin & Virtual Assistant", r"virtual assistant|\bva\b|assistant|admin|data entry|secretar|executive|bookkeep|account|research|scheduler|manager|project"),
]
REMOTE_RE = re.compile(r"remote|work from home|\bwfh\b|home.?based|telecommut|anywhere in the world|fully distributed", re.I)
warnings.filterwarnings("ignore", message=".*match groups.*")
_CAT_RE = [(n, re.compile(p, re.I)) for n, p in CATEGORIES]

# very light canonicalisation so "Logo Design"/"Logo Designer"/"logo design" collapse
_CANON = {"designer": "design", "designers": "design", "developer": "development", "developers": "development",
          "editor": "editing", "editors": "editing", "writer": "writing", "writers": "writing", "assistant": "assistant"}


def norm_title(s: str) -> str:
    s = re.sub(r"[^a-z0-9+#&\s]", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def role_key(s: str) -> str:
    return " ".join(_CANON.get(w, w) for w in norm_title(s).split())


def categorize(titles: pd.Series) -> pd.Series:
    out = pd.Series("Other", index=titles.index, dtype=object)
    todo = pd.Series(True, index=titles.index)
    for name, rx in _CAT_RE:
        hit = todo & titles.str.contains(rx, na=False)
        out[hit] = name
        todo &= ~hit
    return out


def load(path=None) -> pd.DataFrame:
    """Load + clean. Adds: date, category, role, is_remote, hourly_rate, budget_c, title_norm."""
    df = pd.read_csv(path or DATA_PATH)
    df["title"] = df["title"].fillna("").astype(str).map(html.unescape)
    df = df.drop_duplicates("link").reset_index(drop=True)
    df["published"] = pd.to_datetime(df["published_date"], utc=True).dt.tz_convert(None)
    df["date"] = df["published"].dt.normalize()
    df["is_hourly"] = df["is_hourly"].astype(bool)
    for c in ("hourly_low", "hourly_high", "budget"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["country"] = df["country"].fillna("Unknown")
    df["title_norm"] = df["title"].map(norm_title)
    df["role"] = df["title"].map(role_key)
    df["category"] = categorize(df["title_norm"])
    df["is_remote"] = df["title"].str.contains(REMOTE_RE, na=False)
    # hourly rate = midpoint when a range is given, otherwise whichever bound exists
    mid = df[["hourly_low", "hourly_high"]].mean(axis=1)
    df["hourly_rate"] = mid.where(df["is_hourly"])
    # outlier handling: rates $3-$200/h, fixed budgets $5-$100k (max in raw data is $1M)
    df.loc[~df["hourly_rate"].between(3, 200), "hourly_rate"] = np.nan
    df["budget_c"] = df["budget"].where(df["budget"].between(5, 100_000) & ~df["is_hourly"])
    df["day_bin"] = ((df["date"] - START).dt.days // 7).where(df["date"].between(START, TREND_END))
    return df


def window(df, end=LAST_FULL):
    """Complete-coverage window used for all time-series work."""
    return df[df["date"].between(START, end)]


def data_end(df):
    """Last day with (near-)complete coverage: skips a trailing partial day so monthly refreshes stay correct."""
    c = df[df["date"] >= START].groupby("date").size()
    ref = c.tail(15).iloc[:-1].median()
    return c.index[-1] if c.iloc[-1] >= 0.6 * ref else c.index[-2]
