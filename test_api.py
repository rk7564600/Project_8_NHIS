import pandas as pd
import pytest
from jobmarket.app import create_app


@pytest.fixture(scope="module")
def client():
    # small synthetic frame so tests run in seconds; schema mirrors prep.load() output
    n = 300
    t = ["python flask developer", "logo design", "video editor youtube", "virtual assistant", "react developer"] * (n // 5)
    df = pd.DataFrame({"title": t, "title_norm": t, "link": [f"https://x/{i}" for i in range(n)],
                       "published": pd.date_range("2024-03-01", periods=n, freq="h"), "is_hourly": [i % 2 == 0 for i in range(n)],
                       "hourly_low": [10.0 if i % 2 == 0 else None for i in range(n)], "hourly_high": [30.0 if i % 2 == 0 else None for i in range(n)],
                       "budget": [None if i % 2 == 0 else 200.0 for i in range(n)], "country": ["United States", "India"] * (n // 2),
                       "category": ["Web & Software Dev", "Design & Creative", "Video & Audio", "Admin & Virtual Assistant", "Web & Software Dev"] * (n // 5),
                       "is_remote": [False] * n})
    df["date"] = df.published.dt.normalize()
    return create_app(df, output_dir="/tmp/none").test_client()


def test_health(client):
    assert client.get("/api/health").json["status"] == "ok"


def test_recommend_relevant(client):
    r = client.post("/api/recommend", json={"skills": "flask developer", "top_k": 3})
    assert r.status_code == 200 and r.json["count"] > 0
    assert "flask" in r.json["results"][0]["title"]


def test_filters(client):
    r = client.get("/api/recommend?skills=developer&job_type=hourly&country=India")
    assert all(x["type"] == "hourly" and x["country"] == "India" for x in r.json["results"])


def test_validation(client):
    assert client.get("/api/recommend?job_type=weekly").status_code == 400
    assert client.get("/api/recommend?min_hourly=abc").status_code == 400
    assert client.get("/api/recommend?top_k=500").status_code == 400
    assert client.get("/api/metrics/task99").status_code == 404
