# Web/API service: Flask app served by gunicorn
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_PATH=/app/data/data.csv OUTPUT_DIR=/app/output
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY jobmarket ./jobmarket
EXPOSE 8000
# 1 worker: the TF-IDF index (~250 MB RAM) is built once at start-up; use threads for concurrency
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "180", "jobmarket.app:get_app()"]
