# ============================================================
# Dockerfile
# Production container: scheduler + web UI in one process
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# System dependencies (gcc/libpq for psycopg2; curl for the healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Writable dirs for logs, reports, models, and the SQLite fallback DB
RUN mkdir -p logs reports models data_store \
    && useradd --create-home botuser \
    && chown -R botuser:botuser /app
USER botuser

EXPOSE 8000

# Health check hits the web API (verifies web server AND database)
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -sf http://localhost:8000/api/health || exit 1

CMD ["python", "main.py"]
