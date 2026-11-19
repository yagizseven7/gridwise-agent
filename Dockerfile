FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY knowledge/ ./knowledge/
COPY scripts/ ./scripts/
COPY evals/ ./evals/
COPY integrations/ ./integrations/

# Build the telemetry database at image build time so the container starts warm.
RUN python -c "import sys; sys.path.insert(0,'src'); from gridwise.data import build_database; build_database()"

# Run as a non-root user.
RUN useradd --create-home --uid 1000 gridwise && chown -R gridwise:gridwise /app
USER gridwise

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "gridwise.api:app", "--host", "0.0.0.0", "--port", "8000"]
