# ─────────────────────────────────────────────────────────────────────
# Stage 1: builder — installa dipendenze in un venv isolato
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY app/ ./app/

RUN python -m venv /app/venv && \
    /app/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /app/venv/bin/pip install --no-cache-dir .

# ─────────────────────────────────────────────────────────────────────
# Stage 2: runtime — immagine finale leggera
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia il venv dallo stage builder
COPY --from=builder /app/venv /app/venv

# Copia il codice sorgente
COPY app/ ./app/
COPY static/ ./static/
COPY config.yaml ./config.yaml

# Directory per modelli e report (sovrascritta dai volumi Docker)
RUN mkdir -p /app/models /app/eval_reports

# Utente non-root per sicurezza (uid 1000)
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

ENV PATH="/app/venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
