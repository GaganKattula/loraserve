# Stage 1: Build React frontend
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# Stage 2: Python API
FROM python:3.12-slim

WORKDIR /app

# System deps for asyncpg (needs libpq headers at build time)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps — VPS image does NOT include torch/peft/transformers
# (those run on the GPU pod only)
COPY requirements-vps.txt .
RUN pip install --no-cache-dir -r requirements-vps.txt

# Copy application code
COPY config.py db.py storage.py gpu_lock.py watchdog.py sqs.py ./
COPY api/ api/

# Copy built frontend from Stage 1
COPY --from=frontend /app/frontend/dist frontend/dist

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
