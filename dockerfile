# Base: Python 3.12 slim
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for building torch/transformers and optional vLLM
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files (adjust if you use pyproject.toml/requirements.txt)
COPY requirements.txt /app/requirements.txt
# If you have a pyproject.toml use that instead:
# COPY pyproject.toml /app/pyproject.toml

# Install Python deps
# Include FastAPI/uvicorn, transformers, torch, captum, inseq, vllm (optional), pandas, scikit-learn
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt 

# Copy source
COPY src /app/src
COPY script /app/script
COPY config /app/config
# COPY services /app/services
# COPY data /app/data

COPY openapi.yaml /app/openapi.yaml
ENV OPENAPI_YAML_PATH=/app/openapi.yaml

# Expose API port
EXPOSE 8000

# Default command: start the scorer API (adjust module path if different)
# Example: services/explainer_scorer/app.py contains FastAPI app `app`
# CMD ["uvicorn", "services.explainer_scorer.app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]