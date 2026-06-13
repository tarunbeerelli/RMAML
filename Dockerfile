# ── Base image ────────────────────────────────────────────────────────────────
# PyTorch 2.2 with CUDA 12.1 — matches our training environment exactly
# Use -runtime for inference/CI, -devel only if compiling CUDA extensions
FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy dependency files first — Docker layer caching means this layer
# only rebuilds when pyproject.toml or poetry.lock changes, not on code changes
COPY pyproject.toml poetry.lock ./

RUN pip install --no-cache-dir poetry==1.8.2 && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-root

# ── Application code ──────────────────────────────────────────────────────────
# Copy after deps so code changes don't invalidate the dependency layer
COPY src/ ./src/
COPY train.py evaluate.py tune.py ./
COPY configs/ ./configs/

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app/src

# ── Default command ───────────────────────────────────────────────────────────
# Smoke test verifies the full pipeline works without real data or GPU
# Override with: docker run rmaml python train.py --config configs/conv4_miniimagenet.yaml
CMD ["python", "train.py", "--config", "configs/conv4_miniimagenet.yaml", "--smoke-test"]
