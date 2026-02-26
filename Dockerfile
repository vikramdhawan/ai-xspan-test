FROM python:3.12-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code and data
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
COPY .env.example ./

# Entrypoint: auto-ingests on first run, then starts the CLI
ENTRYPOINT ["sh", "scripts/entrypoint.sh"]
