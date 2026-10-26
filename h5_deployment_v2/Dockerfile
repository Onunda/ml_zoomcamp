FROM python:3.11.15-slim-bookworm

# Install uv from the official Astral installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency definition files
# COPY pyproject.toml uv.lock ./
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install locked dependencies without dev tools
# RUN uv sync --locked --no-dev
RUN uv sync --locked

# Copy application code and model artifacts
COPY predict.py pipeline.bin model_metadata.json ./

EXPOSE 9696

# Start the application using uv and uvicorn
ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]

