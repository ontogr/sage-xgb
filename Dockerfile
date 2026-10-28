# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
COPY --from=uv /uv /uvx /usr/local/bin/
WORKDIR /app

# Dependencies go in their own layer, so source edits do not reinstall them.
FROM base AS builder
ENV UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md LICENSE ./
COPY sage_xgb ./sage_xgb
RUN uv sync --locked --no-dev --no-editable

FROM base AS train
COPY --from=builder /app/.venv /app/.venv
COPY pyproject.toml uv.lock .python-version ./
COPY configs ./configs
COPY tests/fixtures/amlworld_hi_small_sample ./tests/fixtures/amlworld_hi_small_sample
# The dev dependency group is absent from the image, so the entrypoint must not
# sync the environment again at start.
ENTRYPOINT ["uv", "run", "--no-sync", "sage-xgb"]
CMD ["--help"]
