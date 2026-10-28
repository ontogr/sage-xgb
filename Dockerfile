# syntax=docker/dockerfile:1

# The lock installs pyg-lib, the neighbor sampler, only on linux x86_64, so the
# image is built for linux/amd64 on every host.
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM --platform=linux/amd64 python:3.12-slim AS base
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
RUN useradd --create-home --uid 10001 sage && chown sage:sage /app
COPY --from=builder --chown=sage:sage /app/.venv /app/.venv
COPY --chown=sage:sage pyproject.toml uv.lock .python-version ./
COPY --chown=sage:sage configs ./configs
COPY --chown=sage:sage tests/fixtures/amlworld_hi_small_sample ./tests/fixtures/amlworld_hi_small_sample
USER sage
# The dev dependency group is absent from the image, so the entrypoint must not
# sync the environment again at start.
ENTRYPOINT ["uv", "run", "--no-sync", "sage-xgb"]
CMD ["--help"]
