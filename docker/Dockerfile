# One image per benchmark family: `--build-arg EXTRA=swebench` etc.
# Build context is the repo root (see Makefile). Every image carries:
#   - vLLM (the `serve` extra), pinned by uv.lock
#   - sage2-evals with the family's extra, at /opt/sage2-evals
#   - the enroot CLI, for benchmarks that sandbox each task in a container
FROM python:3.12-slim-bookworm

ARG EXTRA=swebench
ARG ENROOT_VERSION=4.2.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git jq gawk parallel pigz squashfs-tools zstd \
        libcap2-bin procps build-essential \
    && arch=$(dpkg --print-architecture) \
    && curl -fsSLO "https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/enroot_${ENROOT_VERSION}-1_${arch}.deb" \
    && apt-get install -y --no-install-recommends ./enroot_${ENROOT_VERSION}-1_${arch}.deb \
    && rm -f ./enroot_*.deb && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /opt/sage2-evals
ENV UV_PROJECT_ENVIRONMENT=/opt/sage2-evals/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/sage2-evals/.venv/bin:$PATH

# Dependencies first so code edits don't re-download vLLM.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev --extra serve --extra "${EXTRA}"
COPY src ./src
COPY tests ./tests
RUN uv sync --locked --no-dev --extra serve --extra "${EXTRA}" \
    && uv run --no-sync --with pytest pytest -q tests \
    && sage2-evals list

ENV SAGE2_SANDBOX=enroot
ENTRYPOINT []
CMD ["sage2-evals", "--help"]
