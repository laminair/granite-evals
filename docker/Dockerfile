# One image per benchmark family: `--build-arg EXTRA=swebench` etc.
# Build context is the repo root (see Makefile). Every image carries:
#   - vLLM (the `serve` extra), pinned by uv.lock
#   - granite-evals with the family's extra, at /opt/granite-evals
#   - the enroot CLI, for benchmarks that sandbox each task in a container
#
# Layers up to the vLLM install don't depend on EXTRA (every RUN after `ARG EXTRA` is
# keyed on it), so images built from one commit share them and a push uploads vLLM once.
FROM python:3.12-slim-bookworm

ARG ENROOT_VERSION=4.2.1

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash ca-certificates curl git jq gawk parallel pigz squashfs-tools zstd \
        libcap2-bin procps build-essential \
    && arch=$(dpkg --print-architecture) \
    && curl -fsSLO "https://github.com/NVIDIA/enroot/releases/download/v${ENROOT_VERSION}/enroot_${ENROOT_VERSION}-1_${arch}.deb" \
    && apt-get install -y --no-install-recommends ./enroot_${ENROOT_VERSION}-1_${arch}.deb \
    && rm -f ./enroot_*.deb && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.20@sha256:100047e74f30778ab704942321a09750d6158739573ff58bf3924085cc6cd2d8 /uv /usr/local/bin/uv

WORKDIR /opt/granite-evals
ENV UV_PROJECT_ENVIRONMENT=/opt/granite-evals/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH=/opt/granite-evals/.venv/bin:$PATH

# Dependencies first so code edits don't re-download vLLM.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev --extra serve

ARG EXTRA=swebench

# Family-specific system setup (docker/extras/README.md). UV_NO_CONFIG: the scripts
# run from WORKDIR, and uv would otherwise apply this project's [tool.uv] settings
# (override-dependencies numpy>=2) to any separate env they build.
COPY docker/extras /tmp/extras
RUN if [ -f "/tmp/extras/${EXTRA}.sh" ]; then UV_NO_CONFIG=1 bash -euxo pipefail "/tmp/extras/${EXTRA}.sh"; fi \
    && rm -rf /tmp/extras /var/lib/apt/lists/*

RUN uv sync --locked --no-install-project --no-dev --extra serve --extra "${EXTRA}"
COPY src ./src
COPY tests ./tests
RUN uv sync --locked --no-dev --extra serve --extra "${EXTRA}" \
    && uv run --no-sync --with pytest pytest -q tests \
    && granite-evals list

ENV GRANITE_EVALS_SANDBOX=enroot
ENTRYPOINT []
CMD ["granite-evals", "--help"]
