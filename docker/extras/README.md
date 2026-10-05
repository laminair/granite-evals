Per-family image setup. `docker/extras/<extra>.sh`, if present, runs as root
during the image build (after `uv sync --extra serve`, before the extra's own
`uv sync`) for the image built with `--build-arg EXTRA=<extra>`. Use it for system
packages or tools a harness needs that pip cannot install. It runs from
`/opt/granite-evals` with `UV_NO_CONFIG=1`, so a separate env it builds with `uv venv`
/ `uv pip` does not pick up this project's `[tool.uv]` settings (e.g. the
`numpy>=2` override).
