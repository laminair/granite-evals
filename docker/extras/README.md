Per-family image setup. `docker/extras/<extra>.sh`, if present, runs as root
during the image build (after the base packages, before `uv sync`) for the
image built with `--build-arg EXTRA=<extra>`. Use it for system packages or
tools a harness needs that pip cannot install.
