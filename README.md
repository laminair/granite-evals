# sage2-evals

Runtime for the IBM **Sage2** benchmark suites, built to run as granite.build steps.
Each benchmark is a `sage2-<id>` step in granite.build that calls:

```bash
sage2-evals run <benchmark> --model <hf dir or hub id> --output-dir out [--limit N]
```

Each run serves the model with vLLM in the job, runs one benchmark and writes
`out/results.json`. The upstream harnesses (mini-swe-agent, swebench, …) are pinned
dependencies. This repo only adapts them. Each harness family ships as its own image.

## Suites

`sage2-evals list --suite granite42` (or `granite5`) prints each benchmark and whether
it is implemented. The suite definitions live in `src/sage2_evals/suites/*.yaml`.

| Implemented | Metric | Image extra |
|---|---|---|
| `swebench-verified` | pass@1[avg-of-3] resolve rate | `swebench` |

## Layout

| Path | What |
|---|---|
| `cli.py` | `run` / `list` |
| `registry.py` | `Benchmark` base class, `RunConfig`, `@register` |
| `serving.py` | vLLM server lifecycle (`--tool-call-parser auto --reasoning-parser auto`) |
| `data.py` | dataset loading from the private mirrors `<SAGE2_HF_ORG>/sage2-<id>` |
| `results.py` | `results.json` schema (value, n, smoke flag, versions, details) |
| `sandbox/` | per-task containers: enroot (BlueVela), podman/docker (local) |
| `benchmarks/` | one module per benchmark family |

## Running locally

```bash
uv sync --extra swebench
uv run pytest
# Check images, sandbox and grading without a model or GPU:
SAGE2_SANDBOX=podman uv run sage2-evals run swebench-verified --model none \
  --output-dir /tmp/gold --dataset SWE-bench/SWE-bench_Verified --limit 2 --option patch=gold
```

`--limit N` is the smoke knob. It takes the first N examples by id, so two smoke runs
see the same examples. `results.json` records `smoke: true` for such runs.

## Environment

| Variable | Purpose |
|---|---|
| `SAGE2_HF_ORG` | org that holds the private dataset mirrors |
| `HF_TOKEN` | read access to the mirrors |
| `SAGE2_SANDBOX` | `enroot` (default), `podman`, `docker` |
| `SAGE2_ENROOT_CACHE` | shared squashfs cache for sandbox images |

## Datasets

Every benchmark reads a private HF dataset `<org>/sage2-<id>` in the `gbspace-public`
resource group, snapshotted from upstream by `scripts/mirror_dataset.py`. Each mirror
contains `SAGE2_SOURCE.json`, which records the upstream repo and commit.
`--dataset <hub id>` bypasses the mirror.

## Images

Images are built on hg4os, because BlueVela cannot build them, and pushed to ICR:

```bash
make publish-image REGISTRY=<icr host>/<namespace> EXTRA=swebench
```

Tags are the git short SHA. Pin that tag in the granite.build recipe.

## Adding a benchmark

1. Add `benchmarks/<family>.py` with a `@register`ed `Benchmark` subclass. Add the
   module to `registry._BENCHMARK_MODULES` and its harness to an optional extra.
2. Mirror the dataset: `scripts/mirror_dataset.py <id> <upstream> --org … --resource-group-id …`.
3. Copy `steps/sage2-swebench-verified` in granite.build to `steps/sage2-<id>`. Change
   the name, `BENCHMARK` and the defaults, then add a target to the suite recipes.
