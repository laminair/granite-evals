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
| `tau3-bench` | pass@1 (avg of 3): mean pass^1 over airline, retail, telecom | `tau` |
| `tau3-airline` | pass@1 (pass^1 over 4 trials) | `tau` |
| `tau3-retail` | pass@1 (pass^1 over 4 trials) | `tau` |
| `tau3-telecom` | pass@1 (pass^1 over 4 trials) | `tau` |
| `tau3-banking-knowledge` | pass@1 (pass^1 over 4 trials, BM25 + grep retrieval) | `tau` |

## Layout

| Path | What |
|---|---|
| `cli.py` | `run` / `list` / `spend` |
| `registry.py` | `Benchmark` base class, `RunConfig`, `@register` |
| `serving.py` | vLLM server lifecycle (`--tool-call-parser auto --reasoning-parser auto`) |
| `data.py` | dataset loading (upstream HF datasets, pinned by commit) |
| `meter.py` | metering proxy for paid judge / user-simulator APIs: cost ledger, budget cap |
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
| `HF_TOKEN` | read access to gated upstream datasets |
| `SAGE2_SANDBOX` | `enroot` (default), `podman`, `docker` |
| `SAGE2_ENROOT_CACHE` | shared squashfs cache for sandbox images |
| `SAGE2_SPEND_LEDGER` | JSONL ledger of paid API calls, shared by concurrent jobs |
| `SAGE2_SPEND_BUDGET_USD` | refuse paid API calls once the ledger's total reaches this |

## Paid APIs (judges, user simulators)

Calls to paid endpoints go through `sage2_evals.meter`, a local proxy that records
each call's cost (the gateway's `x-litellm-response-cost`, or a deliberately high
token-price fallback) in the ledger and answers HTTP 402 once the budget is spent.
Options named `*_base_url` are metered automatically; a benchmark sends any other
paid endpoint through `meter.metered(url, role)`. `results.json` carries the run's
spend as `details.api_spend`; `sage2-evals spend <ledger>` totals a ledger by
benchmark, model, role and job.

## Datasets

Each benchmark class pins its upstream HF dataset and commit (`dataset`,
`dataset_revision`) and reads it directly. Public data is not mirrored.
`results.json` records the dataset and revision it was computed on.
`--dataset <hub id or path>` (plus `--dataset-revision`) overrides the pin.

## Images

Images are built on hg4os, because BlueVela cannot build them, and pushed to ICR:

```bash
make publish-image EXTRA=swebench   # -> icr.io/tir-hew-sage2-evals/sage2-evals-swebench:<sha>
```

Tags are the git short SHA. Pin that tag in the granite.build recipe.

## Adding a benchmark

1. Add `benchmarks/<family>.py` with a `@register`ed `Benchmark` subclass (every
   module there is registered automatically). Import the harness inside methods,
   and add it to an optional extra of its own.
2. Pin the upstream dataset on the class: `dataset` and `dataset_revision`.
3. Copy `steps/sage2-swebench-verified` in granite.build to `steps/sage2-<id>`. Change
   the name, `BENCHMARK` and the defaults, then add a target to the suite recipes.
