# sage2-evals

Runtime for the IBM **Sage2** benchmark suites, built to run as granite.build steps.
Each benchmark is a `sage2-<id>` step in granite.build that calls:

```bash
sage2-evals run <benchmark> --model <hf dir or hub id> --output-dir out [--limit N]
```

Each run serves the model with vLLM in the job, runs one benchmark and writes
`out/results.json`. The upstream harnesses (mini-swe-agent, swebench, …) are pinned
dependencies. This repo only adapts them. Each harness family ships as its own image.

`--phase generate` and `--phase score` split a run into two jobs on one output dir, so
grading (sandboxes, verifiers, paid judges) holds no GPU. `generate` serves the model
and writes `out/generation.json`; `score` serves nothing, checks that generation.json
matches its model, limit, repeats and dataset, and grades the saved outputs. It never
generates: an example without a generation fails like a failed generation.
`terminal-bench-2.1` and `mmlu-prox-lite` grade inside generation and only take
`--phase all` (the default); so do options with `judge_model=self`.

## Suites

`sage2-evals list --suite granite42` (or `granite5`) prints each benchmark and whether
it is implemented. The suite definitions live in `src/sage2_evals/suites/*.yaml`.

| Implemented | Metric | Image extra |
|---|---|---|
| `swebench-verified` | pass@1[avg-of-3] resolve rate | `swebench` |
| `aime25`, `hmmt-feb25`, `gpqa` (Diamond), `mmlu-pro`, `arena-hard-v2` | NeMo-Skills metrics (see suite) | `nemoskills` |
| `livecodebench-v6`, `scicode`, `ruler-128k`, `ruler-64k` | NeMo-Skills metrics (see suite) | `nemoskills` |
| `hle`, `omniscience`, `omniscience-hallucination`, `aa-lcr` (LLM-judged), `critpt` (graded by AA's CritPt API), `wmt24pp` (XCOMET-XXL) | NeMo-Skills metrics, pass@1 (see suite) | `nemoskills` |
| `mmlu-prox-lite` (lm-eval, 11 Granite languages) | exact match (custom-extract) | `lmeval` |
| `ifbench` (NeMo-Skills + IFBench verifiers) | pass@1[avg-of-2] loose accuracy | `ifbench` |
| `swebench-pro` | pass@1[avg-of-3] resolve rate | `swebench` |
| `swebench-multilingual` | pass@1[avg-of-3] resolve rate | `swebench` |
| `terminal-bench-2.1` | pass@1[avg-of-8] resolve rate | `tbench` |
| `tau3-bench` | pass@1 (avg of 3): mean pass^1 over airline, retail, telecom | `tau` |
| `tau3-airline` | pass@1 (pass^1 over 4 trials) | `tau` |
| `tau3-retail` | pass@1 (pass^1 over 4 trials) | `tau` |
| `tau3-telecom` | pass@1 (pass^1 over 4 trials) | `tau` |
| `tau3-banking-knowledge` | pass@1 (pass^1 over 4 trials, BM25 + grep retrieval) | `tau` |
| `bfcl-v4` | overall_accuracy accuracy (web search via IBM search MCP, not SerpAPI) | `bfcl` |
| `birdbench` | pass@1 execution match (NeMo-Skills protocol, no evidence) | `bird` |
| `gdpval` | Elo. **Approximation, not GDPval-AA's Elo**: an Elo-style score from the pairwise LLM-judged win rate against the gold set's expert deliverables (expert = 1000); `details.elo_is_approximation` | `judged` |
| `profbench` | overall (ProfBench report generation, lite, LLM-judged rubrics) | `judged` |

NeMo-Skills benchmarks share `benchmarks/nemo_skills.py` (its docstring explains how to
add one). `arena-hard-v2` is judged by `aws/claude-sonnet-5` (IBM LiteLLM, metered), not the
official GPT-4.1 judge, and without style control; results record both judges.
`hle`, `omniscience`(`-hallucination`) and `aa-lcr` use ns's judge prompts with the same
judge instead of the official ones (o3-mini, Gemini 2.5 Flash, Qwen3-235B / gpt-4.1), and
leave an unparseable judgement out of the score (ns scores it wrong). `critpt` needs an
Artificial Analysis API key (`ARTIFICIAL_ANALYSIS_API_KEY`) to score; `wmt24pp` scores
with XCOMET-XXL on a GPU in the image's separate `/opt/comet` env. See
`benchmarks/nemo_skills_g5.py`.

## Layout

| Path | What |
|---|---|
| `cli.py` | `run` / `list` / `spend` |
| `registry.py` | `Benchmark` base class, `RunConfig`, `@register` |
| `serving.py` | vLLM server lifecycle (`--tool-call-parser auto --reasoning-parser auto`) |
| `data.py` | dataset loading (upstream HF datasets, pinned by commit) |
| `meter.py` | metering proxy for paid judge / user-simulator APIs: cost ledger, budget cap |
| `results.py` | `results.json` schema (value, n, smoke flag, versions, details), `generation.json` |
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
| `ARTIFICIAL_ANALYSIS_API_KEY` | `critpt` scoring (Artificial Analysis's CritPt grading API) |
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
make publish-image EXTRA=swebench   # -> us.icr.io/cil15-shared-registry/sage2-evals-swebench:<sha>
```

Tags are the git short SHA. Pin that tag in the granite.build recipe.

[docs/bluevela.md](docs/bluevela.md) covers image families, direct BlueVela runs
(`scripts/bv-smoke.sh`, secrets, gold checks). Favored configs per benchmark are in
granite.build's `recipes/sage2/lsf/eval-granite42/README.md`.

## Adding a benchmark

1. Add `benchmarks/<family>.py` with a `@register`ed `Benchmark` subclass (every
   module there is registered automatically). Import the harness inside methods,
   and add it to an optional extra of its own.
2. Pin the upstream dataset on the class: `dataset` and `dataset_revision`.
3. Copy `steps/sage2-swebench-verified` in granite.build to `steps/sage2-<id>`. Change
   the name, `BENCHMARK` and the defaults, then add a target to the suite recipes.
