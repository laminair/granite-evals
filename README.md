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
| `swebench-verified` | pass@1 resolve rate | `swebench` |
| `aime25`, `hmmt-feb25`, `gpqa` (Diamond) | pass@1 symbolic correct | `nemoskills` |
| `mmlu-pro` | 5-shot CoT symbolic correct (ns's per-category examples; `--option shots=0` for 0-shot) | `nemoskills` |
| `arena-hard-v2` | win rate (judge below) | `nemoskills` |
| `livecodebench-v6`, `scicode` | pass@1 accuracy, pass@1 subtask accuracy | `nemoskills` |
| `ruler-64k`, `ruler-128k`, `ruler-256k`, `ruler-512k`, `ruler-1m` | accuracy, thinking on up to the context cap (below) | `nemoskills` |
| `mmlu-prox-lite` (lm-eval, 11 Granite languages) | exact match (custom-extract); deviations below | `lmeval` |
| `ifbench` (NeMo-Skills + IFBench verifiers) | pass@1 prompt loose accuracy (headline) / strict; instruction loose / strict in `details.metrics` | `ifbench` |
| `swebench-pro` | pass@1 resolve rate | `swebench` |
| `swebench-multilingual` | pass@1 resolve rate | `swebench` |
| `terminal-bench-2.1` | pass@1 resolve rate | `tbench` |
| `tau3-bench` | pass@1 (avg of 3): mean pass^1 over airline, retail, telecom | `tau` |
| `tau3-airline` | pass@1 (pass^1; 1 trial by default, the published protocol is `--repeats 4`) | `tau` |
| `tau3-retail` | pass@1 (pass^1, as airline) | `tau` |
| `tau3-telecom` | pass@1 (pass^1, as airline) | `tau` |
| `tau3-banking-knowledge` | pass@1 (pass^1, as airline; BM25 + grep retrieval) | `tau` |
| `bfcl-v4` | overall_accuracy accuracy (web search via IBM search MCP, not SerpAPI) | `bfcl` |
| `birdbench` | pass@1 execution match (NeMo-Skills protocol, no evidence) | `bird` |
| `gdpval` | Elo. **Approximation, not GDPval-AA's Elo**: an Elo-style score from the pairwise LLM-judged win rate against the gold set's expert deliverables (expert = 1000). It is relative to the expert, not on the scale of IBM's published Granite GDPval numbers; `details.elo_is_approximation` | `judged` |
| `profbench` | overall (ProfBench report generation, lite, LLM-judged rubrics) | `judged` |

NeMo-Skills benchmarks share `benchmarks/nemo_skills.py` (its docstring explains how to
add one).

Judges and simulators. `arena-hard-v2` is judged by `aws/claude-sonnet-5` (IBM LiteLLM,
metered), not the official GPT-4.1 judge, and without style control; results record both
judges. The τ³ user simulator and the retail NL-assertion judge are `aws/claude-sonnet-5`
too, where tau2 upstream hard-codes gpt-4.1. Scores of these benchmarks are therefore not
directly comparable with published numbers.

`mmlu-prox-lite` samples with Granite 4.2's card settings for thinking (temperature 1.0,
top_p 0.95, 8192 tokens), not NeMo Evaluator's lm-eval chat protocol (2048 tokens, near-greedy),
which cuts off almost every thinking trace. "(IBM)" is read as the 11 Granite languages that
MMLU-ProX has (the card's list minus Dutch); neither the card nor the blog defines the set,
so it is unconfirmed. `--option languages=all` or a comma list changes it.

`ruler-*` keeps thinking on (Granite's default) and lets each sample generate up to the
context cap (the served `max_model_len`, or `--option context_cap=N`). A sample over the cap,
with a context-length error, or cut off before an answer scores 0 and is logged to
`<output_dir>/failures.jsonl`; `details.failures` counts them. `enable_thinking=false` runs
ns's RULER exactly. `ruler-256k`, `ruler-512k` and `ruler-1m` need a server with that much
context; granite-4.2 is only verified to 128k.

## Repeats

Every benchmark defaults to one sample per example (`--repeats 1`, pass@1). `--repeats k`
keeps the headline at pass@1[avg-of-k] (the mean over the k samples) and adds
`details.pass_at_k`, one shape for every benchmark:

```json
{"k": 4, "pass_at_1": 0.61, "pass_at_k": 0.78, "n": 30, "how": "..."}
```

`pass_at_k` is the fraction of examples with a correct sample among the k (for a graded
score, the best of the k), and `how` says how the benchmark computed it. `tau3-*` keeps
pass^1 as the headline and adds `pass_hat_k` (all k trials succeed); its per-domain
details carry `pass_hat_<j>` and `pass_at_<j>`. `bfcl-v4`, `profbench` and `gdpval` take
only `--repeats 1`: BFCL keeps one result per test id and its overall is a weighted mix of
categories, and the judged benchmarks score one response per task. Their `pass_at_k` is
the k = 1 record (GDPval's `pass_at_k` is null: an Elo has none).

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
