# Running on BlueVela

Normal runs go through granite.build: the `recipes/sage2/lsf/eval-granite42` recipe,
whose README lists the secrets, the gold checks and the favored config of every
benchmark. This page covers the other two paths: building images and running one
image directly on BlueVela with `scripts/bv-smoke.sh` (smokes, gold checks, debugging).

## Images

BlueVela cannot build images. Build them on hg4os and push them to ICR:

```bash
make publish-image EXTRA=<family>   # -> us.icr.io/cil15-shared-registry/sage2-evals-<family>:<sha>
```

| Family | Benchmarks |
|---|---|
| `swebench` | `swebench-verified`, `swebench-pro`, `swebench-multilingual` |
| `tbench` | `terminal-bench-2.1` |
| `bird` | `birdbench` |
| `tau` | `tau3-*` |
| `bfcl` | `bfcl-v4` |
| `judged` | `gdpval`, `profbench` |
| `nemoskills` | `aime25`, `hmmt-feb25`, `gpqa`, `mmlu-pro`, `arena-hard-v2`, `livecodebench-v6`, `scicode`, `ruler-*` |
| `lmeval` | `mmlu-prox-lite` |
| `ifbench` | `ifbench` |

The tag is the commit's short SHA, so build from a clean commit that is pushed. Then
pin that tag in the recipe's `parameters.yaml`.

## A direct run: `scripts/bv-smoke.sh`

The script imports the image with the host's enroot, using the ICR key in
`~/.config/enroot/.credentials`, and caches it under `$ROOT/images`. It then runs
`sage2-evals run` inside the image, and that run starts its own nested enroot
sandboxes. Results go to `$ROOT/runs/smoke-<jobid>/`, and `ROOT` defaults to
`/proj/data-eng/hew/sage2`.

| Variable | Default | |
|---|---|---|
| `IMAGE` | required | `us.icr.io/cil15-shared-registry/sage2-evals-<family>:<sha>` |
| `BENCHMARK` | `swebench-verified` | |
| `MODEL` | `none` | A hub id, an absolute path (mounted), or `none` for gold/oracle modes |
| `LIMIT` / `REPEATS` / `WORKERS` | `2` / `1` / `2` | `REPEATS=k`: headline pass@1[avg-of-k], plus `details.pass_at_k` (README, Repeats). `bfcl-v4`, `profbench`, `gdpval` take only 1 |
| `OPTIONS` | `""` | Space-separated `key=value` |
| `PHASE` | `all` | `generate` or `score` for a split run (below) |
| `RUN` | `$ROOT/runs/smoke-<jobid>` | Output dir; a score job takes its generate job's |
| `EXTRA_ARGS` | `""` | More `sage2-evals run` flags, e.g. `--max-model-len 262144` |
| `JUDGE_ENV` | `~/.config/sage2/judge.env` | Mode-600 env file holding `SAGE2_JUDGE_API_KEY` / `SAGE2_USER_API_KEY`. `/dev/null` means no paid API |
| `SPEND_LEDGER` / `SPEND_BUDGET_USD` | `$ROOT/spend/ledger.jsonl` / `50` | Every job shares one ledger and one cap |

Secrets reach the container only as named env vars (`HF_TOKEN`, `SAGE2_JUDGE_API_KEY`,
`SAGE2_USER_API_KEY`, `SAGE2_MAVEN_MIRROR`). Never put them under `$ROOT`, which is
shared.

`HF_TOKEN`, which `gpqa` needs, is exported by the account's `~/.zshrc`. A `bash -lc`
submission does not see it. Submit from an interactive zsh with `-env all`, so bsub
copies the environment into the job:

```bash
ssh bv 'zsh -ic "cd /proj/data-eng/hew/sage2/logs && \
  IMAGE=us.icr.io/cil15-shared-registry/sage2-evals-nemoskills:<sha> BENCHMARK=gpqa \
  MODEL=ibm-granite/granite-4.2-3b LIMIT=5 WORKERS=4 JUDGE_ENV=/dev/null \
  bsub -G grp_preemptable -q preemptable -J sage2-gpqa -o %J.out -e %J.err \
       -n 16 -R \"span[hosts=1]\" -M 64G -gpu num=1:mode=exclusive_process \
       -env all ~/bv-smoke.sh"'
```

`~/bv-smoke.sh` is a copy of `scripts/bv-smoke.sh`. Don't replace it while jobs
that use it are still queued or running.

A split run is two jobs on one run dir: `PHASE=generate` with `-gpu`, then
`PHASE=score RUN=$ROOT/runs/smoke-<generate jobid>` without `-gpu` (same `BENCHMARK`,
`MODEL`, `LIMIT`, `REPEATS`, `OPTIONS`). bsub's `-w "done(<jobid>)"` queues the second
behind the first. The paid judge keys are needed only by the score job (the tau user
simulator runs in generate).

For gold and oracle modes (`MODEL=none`), drop `-gpu`. `ruler-*` needs a tokenizer
even in gold mode, so keep `MODEL`; its gold cap is the sample length, so it needs no
`--max-model-len`. A model run's context cap is the served `max_model_len` (the
checkpoint's own unless `EXTRA_ARGS="--max-model-len N"`), which must hold a full sample:
`ruler-256k`, `ruler-512k` and `ruler-1m` need `--max-model-len` 262144, 524288 and
1048576 and a model that supports them (granite-4.2 is only verified to 128k). Thinking
generates up to that cap, so a sample is long; one over the cap or cut off before an
answer scores 0 and is logged to `$RUN/failures.jsonl`.

A smoke passes when:

1. the gold mode scores 1.0, or the documented reference value (see the recipe README
   for each benchmark's gold option and the known upstream failures), and
2. the model run's `results.json` shows real grading: `n` as expected, no parse or
   extraction failures, and `details.api_spend` inside the budget.

`sage2-evals spend $ROOT/spend/ledger.jsonl` totals the spend by benchmark, model,
role and job.
