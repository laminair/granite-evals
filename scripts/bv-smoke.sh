#!/bin/bash
# BlueVela smoke test of a sage2-evals image, outside granite.build:
#   1. host enroot imports the step image (uses ~/.config/enroot/.credentials),
#   2. starts it and runs `sage2-evals run` inside, which starts its own nested
#      enroot sandboxes.
# Submit with e.g.
#   bsub -G <group> -q <queue> -J sage2-gold -o %J.out -e %J.err \
#        -n 16 -R "span[hosts=1]" -M 64G bv-smoke.sh
#   MODEL=/path/to/hf/model bsub ... -gpu num=1 bv-smoke.sh   # model run
# IMAGE is required; BENCHMARK, LIMIT, OPTIONS (space-separated k=v), EXTRA_ARGS optional.
# SWE-bench gold run (no model): MODEL=none OPTIONS=patch=gold.
set -euo pipefail

IMAGE="${IMAGE:?set IMAGE=icr.io/tir-hew-sage2-evals/sage2-evals-<family>:<sha>}"
BENCHMARK="${BENCHMARK:-swebench-verified}"
MODEL="${MODEL:-none}"
LIMIT="${LIMIT:-2}"
REPEATS="${REPEATS:-1}"
WORKERS="${WORKERS:-2}"
# Empty = the dataset the benchmark pins.
DATASET="${DATASET:-}"
OPTIONS="${OPTIONS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"  # further `sage2-evals run` arguments, e.g. --max-model-len 32768
ROOT="${ROOT:-/proj/data-eng/hew/sage2}"
# Judge / user-simulator key: an env file the user keeps in their home (mode 600),
# e.g. SAGE2_JUDGE_API_KEY=...; passed into the container by name, never printed.
# ROOT is world-writable, so keys never go there.
JUDGE_ENV="${JUDGE_ENV:-$HOME/.config/sage2/judge.env}"
if [ -r "$JUDGE_ENV" ]; then set -a; . "$JUDGE_ENV"; set +a; echo "judge env: loaded $JUDGE_ENV"; fi
# Paid API calls go through sage2_evals.meter: one ledger for every job, one budget.
SPEND_LEDGER="${SPEND_LEDGER:-$ROOT/spend/ledger.jsonl}"
SPEND_BUDGET_USD="${SPEND_BUDGET_USD:-50}"
mkdir -p "$(dirname "$SPEND_LEDGER")"
ENV_ARGS=(--env SAGE2_SPEND_LEDGER="$SPEND_LEDGER" --env SAGE2_SPEND_BUDGET_USD="$SPEND_BUDGET_USD")
for v in SAGE2_JUDGE_API_KEY SAGE2_USER_API_KEY; do [ -n "${!v:-}" ] && ENV_ARGS+=(--env "$v"); done
RUN="${RUN:-$ROOT/runs/smoke-${LSB_JOBID:-$$}}"

echo "=== $(hostname) job=${LSB_JOBID:-local} image=$IMAGE model=$MODEL ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

LOCAL=/opt/nvme/enroot-$USER/sage2-${LSB_JOBID:-$$}
NAME=sage2-${LSB_JOBID:-$$}
mkdir -p "$ROOT/images" "$ROOT/enroot-cache" "$ROOT/hf-home" "$RUN" \
    "$LOCAL/data" "$LOCAL/cache" "$LOCAL/runtime" "$LOCAL/temp" "$LOCAL/inner"
# Every step may fail harmlessly (nothing to unmount on a cached image); under set -e
# a failing step would otherwise end the trap and become the job's exit status.
trap 'enroot remove -f "$NAME" >/dev/null 2>&1 || true
      for m in "$LOCAL/flat/merged" "$LOCAL/flat/layers"; do fusermount3 -u "$m" 2>/dev/null || true; done
      rm -rf "$LOCAL" || true' EXIT
# Layer downloads are kept on /proj so a rerun doesn't fetch them again.
export ENROOT_DATA_PATH=$LOCAL/data ENROOT_CACHE_PATH=$ROOT/enroot-layers \
    ENROOT_RUNTIME_PATH=$LOCAL/runtime ENROOT_TEMP_PATH=$LOCAL/temp \
    ENROOT_SQUASH_OPTIONS='-comp lz4 -Xhc -no-xattrs'
mkdir -p "$ENROOT_CACHE_PATH"
# A separate enroot config dir (holding its own .credentials) for the step image
# pull, so the account's ~/.config/enroot stays untouched.
[ -n "${SAGE2_ENROOT_CONFIG:-}" ] && export ENROOT_CONFIG_PATH="$SAGE2_ENROOT_CONFIG"

SQSH="$ROOT/images/$(echo "$IMAGE" | tr '/:' '__').sqsh"
# BlueVela compute nodes can't give enroot-aufs2ovlfs its capabilities, so a plain
# `enroot import` fails. The workaround of SkyPilot's LSF provider
# (sky/provision/lsf/instance.py), with its layer order fixed: tolerate the whiteout
# conversion failing, fall back to a layered squashfs, flatten it with squashfuse +
# fuse-overlayfs. (Inside the image, sage2-evals imports with its own rootless
# helpers instead; see sage2_evals/sandbox/ovlfs.py.)
WRAP=$LOCAL/wrappers
mkdir -p "$WRAP"
printf '#!/bin/bash\n/usr/bin/enroot-aufs2ovlfs "$@" || true\n' > "$WRAP/enroot-aufs2ovlfs"
cat > "$WRAP/enroot-mksquashovlfs" <<'EOF_WRAP'
#!/bin/bash
LAYERS="$1"; OUTFILE="$2"; shift 2
/usr/bin/enroot-mksquashovlfs "$LAYERS" "$OUTFILE" "$@" 2>/dev/null && [ -f "$OUTFILE" ] && exit 0
IFS=':' read -ra DIRS <<< "$LAYERS"
mksquashfs "${DIRS[@]}" "$OUTFILE" "$@" -no-xattrs
EOF_WRAP
chmod +x "$WRAP"/*
command -v fusermount >/dev/null || ln -sf "$(command -v fusermount3)" "$WRAP/fusermount"
export PATH="$WRAP:$PATH"

flatten() {  # layered sqsh (dirs 0/ 1/ ...) -> flat sqsh, in place
    local f=$LOCAL/flat
    mkdir -p "$f/layers" "$f/merged" "$f/upper" "$f/work"
    squashfuse "$1" "$f/layers"
    if [ -d "$f/layers/bin" ] || [ -L "$f/layers/bin" ] || [ ! -d "$f/layers/0" ]; then
        fusermount3 -u "$f/layers"; echo "sqsh already flat"; return 0
    fi
    echo "=== flatten $(ls "$f/layers" | wc -l) layers ==="
    # Top layer first, as enroot-mksquashovlfs stacks them: 0 is enroot's own
    # (image ENV in /etc/environment, /etc/rc), then 1 = the newest image layer
    # down to N = the base (docker.sh reverses the manifest). SkyPilot's provider
    # sorts N..1,0, which stacks the image upside down.
    local lower n
    n=$(ls "$f/layers" | sort -n | tail -1)
    lower=$(seq -s: -f "$f/layers/%g" 0 "$n")
    fuse-overlayfs -o "lowerdir=$lower,upperdir=$f/upper,workdir=$f/work" "$f/merged"
    mksquashfs "$f/merged" "$LOCAL/flat.sqsh" -comp lz4 -Xhc -no-xattrs -noappend -quiet
    fusermount3 -u "$f/merged"; fusermount3 -u "$f/layers"
    mv "$LOCAL/flat.sqsh" "$1"
}

if [ ! -s "$SQSH" ]; then
    echo "=== import $IMAGE ==="
    enroot import -o "$SQSH.partial" "docker://${IMAGE%%/*}#${IMAGE#*/}" || [ -s "$SQSH.partial" ]
    flatten "$SQSH.partial"
    mv "$SQSH.partial" "$SQSH"
    echo "=== imported: $(du -h "$SQSH" | cut -f1) ==="
fi

enroot create --name "$NAME" "$SQSH"

MOUNTS=(--mount "$ROOT:$ROOT" --mount "$LOCAL/inner:/scratch")
case "$MODEL" in /*) MOUNTS+=(--mount "$MODEL:$MODEL") ;; esac
OPTS=""
for kv in $OPTIONS; do OPTS="$OPTS --option $kv"; done

# The GPU env granite.build's SkyPilot LSF provider also sets; it makes enroot's
# nvidia hook mount the job's GPUs.
echo "=== sage2-evals run $BENCHMARK ==="
enroot start --rw "${MOUNTS[@]}" \
    --env HF_HOME="$ROOT/hf-home" \
    --env NVIDIA_VISIBLE_DEVICES=all \
    --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env SAGE2_SANDBOX=enroot \
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
    --env SAGE2_ENROOT_CACHE="$ROOT/enroot-cache" \
    --env ENROOT_DATA_PATH=/scratch/data \
    --env ENROOT_CACHE_PATH=/scratch/cache \
    --env ENROOT_RUNTIME_PATH=/scratch/runtime \
    --env ENROOT_TEMP_PATH=/scratch/temp \
    "$NAME" bash -c "
        set -eo pipefail
        # enroot passes the host PATH through; same line as the granite.build step.
        export PATH=/opt/sage2-evals/.venv/bin:\$PATH
        mkdir -p /scratch/data /scratch/cache /scratch/runtime /scratch/temp
        enroot version
        sage2-evals run $BENCHMARK --model $MODEL --output-dir $RUN \
            --limit $LIMIT --repeats $REPEATS --workers $WORKERS \
            ${DATASET:+--dataset $DATASET} $OPTS $EXTRA_ARGS 2>&1 | tee $RUN/sage2.log
    "
echo "=== results: $RUN/results.json ==="
cat "$RUN/results.json"
