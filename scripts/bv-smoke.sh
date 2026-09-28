#!/bin/bash
# BlueVela smoke test of a sage2-evals image, outside granite.build:
#   1. host enroot imports the step image (uses ~/.config/enroot/.credentials),
#   2. starts it and runs `sage2-evals run` inside, which starts its own nested
#      enroot sandboxes.
# Submit with e.g.
#   bsub -G <group> -q <queue> -J sage2-gold -o %J.out -e %J.err \
#        -n 16 -R "span[hosts=1]" -M 64G bv-smoke.sh
#   MODEL=/path/to/hf/model bsub ... -gpu num=1 bv-smoke.sh   # model run
set -euo pipefail

IMAGE="${IMAGE:-icr.io/tir-hew-sage2-evals/sage2-evals-swebench:9419469}"
BENCHMARK="${BENCHMARK:-swebench-verified}"
MODEL="${MODEL:-none}"
LIMIT="${LIMIT:-2}"
REPEATS="${REPEATS:-1}"
WORKERS="${WORKERS:-2}"
DATASET="${DATASET:-SWE-bench/SWE-bench_Verified}"
OPTIONS="${OPTIONS:-$([ "$MODEL" = none ] && echo patch=gold)}"
ROOT="${ROOT:-/proj/data-eng/hew/sage2}"
RUN="${RUN:-$ROOT/runs/smoke-${LSB_JOBID:-$$}}"

echo "=== $(hostname) job=${LSB_JOBID:-local} image=$IMAGE model=$MODEL ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

LOCAL=/opt/nvme/enroot-$USER/sage2-${LSB_JOBID:-$$}
NAME=sage2-${LSB_JOBID:-$$}
mkdir -p "$ROOT/images" "$ROOT/enroot-cache" "$ROOT/hf-home" "$RUN" \
    "$LOCAL/data" "$LOCAL/cache" "$LOCAL/runtime" "$LOCAL/temp" "$LOCAL/inner"
trap 'enroot remove -f "$NAME" >/dev/null 2>&1 || true; fusermount3 -u "$LOCAL/flat/merged" "$LOCAL/flat/layers" 2>/dev/null; rm -rf "$LOCAL"' EXIT
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

echo "=== sage2-evals run $BENCHMARK ==="
enroot start --rw "${MOUNTS[@]}" \
    --env HF_HOME="$ROOT/hf-home" \
    --env SAGE2_SANDBOX=enroot \
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
            --dataset $DATASET $OPTS 2>&1 | tee $RUN/sage2.log
    "
echo "=== results: $RUN/results.json ==="
cat "$RUN/results.json"
