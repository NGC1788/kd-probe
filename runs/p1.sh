#!/usr/bin/env bash
# Pre-registered P1 probe (PREREGISTRATION.md): DeiT3-B -> DeiT-S on ImageNet-100,
# full vs MaskedKD x CE on/off x seeds 0,1,2. Runs one at a time, waits for GPU memory.
#   EPOCHS=100 nohup bash runs/p1.sh > p1.out 2>&1 &
#   EPOCHS=100 WAIT_PATTERN=aim-lab-test-2 nohup bash runs/p1.sh > p1.out 2>&1 &   # wait for a co-user's job
#   EPOCHS=100 ARMS=mask30 nohup bash runs/p1.sh > p1_secondary.out 2>&1 &   # secondary, if registered
set -uo pipefail
cd "$(dirname "$0")/.."
: "${EPOCHS:?set EPOCHS (fixed from the bench before stage 1 starts)}"
DATA=${DATA:-$HOME/kdprobe/data/in100}
ROOT=${ROOT:-out/p1}
ARMS=${ARMS:-"full mask50"}
MIN_FREE_MB=${MIN_FREE_MB:-14000}
# Shared GPU: do not start a run while a process matching WAIT_PATTERN is alive
# (e.g. WAIT_PATTERN=aim-lab-test-2). A run already started is not interrupted.
WAIT_PATTERN=${WAIT_PATTERN:-}
mkdir -p "$ROOT"

finished() { [ -f "$1/log.txt" ] && [ "$(wc -l < "$1/log.txt")" -ge "$EPOCHS" ]; }
wait_gpu() {
  while [ -n "$WAIT_PATTERN" ] && pgrep -f -- "$WAIT_PATTERN" > /dev/null; do
    echo "$(date +%F_%T) waiting: a process matching '$WAIT_PATTERN' is running"; sleep 300
  done
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$MIN_FREE_MB" ] && return
    echo "$(date +%F_%T) waiting: free ${free}MB < ${MIN_FREE_MB}MB"; sleep 300
  done
}

for s in 0 1 2; do for ce in 1 0; do for arm in $ARMS; do
  dir="$ROOT/${arm}_ce${ce}_s${s}"
  if finished "$dir"; then echo "skip $dir (done)"; continue; fi
  [ -d "$dir" ] && { echo "$dir incomplete -> restart from scratch"; rm -rf "$dir"; }
  case $arm in
    full)   extra="" ;;
    mask50) extra="--maskedkd --len_num_keep 98" ;;
    mask30) extra="--maskedkd --len_num_keep 59" ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  [ "$ce" = 0 ] && extra="$extra --ce-off"
  wait_gpu
  echo "$(date +%F_%T) start $dir"
  python -m probe.launch --model deit_small_patch16_224 --teacher_model deit3_base \
    --data-set IN100 --data-path "$DATA" --epochs "$EPOCHS" --batch-size 128 \
    --distillation-type soft --distillation-alpha 0.5 --distillation-tau 1.0 \
    --no-repeated-aug --num_workers 8 --seed "$s" --output_dir "$dir" $extra > "$dir.log" 2>&1
  echo "$(date +%F_%T) end $dir (exit $?)"
done; done; done
python -m probe.collect --root "$ROOT" --epochs "$EPOCHS"
