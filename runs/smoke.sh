#!/usr/bin/env bash
# 1-epoch smoke test on 10 images/class: checks data, teacher download, both arms, CE on/off.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-$HOME/kdprobe/data/in100}
COMMON="--model deit_small_patch16_224 --teacher_model deit3_base --data-set IN100 --data-path $DATA \
  --epochs 1 --batch-size 128 --distillation-type soft --distillation-alpha 0.5 --distillation-tau 1.0 \
  --no-repeated-aug --num_workers 8 --seed 0 --probe-subset 10"
python -m probe.launch $COMMON --output_dir out/smoke/full_ce1
python -m probe.launch $COMMON --output_dir out/smoke/mask50_ce0 --maskedkd --len_num_keep 98 --ce-off
for d in out/smoke/full_ce1 out/smoke/mask50_ce0; do echo "== $d"; tail -1 $d/log.txt; cat $d/probe_timing.jsonl; done
