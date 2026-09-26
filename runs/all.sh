#!/usr/bin/env bash
# Primary 12 runs, then the pre-registered 1.3x rule, then (if allowed) the secondary mask30 runs.
#   WAIT_PATTERN=aim-lab-test-2 nohup bash runs/all.sh > all.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export EPOCHS=100 ROOT=${ROOT:-out/p1} WAIT_PATTERN=${WAIT_PATTERN:-}
ARMS="full mask50" bash runs/p1.sh
python - <<'PY'
import json, statistics, sys
bench = json.load(open("out/bench.json"))["results"]["full_teacher"]["img_per_s"]
rows = [json.loads(l) for l in open("out/p1/full_ce1_s0/probe_timing.jsonl")]
measured = statistics.median(r["sec"] for r in rows)
expected = 126689 / bench
ratio = measured / expected
ok = ratio <= 1.3
json.dump({"bench_full_teacher_sec_per_epoch": round(expected, 1), "measured_median_sec": round(measured, 1),
           "ratio": round(ratio, 3), "run_secondary": ok}, open("out/p1/secondary_rule.json", "w"), indent=1)
print(f"1.3x rule: measured {measured:.0f}s vs bench {expected:.0f}s per epoch -> ratio {ratio:.2f} -> "
      + ("run secondary" if ok else "skip secondary"))
sys.exit(0 if ok else 3)
PY
if [ $? -eq 0 ]; then ARMS="mask30" bash runs/p1.sh; fi
python -m probe.collect --root "$ROOT" --epochs "$EPOCHS"
