"""Apply the pre-registered P1 decision rule to finished runs.

    python -m probe.collect --root out/p1 --epochs 100

Run directories are named {full|mask50|mask30}_ce{1|0}_s{seed}. The metric is
the LAST epoch's top-1 (not the best epoch). A run counts only if its log has
exactly `--epochs` epochs.
"""
import argparse
import json
import math
import os
import statistics as stats

T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def last_acc(run_dir, epochs):
    p = os.path.join(run_dir, "log.txt")
    if not os.path.exists(p):
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    if len(rows) != epochs or rows[-1]["epoch"] != epochs - 1:
        return None
    return rows[-1]["test_acc1"]


def describe(xs):
    n = len(xs)
    m = stats.mean(xs)
    if n < 2:
        return {"n": n, "mean": m}
    sd = stats.stdev(xs)
    h = T975.get(n, 1.96) * sd / math.sqrt(n)
    return {"n": n, "mean": round(m, 3), "sd": round(sd, 3), "ci95": [round(m - h, 3), round(m + h, 3)],
            "per_seed": [round(x, 3) for x in xs], "same_sign": all(x > 0 for x in xs) or all(x < 0 for x in xs)}


def verdict(gap0, dce):
    c1 = gap0["mean"] >= 1.0 and gap0.get("same_sign", False) and gap0["mean"] > 0
    c2 = dce["mean"] >= 0.5 and dce.get("same_sign", False) and dce["mean"] > 0
    if c1 and c2:
        return "A: P1 reproduced (CE hides the masking loss) -> stage 2"
    if c1:
        return "B: label-independent loss (check MaskedKD reproduction if gap_1 >= 0.5) -> no automatic stage 2"
    if c2:
        return "C: interaction only (masking helps with labels) -> different finding, no stage 2"
    return "D: null -> reject P1"


def analyse(root, epochs, masked, seeds):
    acc = {}
    for arm in ("full", masked):
        for ce in (1, 0):
            for s in seeds:
                acc[(arm, ce, s)] = last_acc(os.path.join(root, f"{arm}_ce{ce}_s{s}"), epochs)
    done = [s for s in seeds if all(acc[(a, c, s)] is not None for a in ("full", masked) for c in (1, 0))]
    out = {"masked_arm": masked, "complete_seeds": done,
           "acc": {f"{a}_ce{c}_s{s}": v for (a, c, s), v in acc.items()}}
    if len(done) < 2:
        out["verdict"] = "incomplete"
        return out
    gap1 = [acc[("full", 1, s)] - acc[(masked, 1, s)] for s in done]
    gap0 = [acc[("full", 0, s)] - acc[(masked, 0, s)] for s in done]
    dce = [g0 - g1 for g0, g1 in zip(gap0, gap1)]
    out.update({"gap_1": describe(gap1), "gap_0": describe(gap0), "delta_ce": describe(dce)})
    out["verdict"] = verdict(out["gap_0"], out["delta_ce"]) if len(done) == len(seeds) else \
        f"provisional ({len(done)}/{len(seeds)} seeds): " + verdict(out["gap_0"], out["delta_ce"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out/p1")
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    res = {"primary_mask50": analyse(args.root, args.epochs, "mask50", args.seeds)}
    if any(d.startswith("mask30") for d in os.listdir(args.root)):
        res["secondary_mask30"] = analyse(args.root, args.epochs, "mask30", args.seeds)
        res["secondary_note"] = "secondary never counts as P1 if the primary fails (pre-registration)"
    print(json.dumps(res, indent=1))
    with open(os.path.join(args.root, "p1_summary.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
