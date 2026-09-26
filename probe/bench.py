"""Synthetic throughput benchmark for budgeting (no dataset needed).

    python -m probe.bench --out out/bench.json

Times the full training step (student fwd/bwd + teacher fwd under autocast,
batch 128) for: student only, full teacher, MaskedKD keeping 98 tokens (~50%),
keeping 59 tokens (~30%). Data loading is excluded, and on a shared GPU the
numbers include contention, so use them for budgeting only (pre-registration).
"""
import probe.common as common

import argparse
import json
import subprocess
import time

import torch
import torch.nn.functional as F

common.add_maskedkd_to_path()
import deit3  # noqa: E402
import models_student  # noqa: E402

TRAIN_IMAGES = 126_689


def other_gpu_load():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=100, help="for the time estimate only")
    ap.add_argument("--out", default="out/bench.json")
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    load_before = other_gpu_load()
    student = models_student.deit_small_patch16_224(num_classes=100, drop_path_rate=0.1).to(dev)
    teacher = deit3.deit_base_patch16_LS(pretrained=True).to(dev).eval()
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(args.batch_size, 3, 224, 224, device=dev)
    y = torch.randint(0, 100, (args.batch_size,), device=dev)
    idx = torch.arange(100, device=dev)

    def step(keep_n):
        with torch.cuda.amp.autocast():
            out, attn = student(x)
            loss = F.cross_entropy(out, y)
            if keep_n is not None:
                B, N = x.shape[0], attn.shape[-1] - 1
                if keep_n < N:
                    keep = torch.topk(attn.mean(dim=1)[:, 0, 1:], keep_n).indices
                else:
                    keep = torch.arange(N, device=dev).unsqueeze(0).expand(B, N)
                with torch.no_grad():
                    t = teacher(x, keep).index_select(1, idx)
                loss = 0.5 * loss + 0.5 * F.kl_div(F.log_softmax(out, 1), F.softmax(t, 1), reduction="batchmean")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    results = {}
    for name, keep_n in [("student_only", None), ("full_teacher", 196), ("masked_keep98", 98), ("masked_keep59", 59)]:
        for _ in range(args.warmup):
            step(keep_n)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.iters):
            step(keep_n)
        torch.cuda.synchronize()
        ips = args.iters * args.batch_size / (time.time() - t0)
        results[name] = {"img_per_s": round(ips, 1),
                         "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                         "train_h_per_epoch": round(TRAIN_IMAGES / ips / 3600, 3),
                         f"train_h_{args.epochs}ep": round(TRAIN_IMAGES / ips / 3600 * args.epochs, 2)}
        print(name, results[name], flush=True)

    report = {"batch_size": args.batch_size, "gpu": torch.cuda.get_device_name(0),
              "gpu_load_before_start": load_before, "gpu_load_after": other_gpu_load(),
              "note": "synthetic data; excludes data loading and eval; includes contention on a shared GPU",
              "results": results}
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
