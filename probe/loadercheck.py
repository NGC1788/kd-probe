"""CPU-only data-loader throughput check on the actual server (no GPU needed).

    python -m probe.loadercheck --data-path ~/kdprobe/data/in100

Decodes ImageNet-100 train images with MaskedKD's DeiT train transform and
reports img/s for several worker counts. The GPU needs up to ~650 img/s
(fastest arm in the bench), so the loader should deliver about 15% more than
that. Worker count changes only speed, not the design (pre-registration).
"""
import probe.common as common

import argparse
import os
import time

import torch
from torchvision.datasets import ImageFolder

common.add_maskedkd_to_path()
import datasets as mkd_datasets  # noqa: E402
import main as mkd_main  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--need", type=float, default=750.0, help="img/s the GPU side can consume, with margin")
    a = ap.parse_args()

    margs = mkd_main.get_args_parser().parse_args([])
    ds = ImageFolder(os.path.join(a.data_path, "train"), transform=mkd_datasets.build_transform(True, margs))
    best = None
    for w in a.workers:
        dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=w, drop_last=True)
        it = iter(dl)
        for _ in range(min(5, a.batches)):
            next(it)
        t0, n = time.time(), 0
        for _ in range(a.batches):
            x, _ = next(it)
            n += x.shape[0]
        ips = n / (time.time() - t0)
        print(f"num_workers={w:3d}: {ips:7.1f} img/s", flush=True)
        if best is None and ips >= a.need:
            best = w
        del it, dl
    cpus = os.cpu_count()
    if best is None:
        print(f"WARNING: no worker count reached {a.need:.0f} img/s on {cpus} CPUs: training will be loader-bound")
    else:
        print(f"RECOMMEND NUM_WORKERS={best} (reaches {a.need:.0f} img/s; {cpus} CPUs)")


if __name__ == "__main__":
    main()
