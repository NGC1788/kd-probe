"""Run MaskedKD's official training loop with the probe's changes patched in.

MaskedKD (github.com/effl-lab/MaskedKD) has no license, so it is not copied
here: setup.sh clones it at a pinned commit and this launcher patches it at
import time. Changes, all needed for the pre-registered P1 probe:

1. --data-set IN100: ImageFolder over the extracted ImageNet-100 tree.
2. Teacher logits restricted to the 100 ImageNet-100 classes before softmax.
3. --ce-off: drop the ground-truth CE term, keep the KD term's weight (alpha)
   unchanged. (Setting alpha=1 instead would double the KD weight.)
4. DeiT III teachers: MaskedKD's loss calls teacher(x, keep, maskedkd) but
   deit3.vit_models.forward takes (x, keep); an adapter handles both.
5. Single-GPU sampler fix: main.py only calls sampler.set_epoch() when
   distributed, so a single GPU would replay one shuffle order every epoch.
   The train sampler now advances its epoch itself and uses --seed.
6. Timing and peak memory per epoch -> probe_timing.jsonl; a copy of the
   checkpoint at 50% of training (seed 0 only) for the residual measurement.
"""
import probe.common as common  # sets torch.load policy before torch import

import argparse
import inspect
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

common.add_maskedkd_to_path()

import datasets as mkd_datasets  # noqa: E402  (MaskedKD's datasets.py)
import engine as mkd_engine  # noqa: E402
import losses as mkd_losses  # noqa: E402
import samplers as mkd_samplers  # noqa: E402

PROBE = {"ce_off": False, "idx100": None, "seed": 0, "subset": 0}


# ---------------------------------------------------------------- dataset
_orig_build_dataset = mkd_datasets.build_dataset


def build_dataset(is_train, args):
    if args.data_set != "IN100":
        return _orig_build_dataset(is_train, args)
    from torchvision.datasets import ImageFolder
    transform = mkd_datasets.build_transform(is_train, args)
    ds = ImageFolder(os.path.join(args.data_path, "train" if is_train else "val"), transform=transform)
    assert len(ds.classes) == common.IN100_NUM_CLASSES
    if is_train and PROBE["subset"] > 0:
        per_class, keep = {}, []
        for i, (_, y) in enumerate(ds.samples):
            if per_class.get(y, 0) < PROBE["subset"]:
                per_class[y] = per_class.get(y, 0) + 1
                keep.append(i)
        ds = torch.utils.data.Subset(ds, keep)
    return ds, common.IN100_NUM_CLASSES


mkd_datasets.build_dataset = build_dataset


# ---------------------------------------------------------------- loss
class ProbeDistillationLoss(torch.nn.Module):
    """Same signature and KD math as MaskedKD's DistillationLoss (soft KD,
    KL(teacher || student) * T^2), plus the probe switches."""

    def __init__(self, base_criterion, teacher_model, distillation_type, alpha, tau, len_num_keep, maskedkd):
        super().__init__()
        assert distillation_type == "soft", "the probe uses soft KD only"
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha, self.tau = alpha, tau
        self.len_num_keep, self.maskedkd = len_num_keep, maskedkd
        self.ce_weight = 0.0 if PROBE["ce_off"] else (1.0 - alpha)
        self.idx100 = PROBE["idx100"]
        self.teacher_takes_flag = "maskedkd" in inspect.signature(teacher_model.forward).parameters

    def teacher_logits(self, inputs, attn):
        B, N = inputs.shape[0], attn.shape[-1] - 1
        if self.maskedkd:
            keep = torch.topk(attn.mean(dim=1)[:, 0, 1:], self.len_num_keep).indices
        else:
            keep = torch.arange(N, device=inputs.device).unsqueeze(0).expand(B, N)
        with torch.no_grad():
            if self.teacher_takes_flag:
                t = self.teacher_model(inputs, keep, self.maskedkd)
            else:
                t = self.teacher_model(inputs, keep)
        if self.idx100 is not None:
            t = t.index_select(1, self.idx100.to(t.device))
        return t

    def forward(self, inputs, outputs, labels, attn):
        t = self.teacher_logits(inputs, attn)
        T = self.tau
        kd = F.kl_div(F.log_softmax(outputs / T, dim=1), F.softmax(t / T, dim=1), reduction="batchmean") * (T * T)
        loss = self.alpha * kd
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * self.base_criterion(outputs, labels)
        return loss


mkd_losses.DistillationLoss = ProbeDistillationLoss


# ---------------------------------------------------------------- samplers
class AutoEpochDistributedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, *a, **kw):
        kw.setdefault("seed", PROBE["seed"])
        super().__init__(*a, **kw)
        self._auto_epoch = -1

    def __iter__(self):
        self._auto_epoch += 1
        self.set_epoch(self._auto_epoch)
        return super().__iter__()


class AutoEpochRASampler(mkd_samplers.RASampler):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._auto_epoch = -1

    def __iter__(self):
        self._auto_epoch += 1
        self.set_epoch(self._auto_epoch + 1000 * PROBE["seed"])
        return super().__iter__()


torch.utils.data.DistributedSampler = AutoEpochDistributedSampler
mkd_samplers.RASampler = AutoEpochRASampler


# ---------------------------------------------------------------- timing
_orig_train_one_epoch = mkd_engine.train_one_epoch


def timed_train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                          max_norm=0, mixup_fn=None, set_training_mode=True, args=None):
    out = Path(args.output_dir) if args is not None and args.output_dir else None
    if out is not None and PROBE["seed"] == 0 and epoch == args.epochs // 2 and (out / "checkpoint.pth").exists():
        shutil.copyfile(out / "checkpoint.pth", out / "checkpoint_mid.pth")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    stats = _orig_train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                                  max_norm, mixup_fn, set_training_mode=set_training_mode, args=args)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = len(data_loader) * data_loader.batch_size
    rec = {"epoch": epoch, "sec": round(dt, 2), "img_per_s": round(n / dt, 1),
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    if out is not None:
        with (out / "probe_timing.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return stats


mkd_engine.train_one_epoch = timed_train_one_epoch

import main as mkd_main  # noqa: E402  (imports the patched names above)


# ---------------------------------------------------------------- entry
def git_rev(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser("kd-probe launcher", parents=[mkd_main.get_args_parser()])
    for a in parser._actions:
        if a.dest == "data_set":
            a.choices = list(a.choices) + ["IN100"]
    parser.add_argument("--ce-off", action="store_true", help="drop ground-truth CE, keep KD weight alpha")
    parser.add_argument("--probe-subset", type=int, default=0, help="images per class (smoke tests only)")
    args = parser.parse_args()

    PROBE["ce_off"] = args.ce_off
    PROBE["seed"] = args.seed
    PROBE["subset"] = args.probe_subset
    random.seed(args.seed)
    if args.data_set == "IN100":
        PROBE["idx100"], classes = common.teacher_index_for_folders(os.path.join(args.data_path, "train"))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = {k: (v if isinstance(v, (int, float, str, bool, type(None), list)) else str(v))
               for k, v in vars(args).items()}
        cfg["probe"] = {"ce_weight": 0.0 if args.ce_off else 1.0 - args.distillation_alpha,
                        "kd_weight": args.distillation_alpha,
                        "teacher_idx100": None if PROBE["idx100"] is None else PROBE["idx100"].tolist(),
                        "maskedkd_commit": git_rev(common.MASKEDKD_DIR),
                        "probe_commit": git_rev(common.REPO)}
        (out / "probe_config.json").write_text(json.dumps(cfg, indent=1))
        assert cfg["probe"]["maskedkd_commit"] in (None, common.MASKEDKD_COMMIT), "MaskedKD commit differs from pinned"

    mkd_main.main(args)


if __name__ == "__main__":
    main()
