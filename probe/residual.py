"""Stage 0: per-sample gradient residual between full and masked teacher targets.

    python -m probe.residual --data-path ~/kdprobe/data/in100 --out out/residual_init.json
    python -m probe.residual --data-path ~/kdprobe/data/in100 --ckpt out/p1/<run>/checkpoint_mid.pth --out ...

For n fixed training images (eval transform, deterministic), with the student
at init or at a checkpoint, per-sample gradients w.r.t. all student params:

  g_off  = alpha * grad KL(p_full || s)                      (CE-off training gradient)
  g_on   = (1-alpha) * grad CE_ls(s, y) + g_off              (CE-on training gradient)
  d_k    = alpha * (grad KL(p_mask_k || s) - grad KL(p_full || s))   for k kept tokens

Reports, per setting and k:
  R_k      = E||d_k||^2                 residual power
  bias2_k  = ||E d_k||^2 (debiased)     systematic part (does not shrink with batch B)
  sigma2   = tr Cov(g)                  SGD noise of the training gradient
  q*(kappa)= R_k / (R_k + kappa*sigma2) cheapest per-sample full-call probability keeping
             correction noise <= kappa * SGD noise (independent calls: B cancels)
  cost     = k/196 + q*                 teacher cost of the two-level estimator, in full-teacher units
  bias2 / (sigma2/B) for B in {128, 512}  (P2 is only a hypothesis; this is a frozen-state number)
"""
import probe.common as common

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

common.add_maskedkd_to_path()
import deit3  # noqa: E402
import models_student  # noqa: E402

KAPPAS = (0.25, 0.5, 1.0)
BATCHES = (128, 512)


class Stat:
    """Streaming mean vector and mean squared norm of flat gradient vectors."""

    def __init__(self):
        self.n, self.sum, self.sq = 0, None, 0.0

    def add(self, g):
        self.n += 1
        self.sum = g.clone() if self.sum is None else self.sum.add_(g)
        self.sq += float(g.pow(2).sum())

    def summary(self):
        mean = self.sum / self.n
        m2 = self.sq / self.n
        mean_sq = float(mean.pow(2).sum())
        trcov = (m2 - mean_sq) * self.n / (self.n - 1)
        return {"E_norm2": m2, "mean_norm2_raw": mean_sq,
                "mean_norm2_debiased": mean_sq - trcov / self.n, "trcov": trcov}


def flat_grad(params):
    return torch.cat([p.grad.reshape(-1) for p in params]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--ckpt", default="", help="checkpoint.pth / checkpoint_mid.pth; empty = init")
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--keeps", type=int, nargs="+", default=[98, 59])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--random-teacher", action="store_true", help="dry runs only")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = torch.device(args.device)
    train_root = os.path.join(args.data_path, "train")
    idx100, _ = common.teacher_index_for_folders(train_root)
    idx100 = idx100.to(dev)

    tf = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    ds = ImageFolder(train_root, transform=tf)
    order = random.Random(args.seed).sample(range(len(ds)), args.n)

    student = models_student.deit_small_patch16_224(num_classes=100, drop_path_rate=0.0).to(dev)
    if args.ckpt:
        student.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"])
    student.eval()  # no dropout / drop-path: measure the loss landscape, not regularizer noise
    teacher = deit3.deit_base_patch16_LS(pretrained=not args.random_teacher).to(dev).eval()
    params = [p for p in student.parameters() if p.requires_grad]
    a = args.alpha

    st = {"g_off": Stat(), "g_on": Stat(), **{f"d_{k}": Stat() for k in args.keeps}}
    for j, i in enumerate(order):
        x, y = ds[i]
        x, y = x.unsqueeze(0).to(dev), torch.tensor([y], device=dev)
        out, attn = student(x)
        logp = F.log_softmax(out.float(), dim=1)
        N = attn.shape[-1] - 1
        with torch.no_grad(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            p_full = F.softmax(teacher(x, torch.arange(N, device=dev).unsqueeze(0)).index_select(1, idx100).float(), 1)
            score = attn.mean(dim=1)[:, 0, 1:]
            p_mask = {k: F.softmax(teacher(x, torch.topk(score, k).indices).index_select(1, idx100).float(), 1)
                      for k in args.keeps}
        kl_full = (p_full * (p_full.clamp_min(1e-12).log() - logp)).sum()
        ce_ls = (1 - args.smoothing) * F.nll_loss(logp, y) - args.smoothing * logp.mean()

        losses = [(f"d_{k}", -a * ((p_mask[k] - p_full) * logp).sum()) for k in args.keeps]
        losses += [("g_off", a * kl_full), ("g_on", (1 - a) * ce_ls + a * kl_full)]
        for m, (name, loss) in enumerate(losses):
            student.zero_grad(set_to_none=False)
            loss.backward(retain_graph=m < len(losses) - 1)
            st[name].add(flat_grad(params))
        if (j + 1) % 256 == 0:
            print(f"{j + 1}/{args.n}", flush=True)

    s = {k: v.summary() for k, v in st.items()}
    report = {"ckpt": args.ckpt or "init", "n": args.n, "alpha": a, "keeps": args.keeps, "raw": s, "derived": {}}
    for setting in ("g_off", "g_on"):
        sigma2 = s[setting]["trcov"]
        for k in args.keeps:
            R = s[f"d_{k}"]["E_norm2"]
            b2 = s[f"d_{k}"]["mean_norm2_debiased"]
            qs = {str(kp): R / (R + kp * sigma2) for kp in KAPPAS}
            report["derived"][f"{setting}_keep{k}"] = {
                "sigma2": sigma2, "R": R, "bias2": b2,
                "q_star": qs, "cost_full_units": {kp: k / 196 + q for kp, q in qs.items()},
                "bias2_over_noise": {str(B): b2 / (sigma2 / B) for B in BATCHES}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    for key, v in report["derived"].items():
        print(key, {"q*(k=1)": round(v["q_star"]["1.0"], 4), "cost(k=1)": round(v["cost_full_units"]["1.0"], 3),
                    "bias2/noise(B=128)": round(v["bias2_over_noise"]["128"], 3)})
    print("saved", args.out)


if __name__ == "__main__":
    main()
