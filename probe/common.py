"""Shared helpers: MaskedKD path, ImageNet-100 class mapping, torch.load policy."""
import os
import sys
from pathlib import Path

# Trusted checkpoints only (official DeiT/DeiT III URLs and our own runs).
# torch>=2.6 defaults torch.load(weights_only=True), which rejects MaskedKD's
# checkpoints that also pickle `args`.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

REPO = Path(__file__).resolve().parents[1]
MASKEDKD_DIR = REPO / "third_party" / "MaskedKD"
MASKEDKD_COMMIT = "96d052da7346441e2425876a9ad37ff8c87fe383"
CMC_URL = "https://raw.githubusercontent.com/HobbitLong/CMC/master/imagenet100.txt"
IN100_NUM_CLASSES = 100


def add_maskedkd_to_path():
    if not MASKEDKD_DIR.exists():
        sys.exit(f"MaskedKD not found at {MASKEDKD_DIR}. Run ./setup.sh first.")
    p = str(MASKEDKD_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def imagenet1k_wnids():
    """The 1000 ImageNet-1k synsets in teacher-logit index order."""
    from timm.data import ImageNetInfo
    wnids = list(ImageNetInfo("imagenet-1k").label_names())
    assert len(wnids) == 1000 and wnids[0] == "n01440764", "unexpected ImageNet-1k synset order"
    return wnids


def teacher_index_for_folders(train_root):
    """Teacher-logit indices for the ImageFolder classes (sorted wnid folders)."""
    import torch
    classes = sorted(d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root, d)))
    assert len(classes) == IN100_NUM_CLASSES, f"expected 100 class folders, got {len(classes)}"
    wnids = imagenet1k_wnids()
    return torch.tensor([wnids.index(c) for c in classes], dtype=torch.long), classes
