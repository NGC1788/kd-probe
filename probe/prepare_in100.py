"""Extract HF `clane9/imagenet-100` parquet shards into an ImageFolder tree.

    python -m probe.prepare_in100 --src ~/kdprobe/data/imagenet-100 --dst ~/kdprobe/data/in100

Label i of the HF dataset is line i of the CMC imagenet100.txt list. Before
writing anything we check each HF label name against timm's ImageNet-1k
description for that wnid, so the teacher-logit mapping cannot silently drift.
"""
import argparse
import glob
import json
import os
import urllib.request

import pyarrow.parquet as pq

from probe.common import CMC_URL, imagenet1k_wnids

EXPECTED = {"train": 126_689, "val": 5_000}
SPLITS = {"train": "train", "validation": "val"}


def hf_label_names(parquet_file):
    meta = pq.read_schema(parquet_file).metadata or {}
    raw = meta.get(b"huggingface")
    if raw is None:
        return None
    feats = json.loads(raw)["info"]["features"]
    return feats["label"].get("names")


def ext_for(b):
    if b[:3] == b"\xff\xd8\xff":
        return ".JPEG"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return ".img"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="snapshot dir containing data/*.parquet")
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    cmc = [l.strip() for l in urllib.request.urlopen(CMC_URL).read().decode().splitlines() if l.strip()]
    assert len(cmc) == 100, len(cmc)

    from timm.data import ImageNetInfo
    info = ImageNetInfo("imagenet-1k")
    wnids = imagenet1k_wnids()

    first = sorted(glob.glob(os.path.join(args.src, "data", "train-*.parquet")))[0]
    names = hf_label_names(first)
    mismatches = []
    if names is None:
        print("WARNING: no HF label names in parquet metadata; skipping name check")
    else:
        assert len(names) == 100
        for i, (name, w) in enumerate(zip(names, cmc)):
            desc = info.index_to_description(wnids.index(w))
            if name.split(",")[0].strip().lower() != desc.split(",")[0].strip().lower():
                mismatches.append((i, w, name, desc))
        if mismatches:
            for m in mismatches[:10]:
                print("MISMATCH", m)
            raise SystemExit(f"{len(mismatches)} label/wnid mismatches: refusing to extract")
        print("label names match CMC wnids: 100/100")

    counts = {}
    for hf_split, out_split in SPLITS.items():
        files = sorted(glob.glob(os.path.join(args.src, "data", f"{hf_split}-*.parquet")))
        assert files, f"no parquet files for split {hf_split}"
        n = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
                img = batch.column("image")
                blobs = img.field("bytes").to_pylist()
                paths = img.field("path").to_pylist()
                labels = batch.column("label").to_pylist()
                for b, p, y in zip(blobs, paths, labels):
                    d = os.path.join(args.dst, out_split, cmc[y])
                    os.makedirs(d, exist_ok=True)
                    stem = os.path.splitext(os.path.basename(p))[0] if p else f"{out_split}_{n:07d}"
                    out = os.path.join(d, f"{out_split}_{n:07d}_{stem}{ext_for(b)}")
                    if not os.path.exists(out):
                        with open(out, "wb") as fh:
                            fh.write(b)
                    n += 1
            print(f"{out_split}: {n} images after {os.path.basename(f)}", flush=True)
        counts[out_split] = n

    manifest = {"source": "hf:clane9/imagenet-100", "cmc_list": CMC_URL, "counts": counts,
                "expected": EXPECTED, "wnids_cmc_order": cmc}
    with open(os.path.join(args.dst, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    ok = counts == EXPECTED
    print("counts", counts, "OK" if ok else f"MISMATCH (expected {EXPECTED})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
