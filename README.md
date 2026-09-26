# kd-probe

**Does ground-truth CE hide the cost of teacher-side masking in knowledge distillation?**

This is a small pre-registered probe on top of [MaskedKD](https://github.com/effl-lab/MaskedKD) (Son et al., ECCV 2024). MaskedKD masks the teacher's input patches using student attention and reports no accuracy drop at 50% teacher cost. Every supervised KD run in that paper keeps the ground-truth CE term. This probe asks whether the result still holds when CE is removed from the same frozen-teacher recipe.

The design and the decision rule are fixed in [PREREGISTRATION.md](PREREGISTRATION.md) before any result is seen.

## Layout
| File | Purpose |
|---|---|
| `setup.sh` | clones MaskedKD at commit `96d052d` into `third_party/` and installs `timm==0.9.16` and `pyarrow` |
| `probe/prepare_in100.py` | turns HF `clane9/imagenet-100` parquet into an ImageFolder tree, after checking every label against its ImageNet wnid |
| `probe/launch.py` | runs MaskedKD's own training loop with the probe's patches (see the docstring) |
| `probe/bench.py` | measures synthetic throughput for budgeting |
| `probe/residual.py` | Stage 0: per-sample gradient residual between full and masked teacher targets |
| `probe/collect.py` | applies the pre-registered decision rule to finished runs |
| `runs/smoke.sh`, `runs/p1.sh` | smoke test and the 12-run probe |

MaskedKD has no license file, so its code is **not** redistributed here. `setup.sh` clones it. The patches in `probe/launch.py` fix a few things:
- MaskedKD's loss calls DeiT III teachers with three arguments, but their forward takes two.
- On one GPU the train sampler never calls `set_epoch`.
- The probe also needs the 100-class teacher logits and the `--ce-off` switch.

## Usage (single GPU)
```bash
mkdir -p ~/kdprobe && cd ~/kdprobe
uv venv --python ~/miniforge3/bin/python3 --system-site-packages .venv && source .venv/bin/activate
git clone https://github.com/NGC1788/kd-probe code && cd code
./setup.sh

# data: HF clane9/imagenet-100 (8.4 GB parquet, non-commercial research terms)
python -m probe.prepare_in100 --src ~/kdprobe/data/imagenet-100 --dst ~/kdprobe/data/in100

python -m probe.bench --out out/bench.json          # budgeting only
bash runs/smoke.sh                                    # 1 epoch, 10 images/class
python -m probe.residual --data-path ~/kdprobe/data/in100 --out out/residual_init.json
EPOCHS=100 nohup bash runs/p1.sh > p1.out 2>&1 &     # EPOCHS fixed before stage 1
python -m probe.collect --root out/p1 --epochs 100
```

## License
MIT for the code in this repository (see `LICENSE`). ImageNet-derived data follows the ImageNet terms of access.
