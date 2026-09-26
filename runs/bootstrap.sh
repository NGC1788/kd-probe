#!/usr/bin/env bash
# Rebuild everything on a fresh server, then run stage 0 (bench, smoke, residual).
#   mkdir -p ~/kdprobe && git clone https://github.com/NGC1788/kd-probe ~/kdprobe/code
#   bash ~/kdprobe/code/runs/bootstrap.sh 2>&1 | tee ~/kdprobe/bootstrap.log
# Safe to re-run: finished steps are skipped. Does not start the 12-run probe.
set -euo pipefail
BASE=${KDPROBE_HOME:-$HOME/kdprobe}
CODE="$BASE/code"
DATA="$BASE/data/in100"
TORCH_PIN=${TORCH_PIN:-"torch==2.7.0 torchvision==0.22.0"}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

echo "== 0. server check"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing: NVIDIA driver not installed yet"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader || true
df -h "$HOME" | tail -1; free -g | sed -n 2p; nproc
command -v git >/dev/null || { echo "git missing"; exit 1; }
command -v uv >/dev/null || { echo "uv missing: install it first (https://docs.astral.sh/uv/getting-started/installation/)"; exit 1; }

echo "== 1. venv + torch ($TORCH_PIN)"
mkdir -p "$BASE" && cd "$BASE"
[ -d .venv ] || uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.7.0') and torch.cuda.is_available() else 1)" 2>/dev/null \
  || uv pip install $TORCH_PIN --index-url "$TORCH_INDEX"
# brotli: an old brotli breaks huggingface_hub downloads ("process() takes no keyword arguments")
uv pip install -U huggingface_hub brotli

echo "== 2. code + MaskedKD"
[ -d "$CODE/.git" ] || git clone https://github.com/NGC1788/kd-probe "$CODE"
cd "$CODE" && git pull -q && ./setup.sh

echo "== 3. ImageNet-100"
if [ -f "$DATA/manifest.json" ] && python -c "import json,sys; sys.exit(0 if json.load(open('$DATA/manifest.json'))['counts']=={'train':126689,'val':5000} else 1)"; then
  echo "already prepared: $DATA"
else
  python -c "from huggingface_hub import snapshot_download as d; d('clane9/imagenet-100', repo_type='dataset', local_dir='$BASE/data/imagenet-100')"
  python -m probe.prepare_in100 --src "$BASE/data/imagenet-100" --dst "$DATA"
  rm -rf "$BASE/data/imagenet-100"
fi

echo "== 4. stage 0"
mkdir -p out
python -m probe.loadercheck --data-path "$DATA" 2>&1 | grep -v -i -E "warn|register_model" | tee out/loadercheck.txt
python -m probe.bench --out out/bench.json 2>&1 | grep -v -i -E "warn|register_model"
bash runs/smoke.sh > out/smoke.log 2>&1 && tail -6 out/smoke.log && rm -f out/smoke/*/checkpoint.pth
python -m probe.residual --data-path "$DATA" --out out/residual_init.json 2>&1 | grep -v -i -E "warn|register_model" | tail -5
df -h "$HOME" | tail -1
echo "== done. Start the probe only when the GPU is free (see README)."
