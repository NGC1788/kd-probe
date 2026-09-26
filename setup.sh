#!/usr/bin/env bash
# Clone MaskedKD at the pinned commit and install the probe's extra deps into the ACTIVE venv.
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=96d052da7346441e2425876a9ad37ff8c87fe383
if [ -z "${VIRTUAL_ENV:-}" ]; then echo "activate the venv first: source ~/kdprobe/.venv/bin/activate"; exit 1; fi
mkdir -p third_party
[ -d third_party/MaskedKD/.git ] || git clone https://github.com/effl-lab/MaskedKD third_party/MaskedKD
git -C third_party/MaskedKD checkout -q "$COMMIT"
uv pip install "timm==0.9.16" pyarrow
python -c "import torch, timm, pyarrow; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'timm', timm.__version__, 'pyarrow', pyarrow.__version__)"
python -c "import probe.common as c; c.add_maskedkd_to_path(); import main, deit3, models_student; print('MaskedKD imports OK at', c.MASKEDKD_COMMIT[:7])"
