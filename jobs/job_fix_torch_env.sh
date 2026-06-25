#!/bin/bash
#SBATCH --job-name=fix_torch_env
#SBATCH --account=t4_ai
#SBATCH --partition=t4_ai
#SBATCH --qos=t4_ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

cd /scratch/shassanizorgabad22/SANs
mkdir -p outputs/logs

PYTHON=/home/shassanizorgabad22/.conda/envs/cv_env/bin/python

echo "===== USING PYTHON ====="
echo $PYTHON
$PYTHON --version

echo "===== BEFORE INSTALL ====="
$PYTHON - <<'PY'
import sys
print("Python:", sys.version)

try:
    import torch
    print("torch:", torch.__version__)
except Exception as e:
    print("torch error:", e)

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("torchvision error:", e)
PY

echo "===== REINSTALLING TORCH / TORCHVISION / TORCHAUDIO ====="
$PYTHON -m pip uninstall -y torch torchvision torchaudio

$PYTHON -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo "===== AFTER INSTALL ====="
$PYTHON - <<'PY'
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "===== DONE ====="