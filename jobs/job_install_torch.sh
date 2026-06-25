#!/bin/bash
#SBATCH --job-name=install_torch
#SBATCH --account=t4_ai
#SBATCH --partition=t4_ai
#SBATCH --qos=t4_ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

cd /scratch/shassanizorgabad22/SANs
mkdir -p outputs/logs

PYTHON=/home/shassanizorgabad22/.conda/envs/cv_env/bin/python

echo "===== PYTHON ====="
$PYTHON --version
$PYTHON -m pip --version

echo "===== CLEANING BROKEN TORCH INSTALLATION ====="
$PYTHON -m pip uninstall -y torch torchvision torchaudio || true

echo "===== INSTALLING MATCHING TORCH PACKAGES ====="
$PYTHON -m pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "===== VERIFY INSTALLATION ====="
$PYTHON - <<'PY'
import torch
import torchvision
import torchaudio

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "===== TORCH INSTALL JOB DONE ====="