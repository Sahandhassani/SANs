#!/bin/bash
#SBATCH --job-name=test_kermany_loader
#SBATCH --account=t4_ai
#SBATCH --partition=t4_ai
#SBATCH --qos=t4_ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

cd /scratch/shassanizorgabad22/SANs
mkdir -p outputs/logs

PYTHON=/home/shassanizorgabad22/.conda/envs/cv_env/bin/python

echo "===== PYTHON ====="
$PYTHON --version

echo "===== TESTING KERMANY DATASET LOADER ====="

$PYTHON - <<'PY'
from utils.dataset import KermanyBinaryDataset

train_ds = KermanyBinaryDataset(
    root_dir="Data/raw/Kermany2018",
    split="train",
    image_size=224
)

val_ds = KermanyBinaryDataset(
    root_dir="Data/raw/Kermany2018",
    split="val",
    image_size=224
)

test_ds = KermanyBinaryDataset(
    root_dir="Data/raw/Kermany2018",
    split="test",
    image_size=224
)

print("Train samples:", len(train_ds))
print("Val samples:", len(val_ds))
print("Test samples:", len(test_ds))

img, label = train_ds[0]
print("Image shape:", img.shape)
print("Label:", label)

print("===== DATASET LOADER WORKS =====")
PY