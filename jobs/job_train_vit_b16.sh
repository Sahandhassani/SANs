#!/bin/bash
#SBATCH --job-name=sans_vit
#SBATCH --account=t4_ai
#SBATCH --partition=t4_ai
#SBATCH --qos=t4_ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

cd /scratch/shassanizorgabad22/SANS_MultiStage_TL

mkdir -p outputs/logs outputs/checkpoints outputs/results

module purge
module load cuda/12.8.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cv_env

python main.py \
    --dataset kermany \
    --data_root Data/raw/Kermany2018 \
    --model vit_b16 \
    --pretrained \
    --image_size 224 \
    --batch_size 32 \
    --epochs 30 \
    --lr 1e-4 \
    --seed 42
