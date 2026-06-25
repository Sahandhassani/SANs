#!/bin/bash
#SBATCH --job-name=download_osdr
#SBATCH --account=t4_ai
#SBATCH --partition=t4_ai
#SBATCH --qos=t4_ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/%x-%j.out
#SBATCH --error=outputs/logs/%x-%j.err

set -euo pipefail

cd /scratch/shassanizorgabad22/SANS_MultiStage_TL

mkdir -p outputs/logs
mkdir -p Data/raw/OSDR/OSD-679
mkdir -p Data/raw/OSDR/OSD-680

module purge
module load cuda/12.8.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cv_env

python Data/download/download_osdr.py \
    --study_id OSD-679 \
    --output_dir Data/raw/OSDR/OSD-679

python Data/download/download_osdr.py \
    --study_id OSD-680 \
    --output_dir Data/raw/OSDR/OSD-680
