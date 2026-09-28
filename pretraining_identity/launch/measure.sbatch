#!/bin/bash
#SBATCH --job-name=identity
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
# One cell per array task: `sbatch --array=... launch/measure.sbatch <figure>`.
# Submitted by launch/sft.sh and launch/families.sh.
set -euo pipefail
: "${SCRATCH_DIR:?set SCRATCH_DIR to the directory prefetch.py downloaded to}"
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
"${PYTHON:-.venv/bin/python}" measure.py --figure "$1" --index "$SLURM_ARRAY_TASK_ID" --out_dir "${OUT_DIR:-$SCRATCH_DIR/identity}"
