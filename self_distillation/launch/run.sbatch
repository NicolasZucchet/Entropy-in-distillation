#!/bin/bash
#SBATCH --nodes=1
#SBATCH --export=ALL
# One run on one node, submitted by `submit` in launch/common.sh.
set -euo pipefail
: "${SCRATCH_DIR:?set SCRATCH_DIR}"
cd "$SLURM_SUBMIT_DIR"
export PATH="${VENV:-$SLURM_SUBMIT_DIR/.venv}/bin:$PATH"  # from `uv sync --group experiments`

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$SCRATCH_DIR/hf" HF_HUB_OFFLINE=1  # filled by fetch_data.py
export WANDB_DIR="$SCRATCH_DIR/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_WORKER_MULTIPROC_METHOD=spawn
# Compilation caches default to the home directory, which is often small.
export TRITON_CACHE_DIR="$SCRATCH_DIR/triton-cache" VLLM_CACHE_ROOT="$SCRATCH_DIR/vllm-cache"
export TORCH_EXTENSIONS_DIR="$SCRATCH_DIR/torch-extensions" VLLM_NO_USAGE_STATS=1

PORT=$((20000 + SLURM_JOB_ID % 10000))  # two jobs on one node need different ports
srun accelerate launch --config_file launch/zero3.yaml \
    --num_processes "${GPUS:-8}" --main_process_port "$PORT" train.py "$@"
