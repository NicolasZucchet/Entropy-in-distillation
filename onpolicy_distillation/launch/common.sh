# Sourced by the launchers. Settings, all overridable in the environment:
#
#   SCRATCH_DIR  directory for the corpora, checkpoints and logs (required)
#   PYTHON       interpreter (default: .venv/bin/python, from `uv sync --group experiments`)
#   SBATCH_ARGS  site flags, e.g. "--account=abc --partition=gpu"
#   LOCAL=1      run in the foreground instead of submitting to Slurm
: "${SCRATCH_DIR:?set SCRATCH_DIR}"
export SCRATCH_DIR
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"  # 1 on compute nodes without internet, after `uv run python data.py`
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 VLLM_CACHE_ROOT="$SCRATCH_DIR/vllm-cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON="${PYTHON:-$HERE/.venv/bin/python}"
mkdir -p "$SCRATCH_DIR/logs"

# submit NAME WALLTIME COMMAND: one job on one GPU.
submit() {
  if [ -n "${LOCAL:-}" ]; then
    (cd "$HERE" && bash -c "$3")
    return
  fi
  # Ports derived from the job id, so that jobs sharing a node do not collide.
  sbatch ${SBATCH_ARGS:-} --gpus=1 --cpus-per-task=16 --mem=200G --time="$2" \
    --job-name="$1" --output="$SCRATCH_DIR/logs/%x-%j.out" \
    --wrap "cd $HERE
export MASTER_PORT=\$((29500 + SLURM_JOB_ID % 20000)) VLLM_GROUP_PORT=\$((51216 + SLURM_JOB_ID % 10000))
$3"
}
