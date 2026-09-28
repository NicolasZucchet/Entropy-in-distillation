# Sourced by the launchers. From this folder:
#
#   export SCRATCH_DIR=/path/to/scratch   # VENV defaults to .venv, from `uv sync --group experiments`
#   export SBATCH_ACCOUNT=... SBATCH_PARTITION=...   # as your cluster needs
#   bash launch/gsm8k.sh --dry-run                # print the runs; drop --dry-run to submit
set -euo pipefail
: "${SCRATCH_DIR:?set SCRATCH_DIR}"
mkdir -p "$SCRATCH_DIR/logs"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY=1

# Chemistry: Olmo-3-7B-Instruct on one node of 8 GPUs, the defaults of config.py.
CHEM=(--task chemistry)
CHEM_JOB=(--gpus=8 --cpus-per-task=64 --mem=800G --time=04:00:00 --export=ALL,GPUS=8)

# GSM8K: Qwen3-0.6B on one GPU, 50 steps, validation every 5.
GSM8K=(--task gsm8k --model Qwen/Qwen3-0.6B --max_prompt_length 512 --max_completion_length 640
       --per_device_batch_size 8 --gpus 1 --eval_batch_groups 8 --vllm_gpu_memory_utilization 0.4
       --max_steps 50 --eval_every 5)
GSM8K_JOB=(--gpus=1 --cpus-per-task=8 --mem=100G --time=03:00:00 --export=ALL,GPUS=1)

# submit <run name> <job options array> <run options array> [--field value ...]
submit() {
  local name="$1" job="$2[@]" run="$3[@]"; shift 3
  if [ -n "$DRY" ]; then echo "$name: ${!run} $*"; return; fi
  sbatch --job-name="$name" --output="$SCRATCH_DIR/logs/%x-%j.out" "${!job}" \
      launch/run.sbatch "${!run}" "$@" --run_name "$name"
}
