#!/bin/bash
# Figure 9: 16 cells, one GPU each except the two 32B cells, which need two; about 2 GPU-hours.
# Run from the folder root with SCRATCH_DIR set, and SBATCH_ACCOUNT / SBATCH_PARTITION as your cluster needs.
set -euo pipefail
mkdir -p "$SCRATCH_DIR/logs"
indices() {  # indices of the Figure 9 cells that need $1 GPUs
  uv run --frozen python -c "from cells import CELLS, FIGURES; print(','.join(str(i) for i, n in enumerate(FIGURES['families']) if CELLS[n].gpus == $1))"
}
sbatch --array="$(indices 1)" --output="$SCRATCH_DIR/logs/%x-%A_%a.out" launch/measure.sbatch families
sbatch --array="$(indices 2)" --gpus=2 --output="$SCRATCH_DIR/logs/%x-%A_%a.out" launch/measure.sbatch families
