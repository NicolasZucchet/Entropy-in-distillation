#!/bin/bash
# Figure 4: 12 cells on one GPU each, about 0.5 GPU-hours in total.
# Run from the folder root with SCRATCH_DIR set, and SBATCH_ACCOUNT / SBATCH_PARTITION as your cluster needs.
set -euo pipefail
mkdir -p "$SCRATCH_DIR/logs"
sbatch --array=0-11 --output="$SCRATCH_DIR/logs/%x-%A_%a.out" launch/measure.sbatch sft
