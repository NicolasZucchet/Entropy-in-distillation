#!/bin/bash
# Trains the sweeps of Figures 3 and 8 into outputs/, then merges them into outputs/<sweep>.npz.
# On one machine: bash launch/topk.sh
# On Slurm, split over array tasks: sbatch --array=0-7 launch/topk.sh
# then, once every task has finished, uv run python sweep.py merge <sweep> for each sweep below.
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
set -e
PYTHON=${PYTHON:-.venv/bin/python}  # the environment of `uv sync --group experiments`
for sweep in topk; do
    $PYTHON sweep.py run $sweep
done
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    for sweep in topk; do
        $PYTHON sweep.py merge $sweep
    done
fi
