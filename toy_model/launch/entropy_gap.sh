#!/bin/bash
# Trains the sweeps of Figure 2 (panels b-e) into outputs/, then merges them into outputs/<sweep>.npz.
# On one machine: bash launch/entropy_gap.sh
# On Slurm, split over array tasks: sbatch --array=0-7 launch/entropy_gap.sh
# then, once every task has finished, uv run python sweep.py merge <sweep> for each sweep below.
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
set -e
PYTHON=${PYTHON:-.venv/bin/python}  # the environment of `uv sync --group experiments`
for sweep in context_load perplexity_gap jsd_family; do
    $PYTHON sweep.py run $sweep
done
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    for sweep in context_load perplexity_gap jsd_family; do
        $PYTHON sweep.py merge $sweep
    done
fi
