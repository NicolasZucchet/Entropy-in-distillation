#!/bin/bash
# The twelve arms: {off, on}-policy x {forward, reverse} KL x three teachers,
# about 6.3 hours each on one H100. Arms checkpoint every 50 steps, so rerunning
# this script resumes killed arms and skips finished ones.
set -euo pipefail
source "$(dirname "$0")/common.sh"
for T in Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B; do
  for SRC in off on; do
    for DIV in forward reverse; do
      NAME="$SRC-$DIV-${T##*/}"
      [ -d "$SCRATCH_DIR/runs/$NAME/final" ] && { echo "$NAME done"; continue; }
      LMBDA=$([ $SRC = on ] && echo 1 || echo 0)
      BETA=$([ $DIV = reverse ] && echo 1 || echo 0)
      submit "$NAME" 12:00:00 "$PYTHON train.py --teacher $T --lmbda $LMBDA --beta $BETA"
    done
  done
done
