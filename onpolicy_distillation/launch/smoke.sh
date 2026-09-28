#!/bin/bash
# A short check of the setup: the on-policy reverse KL arm with the Qwen3-1.7B
# teacher, stopped after its step-25 evaluation (about 0.5 GPU-hours; needs the
# Qwen3-1.7B corpora). At step 0 the paper's run reads 0.5754, 0.5365 and 0.280
# for the entropy on teacher completions, on student completions, and accuracy.
set -euo pipefail
source "$(dirname "$0")/common.sh"
submit smoke-on-reverse-Qwen3-1.7B ${WALL:-01:30:00} "$PYTHON train.py \
--teacher Qwen/Qwen3-1.7B --lmbda 1 --beta 1 --run_name smoke-on-reverse-Qwen3-1.7B \
--stop_after 25 --save_interval 1000"
