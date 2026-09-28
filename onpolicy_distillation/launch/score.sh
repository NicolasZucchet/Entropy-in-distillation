#!/bin/bash
# Endpoint accuracy of the base student and the twelve final checkpoints on
# MATH500 (avg@3), GSM8K (avg@3) and AMC23 (avg@5), and of the base student in
# the chat template on MATH500. About 20 minutes per model on one H100.
set -euo pipefail
source "$(dirname "$0")/common.sh"
OUT=$SCRATCH_DIR/scores
BENCHES=("math500 500 3" "gsm8k 1319 3" "amc23 40 5")
MODELS=("student:Qwen/Qwen3-0.6B-Base")
for RUN in "$SCRATCH_DIR"/runs/o*-*-*/final; do
  [ -d "$RUN" ] && MODELS+=("$(basename "$(dirname "$RUN")"):$RUN")
done
for M in "${MODELS[@]}"; do
  LABEL="${M%%:*}"; MODEL="${M#*:}"; CMD=""
  for B in "${BENCHES[@]}"; do
    read -r DS N K <<< "$B"
    CMD="$CMD
$PYTHON score_endpoint.py --model $MODEL --dataset $DS --num_prompts $N --k $K --label $LABEL-$DS --out $OUT/$LABEL.jsonl"
  done
  if [ "$LABEL" = student ]; then
    CMD="$CMD
$PYTHON score_endpoint.py --model $MODEL --dataset math500 --num_prompts 500 --k 3 --chat --thinking --label student-chat-math500 --out $OUT/student-chat.jsonl"
  fi
  submit "score-$LABEL" 05:00:00 "$CMD"
done
