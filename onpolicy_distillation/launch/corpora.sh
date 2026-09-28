#!/bin/bash
# Teacher corpora: 40,000 DeepScaleR prompts (96 held out) and the 200 MATH500
# problems, per teacher. 4.7, 6.5 and 8.1 hours on one H100 for 1.7B, 4B and 8B.
set -euo pipefail
source "$(dirname "$0")/common.sh"
for T in Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B; do
  submit "corpus-${T##*/}" 12:00:00 "
$PYTHON gen_teacher_corpus.py --teacher $T --dataset deepscaler --num_prompts 40000 --num_eval_prompts 96
$PYTHON gen_teacher_corpus.py --teacher $T --dataset math500 --num_prompts 200 --num_eval_prompts 200 --eval_only --drop_unboxed 0"
done
