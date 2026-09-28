#!/bin/bash
# Figure 12 (and the 1e-5 column of Figure 13): 10 GSM8K configurations x 3 seeds.
source launch/common.sh "$@"

ARMS=(
  "default|"
  "alpha0.00|--distillation_alpha 0.0"
  "alpha0.25|--distillation_alpha 0.25"
  "alpha0.75|--distillation_alpha 0.75"
  "alpha1.00|--distillation_alpha 1.0"
  "teacher-live|--teacher_kind live"
  "teacher-base|--teacher_kind base"
  "full|--distillation_mode full_logits"
  "topk20|--distillation_topk 20"
  "topk100-notail|--distillation_add_tail false"
)
for entry in "${ARMS[@]}"; do
  for seed in 0 1 2; do
    # shellcheck disable=SC2086
    submit "gsm8k-lr1e-5-${entry%%|*}-s$seed" GSM8K_JOB GSM8K ${entry#*|} --seed "$seed"
  done
done
