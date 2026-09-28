#!/bin/bash
# Figure 13 beyond its 1e-5 column (which gsm8k.sh provides): 5 weights alpha
# and 3 teachers at 3 learning rates x 3 seeds. `teacher-ema` is the default
# configuration, as is `alpha0.50`; both were run.
source launch/common.sh "$@"

for lr in 1e-6 3e-6 3e-5; do
  for seed in 0 1 2; do
    for a in 0.00 0.25 0.50 0.75 1.00; do
      submit "gsm8k-lr$lr-alpha$a-s$seed" GSM8K_JOB GSM8K \
          --distillation_alpha "$a" --learning_rate "$lr" --seed "$seed"
    done
    for kind in ema live base; do
      submit "gsm8k-lr$lr-teacher-$kind-s$seed" GSM8K_JOB GSM8K \
          --teacher_kind "$kind" --learning_rate "$lr" --seed "$seed"
    done
  done
done
