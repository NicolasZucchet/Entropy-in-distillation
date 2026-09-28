#!/bin/bash
# Figure 14: 5 weights alpha, 500 GSM8K steps, one seed, no validation.
source launch/common.sh "$@"

LONG_JOB=(--gpus=1 --cpus-per-task=8 --mem=100G --time=04:00:00 --export=ALL,GPUS=1)
for a in 0.00 0.25 0.50 0.75 1.00; do
  submit "gsm8k-long-alpha$a-s0" LONG_JOB GSM8K --max_steps 500 --eval_every 0 --distillation_alpha "$a"
done
