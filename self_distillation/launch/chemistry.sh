#!/bin/bash
# Figures 6 and 11: 14 Chemistry runs, the default and one field moved at a time.
source launch/common.sh "$@"

submit chem-default CHEM_JOB CHEM
for a in 0.00 0.25 0.75 1.00; do
  submit "chem-alpha$a" CHEM_JOB CHEM --distillation_alpha "$a"
done
submit chem-teacher-live CHEM_JOB CHEM --teacher_kind live
submit chem-teacher-base CHEM_JOB CHEM --teacher_kind base
submit chem-teacher-ema0.99 CHEM_JOB CHEM --teacher_update_rate 0.01
for k in 1 10 1000; do
  submit "chem-topk$k" CHEM_JOB CHEM --distillation_topk "$k"
done
for k in 10 100 1000; do
  submit "chem-topk$k-notail" CHEM_JOB CHEM --distillation_topk "$k" --distillation_add_tail false
done
