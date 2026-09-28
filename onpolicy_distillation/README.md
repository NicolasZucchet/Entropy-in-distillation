# On- vs. off-policy distillation

Qwen3-1.7B, 4B and 8B (thinking disabled) are distilled into Qwen3-0.6B-Base on DeepScaleR math problems.
Each training run crosses where the prefixes come from (cached teacher completions, or the student's own samples) with the divergence (forward or reverse KL), which gives twelve runs.
This is Section 4.2 of the paper (setup in Appendix C.3).

## Figures

The evaluations of every run are in `data/`, so the figures only need numpy and matplotlib, which [uv](https://docs.astral.sh/uv/) installs on first use.

```bash
uv run python plot.py all    # writes figures/onpolicy_distillation_<name>.pdf and prints the accuracy table
```

| Paper figure | Command |
|---|---|
| 5 | `uv run python plot.py entropy` |
| 10 | `uv run python plot.py entropy_accuracy` |
| endpoint accuracies of Appendices C.2 and C.3 | `uv run python plot.py table` |

## Rerunning

```bash
uv sync --group experiments                                        # Linux x86_64 with NVIDIA GPUs
export SCRATCH_DIR=/path/to/scratch HF_HOME=/path/to/scratch/hf    # outputs, and the Hugging Face cache
export SBATCH_ACCOUNT=... SBATCH_PARTITION=...                     # as your cluster needs
uv run python data.py                                              # downloads the models and datasets
bash launch/corpora.sh                                             # teacher corpora, ~19 H100-hours
bash launch/train.sh                                               # the twelve runs, ~100 H100-hours
bash launch/score.sh                                               # endpoint accuracies, ~4 H100-hours
uv run python collect.py                                           # gathers the evaluations into data/curves.json
uv run python plot.py all
```

Every job runs on one H100; `LOCAL=1` runs it in the foreground instead of submitting it.
`bash launch/smoke.sh` runs one training run up to its step-25 evaluation (~0.5 H100-hours), to check a setup against the paper.

## Files

| File | Content |
|---|---|
| `data.py` | prompts, pinned models and datasets |
| `gen_teacher_corpus.py` | samples a teacher corpus |
| `train.py` | one training run, with TRL's general online logit distillation trainer |
| `entropy.py` | student and teacher entropies |
| `score_endpoint.py`, `grading.py` | endpoint accuracy |
| `collect.py`, `plot.py` | gathering results, and the figures |
| `launch/` | Slurm scripts |

## Notes

- The teacher corpora were sampled without a seed, so regenerated corpora are new samples from the same distribution.
- The evaluations use a fixed seed: a rerun reproduces the paper's step-0 values exactly, and later values within sampling noise.
- The AMC23 accuracies of the teachers come from an earlier evaluation in the chat template: `score_endpoint.py --chat --k 8 --max_new_tokens 4096 --dataset amc23 --num_prompts 40`.
