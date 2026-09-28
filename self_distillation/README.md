# Self-distillation

A model is trained on its own samples to match itself given a successful solution as extra context, with TRL's self-distillation trainer and the defaults of Hübotter et al. (2026).
We vary the divergence, how the teacher follows the student, and how many tokens the divergence covers, on SciKnowEval chemistry (Olmo-3-7B-Instruct) and GSM8K (Qwen3-0.6B).
This is Section 4.3 of the paper (hyperparameters in Appendix C.6).

## Figures

The trajectories of every run are in `data/`, so the figures only need numpy and matplotlib, which [uv](https://docs.astral.sh/uv/) installs on first use.

```bash
uv run python plot.py all    # writes figures/self_distillation_<name>.pdf
```

| Paper figure | Command |
|---|---|
| 6 | `uv run python plot.py chemistry` |
| 11 | `uv run python plot.py chemistry_accuracy` |
| 12 | `uv run python plot.py gsm8k` |
| 13 | `uv run python plot.py gsm8k_learning_rate` |
| 14 | `uv run python plot.py gsm8k_long` |

## Rerunning

```bash
uv sync --group experiments                        # Linux x86_64 with NVIDIA GPUs
export SCRATCH_DIR=/path/to/scratch                # models, data and runs
export SBATCH_ACCOUNT=... SBATCH_PARTITION=...     # as your cluster needs
uv run python fetch_data.py                        # downloads the models and data
bash launch/chemistry.sh                           # Figures 6 and 11: 14 runs on 8 H100s each, ~215 H100-hours
bash launch/gsm8k.sh                               # Figure 12: 30 runs on 1 H100 each, ~12 H100-hours
bash launch/gsm8k_learning_rate.sh                 # Figure 13: 72 more runs, ~30 H100-hours
bash launch/gsm8k_long.sh                          # Figure 14: 5 runs, ~12 H100-hours
uv run python collect.py gsm8k                     # gathers the runs into data/gsm8k.json (or chemistry)
uv run python plot.py all
```

Add `--dry-run` to a launch script to print its runs without submitting them.
Each run writes its metrics to `$SCRATCH_DIR/runs/<run>/metrics.jsonl`.

## Files

| File | Content |
|---|---|
| `train.py` | one run: TRL's `SDPOTrainer` with entropy logging, a fix for a multi-GPU hang, and evaluation at temperature 0.6 |
| `config.py` | options and their defaults, and pinned model revisions |
| `tasks.py` | prompts and rewards |
| `entropy.py` | entropy of the student and the teacher |
| `fetch_data.py`, `collect.py`, `plot.py` | downloads, gathering results, and the figures |
| `launch/` | one Slurm script per figure |

## Notes

- The Chemistry dataset has 1890 training and 210 test questions. The Chemistry runs stop at 150 steps and 2048 completion tokens (8192 in the reference code).
- The Chemistry curves of Figure 6 are smoothed with a 5-step rolling mean.
- The 500-step runs of Figure 14 have no evaluation.
- Reruns are bit-for-bit identical only with the same compiled vLLM kernels. A fresh compilation changes results slightly: 22.85% instead of 22.38% at step 0 of the Chemistry default, and entropies within 0.05 nats over the first 15 steps.
- The top-1 Chemistry run did not reproduce: a second run reached 4.31 nats and about 65% accuracy, against 1.63 nats and 0% in the figure.
