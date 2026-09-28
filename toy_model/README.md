# Toy model

A linear softmax student is distilled from synthetic teachers, on fixed random contexts, with each divergence of the paper.
This is the toy model of Section 3 (setup in Appendix B.7).

## Figures

The results of every sweep are in `data/`, so the figures only need numpy and matplotlib, which [uv](https://docs.astral.sh/uv/) installs on first use.

```bash
uv run python plot.py all    # writes figures/toy_model_<name>.pdf
```

| Paper figure | Command |
|---|---|
| 2 (panels b–e) | `uv run python plot.py entropy_gap` |
| 3 | `uv run python plot.py topk_forward_kl` |
| 7 | `uv run python plot.py jsd_heatmaps` |
| 8 | `uv run python plot.py topk_jsd` |

## Rerunning

```bash
uv sync --group experiments                  # CPU; add --group cuda on an NVIDIA GPU (Linux)
bash launch/entropy_gap.sh                   # Figure 2, ~4 CPU-hours
bash launch/topk.sh                          # Figures 3 and 8, ~1.5 CPU-hours
bash launch/jsd_heatmaps.sh                  # Figure 7, ~3 CPU-hours
uv run python plot.py all --data outputs     # plot your results instead of data/
```

Each sweep is a list of settings, each trained with five seeds.
On Slurm, `sbatch --array=0-7 launch/<script>.sh` splits a sweep over eight tasks; run `uv run python sweep.py merge <sweep>` once they have finished.
`SMOKE=1` shrinks every sweep to test the whole pipeline in about a minute, and `uv run python sweep.py check <sweep> <i>` retrains setting `i` of a sweep and compares it with `data/`.

## Files

| File | Content |
|---|---|
| `toy.py` | teachers, student, divergences and training |
| `sweep.py` | the sweeps: run, merge, check |
| `plot.py` | the figures |
| `launch/` | one script per figure |
| `data/` | the results behind the paper figures |

## Notes

- `data/` was trained with jax 0.8.1 on NVIDIA GPUs. Other hardware does not reproduce it bit for bit, but entropy gaps agree to about 0.01 nats, well within the spread over seeds.
- The trajectories of Figure 2e are one seed; every other panel averages five.
- The top-k runs (Figures 3 and 8) train for 40,000 steps, but the figures stop at step 10,000.
- Adam uses the optax defaults (beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8).
