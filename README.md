# Divergence controls entropy in distillation

Code for the paper *Divergence controls entropy in distillation* (Nicolas Zucchet and Scott Linderman).

Each experiment has its own folder, environment and README.

| Folder | Paper figures | What it does | Compute to rerun |
|---|---|---|---|
| [`toy_model/`](toy_model) | 2, 3, 7, 8 | a linear softmax student distilled from synthetic teachers | ~10 CPU-hours |
| [`pretraining_identity/`](pretraining_identity) | 4, 9 | entropy and cross-entropy of OLMo 2 and Pythia on their training data | ~5 H100-hours |
| [`onpolicy_distillation/`](onpolicy_distillation) | 5, 10 | on- vs. off-policy distillation with forward and reverse KL, Qwen3 | ~125 H100-hours |
| [`self_distillation/`](self_distillation) | 6, 11–14 | self-distillation on chemistry and GSM8K | ~270 H100-hours |

Figure 1 is a diagram.

## Figures

Every folder ships the results behind its figures, so they can be redrawn on a laptop with [uv](https://docs.astral.sh/uv/):

```bash
for d in toy_model pretraining_identity onpolicy_distillation self_distillation; do (cd $d && uv run python plot.py all); done
```

## Rerunning

Each folder's README gives the commands.
`uv sync --group experiments` installs the exact environment an experiment ran in.
The language model experiments ran on a Slurm cluster of H100 GPUs.

## Licence

MIT, see [`LICENSE`](LICENSE).
