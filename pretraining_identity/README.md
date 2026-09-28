# Entropy and cross-entropy of pretrained and finetuned models

Released OLMo 2 and Pythia checkpoints are evaluated, with forward passes only, on the data they were trained on, recording their mean entropy and mean cross-entropy.
This is Section 4.1 of the paper (details in Appendix C.1).
Each measurement of one checkpoint on one dataset is called a cell.

## Figures

The result of every cell is in `data/`, so the figures only need numpy and matplotlib, which [uv](https://docs.astral.sh/uv/) installs on first use.

```bash
uv run python plot.py all    # writes figures/pretraining_identity_<name>.pdf
```

| Paper figure | Cells | Command |
|---|---|---|
| 4 | OLMo 2 1B, 7B, 13B, base and SFT, on the stage-2 pretraining mix and on their SFT mixture (12 cells) | `uv run python plot.py sft` |
| 9 | OLMo 2 1B to 32B at the end of stage 1 and at the end of pretraining, and Pythia 70M to 12B (16 cells) | `uv run python plot.py families` |

`uv run python plot.py table` prints the numbers behind both figures.

## Rerunning

```bash
uv sync --group experiments                                        # Linux x86_64 with NVIDIA GPUs
export SCRATCH_DIR=/path/to/scratch HF_HOME=/path/to/scratch/hf    # results, and the Hugging Face cache
export SBATCH_ACCOUNT=... SBATCH_PARTITION=...                     # as your cluster needs
uv run python prefetch.py                                          # downloads everything, ~650GB
bash launch/sft.sh                                                 # Figure 4, ~0.5 H100-hours
bash launch/families.sh                                            # Figure 9, ~2 H100-hours
IDENTITY_DATA=$SCRATCH_DIR/identity uv run python plot.py all      # plot your results
```

One cell alone: `uv run python measure.py --cell 1B-sft-on-tulu3_sft_0225` (writes to `outputs/`).
Rerunning a cell reproduces its file in `data/` exactly.

## Files

| File | Content |
|---|---|
| `cells.py` | the cells of each figure, with checkpoints pinned to commits |
| `corpora.py` | the corpus files, pinned to commits |
| `prefetch.py` | downloads checkpoints and corpora |
| `measure.py` | measures one cell |
| `plot.py` | the figures and the table |
| `launch/` | Slurm scripts |

## Notes

- Every cell uses 2M tokens (only the assistant tokens on the SFT mixtures), in blocks of 2048 tokens (2049 for Pythia).
- Every source of an OLMo 2 pretraining mix gets the same number of tokens, and `plot.py` averages the sources at the proportions of the mix published with OLMo 2.
- The latest Pythia chunk starts at shard 19 of 21, about 90% of the way through training.
