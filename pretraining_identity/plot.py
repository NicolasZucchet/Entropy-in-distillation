"""Figures 4 and 9 of the paper, and the numbers behind them, from the JSONs in `data/`.

    python plot.py sft        # Figure 4 -> figures/pretraining_identity_sft.pdf
    python plot.py families   # Figure 9 -> figures/pretraining_identity_families.pdf
    python plot.py table         # every cell, raw and pooled at the mixture proportions
    python plot.py all

Set `IDENTITY_DATA` to plot your own measurements instead of `data/`.
"""
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from cells import FIGURES

DATA = os.environ.get("IDENTITY_DATA", "data")
FIGDIR = "figures"

# The identity holds on average over the training distribution, whereas every
# source of a pretraining mix is measured on the same number of tokens, so the
# per-source means are re-pooled at the published proportions of the mix.
# Stage 2 (dolmino-mix-1124, Mix %): the 7B and 1B were trained on the 50B mix, the
# 13B and 32B on the 100B one.
DOLMINO_50B = {"dclm": 47.2, "flan": 16.6, "math": 20.8, "wiki": 7.11,
               "pes2o": 5.85, "stackexchange": 2.45}
DOLMINO_100B = {"dclm": 50.2, "flan": 16.7, "math": 17.5, "wiki": 3.57,
                "pes2o": 9.52, "stackexchange": 2.47}
# Stage-2 math splits between its seven sources by their token counts in millions.
DOLMINO_MATH = {"tinyGSM-MIND": 6480.0, "mathcoder2-synthmath": 3870.0, "tulu_math": 230.0,
                "metamath-owmfilter": 84.2, "dolmino_math_synth": 28.7, "gsm8k": 2.74,
                "codesearchnet-owmfilter": 1.78}
# Stage 1 (olmo-mix-1124) is used whole: its source token counts in billions.
# Algebraic Stack is not measured and drops out of the average.
OLMO_MIX = {"dclm": 3700.0, "starcoder": 83.0, "pes2o": 58.6, "arxiv": 20.8,
            "algebraic-stack": 11.8, "open-web-math": 12.2, "wiki": 3.66}


def size_tag(model):
    """The parameter-count field of a model name: `7B` for `allenai/OLMo-2-1124-7B-SFT`."""
    parts = model.split("/")[-1].split("-")
    while len(parts) > 1 and parts[-1] == "SFT":
        parts.pop()
    return parts[-1]


def params(model):
    """Parameter count from a model name."""
    tag = size_tag(model)
    return float(tag[:-1]) * (1e9 if tag[-1] in "bB" else 1e6)


def mix_weights(run):
    """Source weights of the corpus a run was measured on, or None if it needs no re-pooling."""
    if run["mix"] == "olmo_mix":
        return OLMO_MIX
    if run["mix"] != "dolmino":
        return None  # the Pile chunks and the SFT mixtures are read at their own proportions
    tokens = re.search(r"tokens(\d+)B", run["revision"])
    if tokens:
        weights = DOLMINO_100B if int(tokens.group(1)) >= 75 else DOLMINO_50B
    else:  # `main` does not name its mix; the 13B and 32B stage-2 runs used the 100B one
        weights = DOLMINO_100B if size_tag(run["model"]) in ("13B", "32B") else DOLMINO_50B
    if "math" in run["per_source"]:  # measured with math as a single source
        return weights
    weights = dict(weights)
    math = weights.pop("math")
    total = sum(DOLMINO_MATH.values())
    weights.update({f"math/{k}": math * v / total for k, v in DOLMINO_MATH.items()})
    return weights


def pooled(run):
    """`(entropy, cross_entropy)` of a run, re-pooled at the mixture proportions where needed."""
    weights = mix_weights(run)
    per = run["per_source"]
    if not weights or not per:
        return run["entropy"], run["cross_entropy"]
    keys = [k for k in weights if k in per]
    total = sum(weights[k] for k in keys)
    return (sum(weights[k] * per[k]["entropy"] for k in keys) / total,
            sum(weights[k] * per[k]["cross_entropy"] for k in keys) / total)


def load(figure):
    """The runs of a figure, each with its pooled `H` and `CE`."""
    runs = []
    for name in FIGURES[figure]:
        with open(os.path.join(DATA, f"{name}.json")) as f:
            run = json.load(f)
        run["name"], run["size"] = name, size_tag(run["model"])
        run["H"], run["CE"] = pooled(run)
        runs.append(run)
    return runs


def diagonal(ax, runs):
    """Equal axes around the runs, with the line entropy = cross-entropy."""
    values = [v for r in runs for v in (r["H"], r["CE"])]
    pad = 0.1 * (max(values) - min(values)) + 0.03
    lim = (min(values) - pad, max(values) + pad)
    ax.plot(lim, lim, color="grey", lw=0.8, zorder=0)
    ax.set(xlim=lim, ylim=lim, aspect="equal", xlabel="cross-entropy")


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


def sft():
    """Figure 4: base (hollow) and SFT (filled) OLMo 2 models, on pretraining and SFT data."""
    runs = load("sft")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    panels = [("pretraining data (stage-2 mix)", "dolmino", "C0"),
              ("SFT data", "tulu3_sft", "C1")]
    for ax, (title, mix, colour) in zip(axes, panels):
        cells = [r for r in runs if r["mix"].startswith(mix)]
        for size in ("1B", "7B", "13B"):
            base = next(r for r in cells if r["size"] == size and r["stage"] == "base")
            tuned = next(r for r in cells if r["size"] == size and r["stage"] == "sft")
            ax.scatter(base["CE"], base["H"], facecolors="none", edgecolors=colour)
            ax.scatter(tuned["CE"], tuned["H"], color=colour)
            ax.annotate("", xy=(tuned["CE"], tuned["H"]), xytext=(base["CE"], base["H"]),
                        arrowprops=dict(arrowstyle="->", color=colour))
            ax.annotate(size, (base["CE"], base["H"]), textcoords="offset points",
                        xytext=(6, 3), fontsize=8)
        diagonal(ax, cells)
        ax.set_title(title)
    axes[0].set_ylabel("entropy")
    axes[0].legend(handles=[
        plt.Line2D([], [], ls="", marker="o", mfc="none", mec="k", label="base"),
        plt.Line2D([], [], ls="", marker="o", color="k", label="after SFT")], loc="upper left")
    save(fig, "pretraining_identity_sft.pdf")


def families():
    """Figure 9: OLMo 2 at the end of each pretraining stage, and Pythia, on their training data."""
    runs = load("families")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    panels = [("OLMo 2", [r for r in runs if r["mix"] != "pile"]),
              ("Pythia", [r for r in runs if r["mix"] == "pile"])]
    for ax, (title, cells) in zip(axes, panels):
        for r in cells:
            colour = "C9" if r["stage"] == "stage1" else "C0"
            ax.scatter(r["CE"], r["H"], color=colour, s=10 * np.log10(params(r["model"]) / 1e7))
            ax.annotate(r["size"], (r["CE"], r["H"]), textcoords="offset points",
                        xytext=(4, -8), fontsize=7)
        diagonal(ax, cells)
        ax.set_title(title)
    axes[0].set_ylabel("entropy")
    axes[0].legend(handles=[
        plt.Line2D([], [], ls="", marker="o", color="C9", label="end of stage 1"),
        plt.Line2D([], [], ls="", marker="o", color="C0", label="final model (end of stage 2)")], loc="upper left")
    save(fig, "pretraining_identity_families.pdf")


def table():
    """Print every cell: raw and pooled entropy, cross-entropy and their gap."""
    row = "%-28s %8s %8s %8s %8s %9s"
    print(row % ("cell", "H raw", "CE raw", "H", "CE", "gap (%)"))
    for figure in FIGURES:
        for r in load(figure):
            gap = 100 * (r["H"] - r["CE"]) / r["CE"]
            print("%-28s %8.4f %8.4f %8.4f %8.4f %+9.2f" % (
                r["name"], r["entropy"], r["cross_entropy"], r["H"], r["CE"], gap))


if __name__ == "__main__":
    figures = {"sft": sft, "families": families, "table": table}
    choice = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name in figures if choice == "all" else [choice]:
        figures[name]()
