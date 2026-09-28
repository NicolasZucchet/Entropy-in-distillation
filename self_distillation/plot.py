"""Figures 6 and 11-14 of the paper, from data/chemistry.json and data/gsm8k.json.

    python plot.py chemistry             # Figure 6
    python plot.py chemistry_accuracy    # Figure 11
    python plot.py gsm8k                 # Figure 12
    python plot.py gsm8k_learning_rate   # Figure 13
    python plot.py gsm8k_long            # Figure 14
    python plot.py all

Each data file maps a run name (see launch/) to its series: `entropy` once per
optimizer step, `acc` (avg@16) once per validation pass.
"""
import json
import os
import statistics
import sys

import matplotlib.pyplot as plt
import numpy as np

ALPHAS = ["0.00", "0.25", "0.50", "0.75", "1.00"]
LRS = ["1e-6", "3e-6", "1e-5", "3e-5"]
SEEDS = [0, 1, 2]
ALPHA_COLOR = {a: plt.cm.viridis(0.9 * float(a)) for a in ALPHAS}


def load(setup):
    with open(os.path.join("data", f"{setup}.json")) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs("figures", exist_ok=True)
    fig.savefig(os.path.join("figures", name), bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/" + name)


def rolling_mean(xs, w=5):
    """Centred rolling mean, with the window shrinking at the ends."""
    return [np.mean(xs[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(xs))]


def seed_band(series):
    """Mean, min and max across seeds, truncated to the shortest series."""
    n = min(len(s) for s in series)
    a = np.array([s[:n] for s in series])
    return a.mean(0), a.min(0), a.max(0)


# Chemistry: one seed per run, entropy smoothed over 5 steps.

CHEM_PANELS = [
    ("generalized JS weight (0 forward, 1 reverse)",
     [(f"chem-alpha{a}" if a != "0.50" else "chem-default", a, ALPHA_COLOR[a], "-", "o")
      for a in ALPHAS]),
    ("teacher",
     [("chem-default", "EMA 0.95", "C0", "-", "o"), ("chem-teacher-ema0.99", "EMA 0.99", "C1", "-", "o"),
      ("chem-teacher-base", "frozen", "C2", "-", "o"), ("chem-teacher-live", "live", "C3", "-", "o")]),
    ("top-k",
     [("chem-topk1", "1", "C0", "-", "o"), ("chem-topk10", "10", "C1", "-", "o"),
      ("chem-default", "100", "C2", "-", "o"), ("chem-topk1000", "1000", "C3", "-", "o"),
      ("chem-topk10-notail", "10, renormalized", "C1", "--", "s"),
      ("chem-topk100-notail", "100, renormalized", "C2", "--", "s"),
      ("chem-topk1000-notail", "1000, renormalized", "C3", "--", "s")]),
]


def chemistry():
    """Figure 6: student entropy over training on Chemistry, one panel per ablation."""
    runs = load("chemistry")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (title, arms) in zip(axes, CHEM_PANELS):
        for name, label, color, ls, _ in arms:
            y = rolling_mean(runs[name]["entropy"])
            ax.plot(range(1, len(y) + 1), y, color=color, ls=ls, label=label)
        ax.set(title=title, xlabel="step", xlim=(1, 150))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("average entropy")
    for ax in axes[1:]:  # the range of the divergence panel; top-1 spikes above it
        ax.set_ylim(axes[0].get_ylim())
    fig.tight_layout()
    save(fig, "self_distillation_chemistry.pdf")


def chem_eval_points(run, every=50, half=10):
    """(entropy, accuracy) at each validation; entropy is the median over steps within `half`."""
    ent, points = run["entropy"], [(run["entropy"][0], 100 * run["acc"][0])]
    for i in range(1, len(run["acc"])):
        step = every * i
        points.append((statistics.median(ent[max(0, step - 1 - half):step + half]), 100 * run["acc"][i]))
    return points


def chemistry_accuracy():
    """Figure 11: accuracy against entropy at each Chemistry validation (larger is later)."""
    runs = load("chemistry")
    base = chem_eval_points(runs["chem-default"])[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (title, arms) in zip(axes, CHEM_PANELS):
        for name, label, color, _, marker in arms:
            points = [base] + chem_eval_points(runs[name])[1:]
            x, y = zip(*points)
            ax.plot(x, y, color=color, lw=0.8, alpha=0.5)
            ax.scatter(x[1:], y[1:], s=[15, 35, 60], color=color, marker=marker, label=label)
        ax.scatter(*base, color="grey", zorder=3, label="base")
        ax.set(title=title, xlabel="average entropy")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("accuracy (%)")
    fig.tight_layout()
    save(fig, "self_distillation_chemistry_accuracy.pdf")


# GSM8K: three seeds per configuration, drawn as their mean and range.

def gsm8k_runs(runs, arm, lr="1e-5"):
    """The three seeds of one configuration."""
    return [runs[f"gsm8k-lr{lr}-{arm}-s{s}"] for s in SEEDS]


GSM_PANELS = [
    ("generalized JS weight (0 forward, 1 reverse)",
     [("default" if a == "0.50" else f"alpha{a}", a, ALPHA_COLOR[a], "-") for a in ALPHAS]),
    ("teacher", [("default", "EMA 0.95", "C0", "-"), ("teacher-base", "frozen", "C2", "-"),
                 ("teacher-live", "live", "C3", "-")]),
    ("top-k", [("topk20", "20", "C0", "-"), ("default", "100", "C2", "-"), ("full", "full", "C3", "-"),
               ("topk100-notail", "100, renormalized", "C2", "--")]),
]


def gsm8k():
    """Figure 12: entropy (top) and accuracy (bottom) over training on GSM8K."""
    runs = load("gsm8k")
    base = 100 * runs["gsm8k-lr1e-5-default-s0"]["acc"][0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey="row")
    for col, (title, arms) in enumerate(GSM_PANELS):
        ax_h, ax_a = axes[0, col], axes[1, col]
        for arm, label, color, ls in arms:
            seeds = gsm8k_runs(runs, arm)
            m, lo, hi = seed_band([r["entropy"] for r in seeds])
            steps = np.arange(1, len(m) + 1)
            ax_h.fill_between(steps, lo, hi, color=color, alpha=0.15, lw=0)
            ax_h.plot(steps, m, color=color, ls=ls, label=label)
            m, lo, hi = seed_band([[100 * v for v in r["acc"]] for r in seeds])
            steps = 5 * np.arange(len(m))
            ax_a.fill_between(steps, lo, hi, color=color, alpha=0.15, lw=0)
            ax_a.plot(steps, m, color=color, ls=ls)
        ax_a.axhline(base, color="grey", ls=":", lw=1)
        ax_h.set_title(title)
        ax_h.legend(fontsize=8)
        ax_a.set(xlabel="step", xlim=(0, 50))
    axes[0, 0].set_ylabel("average entropy")
    axes[1, 0].set_ylabel("avg@16 accuracy (%)")
    fig.tight_layout()
    save(fig, "self_distillation_gsm8k.pdf")


def lr_cell(runs, arm, lr):
    """The seeds of one learning-rate cell; at 1e-5 these are the runs of `gsm8k`."""
    if lr == "1e-5" and arm in ("alpha0.50", "teacher-ema"):
        arm = "default"
    return gsm8k_runs(runs, arm, lr)


def gsm8k_learning_rate():
    """Figure 13: entropy at step 50 and final accuracy against the learning rate."""
    runs = load("gsm8k")
    base = 100 * runs["gsm8k-lr1e-5-default-s0"]["acc"][0]
    columns = [
        ("generalized JS weight (0 forward, 1 reverse)", [(f"alpha{a}", a, ALPHA_COLOR[a]) for a in ALPHAS]),
        ("teacher", [("teacher-ema", "EMA 0.95", "C0"), ("teacher-base", "frozen", "C2"),
                     ("teacher-live", "live", "C3")]),
    ]
    stats = {"entropy at step 50": lambda r: r["entropy"][-1],
             "avg@16 over the last 5 validations (%)": lambda r: 100 * np.mean(r["acc"][-5:])}
    x = [float(lr) for lr in LRS]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey="row")
    for col, (title, arms) in enumerate(columns):
        for row, (ylabel, stat) in enumerate(stats.items()):
            ax = axes[row, col]
            for arm, label, color in arms:
                values = [[stat(r) for r in lr_cell(runs, arm, lr)] for lr in LRS]
                mean = np.array([np.mean(v) for v in values])
                err = [mean - [min(v) for v in values], [max(v) for v in values] - mean]
                ax.errorbar(x, mean, yerr=err, color=color, marker="o", capsize=2, label=label)
            ax.set_xscale("log")
            if col == 0:
                ax.set_ylabel(ylabel)
        axes[0, col].set_title(title)
        axes[0, col].legend(fontsize=8)
        axes[1, col].axhline(base, color="grey", ls=":", lw=1)
        axes[1, col].set_xlabel("learning rate")
        axes[1, col].set_xticks(x, LRS)
    fig.tight_layout()
    save(fig, "self_distillation_gsm8k_learning_rate.pdf")


def gsm8k_long():
    """Figure 14: entropy over 500 GSM8K steps, one seed per weight alpha."""
    runs = load("gsm8k")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for a in ALPHAS:
        y = runs[f"gsm8k-long-alpha{a}-s0"]["entropy"]
        ax.plot(range(1, len(y) + 1), y, color=ALPHA_COLOR[a], label=a)
    ax.set(xlabel="step", ylabel="average entropy", xlim=(1, 500))
    ax.legend(title="generalized JS weight", fontsize=8)
    fig.tight_layout()
    save(fig, "self_distillation_gsm8k_long.pdf")


FIGURES = {"chemistry": chemistry, "chemistry_accuracy": chemistry_accuracy, "gsm8k": gsm8k,
           "gsm8k_learning_rate": gsm8k_learning_rate, "gsm8k_long": gsm8k_long}

if __name__ == "__main__":
    names = FIGURES if sys.argv[1:] == ["all"] else sys.argv[1:]
    for name in names:
        FIGURES[name]()
