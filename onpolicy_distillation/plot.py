"""Figures 5 and 10 of the paper, and the endpoint accuracies quoted in its appendix.

    python plot.py entropy            # Figure 5: student entropy on its own MATH500 completions
    python plot.py entropy_accuracy   # Figure 10: also on frozen teacher completions, and accuracy
    python plot.py table              # endpoint accuracies, from data/scores/
    python plot.py all
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
TEACHERS = ["Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
COLOR = {"forward": "tab:blue", "reverse": "tab:red"}
LINESTYLE = {"on": "-", "off": "--"}

ON_STUDENT = ("eval/math500/student_data/student_entropy", "entropy, student completions", 1)
ON_TEACHER = ("eval/math500/teacher_data/student_entropy", "entropy, teacher completions", 1)
ACCURACY = ("eval/math500/accuracy/accuracy", "MATH500 accuracy (%)", 100)


def curves(rows, name):
    """One row of panels per quantity, one column per teacher, one line per arm."""
    runs = json.load(open(os.path.join(HERE, "data", "curves.json")))["runs"]
    fig, axes = plt.subplots(len(rows), len(TEACHERS), squeeze=False, sharex=True, sharey="row",
                             figsize=(10, 2.5 * len(rows) + 0.5))
    for r, (key, label, scale) in enumerate(rows):
        for c, teacher in enumerate(TEACHERS):
            ax = axes[r][c]
            for src in ("on", "off"):
                for div in ("forward", "reverse"):
                    h = runs[f"{src}-{div}-{teacher}"]["history"]
                    points = [(s, scale * v) for s, v in zip(h["_step"], h[key])
                              if v is not None and v == v]  # skip missing and NaN
                    ax.plot(*zip(*points), marker="o", color=COLOR[div], ls=LINESTYLE[src],
                            label=f"{div} KL, {src}-policy")
            if r == 0:
                ax.set_title(f"teacher {teacher}")
            if r == len(rows) - 1:
                ax.set_xlabel("distillation step")
        axes[r][0].set_ylabel(label)
    axes[0][-1].legend()
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name))
    print("wrote", os.path.join(FIGDIR, name))


def _accuracies(path):
    """`{label: accuracy in %}` of one scores file."""
    return {r["label"]: 100 * r["t0.6"]["accuracy"] for r in map(json.loads, open(path))}


def table():
    """The endpoint accuracies of the appendix."""
    scores = os.path.join(HERE, "data", "scores")
    benches = ["math500", "gsm8k", "amc23"]
    rows = {}
    for path in sorted(glob.glob(os.path.join(scores, "o*-*-Qwen3-*.jsonl"))) + [
            os.path.join(scores, "student.jsonl")]:
        name = os.path.basename(path).removesuffix(".jsonl")
        acc = _accuracies(path)
        rows[name] = [acc[f"{name}-{b}"] for b in benches]
    print(f"{'model':28s}" + "".join(f"{b:>10s}" for b in benches))
    for name, accs in rows.items():
        print(f"{name:28s}" + "".join(f"{a:10.2f}" for a in accs))
    arms = [v[0] for k, v in rows.items() if k.startswith(("on-", "off-"))]
    base = rows["student"][0]
    chat = _accuracies(os.path.join(scores, "student-chat.jsonl"))
    print(f"\nMATH500: the {len(arms)} arms score between {min(arms):.2f}% and {max(arms):.2f}%, "
          f"the base student {base:.2f}% ({next(iter(chat.values())):.2f}% in the chat template)")
    for t in ("4B", "8B"):
        acc = _accuracies(os.path.join(scores, f"teacher-Qwen3-{t}-amc23-chat.jsonl"))
        print(f"AMC23, Qwen3-{t} teacher: {next(iter(acc.values())):.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", nargs="?", default="all", choices=["entropy", "entropy_accuracy", "table", "all"])
    what = p.parse_args().what
    if what in ("entropy", "all"):
        curves([ON_STUDENT], "onpolicy_distillation_entropy.pdf")
    if what in ("entropy_accuracy", "all"):
        curves([ON_TEACHER, ON_STUDENT, ACCURACY], "onpolicy_distillation_entropy_accuracy.pdf")
    if what in ("table", "all"):
        table()
