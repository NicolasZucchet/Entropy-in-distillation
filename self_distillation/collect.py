"""Gather the `metrics.jsonl` of finished runs into the data file the figures read.

    python collect.py chemistry    # $SCRATCH_DIR/runs/chem-*/  -> data/chemistry.json
    python collect.py gsm8k        # $SCRATCH_DIR/runs/gsm8k-*/ -> data/gsm8k.json
"""
import glob
import json
import os
import sys

from config import SCRATCH_DIR

# Short name in the data file -> key logged by the trainer. The first three
# come once per validation pass, the others once per optimizer step.
SERIES = {
    "acc": "eval_self_distillation/reward_mean",
    "len": "eval_completions/mean_length",
    "reprompt": "eval_self_distillation/reprompt_sample_fraction",
    "entropy": "policy/entropy",
    "teacher_entropy": "policy/teacher_entropy",
    "entropy_distilled": "policy/student_entropy_distilled",
    "frac_deflated": "policy/frac_deflated",
}
PREFIX = {"chemistry": "chem", "gsm8k": "gsm8k"}


def read_run(path):
    """Every series of one run, in logging order."""
    out = {k: [] for k in SERIES}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            for short, key in SERIES.items():
                if key in record:
                    out[short].append(float(record[key]))
    return out


def main():
    setup = sys.argv[1]
    runs = {}
    for path in sorted(glob.glob(os.path.join(SCRATCH_DIR, "runs", f"{PREFIX[setup]}-*", "metrics.jsonl"))):
        runs[os.path.basename(os.path.dirname(path))] = read_run(path)
    with open(os.path.join("data", f"{setup}.json"), "w") as f:
        json.dump(runs, f, indent=1)
    print(f"wrote data/{setup}.json with {len(runs)} runs")


if __name__ == "__main__":
    main()
