"""Gather the `eval.jsonl` of every arm into `data/curves.json`, which `plot.py` reads.

    python collect.py --runs $SCRATCH_DIR/runs
"""
import argparse
import glob
import json
import os


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=os.path.join(os.environ.get("SCRATCH_DIR", "outputs"), "runs"))
    p.add_argument("--out", default="data/curves.json")
    a = p.parse_args()
    runs = {}
    for path in sorted(glob.glob(os.path.join(a.runs, "o*-*-*", "eval.jsonl"))):
        name = os.path.basename(os.path.dirname(path))
        # Keyed on the step, so that a step evaluated twice (a resumed arm) counts once.
        rows = {row["_step"]: row for row in map(json.loads, open(path))}
        steps = sorted(rows)
        keys = sorted({k for row in rows.values() for k in row} - {"_step"})
        history = {"_step": steps} | {k: [rows[s].get(k) for s in steps] for k in keys}
        runs[name] = {"history": history}
        print(f"{name:28s} {len(steps):3d} evaluations")
    with open(a.out, "w") as f:
        json.dump({"runs": runs}, f)
    print(f"wrote {len(runs)} runs to {a.out}")


if __name__ == "__main__":
    main()
