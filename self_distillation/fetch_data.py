"""Download the data and models at the revisions the paper used.

Run once on a machine with internet access; the jobs run offline.

    SCRATCH_DIR=/path/to/scratch python fetch_data.py [--skip-olmo]
"""
import argparse
import hashlib
import os
import urllib.request

from config import MODEL_REVISIONS, SCRATCH_DIR

# SciKnowEval chemistry as shipped with the code of Hubotter et al. (2026).
SDPO_COMMIT = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"
CHEMISTRY_SHA256 = {
    "train": "dc841dc92a16a6af3944336ecd887e80907bc9244f2bd51cd2e3869959a84029",
    "test": "772adb9f2bdb1bbc2a542f350b55a1b23091fa98e793f5e24c5dba410ce299bf",
}


def fetch_chemistry():
    """The two Chemistry files, into $SCRATCH_DIR/datasets/sciknoweval/chemistry/."""
    out = os.path.join(SCRATCH_DIR, "datasets", "sciknoweval", "chemistry")
    os.makedirs(out, exist_ok=True)
    for split, digest in CHEMISTRY_SHA256.items():
        path = os.path.join(out, f"{split}.json")
        if not os.path.exists(path):
            url = (f"https://raw.githubusercontent.com/lasgroup/SDPO/{SDPO_COMMIT}"
                   f"/datasets/sciknoweval/chemistry/{split}.json")
            urllib.request.urlretrieve(url, path)
        with open(path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != digest:
                raise SystemExit(f"{path} is not the file the paper used")
        print("ok", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-olmo", action="store_true", help="GSM8K runs only")
    args = ap.parse_args()
    os.environ["HF_HOME"] = os.path.join(SCRATCH_DIR, "hf")

    from huggingface_hub import snapshot_download

    from tasks import GSM8K_REVISION, load_dataset

    fetch_chemistry()
    for split in ("train", "test"):
        load_dataset("openai/gsm8k", "main", split=split, revision=GSM8K_REVISION)
    for model, revision in MODEL_REVISIONS.items():
        if not (args.skip_olmo and "Olmo" in model):
            print("ok", snapshot_download(model, revision=revision))


if __name__ == "__main__":
    main()
