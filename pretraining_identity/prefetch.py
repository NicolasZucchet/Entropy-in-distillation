"""Download every checkpoint and corpus file the cells need.

Run once where there is network access, with `HF_HOME` where the compute nodes
will read the cache from:

    HF_HOME=$SCRATCH_DIR/hf python prefetch.py                 # everything, ~650GB
    HF_HOME=$SCRATCH_DIR/hf python prefetch.py --figure sft    # Figure 4 only
"""
import argparse
import hashlib
import os

import requests
from huggingface_hub import hf_hub_download, hf_hub_url, snapshot_download

from cells import CELLS, FIGURES
from corpora import (OLMO_SHARDS, PILE_CHUNK_TOKENS, PILE_CHUNKS, PILE_REPO, PILE_SEQ,
                     PILE_SHARD_BYTES, SFT_MIXTURES, pile_path)


def fetch_pile_chunk(shard, sha256, out_path):
    """Range-download `PILE_CHUNK_TOKENS` tokens of a Pile shard, from its first sequence boundary."""
    # A shard is a flat uint16 array starting at global token `shard * 15e9`,
    # which falls inside a 2049-token sequence.
    skip = (-(shard * (PILE_SHARD_BYTES // 2))) % PILE_SEQ
    start = skip * 2
    end = start + PILE_CHUNK_TOKENS * 2 - 1
    url = hf_hub_url(PILE_REPO, f"document-{shard:05d}-of-00020.bin", repo_type="dataset")
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=600)
    r.raise_for_status()
    if hashlib.sha256(r.content).hexdigest() != sha256:
        raise RuntimeError(f"pile shard {shard}: checksum mismatch")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(r.content)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figure", choices=sorted(FIGURES), help="only what this figure needs")
    args = ap.parse_args()
    names = FIGURES[args.figure] if args.figure else sorted(CELLS)
    cells = [CELLS[name] for name in names]

    repos = {(c.model, c.commit) for c in cells} | {c.tokenizer for c in cells if c.tokenizer}
    for repo, commit in sorted(repos):
        # Some of these repos ship `*.bin` weights without a safetensors copy.
        snapshot_download(repo, revision=commit,
                          allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"])
        print("OK", repo, commit, flush=True)

    mixes = {c.mix for c in cells}
    for mix in sorted(mixes & set(OLMO_SHARDS)):
        repo, commit, files = OLMO_SHARDS[mix]
        for entry in files:
            hf_hub_download(repo, entry[1], repo_type="dataset", revision=commit)
        print("OK", mix, flush=True)
    if "pile" in mixes:
        for label, shard, sha256 in PILE_CHUNKS:
            if not os.path.exists(pile_path(label)):
                fetch_pile_chunk(shard, sha256, pile_path(label))
            print("OK", label, flush=True)

    from datasets import load_dataset

    for mix in sorted(mixes & set(SFT_MIXTURES)):
        repo, commit = SFT_MIXTURES[mix]
        print("OK", mix, len(load_dataset(repo, split="train", revision=commit)), "rows", flush=True)


if __name__ == "__main__":
    main()
