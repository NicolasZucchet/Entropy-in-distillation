"""Measure a checkpoint's mean entropy and mean cross-entropy on a corpus.

At a stationary point of the cross-entropy objective, the mean predictive entropy
of a model equals its mean cross-entropy on its training data (Section 3.3 of the
paper). This script measures both, pooled and per data source, for one cell of
`cells.py`, and writes `<out_dir>/<cell>.json`:

    python measure.py --cell 1B-sft-on-tulu3_sft_0225
    python measure.py --figure sft --index 3      # the 4th cell of Figure 4

Run `prefetch.py` first; this script then needs no network (`HF_HUB_OFFLINE=1`).
"""
import argparse
import gzip
import io
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cells import CELLS, FIGURES
from corpora import SFT_MIXTURES, shard_paths


# --- Per-token statistics -----------------------------------------------------


def logit_stats(logits, targets):
    """Per-token entropy, cross-entropy and mean-logit mismatch g, with H - CE = -g."""
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(-1)
    tgt = targets.unsqueeze(-1)
    cross_entropy = -logp.gather(-1, tgt).squeeze(-1)
    g = (p * logits).sum(-1) - logits.gather(-1, tgt).squeeze(-1)
    return entropy, cross_entropy, g


@torch.no_grad()
def sequence_stats(model, input_ids, attention_mask, loss_mask, chunk=256):
    """`(entropy, cross_entropy, g)` of every supervised token of a batch, flattened."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    targets, mask = input_ids[:, 1:], loss_mask[:, 1:]
    targets, mask = targets.to(logits.device), mask.to(logits.device)
    # Chunked along the sequence so the float32 softmax over the vocabulary is
    # never materialised for the whole batch at once.
    parts = [logit_stats(logits[:, i : i + chunk], targets[:, i : i + chunk])
             for i in range(0, logits.shape[1], chunk)]
    keep = mask.reshape(-1).bool()
    return [torch.cat([p[k] for p in parts], dim=1).reshape(-1)[keep] for k in range(3)]


# --- Pretraining corpora: packed blocks ---------------------------------------


def _open_shard(path):
    if path.endswith(".zst") or path.endswith(".zstd"):
        import zstandard

        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")))
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def token_stream(tokenizer, mix, seq_len, num_tokens):
    """Yield `(block, source)` packed blocks, with the same token budget for every source."""
    shards = shard_paths(mix)
    per_source = num_tokens // len({source for source, _, _ in shards})
    if mix == "pile":
        # Already tokenized and packed into sequences of `seq_len` tokens.
        for source, path, _ in shards:
            toks = np.memmap(path, dtype=np.uint16, mode="r")
            n = min(per_source, (len(toks) // seq_len) * seq_len)
            for i in range(0, n, seq_len):
                yield toks[i : i + seq_len].astype(np.int64).tolist(), source
        return
    for source, path, share in shards:
        budget = max(seq_len, int(per_source * share))
        buf, emitted = [], 0
        with _open_shard(path) as fh:
            for line in fh:
                text = json.loads(line).get("text")
                if not text:
                    continue
                # Documents are packed with an EOS between them, as in pretraining.
                buf.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
                buf.append(tokenizer.eos_token_id)
                while len(buf) >= seq_len:
                    yield buf[:seq_len], source
                    buf = buf[seq_len:]
                    emitted += seq_len
                    if emitted >= budget:
                        break
                if emitted >= budget:
                    break


# --- SFT mixtures: chat template, assistant tokens only -----------------------


def assistant_spans(tokenizer, messages):
    """The rendered conversation and the character spans of its assistant turns.

    As in open-instruct, a turn's loss covers its content and end-of-turn token:
    from `apply_chat_template(messages[:i], add_generation_prompt=True)` to
    `apply_chat_template(messages[:i + 1])`. Rows whose template is not
    append-only are dropped.
    """
    try:
        full = tokenizer.apply_chat_template(messages, tokenize=False)
    except Exception:
        return None
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        head = tokenizer.apply_chat_template(messages[:i], tokenize=False,
                                             add_generation_prompt=True)
        upto = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False)
        if not (full.startswith(head) and full.startswith(upto) and len(head) < len(upto)):
            return None
        spans.append((len(head), len(upto)))
    return (full, spans) if spans else None


def chat_example(tokenizer, messages, max_len):
    """Token ids and assistant-only loss mask, truncated at `max_len` as in open-instruct."""
    rendered = assistant_spans(tokenizer, messages)
    if rendered is None:
        return None
    full, spans = rendered
    # The template already emits the BOS token.
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True,
                    truncation=True, max_length=max_len)
    mask = [1 if end > start and any(a <= start < b for a, b in spans) else 0
            for start, end in enc["offset_mapping"]]
    return (enc["input_ids"], mask) if any(mask) else None


def sft_stream(tokenizer, mixture, num_tokens, max_len, seed):
    """Yield `(ids, loss_mask, source)` from the shuffled mixture until `num_tokens` supervised tokens."""
    from datasets import load_dataset

    if tokenizer.chat_template is None:
        raise SystemExit(f"{tokenizer.name_or_path} has no chat template")
    repo, commit = SFT_MIXTURES[mixture]
    # Shuffled, since the rows are grouped by source.
    ds = load_dataset(repo, split="train", revision=commit).shuffle(seed=seed)
    emitted = 0
    for row in ds:
        example = chat_example(tokenizer, row["messages"], max_len)
        if example is None:
            continue
        ids, mask = example
        yield ids, mask, row["source"]
        emitted += sum(mask)
        if emitted >= num_tokens:
            return


# --- Measurement ----------------------------------------------------------------


def corpus_stream(tokenizer, cell, args):
    """Yield `(ids, loss_mask, source)` for the corpus of `cell`."""
    if cell.mix in SFT_MIXTURES:
        yield from sft_stream(tokenizer, cell.mix, args.num_tokens, args.max_len, args.seed)
        return
    for block, source in token_stream(tokenizer, cell.mix, cell.seq_len, args.num_tokens):
        yield block, [1] * len(block), source


def batches(stream, batch_size, pad_id):
    """Right-padded batches `(ids, attention_mask, loss_mask, sources)`."""
    buf = []
    for item in stream:
        buf.append(item)
        if len(buf) == batch_size:
            yield _pad(buf, pad_id)
            buf = []
    if buf:
        yield _pad(buf, pad_id)


def _pad(items, pad_id):
    width = max(len(ids) for ids, _, _ in items)
    return (
        [ids + [pad_id] * (width - len(ids)) for ids, _, _ in items],
        [[1] * len(ids) + [0] * (width - len(ids)) for ids, _, _ in items],
        [mask + [0] * (width - len(mask)) for _, mask, _ in items],
        [source for _, _, source in items],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--cell", choices=sorted(CELLS))
    group.add_argument("--figure", choices=sorted(FIGURES), help="with --index, for job arrays")
    ap.add_argument("--index", type=int, help="position of the cell in its figure's list")
    ap.add_argument("--num_tokens", type=int, default=2_000_000,
                    help="token budget; supervised tokens on the SFT mixtures")
    ap.add_argument("--max_len", type=int, default=4096,
                    help="SFT truncation length, open-instruct's max_seq_length")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed of the SFT mixture")
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    cell = CELLS[args.cell or FIGURES[args.figure][args.index]]
    print("cell", cell.name, flush=True)

    tok_repo, tok_commit = cell.tokenizer or (cell.model, cell.commit)
    tok = AutoTokenizer.from_pretrained(tok_repo, revision=tok_commit)
    model = AutoModelForCausalLM.from_pretrained(
        cell.model, revision=cell.commit, dtype=getattr(torch, cell.dtype),
        device_map="auto" if cell.gpus > 1 else "cuda").eval()
    device = next(model.parameters()).device
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Token-weighted sums, pooled and per source.
    totals = {"entropy": 0.0, "cross_entropy": 0.0, "g": 0.0}
    counted, seqs, per_source = 0, 0, {}
    for ids, attn, loss, sources in batches(corpus_stream(tok, cell, args), cell.batch_size, pad_id):
        entropy, cross_entropy, g = sequence_stats(
            model, torch.tensor(ids, device=device), torch.tensor(attn, device=device),
            torch.tensor(loss, device=device))
        for key, value in zip(totals, (entropy, cross_entropy, g)):
            totals[key] += value.sum().item()
        counted += entropy.numel()
        seqs += len(ids)
        # Each row owns the next `kept` entries of the flattened tensors; the mask
        # is shifted by one position, like the targets.
        offset = 0
        for source, mask in zip(sources, loss):
            kept = sum(mask[1:])
            acc = per_source.setdefault(source, [0.0, 0.0, 0])
            acc[0] += entropy[offset : offset + kept].sum().item()
            acc[1] += cross_entropy[offset : offset + kept].sum().item()
            acc[2] += kept
            offset += kept

    if not counted:
        raise SystemExit(f"no tokens measured on {cell.mix}")
    entropy, cross_entropy = totals["entropy"] / counted, totals["cross_entropy"] / counted
    summary = {
        "entropy": entropy,
        "cross_entropy": cross_entropy,
        "g": totals["g"] / counted,
        "identity_gap": entropy - cross_entropy,
        "self_perplexity": float(np.exp(entropy)),
        "perplexity": float(np.exp(cross_entropy)),
        "tokens": counted,
        "sequences": seqs,
        "stage": cell.stage,
        "mix": cell.mix,
        "model": cell.model,
        "revision": cell.revision,
        "commit": cell.commit,
        "per_source": {
            s: {"entropy": h / n, "cross_entropy": c / n, "identity_gap": (h - c) / n, "tokens": n}
            for s, (h, c, n) in per_source.items()
        },
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{cell.name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_source"}, indent=2))


if __name__ == "__main__":
    main()
