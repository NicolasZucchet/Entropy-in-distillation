"""Prompt sets, teacher corpora and stop tokens.

Every math prompt is the problem text followed by the boxed-answer instruction,
with no chat template, so the teacher corpus, the student's rollouts and the
endpoint scorer all see the same prompts.

    python data.py    # download the pinned models and datasets (needs internet)
"""
import hashlib
import json
import os

SCRATCH = os.environ.get("SCRATCH_DIR", os.path.join(os.getcwd(), "outputs"))

REVISIONS = {
    "Qwen/Qwen3-0.6B-Base": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "agentica-org/DeepScaleR-Preview-Dataset": "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "math-ai/amc23": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
}

BOXED_INSTRUCTION = "\n\nLet's think step by step and output the final answer within \\boxed{}."


def revision(name: str) -> str | None:
    """The pinned revision of a hub id, or None for a local checkpoint."""
    return REVISIONS.get(name)


def _load(name, *args, **kwargs):
    from datasets import load_dataset

    return load_dataset(name, *args, revision=revision(name), **kwargs)


def _unbox(text: str) -> str:
    """The content of the last `\\boxed{...}`, brace-matched."""
    i = text.rfind(r"\boxed{")
    if i < 0:
        return text.strip()
    depth, out = 0, []
    for c in text[i + len(r"\boxed{") - 1 :]:
        if c == "{":
            depth += 1
            if depth == 1:
                continue
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
    return "".join(out).strip()


def load_prompts(dataset: str, n: int, split: str = "train") -> list[dict]:
    """The first `n` problems of a dataset as `{"prompt", "answer"}` rows."""
    if dataset == "deepscaler":
        ds = _load("agentica-org/DeepScaleR-Preview-Dataset", split="train")
        ds = ds.select(range(min(n, len(ds))))
        return [{"prompt": r["problem"] + BOXED_INSTRUCTION, "answer": r["answer"]} for r in ds]
    if dataset == "math500":  # a single `test` split
        ds = _load("HuggingFaceH4/MATH-500", split="test")
        ds = ds.select(range(min(n, len(ds))))
        return [{"prompt": r["problem"] + BOXED_INSTRUCTION, "answer": _unbox(r["answer"])}
                for r in ds]
    if dataset == "gsm8k":
        ds = _load("openai/gsm8k", "main", split=split).select(range(n))
        return [{"prompt": r["question"], "answer": r["answer"].partition("####")[2].strip()}
                for r in ds]
    if dataset == "amc23":
        ds = _load("math-ai/amc23", split="test")
        ds = ds.select(range(min(n, len(ds))))
        return [{"prompt": r["question"], "answer": str(r["answer"])} for r in ds]
    raise ValueError(f"unknown dataset {dataset!r}")


def corpus_path(root: str, teacher: str, dataset: str, n: int, max_new_tokens: int) -> str:
    """Content-addressed path, so that every arm of a teacher reads the same corpus."""
    key = f"{teacher}|{dataset}|{n}|{max_new_tokens}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    return os.path.join(root, f"{teacher.split('/')[-1]}__{dataset}__n{n}__{digest}.jsonl")


def load_corpus(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def turn_stop_ids(tokenizer) -> set[int]:
    """Special tokens that end an assistant turn under the chat template.

    The Qwen base checkpoints declare `<|endoftext|>` as their only EOS, while
    their template ends a turn with `<|im_end|>`, so both have to stop generation.
    """
    probe = "ZQXTURNPROBEXQZ"
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": probe}]
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
    except Exception:
        return set()
    _, sep, tail = text.rpartition(probe)
    if not sep:
        return set()
    special = set(tokenizer.all_special_ids)
    return {i for i in tokenizer(tail, add_special_tokens=False)["input_ids"] if i in special}


def stop_ids(tokenizer, generation_config=None) -> set[int]:
    """The model's declared EOS ids plus the template's end of turn."""
    raw = getattr(generation_config, "eos_token_id", None)
    if raw is None:
        raw = tokenizer.eos_token_id
    declared = set(raw if isinstance(raw, (list, tuple)) else [raw]) - {None}
    return declared | turn_stop_ids(tokenizer)


if __name__ == "__main__":
    from huggingface_hub import snapshot_download

    for name, rev in REVISIONS.items():
        if name.startswith("Qwen/"):
            print("model", name, snapshot_download(name, revision=rev))
    for dataset in ("deepscaler", "math500", "gsm8k", "amc23"):
        load_prompts(dataset, 1, split="test")
        print("dataset", dataset)
