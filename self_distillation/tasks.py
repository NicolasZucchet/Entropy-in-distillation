"""The two tasks: prompts as TRL conversational datasets, and binary rewards."""
import json
import os
import re

from datasets import Dataset, load_dataset

GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
GSM8K_SYSTEM = "You are a helpful assistant. Answer concisely."


def gsm8k_rows(split: str, n: int) -> list[dict]:
    """The first `n` rows of a GSM8K split, unshuffled, with the final answer only."""
    ds = load_dataset("openai/gsm8k", "main", split=split, revision=GSM8K_REVISION)
    return [{"prompt": r["question"], "answer": r["answer"].partition("####")[2].strip()}
            for r in ds.select(range(n))]


def chemistry_rows(data_dir: str, split: str) -> list[dict]:
    """SciKnowEval chemistry, in the files shipped with the code of Hubotter et al. (2026).

    Used verbatim: their train/test split is an unseeded random draw, so it
    cannot be rebuilt from the original dataset.
    """
    path = os.path.join(data_dir, "sciknoweval", "chemistry", f"{split}.json")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(cfg, split: str) -> Dataset:
    """One split (`train` or `test`) of `cfg.task` as TRL conversational rows."""
    if cfg.task == "gsm8k":
        n = cfg.num_train_prompts if split == "train" else cfg.num_eval_prompts
        return Dataset.from_list([
            {"prompt": [{"role": "system", "content": GSM8K_SYSTEM},
                        {"role": "user", "content": r["prompt"]}],
             "answer": r["answer"],
             "privileged_context": None}
            for r in gsm8k_rows(split, n)
        ])
    return Dataset.from_list([
        {"prompt": [{"role": "system", "content": r["system"]},
                    {"role": "user", "content": r["prompt"]}],
         "answer": r["answer"]}
        for r in chemistry_rows(cfg.data_dir, split)
    ])


def _texts(completions) -> list[str]:
    return [c[0]["content"] if isinstance(c, list) else c for c in completions]


def chemistry_reward(completions, answer, **kwargs) -> list[float]:
    """1 if the letter in the last `<answer>` block matches, as in the reference code."""
    def extract(text):
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    return [float(extract(t) == a) for t, a in zip(_texts(completions), answer)]


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _as_number(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def gsm8k_reward(completions, answer, **kwargs) -> list[float]:
    """1 if the last number in the completion equals the label numerically."""
    out = []
    for text, label in zip(_texts(completions), answer):
        found = _NUMBER.findall(text)
        got, want = (_as_number(found[-1]) if found else None), _as_number(label)
        out.append(float(got is not None and want is not None and abs(got - want) < 1e-6))
    return out


REWARDS = {"chemistry": chemistry_reward, "gsm8k": gsm8k_reward}
