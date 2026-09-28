"""Endpoint accuracy of one model on one benchmark: avg@k at temperature 0.6, top-p 0.95.

The result is appended to `--out` as one JSON line. One model per process, as
vLLM does not reliably release the GPU to a second engine.

    python score_endpoint.py --model <checkpoint> --dataset math500 --num_prompts 500 --k 3 \\
        --label on-reverse-Qwen3-1.7B-math500 --out scores/on-reverse-Qwen3-1.7B.jsonl

Prompts are the problem text without chat template, as in training; `--chat`
renders them in the chat template instead (the chat-template baseline of the
appendix), and `--thinking` then leaves the template's thinking switch on.
"""
import argparse
import json
import os
import signal
import sys

from data import load_prompts, revision, stop_ids
from grading import accuracy_stats, hits


def score(a) -> dict:
    from transformers import AutoTokenizer, GenerationConfig
    from vllm import LLM, SamplingParams

    rows = load_prompts(a.dataset, a.num_prompts, split="test")
    rev = revision(a.model)
    tok = AutoTokenizer.from_pretrained(a.model, revision=rev)
    prompts = [r["prompt"] for r in rows]
    if a.chat:
        prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=a.thinking)
                   for p in prompts]
    try:
        gen_cfg = GenerationConfig.from_pretrained(a.model, revision=rev)
    except Exception:
        gen_cfg = None
    stop = sorted(stop_ids(tok, gen_cfg))

    llm = LLM(model=a.model, revision=rev, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_frac)
    params = SamplingParams(n=a.k, temperature=0.6, top_p=0.95, max_tokens=a.max_new_tokens,
                            seed=a.seed, stop_token_ids=stop)
    completions, truncated, lengths = [], [], []
    for req in llm.generate(prompts, params):
        for o in req.outputs:
            completions.append(o.text)
            truncated.append(o.finish_reason == "length")
            lengths.append(len(o.token_ids))
    correct = hits(completions, [r["answer"] for r in rows for _ in range(a.k)])
    stats = accuracy_stats(correct, truncated, lengths)
    per_question = [sum(correct[i : i + a.k]) / a.k for i in range(0, len(correct), a.k)]
    print(f"[{a.label}] " + "  ".join(f"{n} {v:.4f}" for n, v in stats.items()), flush=True)
    return {"label": a.label, "model": a.model, "n_prompts": len(rows), "k": a.k,
            "max_new_tokens": a.max_new_tokens, "chat": a.chat, "thinking": a.thinking,
            "t0.6": stats, "per_question": {"t0.6": per_question}}


def hard_exit() -> None:
    """Kill our child processes and exit: vLLM's engine process can outlive the interpreter."""
    sys.stdout.flush()
    sys.stderr.flush()
    me = os.getpid()
    for entry in os.listdir("/proc") if os.path.isdir("/proc") else []:
        try:
            if entry.isdigit() and int(open(f"/proc/{entry}/stat").read().rsplit(")", 1)[1].split()[1]) == me:
                os.kill(int(entry), signal.SIGKILL)
        except (OSError, IndexError, ValueError):
            continue
    os._exit(0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="hub id or path to a saved checkpoint")
    p.add_argument("--dataset", default="math500", choices=["math500", "gsm8k", "amc23"])
    p.add_argument("--num_prompts", type=int, default=500)
    p.add_argument("--k", type=int, default=3, help="samples per prompt")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--label", default="")
    p.add_argument("--out", default="scores.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_frac", type=float, default=0.85)
    p.add_argument("--chat", action="store_true", help="render the prompts in the chat template")
    p.add_argument("--thinking", action="store_true", help="with --chat, enable_thinking=True")
    a = p.parse_args()
    a.label = a.label or f"{os.path.basename(a.model.rstrip('/'))}-{a.dataset}"
    rec = score(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"appended {rec['label']} to {a.out}")
    hard_exit()


if __name__ == "__main__":
    main()
