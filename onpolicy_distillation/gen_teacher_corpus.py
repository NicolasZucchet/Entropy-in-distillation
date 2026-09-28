"""Sample and cache one teacher corpus, shared by every arm of that teacher.

The teacher is sampled at temperature 1 without top-p or top-k truncation. Each
completion is cut right after its first balanced `\\boxed{...}`, which gives an
instruct model prompted without its chat template an end of turn. Completions
without one within the budget are dropped, except with `--drop_unboxed 0`, which
the MATH500 corpus uses so that every problem stays in the evaluation set. The
first `--num_eval_prompts` prompts form the held-out split.

    python gen_teacher_corpus.py --teacher Qwen/Qwen3-1.7B --dataset deepscaler \\
        --num_prompts 40000 --num_eval_prompts 96
    python gen_teacher_corpus.py --teacher Qwen/Qwen3-1.7B --dataset math500 \\
        --num_prompts 200 --num_eval_prompts 200 --eval_only --drop_unboxed 0
"""
import argparse
import json
import os
import re

from data import SCRATCH, corpus_path, load_prompts, revision

THINK = re.compile(r"^.*?</think>", re.DOTALL)  # a reasoning block, if the teacher emits one


def answer_end(text: str) -> int | None:
    """Index just past the first balanced `\\boxed{...}`, or None."""
    i = text.find("\\boxed")
    j = text.find("{", i) if i >= 0 else -1
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", required=True)
    p.add_argument("--dataset", default="deepscaler")
    p.add_argument("--num_prompts", type=int, default=40000)
    p.add_argument("--num_eval_prompts", type=int, default=96)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--drop_unboxed", type=int, default=1)
    p.add_argument("--eval_only", action="store_true", help="write only the held-out split")
    p.add_argument("--tag", default="_rawcut", help="part of the corpus address, as in train.py")
    p.add_argument("--root", default=f"{SCRATCH}/teacher_corpora")
    p.add_argument("--seed", type=int, default=None, help="the paper's corpora were sampled without one")
    args = p.parse_args()

    os.makedirs(args.root, exist_ok=True)
    path = corpus_path(args.root, args.teacher, args.dataset + args.tag, args.num_prompts,
                       args.max_new_tokens)
    if os.path.exists(path):
        print(f"corpus already at {path}")
        return

    n_eval = args.num_eval_prompts
    n_train = 0 if args.eval_only else args.num_prompts
    pool = load_prompts(args.dataset, n_eval + n_train)
    eval_rows, train_rows = pool[:n_eval], pool[n_eval:][:n_train]
    rows = train_rows + eval_rows
    print(f"{len(train_rows)} training and {len(eval_rows)} held-out prompts -> {path}", flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.teacher, revision=revision(args.teacher), dtype="bfloat16",
              gpu_memory_utilization=0.85, **({} if args.seed is None else {"seed": args.seed}))
    sampling = SamplingParams(n=1, temperature=1.0, top_p=1.0, top_k=-1,
                              max_tokens=args.max_new_tokens)
    samples = [req.outputs[0].text for req in llm.generate([r["prompt"] for r in rows], sampling)]

    kept = 0
    with open(path, "w") as f:
        for i, (row, text) in enumerate(zip(rows, samples)):
            text = THINK.sub("", text).lstrip()
            end = answer_end(text)
            if end:
                text = text[:end]
            elif args.drop_unboxed:
                continue
            kept += 1
            f.write(json.dumps({"prompt": row["prompt"], "answer": row["answer"], "completion": text,
                                "split": "train" if i < len(train_rows) else "eval"}) + "\n")
    print(f"kept {kept} of {len(rows)} completions")


if __name__ == "__main__":
    main()
