"""Train one arm of the grid: prefix source x KL direction, for one teacher.

TRL's general online logit distillation (GOLD) trainer covers the four
objectives with two numbers: `lmbda` is the fraction of on-policy
(student-sampled) prefixes, and `beta` moves the generalized Jensen-Shannon
divergence from forward KL (0) to reverse KL (1). The divergence is computed on
the full vocabulary at every position. Off-policy prefixes are the cached
teacher completions of `gen_teacher_corpus.py`, to which TRL appends the
student's end-of-text token.

    python train.py --teacher Qwen/Qwen3-1.7B --lmbda 1 --beta 1

The in-training evaluations are appended to `<output_dir>/<run_name>/eval.jsonl`.
"""
import argparse
import contextlib
import json
import os
import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl.experimental.gold import GOLDConfig, GOLDTrainer

from data import SCRATCH, corpus_path, load_corpus, revision, stop_ids
from entropy import logit_scale, paired_stats, summarise
from grading import accuracy_stats, hits

# Hyperparameters of Table 2 in the appendix.
STUDENT = "Qwen/Qwen3-0.6B-Base"
NUM_PROMPTS, NUM_EVAL_PROMPTS = 40000, 96  # DeepScaleR corpus, of which the first 96 are held out
NUM_BENCH_PROMPTS = 200  # MATH500
MAX_PROMPT_TOKENS, MAX_NEW_TOKENS = 1024, 4096
CORPUS_TAG = "_rawcut"  # part of the corpus address, as in `gen_teacher_corpus.py`
LEARNING_RATE, WARMUP_STEPS, TOTAL_STEPS = 5e-7, 45, 800
GRAD_ACCUM = 32  # questions per optimizer step, one per micro-batch
EVAL_STEPS = (0, 25, 50, 100, 150, 250, 400, 600, 800)
EVAL_MAX_NEW_TOKENS = 2048
EVAL_BATCH, EVAL_GEN_BATCH = 4, 64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default="Qwen/Qwen3-4B")
    p.add_argument("--lmbda", type=float, default=1.0, help="0 off-policy, 1 on-policy")
    p.add_argument("--beta", type=float, default=1.0, help="0 forward KL, 1 reverse KL")
    p.add_argument("--run_name", default="", help="defaults to e.g. on-reverse-Qwen3-4B")
    p.add_argument("--corpus_dir", default=f"{SCRATCH}/teacher_corpora")
    p.add_argument("--output_dir", default=f"{SCRATCH}/runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_interval", type=int, default=50,
                   help="checkpoint every this many steps, so that a killed arm resumes")
    p.add_argument("--stop_after", type=int, default=0,
                   help="stop after this step's evaluation, without a final model (smoke test)")
    args = p.parse_args()
    if not args.run_name:
        src = "on" if args.lmbda > 0.5 else "off"
        div = "reverse" if args.beta > 0.5 else "forward"
        args.run_name = f"{src}-{div}-{args.teacher.split('/')[-1]}"
    return args


class EvalSet:
    """A named prompt set, with the frozen teacher completions paired with it."""

    def __init__(self, name: str, rows: list[dict]):
        self.name = name
        self.prompts = [r["prompt"] for r in rows]
        self.teacher_texts = [r["prompt"] + r["completion"] for r in rows]
        self.answers = [r["answer"] for r in rows]
        correct = hits([r["completion"] for r in rows], self.answers)
        self.teacher_accuracy = sum(correct) / max(len(correct), 1)


def build_datasets(args):
    """The training set, and the held-out DeepScaleR and MATH500 evaluation sets."""
    from datasets import Dataset

    def corpus(dataset, n):
        path = corpus_path(args.corpus_dir, args.teacher, dataset + CORPUS_TAG, n, MAX_NEW_TOKENS)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no teacher corpus at {path}; run gen_teacher_corpus.py first")
        return load_corpus(path)

    rows = corpus("deepscaler", NUM_PROMPTS)
    train = [r for r in rows if r["split"] == "train"]
    held_out = [r for r in rows if r["split"] == "eval"][:NUM_EVAL_PROMPTS]
    bench = [r for r in corpus("math500", NUM_BENCH_PROMPTS) if r["split"] == "eval"]
    evalsets = [EvalSet("train", held_out), EvalSet("math500", bench[:NUM_BENCH_PROMPTS])]
    print(f"[data] {len(train)} training rows", flush=True)
    for es in evalsets:
        print(f"[evalset] {es.name}: {len(es.prompts)} prompts, "
              f"teacher accuracy {es.teacher_accuracy:.3f}", flush=True)
    # Plain strings are TRL's standard format, so no chat template is applied.
    train_ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]}
                                  for r in train])
    return train_ds, evalsets


@torch.no_grad()
def teacher_forced(student, teacher, tok, texts, prompts, device):
    """Paired statistics on prompt + completion texts, on the completion tokens only."""
    cols: dict[str, list[torch.Tensor]] = {}
    for i in range(0, len(texts), EVAL_BATCH):
        enc = tok(texts[i : i + EVAL_BATCH], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_PROMPT_TOKENS + EVAL_MAX_NEW_TOKENS).to(device)
        loss_mask = enc["attention_mask"].clone()
        for j, p in enumerate(prompts[i : i + EVAL_BATCH]):
            loss_mask[j, : len(tok(p, add_special_tokens=False)["input_ids"])] = 0
        out = paired_stats(student, teacher, enc["input_ids"], enc["attention_mask"], loss_mask)
        out["_seq"] = out["_seq"] + i  # sequence ids unique across batches
        for k, v in out.items():
            cols.setdefault(k, []).append(v.cpu())
    return {k: torch.cat(v) for k, v in cols.items()}


@contextlib.contextmanager
def fixed_rng(seed: int):
    """Run a block with a pinned torch RNG, then restore the training loop's stream."""
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@torch.no_grad()
def sample(model, tok, prompts, device):
    """One completion per prompt at temperature 1, and its length up to the first stop token."""
    eos = stop_ids(tok, model.generation_config)
    outs, lengths = [], []
    for i in range(0, len(prompts), EVAL_GEN_BATCH):
        enc = tok(prompts[i : i + EVAL_GEN_BATCH], return_tensors="pt", padding=True,
                  padding_side="left").to(device)
        gen = model.generate(**enc, max_new_tokens=EVAL_MAX_NEW_TOKENS, do_sample=True,
                             temperature=1.0, top_p=1.0, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1] :]
        outs += tok.batch_decode(new, skip_special_tokens=True)
        lengths += [next((j + 1 for j, t in enumerate(row) if t in eos), len(row))
                    for row in new.tolist()]
    return outs, lengths


class EntropyEval(TrainerCallback):
    """Student entropy on frozen teacher completions and on its own samples, plus accuracy."""

    def __init__(self, args, tok, teacher, evalsets: list[EvalSet], log_path: str):
        self.args, self.tok, self.teacher, self.evalsets = args, tok, teacher, evalsets
        self.log_path = log_path

    def run(self, student, step: int):
        device = next(student.parameters()).device
        was_training = student.training
        student.eval()
        logs = {}
        for es in self.evalsets:
            pre = f"eval/{es.name}/"
            on_teacher = teacher_forced(student, self.teacher, self.tok, es.teacher_texts,
                                        es.prompts, device)
            logs |= summarise(on_teacher, f"{pre}teacher_data/")
            # The same seed at every evaluation, so that neighbouring points share
            # contexts as far as the weights allow. crc32, as `hash` is salted per process.
            with fixed_rng(self.args.seed + zlib.crc32(es.name.encode())):
                completions, lengths = sample(student, self.tok, es.prompts, device)
            on_student = teacher_forced(student, self.teacher, self.tok,
                                        [p + c for p, c in zip(es.prompts, completions)],
                                        es.prompts, device)
            logs |= summarise(on_student, f"{pre}student_data/")
            stats = accuracy_stats(hits(completions, es.answers),
                                   [n >= EVAL_MAX_NEW_TOKENS for n in lengths], lengths)
            logs |= {f"{pre}accuracy/{k}": v for k, v in stats.items()}
            logs[f"{pre}accuracy/teacher"] = es.teacher_accuracy
        logs |= {f"eval/{k}": v for k, v in logit_scale(student).items()}
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"_step": step, **logs}) + "\n")
        print(f"[eval] step {step}: " + "  ".join(
            f"{k.removeprefix('eval/')} {logs[k]:.4f}" for k in (
                "eval/math500/teacher_data/student_entropy",
                "eval/math500/student_data/student_entropy",
                "eval/math500/accuracy/accuracy")), flush=True)
        if was_training:
            student.train()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step in EVAL_STEPS and step < TOTAL_STEPS:  # the last step is evaluated by main()
            self.run(model, step)
        if self.args.stop_after and step >= self.args.stop_after:
            control.should_training_stop = True


def main():
    args = parse_args()
    out = os.path.join(args.output_dir, args.run_name)
    os.makedirs(out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(STUDENT, revision=revision(STUDENT))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load(path):
        # Pure bfloat16, without float32 master weights, as in the reference implementation.
        return AutoModelForCausalLM.from_pretrained(
            path, revision=revision(path), dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa")

    student = load(STUDENT)
    teacher = load(args.teacher).eval()
    train_ds, evalsets = build_datasets(args)
    callback = EntropyEval(args, tok, teacher, evalsets, os.path.join(out, "eval.jsonl"))

    config = GOLDConfig(
        output_dir=out,
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=1.0,
        seq_kd=False,
        max_completion_length=MAX_NEW_TOKENS,
        max_length=MAX_PROMPT_TOKENS + MAX_NEW_TOKENS,
        top_k=0,  # unfiltered sampling of the on-policy rollouts
        top_p=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.25,
        vllm_enable_sleep_mode=True,
        vllm_group_port=int(os.environ.get("VLLM_GROUP_PORT", 51216)),
        save_strategy="steps",
        save_steps=args.save_interval,
        save_total_limit=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=WARMUP_STEPS,
        max_steps=TOTAL_STEPS,
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=10,
        bf16=True,
        seed=args.seed,
        report_to=[],
    )
    # GOLD copies `eos_token_id` into its generation config once, at construction,
    # and uses it both to stop rollouts and to mask the loss after the first EOS,
    # so the full stop set has to be in place before.
    stop = sorted(stop_ids(tok, student.generation_config))
    student.generation_config.eos_token_id = stop
    print(f"[stop] rollout/mask eos ids: { {i: tok.decode([i]) for i in stop} }", flush=True)

    trainer = GOLDTrainer(model=student, teacher_model=teacher, args=config,
                          processing_class=tok, train_dataset=train_ds, callbacks=[callback])
    # The colocated vLLM sampler does not inherit that stop set.
    if getattr(trainer, "vllm_generation", None) is not None:
        trainer.vllm_generation.generation_kwargs["stop_token_ids"] = stop

    resume = any(d.startswith("checkpoint-") for d in os.listdir(out))
    if not resume:
        callback.run(student, step=0)
    trainer.train(resume_from_checkpoint=resume)
    if args.stop_after:
        return
    callback.run(student, step=TOTAL_STEPS)
    trainer.save_model(os.path.join(out, "final"))


if __name__ == "__main__":
    main()
