"""Run configuration, with a `--field value` command line generated from it.

The defaults are the Chemistry setup, i.e. the released configuration of
Hubotter et al. (2026) with the changes listed in the README. The launchers in
`launch/` override the model and lengths for GSM8K, and each ablation moves one
field away from these defaults.
"""
import argparse
import dataclasses
import os
from dataclasses import dataclass
from typing import get_type_hints

# Everything large (Hugging Face cache, datasets, run outputs, logs) lives here.
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", os.path.abspath("outputs"))

# The Hugging Face revisions every run of the paper loaded.
MODEL_REVISIONS = {
    "allenai/Olmo-3-7B-Instruct": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
}


@dataclass
class RunConfig:
    model: str = "allenai/Olmo-3-7B-Instruct"
    task: str = "chemistry"  # chemistry | gsm8k
    data_dir: str = os.path.join(SCRATCH_DIR, "datasets")
    num_train_prompts: int = 4000  # GSM8K only; Chemistry uses fixed files
    num_eval_prompts: int = 200

    # Rollouts: `train_batch_size` questions per step, `num_generations` answers
    # each, and one optimizer step per batch, so training is strictly on-policy.
    train_batch_size: int = 32
    num_generations: int = 8
    max_prompt_length: int = 2048
    max_completion_length: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0

    # Validation: avg@16 at its own sampling temperature and top-p.
    num_generations_eval: int = 16
    eval_temperature: float = 0.6
    eval_top_p: float = 0.95
    eval_every: int = 50  # 0 disables validation
    eval_batch_groups: int = 16  # avg@16 groups per vLLM call; throughput only

    # Self-distillation objective.
    distillation_alpha: float = 0.5  # 0 forward KL, 1 reverse KL
    distillation_mode: str = "topk_logits"  # topk_logits | full_logits
    distillation_topk: int = 100
    distillation_add_tail: bool = True
    distillation_is_clip: float = 2.0
    teacher_kind: str = "ema"  # ema | live | base
    teacher_update_rate: float = 0.05  # EMA decay 0.95
    max_reprompt_len: int = 10240
    success_reward_threshold: float = 0.5

    # Optimization.
    learning_rate: float = 1e-5
    warmup_steps: int = 10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 150

    # Engine.
    gpus: int = 8
    per_device_batch_size: int = 1
    vllm_gpu_memory_utilization: float = 0.3
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"

    wandb_project: str = "self-distillation-entropy"
    wandb_mode: str = "disabled"  # disabled | offline | online
    seed: int = 0
    output_dir: str = os.path.join(SCRATCH_DIR, "runs")
    run_name: str = "run"


def parse_config() -> RunConfig:
    """A `RunConfig` with the `--field value` overrides of the command line."""
    parser = argparse.ArgumentParser(description="Self-distillation with TRL")
    hints = get_type_hints(RunConfig)
    for field in dataclasses.fields(RunConfig):
        kind = hints[field.name]
        parse = (lambda x: x.lower() in ("true", "1", "yes")) if kind is bool else kind
        parser.add_argument(f"--{field.name}", type=parse, default=field.default)
    return RunConfig(**vars(parser.parse_args()))
