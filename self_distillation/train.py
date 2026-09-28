"""Self-distillation with TRL's `SDPOTrainer`, following Hubotter et al. (2026).

The reward is binary. The teacher is the student (or a moving average of it)
whose prompt also contains a successful rollout from the same group, and the
student is trained towards it with a generalized Jensen-Shannon divergence on
its top-k tokens. Run through `launch/run.sbatch`, or directly:

    accelerate launch --config_file launch/zero3.yaml --num_processes 1 train.py \
        --task gsm8k --model Qwen/Qwen3-0.6B --gpus 1 ...

Metrics are printed and appended to `<output_dir>/<run_name>/metrics.jsonl`.
"""
import json
import os

import torch
from transformers import TrainerCallback
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer

from config import MODEL_REVISIONS, parse_config
from entropy import entropy_means, entropy_sums
from tasks import REWARDS, load_split


class JsonlMetrics(TrainerCallback):
    """Append every logged dict, with its step, to a JSON-lines file."""

    def __init__(self, path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            with open(self.path, "a") as f:
                f.write(json.dumps({"step": state.global_step, **logs}) + "\n")


class Trainer(SDPOTrainer):
    """`SDPOTrainer` with entropy logging, a ZeRO-3 fix, and validation-only sampling settings."""

    def __init__(self, *args, eval_temperature, eval_top_p, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_temperature, self.eval_top_p = eval_temperature, eval_top_p

    def _log_entropy(self, mode, student_logits, teacher_logits, completion_mask, loss_mask):
        """Log student and teacher entropies (`policy/entropy` is the one plotted).

        One all-gather of a fixed-size vector, so every rank runs the same
        collectives whatever its batch holds.
        """
        stats = entropy_sums(student_logits, teacher_logits, completion_mask, loss_mask)
        stats = self.accelerator.gather(stats.unsqueeze(0)).sum(0).tolist()
        for key, value in entropy_means(stats).items():
            self._metrics[mode][f"policy/{key}"].append(value)

    def _compute_self_distillation_loss(self, model, inputs, distillation_logits):
        """Log entropies, and fix a hang when a rank has no successful rollout to distill.

        TRL 1.9.2 returns early on such a rank and skips the `gather` the other
        ranks perform, which desynchronizes every later ZeRO-3 collective.
        """
        mode = "train" if model.training else "eval"
        d = distillation_logits
        self._log_entropy(mode, d.student_logits, d.teacher_logits, d.completion_mask, d.loss_mask)
        if d.loss_mask.sum() == 0:
            zero = torch.zeros((), device=d.student_logits.device)
            self._log_self_distillation_metric(mode, self.accelerator.gather(zero).mean().item())
            return d.student_logits.sum() * 0.0
        return super()._compute_self_distillation_loss(model, inputs, distillation_logits)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Validation only needs the rewards, which `_prepare_inputs` records; skip the loss."""
        if not isinstance(inputs, dict):
            self._prepare_inputs(inputs)
        return torch.zeros((), device=self.accelerator.device), None, None

    def _generate_vllm(self, prompt_ids):
        """Sample validation rollouts at the validation temperature and top-p."""
        if self.model.training:
            return super()._generate_vllm(prompt_ids)
        gen = self.vllm_generation
        train_t, train_p = gen.temperature, gen.top_p
        gen.temperature, gen.top_p = self.eval_temperature, self.eval_top_p
        try:
            return super()._generate_vllm(prompt_ids)
        finally:
            gen.temperature, gen.top_p = train_t, train_p


def resolve_model(model):
    """The local snapshot of a pinned model, so weights, tokenizer and vLLM share its revision."""
    if model not in MODEL_REVISIONS or os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=MODEL_REVISIONS[model])


def sdpo_config(cfg):
    """TRL's `SDPOConfig` for a run."""
    # One optimizer step per generation batch, which keeps training on-policy.
    generation_batch = cfg.train_batch_size * cfg.num_generations
    per_step = cfg.per_device_batch_size * cfg.gpus
    if generation_batch % per_step:
        raise ValueError(f"{generation_batch} sequences per step do not split into {per_step}")
    grad_accum = generation_batch // per_step
    if (cfg.num_generations_eval * cfg.eval_batch_groups) % cfg.gpus:
        raise ValueError("the eval batch must hold whole avg@16 groups on every GPU")

    return SDPOConfig(
        output_dir=os.path.join(cfg.output_dir, cfg.run_name),
        run_name=cfg.run_name,
        seed=cfg.seed,
        # Rollouts.
        num_generations=cfg.num_generations,
        num_generations_eval=cfg.num_generations_eval,
        steps_per_generation=grad_accum,
        gradient_accumulation_steps=grad_accum,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.num_generations_eval * cfg.eval_batch_groups // cfg.gpus,
        num_iterations=1,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        chat_template_kwargs={"enable_thinking": False},
        # Objective.
        distillation_weight=1.0,
        distillation_mode=cfg.distillation_mode,
        distillation_topk=cfg.distillation_topk if cfg.distillation_mode == "topk_logits" else None,
        distillation_add_tail=cfg.distillation_add_tail,
        distillation_alpha=cfg.distillation_alpha,
        distillation_is_clip=cfg.distillation_is_clip or None,  # inert: training is on-policy
        importance_sampling_level="token",
        teacher_model_kind=cfg.teacher_kind,
        teacher_update_rate=cfg.teacher_update_rate,
        teacher_sync_steps=1,
        max_reprompt_len=cfg.max_reprompt_len,
        dont_reprompt_on_self_success=True,
        remove_thinking_from_demonstration=True,
        include_environment_feedback=False,  # the only privileged information is a successful rollout
        use_successful_as_teacher=True,
        success_reward_threshold=cfg.success_reward_threshold,
        beta=0.0,  # no KL to a reference policy
        scale_rewards="none",
        epsilon_high=0.28,
        loss_type="dapo",
        # Optimization.
        learning_rate=cfg.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.grad_clip,
        max_steps=cfg.max_steps,
        bf16=cfg.dtype == "bfloat16",
        gradient_checkpointing=True,
        # Generation engine.
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
        vllm_max_model_length=cfg.max_prompt_length + cfg.max_completion_length,
        # Reporting.
        eval_strategy="steps" if cfg.eval_every else "no",
        eval_steps=cfg.eval_every or None,
        eval_on_start=bool(cfg.eval_every),
        logging_steps=1,
        save_strategy="no",
        report_to="none" if cfg.wandb_mode == "disabled" else "wandb",
        model_init_kwargs={"dtype": getattr(torch, cfg.dtype),
                           "attn_implementation": cfg.attn_implementation},
    )


def main():
    cfg = parse_config()
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)
    os.environ.setdefault("WANDB_MODE", cfg.wandb_mode)
    args = sdpo_config(cfg)
    trainer = Trainer(
        model=resolve_model(cfg.model),
        reward_funcs=REWARDS[cfg.task],
        args=args,
        train_dataset=load_split(cfg, "train"),
        eval_dataset=load_split(cfg, "test"),
        eval_temperature=cfg.eval_temperature,
        eval_top_p=cfg.eval_top_p,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.add_callback(JsonlMetrics(os.path.join(args.output_dir, "metrics.jsonl")))
    trainer.train()


if __name__ == "__main__":
    main()
