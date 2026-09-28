"""Per-token student and teacher statistics on teacher-forced completions."""
import torch


@torch.no_grad()
def paired_stats(student, teacher, input_ids, attention_mask, loss_mask, chunk=256):
    """Entropies, KLs and cross-entropies of both models on the tokens of `loss_mask`.

    Both models see the same sequences, so every quantity compares them on the
    same contexts. The float32 softmax is taken in chunks along the sequence, so
    it is never materialised over the full sequence and 152k-token vocabulary.
    """
    s_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    t_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    targets, mask = input_ids[:, 1:], loss_mask[:, 1:]
    cols: dict[str, list[torch.Tensor]] = {}
    for i in range(0, s_logits.shape[1], chunk):
        sl = torch.log_softmax(s_logits[:, i : i + chunk].float(), dim=-1)
        tl = torch.log_softmax(t_logits[:, i : i + chunk].float(), dim=-1)
        sp, tp = sl.exp(), tl.exp()
        tgt = targets[:, i : i + chunk].unsqueeze(-1)
        piece = {
            "student_entropy": -(sp * sl).sum(-1),
            "teacher_entropy": -(tp * tl).sum(-1),
            "forward_kl": (tp * (tl - sl)).sum(-1),  # KL(teacher || student)
            "reverse_kl": (sp * (sl - tl)).sum(-1),  # KL(student || teacher)
            "student_ce": -sl.gather(-1, tgt).squeeze(-1),
            "teacher_ce": -tl.gather(-1, tgt).squeeze(-1),
        }
        for k, v in piece.items():
            cols.setdefault(k, []).append(v)
    # Concatenate along positions before flattening, so the order matches `mask`.
    keep = mask.reshape(-1).bool()
    out = {k: torch.cat(v, dim=1).reshape(-1)[keep] for k, v in cols.items()}
    out["entropy_gap"] = out["student_entropy"] - out["teacher_entropy"]
    # The sequence each kept token came from, for the per-sequence averages.
    b, seq_len = mask.shape
    out["_seq"] = torch.arange(b, device=mask.device).unsqueeze(1).expand(b, seq_len).reshape(-1)[keep]
    return out


def _per_seq_mean(values: torch.Tensor, seq: torch.Tensor) -> float:
    """Mean over sequences of the per-sequence token mean, in float64."""
    n = int(seq.max().item()) + 1 if seq.numel() else 0
    if n == 0:
        return float("nan")
    sums = torch.zeros(n, dtype=torch.float64, device=values.device)
    counts = torch.zeros(n, dtype=torch.float64, device=values.device)
    sums.scatter_add_(0, seq, values.double())
    counts.scatter_add_(0, seq, torch.ones_like(values, dtype=torch.float64))
    keep = counts > 0
    return (sums[keep] / counts[keep]).mean().item()


def summarise(cols: dict[str, torch.Tensor], prefix: str) -> dict[str, float]:
    """Token-pooled means (`<name>`, what the paper reports) and per-sequence means (`<name>_per_seq`)."""
    seq = cols["_seq"]
    out = {}
    for k, v in cols.items():
        if k != "_seq":
            out[f"{prefix}{k}"] = v.mean().item()
            out[f"{prefix}{k}_per_seq"] = _per_seq_mean(v, seq)
    return out


def logit_scale(model) -> dict[str, float]:
    """Norms of the final normalization gain and of the output layer, which set the logit scale."""
    out = {}
    for name, param in model.named_parameters():
        if name.endswith("model.norm.weight"):
            out["final_norm_gain"] = param.detach().float().norm().item()
        elif name.endswith("lm_head.weight"):
            out["lm_head_norm"] = param.detach().float().norm().item()
    return out
