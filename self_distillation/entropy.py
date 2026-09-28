"""Token-level entropy of the student and the teacher, from logits already in hand."""
import torch

# Order of the statistics in the vector returned by `entropy_sums`.
FIELDS = ("h_sum", "h_count", "dist_student", "dist_teacher", "dist_count", "deflated")


@torch.no_grad()
def entropy_sums(student_logits, teacher_logits, completion_mask, loss_mask, chunk=256):
    """Unnormalized entropy sums and token counts, in the order of `FIELDS`.

    Sums rather than means, so that the caller can add them across ranks and get
    token-weighted averages. `completion_mask` covers every sampled token (the
    policy's entropy); `loss_mask` the tokens the divergence is taken on, where
    the teacher sees the privileged context. The softmax is taken `chunk`
    positions at a time to bound the float32 memory over the vocabulary.
    """
    stats = torch.zeros(len(FIELDS), device=student_logits.device, dtype=torch.float32)
    comp = completion_mask.float()
    loss_m = loss_mask.float()
    for i in range(0, student_logits.shape[1], chunk):
        sl = slice(i, i + chunk)
        lp = torch.log_softmax(student_logits[:, sl].float(), dim=-1)
        h_student = -(lp.exp() * lp).sum(-1)
        del lp
        stats[0] += (h_student * comp[:, sl]).sum()
        stats[1] += comp[:, sl].sum()
        lp = torch.log_softmax(teacher_logits[:, sl].float(), dim=-1)
        h_teacher = -(lp.exp() * lp).sum(-1)
        del lp
        stats[2] += (h_student * loss_m[:, sl]).sum()
        stats[3] += (h_teacher * loss_m[:, sl]).sum()
        stats[4] += loss_m[:, sl].sum()
        stats[5] += ((h_student < h_teacher).float() * loss_m[:, sl]).sum()
    return stats


def entropy_means(stats) -> dict[str, float]:
    """Means from the summed statistics; a mask with no tokens gives no entry, not a zero."""
    out = {}
    if stats[1] > 0:
        out["entropy"] = stats[0] / stats[1]
    if stats[4] > 0:
        out["student_entropy_distilled"] = stats[2] / stats[4]
        out["teacher_entropy"] = stats[3] / stats[4]
        out["frac_deflated"] = stats[5] / stats[4]
    return out
