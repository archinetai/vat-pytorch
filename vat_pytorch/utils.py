import torch
import torch.nn.functional as F


def exists(value):
    return value is not None


def default(value, default):
    if exists(value):
        return value
    return default


def inf_norm(x):
    return torch.norm(x, p=float("inf"), dim=-1, keepdim=True)


def kl_loss(input, target, reduction="batchmean"):
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )


def sym_kl_loss(input, target, reduction="batchmean", alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )


def js_loss(input, target, reduction="batchmean", alpha=1.0):
    mean_proba = 0.5 * (
        F.softmax(input.detach(), dim=-1) + F.softmax(target.detach(), dim=-1)
    )
    return alpha * (
        F.kl_div(F.log_softmax(input, dim=-1), mean_proba, reduction=reduction)
        + F.kl_div(F.log_softmax(target, dim=-1), mean_proba, reduction=reduction)
    )
