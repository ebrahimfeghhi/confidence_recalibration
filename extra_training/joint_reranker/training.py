"""Compose original Protenix PAE cross-entropy with a joint listwise ranking loss.

Inputs are live tensors. This module is head-agnostic; caller is responsible for
obtaining each per-pose CE from its ORIGINAL Protenix loss implementation.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import PAEPoseReranker


def acceptable_mass_loss(
    scores: Tensor,
    dockq: Tensor,
    *,
    threshold: float = 0.23,
    temperature: float = 0.05,
) -> tuple[Tensor, bool]:
    """Negative log of softmax mass on poses with true DockQ >= threshold.

    No DockQ information reaches the model's feature computation. Ranking is
    inactive on all-good, all-bad, or K<2 groups; PAE CE is still applied to
    every pose by the caller.
    """
    if scores.ndim != 1 or dockq.shape != scores.shape:
        raise ValueError("scores and dockq must have matching shape [K]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(scores).all() or not torch.isfinite(dockq).all():
        raise ValueError("ranking scores and DockQ labels must be finite")
    good = dockq >= threshold
    if scores.numel() < 2 or good.all() or (~good).all():
        return scores.sum() * 0.0, False
    z = scores.float() / temperature
    return torch.logsumexp(z, dim=0) - torch.logsumexp(z[good], dim=0), True


@dataclass
class JointLoss:
    total: Tensor
    pae: Tensor
    ranking: Tensor
    ranking_active: bool
    scores: Tensor


def joint_objective(
    reranker: PAEPoseReranker,
    pae_logits: Tensor | Sequence[Tensor],
    pair_masks: Tensor | Sequence[Tensor],
    iptm_scores: Tensor,
    per_pose_pae_losses: Tensor | Sequence[Tensor],
    dockq: Tensor,
    *,
    alpha: float = 1.0,
    temperature: float = 0.05,
    threshold: float = 0.23,
) -> JointLoss:
    """Compose both end-to-end objectives, leaving stepping to the caller.

    per_pose_pae_losses MUST be original per-pose, differentiable Protenix
    PAE CE scalars using unchanged masks and reductions. Do not use floats
    from metric logs or detach head outputs. All K PAE terms contribute even
    when a ranking group has no acceptable/unacceptable contrast.
    """
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    scores = reranker(pae_logits, pair_masks, iptm_scores)
    if isinstance(per_pose_pae_losses, Tensor):
        pae_losses = per_pose_pae_losses
    else:
        pae_losses = torch.stack(list(per_pose_pae_losses))
    if pae_losses.shape != scores.shape:
        raise ValueError("per_pose_pae_losses must contain one scalar per pose")
    pae = pae_losses.mean()
    ranking, active = acceptable_mass_loss(
        scores, dockq, temperature=temperature, threshold=threshold
    )
    return JointLoss(
        total=pae + alpha * ranking,
        pae=pae,
        ranking=ranking,
        ranking_active=active,
        scores=scores,
    )


def shared_gradient_stats(
    pae_loss: Tensor,
    rank_loss: Tensor,
    shared_head_params: Sequence[torch.nn.Parameter],
) -> dict[str, float]:
    """Inspect the TWO unweighted gradients on shared trainable head weights.

    Diagnostic only: call occasionally before backward and on a representative
    group/update. Uses retain_graph=True; the regular backward can still follow.
    To match gradient accumulation, sum each objective across the SAME full
    accumulation window before calling this, not mean per-pose gradient norms.
    Do not include reranker-only params: PAE does not differentiate through them.
    """
    params = [p for p in shared_head_params if p.requires_grad]
    if not params:
        raise ValueError("no trainable confidence-head parameters provided")
    g_pae = torch.autograd.grad(pae_loss, params, allow_unused=True, retain_graph=True)
    g_rank = torch.autograd.grad(rank_loss, params, allow_unused=True, retain_graph=True)
    origin = torch.zeros((), device=params[0].device)
    dot = origin
    for a, b in zip(g_pae, g_rank):
        if a is not None and b is not None:
            dot = dot + (a.detach().float() * b.detach().float()).sum()
    sqp = sum(
        (g.detach().float().square().sum() for g in g_pae if g is not None),
        start=torch.zeros((), device=params[0].device),
    )
    sqr = sum(
        (g.detach().float().square().sum() for g in g_rank if g is not None),
        start=torch.zeros((), device=params[0].device),
    )
    np = sqp.sqrt().item()
    nr = sqr.sqrt().item()
    cosine = float((dot / (sqp * sqr).sqrt()).item()) if np > 0 and nr > 0 else float("nan")
    return {"pae_norm": np, "rank_norm": nr, "cosine": cosine}


def alpha_for_shared_gradient_ratio(stats: dict[str, float], ratio: float = 0.3) -> float:
    """Initial alpha making ||alpha * g_rank|| / ||g_PAE|| approximately ratio.

    This is a starting calibration, NOT a trained optimum. Tune with validation.
    Do not reuse an alpha from a differently normalized ranking objective.
    """
    if ratio <= 0 or stats["pae_norm"] <= 0 or stats["rank_norm"] <= 0:
        raise ValueError("ratio and both head gradient norms must be positive")
    return ratio * stats["pae_norm"] / stats["rank_norm"]
