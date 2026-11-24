"""Measure alpha from accumulated PAE-versus-ranking gradient norms on the SHARED head.

Per issue #16: do NOT reuse alpha_rank=59. That number was calibrated for a coefficient-delivered
ranking gradient on a different objective with a different candidate-set size and score
derivative; none of those units carry over to a listwise loss whose gradient flows through a
learned reranker.

THE PROCEDURE THIS IMPLEMENTS, from the README:
  1. several MIXED-LABEL real groups across structures and win-rate bins, weights NOT updated;
  2. g_PAE and g_rank accumulated over the SAME window on the SAME trainable head parameters,
     with the cosine recorded too;
  3. alpha0 = r * ||g_PAE|| / ||g_rank||, r = 0.3, as a robust MEDIAN across windows;
  4. a first sweep over r in {0.1, 0.3, 1.0}.

WHY ACCUMULATED AND NOT PER-POSE. The mean of per-pose gradient norms is not the norm of the sum,
and the optimiser sees the sum. An earlier alpha probe in this project measured a 16x spread
across single steps (p10 0.48, p90 7.96, median 1.06) -- a single-window number would have been
worthless, which is why `windows` is a list and the summary is a median.

WHY THE COSINE MATTERS. That same probe measured median cosine -0.170 once its own coverage bias
was removed: the two terms LEAN AGAINST each other, so parity is the largest defensible weight
rather than the neutral one. A positive cosine would mean the ranking term is mostly re-stating
the PAE term and alpha could go higher.

STEP-0 CAVEAT, and it is not small. The reranker's final correction layer is zero-initialised, so
at step 0 the scores are exactly the input ipTM and the gradient reaching the head through the
ranking path is the ipTM path alone -- the pair encoder and attention block receive gradient, but
the head sees the ranking term as if the reranker were the identity. The README therefore says to
RE-MEASURE after the residual stops being zero; `measure_window` records `residual_rms` so a
calibration taken while it is still ~0 is identifiable afterwards rather than mistaken for a
converged ratio.

This module runs no training and updates no weights: every gradient is taken with
torch.autograd.grad and discarded.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import statistics
from typing import Any

import torch

from .integration import joint_group_loss
from .training import alpha_for_shared_gradient_ratio, shared_gradient_stats


@dataclass
class WindowStat:
    pae_norm: float
    rank_norm: float
    cosine: float
    alpha_parity: float
    residual_rms: float
    n_groups: int
    ranking_active: int
    detail: dict = field(default_factory=dict)


def measure_window(
    reranker: Any,
    groups: Sequence[tuple],
    predict_one: Callable,
    shared_head_params: Sequence[torch.nn.Parameter],
    *,
    temperature: float = 0.05,
    threshold: float = 0.23,
) -> WindowStat:
    """One accumulation window: sum both objectives over `groups`, then take the two gradients.

    `groups` is a sequence of (pose_inputs, dockq) for ONE window -- several groups summed, which
    is what the optimiser actually steps on under --accum. alpha is applied as 1.0 here because
    the point is to measure the UNWEIGHTED ratio.

    A group whose labels are all-good or all-bad contributes its PAE term and no ranking term;
    those are counted, not dropped, because the PAE side of the ratio must cover the same window
    the optimiser would.
    """
    if not groups:
        raise ValueError("a calibration window needs at least one group")
    pae_total = None
    rank_total = None
    active = 0
    resid = []
    for pose_inputs, dockq in groups:
        jl = joint_group_loss(reranker, pose_inputs, predict_one, dockq,
                              alpha=1.0, temperature=temperature, threshold=threshold)
        pae_total = jl.pae if pae_total is None else pae_total + jl.pae
        rank_total = jl.ranking if rank_total is None else rank_total + jl.ranking
        active += int(jl.ranking_active)
        # HOW FAR THE RESIDUAL HAS MOVED OFF ZERO. JointLoss carries the input ipTM, so
        # delta = scores - iptm exactly, with no second forward. Its RMS is ~0 at step 0 because
        # the final correction layer is zero-initialised; a calibration taken there describes the
        # reranker-as-identity and the README requires re-measuring once it is non-zero.
        with torch.no_grad():
            resid.append(float((jl.scores.detach() - jl.iptm.detach()).pow(2).mean().sqrt()))
    if active == 0:
        raise ValueError(
            "no group in this window had mixed labels, so the ranking gradient is identically "
            "zero and the ratio is undefined. Sample groups with at least one acceptable and one "
            "unacceptable pose; the README calls this a legitimate TRAINING sampling policy.")
    stats = shared_gradient_stats(pae_total, rank_total, shared_head_params)
    return WindowStat(
        pae_norm=stats["pae_norm"], rank_norm=stats["rank_norm"], cosine=stats["cosine"],
        alpha_parity=alpha_for_shared_gradient_ratio(stats, ratio=1.0),
        residual_rms=statistics.fmean(resid) if resid else float("nan"),
        n_groups=len(groups), ranking_active=active,
        detail={"pae_sum": float(pae_total.detach()), "rank_sum": float(rank_total.detach())},
    )


def summarise(windows: Sequence[WindowStat], ratios=(0.1, 0.3, 1.0)) -> dict:
    """The median across windows, and the alpha each requested head-gradient ratio implies.

    MEDIAN, NOT MEAN. The per-window ratio is heavy-tailed -- the earlier probe spanned 0.48 to
    7.96 at p10/p90 -- so a mean is dragged by one window and the median is what the README asks
    for ("a robust median across representative windows, not a single pose").
    """
    if not windows:
        raise ValueError("no calibration windows were measured")
    par = statistics.median([w.alpha_parity for w in windows])
    out = {
        "n_windows": len(windows),
        "median_alpha_parity": par,
        "median_cosine": statistics.median([w.cosine for w in windows]),
        "median_pae_norm": statistics.median([w.pae_norm for w in windows]),
        "median_rank_norm": statistics.median([w.rank_norm for w in windows]),
        "p10_alpha_parity": min(w.alpha_parity for w in windows),
        "p90_alpha_parity": max(w.alpha_parity for w in windows),
        "median_residual_rms": statistics.median([w.residual_rms for w in windows]),
        "groups": sum(w.n_groups for w in windows),
        "ranking_active_groups": sum(w.ranking_active for w in windows),
    }
    out["alpha_by_ratio"] = {str(r): r * par for r in ratios}
    out["recommended_alpha"] = out["alpha_by_ratio"]["0.3"]
    # THE SPREAD IS PART OF THE ANSWER. A 16x spread means alpha is known to ~1.5x, not two
    # digits, and reporting a single number without it invites over-reading the next arm.
    out["spread_factor"] = (out["p90_alpha_parity"] / out["p10_alpha_parity"]
                            if out["p10_alpha_parity"] > 0 else float("inf"))
    return out


def format_report(summary: dict) -> str:
    a = summary["alpha_by_ratio"]
    lines = [
        "## Alpha calibration from shared-head gradient norms",
        "",
        f"- windows: {summary['n_windows']}   groups: {summary['groups']}"
        f"   with an active ranking term: {summary['ranking_active_groups']}",
        f"- median ||g_PAE|| = {summary['median_pae_norm']:.4e}",
        f"- median ||g_rank|| = {summary['median_rank_norm']:.4e}",
        f"- median cosine(g_PAE, g_rank) = {summary['median_cosine']:+.4f}",
        f"- alpha at parity (ratio 1.0) = {summary['median_alpha_parity']:.4g}"
        f"   [p10 {summary['p10_alpha_parity']:.4g}, p90 {summary['p90_alpha_parity']:.4g},"
        f" spread {summary['spread_factor']:.1f}x]",
        f"- median residual spread at measurement time = {summary['median_residual_rms']:.4e}",
        "",
        "| head-gradient ratio r | alpha |",
        "|---|---|",
    ] + [f"| {r} | {v:.4g} |" for r, v in a.items()] + [
        "",
        f"**Recommended start: alpha = {summary['recommended_alpha']:.4g}** (r = 0.3).",
        "",
        "A negative cosine means the two terms lean against each other, so parity is the largest "
        "defensible weight rather than the neutral one. If the residual spread above is ~0 this "
        "is the STEP-0 calibration and must be re-measured once the reranker's correction is "
        "non-zero.",
    ]
    return "\n".join(lines)
