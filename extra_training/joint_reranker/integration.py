"""Minimal integration bridge: live Protenix pose outputs -> joint training loss.

This is intentionally separate from phase_k_train.py. The existing production
trainer can opt into this bridge after the CUDA capacity probes, without
changing legacy PAE/ranking runs.

Usage:
    def predict_one(pose):  # supplied by the Protenix training adapter
        # 1. Forward the TRAINABLE confidence head with graph enabled.
        # 2. Slice the PAE head's frame/sample dimensions to [F,T,64].
        # 3. Compute ORIGINAL Protenix PAE CE for THIS pose (scalar).
        # 4. Compute differentiable ipTM for THIS AB/AG pair from SAME logits;
        #    retain uncropped d0 convention.
        # 5. Compute model-chain AB/AG mask without using native/ref mapping.
        return PosePrediction(pae_logits, pair_mask, iptm, pae_ce)

    group = joint_group_loss(reranker, poses, predict_one, true_dockq,
                             alpha=alpha, temperature=tau)
    group.total.backward()  # updates head and reranker; optimizer steps once

The forward adapter must NEVER return .detach() or .item() versions of these.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .model import PAEPoseReranker
from .training import JointLoss, joint_objective


@dataclass
class PosePrediction:
    pae_logits: Tensor   # [N_frame,N_token,64], LIVE graph
    pair_mask: Tensor    # [N_frame,N_token] bool from MODEL chain IDs
    iptm: Tensor         # differentiable scalar from SAME logits
    pae_ce: Tensor       # differentiable ORIGINAL Protenix PAE CE scalar


def joint_group_loss(
    reranker: PAEPoseReranker,
    pose_inputs: Sequence[Any],
    predict_one: Callable[[Any], PosePrediction],
    dockq: Tensor,
    *,
    alpha: float = 1.0,
    temperature: float = 0.05,
    threshold: float = 0.23,
) -> JointLoss:
    """Joint graph over K poses, retaining all forward activations until backward.

    Caller must allocate memory for ALL K graphs and pair embeddings.
    'pose_inputs' are different predictions of ONE scored interface; inputs
    must not mix interfaces, even if those interfaces share a PDB.
    """
    if len(pose_inputs) < 2:
        raise ValueError("joint ranking requires at least 2 unique poses in a group")
    if dockq.shape != (len(pose_inputs),):
        raise ValueError("dockq must contain one training-only label per pose")
    predictions = [predict_one(pose) for pose in pose_inputs]
    if any(not isinstance(pred, PosePrediction) for pred in predictions):
        raise TypeError("predict_one must return PosePrediction for every pose")
    for j, pred in enumerate(predictions):
        if pred.pae_logits.ndim != 3 or pred.pae_logits.shape[-1] != 64:
            raise ValueError(f"pose {j}: expected logits [F,T,64]")
        if pred.pair_mask.shape != pred.pae_logits.shape[:2]:
            raise ValueError(f"pose {j}: pair mask does not match its PAE grid")
        if pred.iptm.ndim != 0 or pred.pae_ce.ndim != 0:
            raise ValueError(f"pose {j}: ipTM and PAE CE must be scalar tensors")
        if not (pred.pae_logits.requires_grad and pred.iptm.requires_grad
                and pred.pae_ce.requires_grad):
            raise ValueError(f"pose {j}: head output/ipTM/PAE CE was detached")
    return joint_objective(
        reranker=reranker,
        pae_logits=[pred.pae_logits for pred in predictions],
        pair_masks=[pred.pair_mask for pred in predictions],
        iptm_scores=torch.stack([pred.iptm for pred in predictions]),
        per_pose_pae_losses=torch.stack([pred.pae_ce for pred in predictions]),
        dockq=dockq,
        alpha=alpha,
        temperature=temperature,
        threshold=threshold,
    )
