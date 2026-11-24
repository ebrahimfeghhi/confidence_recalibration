"""The live Protenix forward, turned into the `predict_one` the joint reranker needs.

This is the adapter `integration.joint_group_loss` asks for. It owns ONE job: given the training
state the trainer already has in hand for a pose, produce a `PosePrediction` whose four fields all
carry a live graph back to the trainable confidence head -- the head's PAE logits, the AB/AG pair
mask, the differentiable ipTM, and the ORIGINAL Protenix PAE cross-entropy.

NOTHING HERE REIMPLEMENTS A LOSS OR A KERNEL. Every number comes from the function the existing
arms already use:
  * the head            `model.run_confidence_head(...)`            phase_k_train.py:6919
  * the PAE CE          `loss_fn.calculate_losses(...)`             phase_k_train.py:7171
  * the ipTM            `rank_loss.iptm_pooled(...)`                rank_loss.py:356
A reimplementation that drifted would answer "does a learned readout beat the fixed one" with a
different quantity on each side, which is the failure mode this project has paid for twice.

THE ipTM IS MAX-POOLED WITH A PINNED UNCROPPED d0. Decided in issue #16 on 2026-11-24: the
reranker scores a pose as `ipTM + correction`, so the baseline has to be the ipTM that selection
deploys, which is the max. `iptm_pooled(pool="max", n_pair_tokens=<uncropped count>)` is the
fixed-d0 path added in 60020ce5 -- without the pin, the vendor kernel recomputes d0 from the
CROPPED asym_id and one interface's poses get different TM kernels at 256/384/512. Measured on a
synthetic fixture: 0.0163 against 0.1614 at the smallest crop, a ~10x error.

THE ROW SET IS iptm_pooled's OWN. `live = has_frame & ~is_ligand`, the ligand derived by
scatter_add over `atom_to_token_idx` exactly as rank_loss.py does it. The reranker pools over the
pairs it is correcting, not a different set -- otherwise the correction and the baseline describe
different pair populations.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .integration import PosePrediction


@dataclass
class PoseInput:
    """Everything the head needs for ONE pose, taken from the trainer's own pose loop.

    These are the names the trainer already uses, so a caller inside phase_k_train can pass its
    local variables through without renaming and without a second source of truth:
      feat_head   the head's input_feature_dict (CROPPED token set under --crop-pool)
      feat_cache  the feature dict calculate_losses reads (the trainer's `feat_cache`)
      labels      the label dict calculate_losses reads (the trainer's `ld`)
      s_inputs/s_trunk/z_trunk/coords   the trunk tensors and this pose's coordinates
      pose_id     (pdb, seed, sample) -- the SAMPLE index, not the pose index; see below
    """
    feat_head: dict
    feat_cache: dict
    labels: dict
    s_inputs: Tensor
    s_trunk: Tensor
    z_trunk: Tensor
    coords: Tensor
    pose_id: tuple


def live_row_mask(feat: dict) -> Tensor:
    """iptm_pooled's `live`, recomputed here from the SAME fields so the two cannot drift.

    rank_loss.py:394-399 builds it as has_frame AND NOT ligand, with the ligand flag scattered
    from atoms onto tokens. Copied rather than imported because iptm_pooled computes it inline and
    does not expose it; the test gate asserts the two agree.
    """
    asym = feat["asym_id"].reshape(-1)
    hf = feat["has_frame"].reshape(-1).bool()
    lig = (torch.zeros_like(feat["asym_id"]).scatter_add(
        0, feat["atom_to_token_idx"], feat["is_ligand"].long()) > 0).reshape(-1)
    if hf.shape != lig.shape or hf.shape != asym.shape:
        raise ValueError(
            f"feat fields disagree on token count: asym_id {tuple(asym.shape)}, has_frame "
            f"{tuple(hf.shape)}, ligand {tuple(lig.shape)}")
    return hf & (~lig)


def ab_ag_pair_mask(feat: dict, antibody_asym_ids, antigen_asym_ids) -> Tensor:
    """[F,T] AB<->AG mask on iptm_pooled's row set, from MODEL chain IDs only.

    NO REFERENCE INPUT. The chain roles come from the model's own asym_id, never from an OST
    reference-to-model map, which can depend on the ground truth and is forbidden at inference.
    The row axis is the `live` subset (F rows) and the column axis is every token (T), matching
    how iptm_pooled's `tm` is built: rows are has_frame-and-not-ligand, columns are not filtered.
    """
    asym = feat["asym_id"].reshape(-1)
    live = live_row_mask(feat)
    ab_ids = torch.as_tensor(list(antibody_asym_ids), device=asym.device, dtype=asym.dtype)
    ag_ids = torch.as_tensor(list(antigen_asym_ids), device=asym.device, dtype=asym.dtype)
    if not ab_ids.numel() or not ag_ids.numel():
        raise ValueError("antibody and antigen chain ID lists must be nonempty")
    if bool(torch.isin(ab_ids, ag_ids).any()):
        raise ValueError("antibody and antigen chain IDs overlap")
    ab, ag = torch.isin(asym, ab_ids), torch.isin(asym, ag_ids)
    ab_rows, ag_rows = ab[live], ag[live]
    mask = (ab_rows[:, None] & ag[None, :]) | (ag_rows[:, None] & ab[None, :])
    if not bool(mask.any()):
        raise ValueError("no scored antibody-antigen pairs remain on this crop's live rows")
    return mask


def make_predict_one(
    model: Any,
    configs: Any,
    loss_fn: Any,
    *,
    iptm_pair: tuple,
    n_pair_tokens: int,
    antibody_asym_ids,
    antigen_asym_ids,
    pool: str = "max",
    autocast: bool = True,
) -> Callable[[PoseInput], PosePrediction]:
    """Build the `predict_one` closure for ONE scored interface.

    `iptm_pair` is that interface's scored asym pair and `n_pair_tokens` its UNCROPPED chain-pair
    token count -- the trainer gets the latter from `_uncrop_npt(pdb, pair)`. Both are properties
    of the interface, not of the pose, which is why they are bound here once rather than passed per
    pose: binding them per pose is how a group could silently mix two interfaces.

    pool defaults to "max" per the issue-#16 decision. "mean" is accepted so the control arm can
    be run on the kernel the cropped 7mo52r chain trains on without a second adapter.
    """
    if pool not in ("max", "mean"):
        raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")
    if not n_pair_tokens or int(n_pair_tokens) <= 0:
        raise ValueError(
            "n_pair_tokens must be the UNCROPPED scored chain-pair token count. Passing 0/None "
            "would let iptm_pooled fall back to the cropped count, which is the crop-dependent d0 "
            "the fixed-d0 path exists to prevent.")

    import rank_loss as RKL

    def predict_one(pose: PoseInput) -> PosePrediction:
        if not isinstance(pose, PoseInput):
            raise TypeError("predict_one expects a PoseInput carrying the trainer's live tensors")
        # THE HEAD, EXACTLY AS THE TRAINER CALLS IT. Same keyword names, same kernels from
        # configs, inplace_safe False, chunk_size None -- Protenix's own config says chunk_size
        # "should set to null for normal training" and --chunk-size 64 measured 5.8x slower here
        # for no memory saving.
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if autocast
               else torch.autocast(device_type="cuda", enabled=False))
        with ctx:
            plddt, pae, pde, resolved = model.run_confidence_head(
                input_feature_dict=pose.feat_head, s_inputs=pose.s_inputs,
                s_trunk=pose.s_trunk, z_trunk=pose.z_trunk, pair_mask=None,
                x_pred_coords=pose.coords,
                triangle_multiplicative=configs.triangle_multiplicative,
                triangle_attention=configs.triangle_attention,
                inplace_safe=False, chunk_size=None)
        # FLOAT, AS THE TRAINER DOES. pred["pae"] = pae.float() at phase_k_train.py:6946; the bins
        # are weighted and summed over N_token^2 and bf16 accumulation there is not what the
        # reference path does.
        pred = {"plddt": plddt.float(), "pae": pae.float(), "pde": pde.float(),
                "resolved": resolved.float(), "coordinate_mini": pose.coords}

        # THE ORIGINAL PAE CE, FROM THE ORIGINAL FUNCTION. `cum` is the differentiable scalar the
        # existing arms descend; under --pae-only it IS the PAE cross-entropy. Nothing here
        # touches its mask, denominator or weighting -- the whole point of calling it.
        cum, met = loss_fn.calculate_losses(pose.feat_cache, pred, pose.labels, mode="train")
        if not isinstance(cum, torch.Tensor) or cum.ndim != 0:
            raise ValueError(f"calculate_losses returned {type(cum).__name__} for the loss, "
                             f"expected a 0-dim tensor")
        if not cum.requires_grad:
            raise ValueError(
                "the PAE CE came back detached. The joint objective needs it live: it is half the "
                "gradient reaching the head, and a detached scalar would train the reranker alone.")

        # THE ipTM, FROM THE FULL-TOKEN-GRID LOGITS. iptm_pooled indexes frame rows itself via
        # has_frame, so it must receive the whole [.., T, T, 64] tensor -- the reranker's [F,T,64]
        # view is NOT a drop-in substitute for it, and handing it the sliced view would silently
        # score a different grid. Raised in review on issue #16.
        iptm = RKL.iptm_pooled(pred["pae"], pose.feat_head, configs, iptm_pair,
                               pool=pool, n_pair_tokens=int(n_pair_tokens))
        if iptm is None:
            raise ValueError(
                f"{pose.pose_id}: iptm_pooled found no scored chain pair for {iptm_pair} on this "
                f"crop. A pose with no ipTM has no residual baseline, so the group is invalid; the "
                f"crop pool guarantees every selected crop keeps a scorable cross-chain pair, so "
                f"this means the crop or the pair is wrong rather than merely unlucky.")

        # THE RERANKER'S VIEW: [F,T,64] on iptm_pooled's own live rows.
        live = live_row_mask(pose.feat_head)
        logits = pred["pae"]
        while logits.ndim > 3:
            if logits.shape[0] != 1:
                raise ValueError(
                    f"{pose.pose_id}: the head returned {tuple(logits.shape)}; this adapter takes "
                    f"ONE pose per forward, so a leading axis > 1 means several samples were "
                    f"batched and the group would attribute one pose's logits to another.")
            logits = logits.squeeze(0)
        if logits.ndim != 3 or logits.shape[-1] != 64:
            raise ValueError(f"{pose.pose_id}: expected [T,T,64] PAE logits, got "
                             f"{tuple(logits.shape)}")
        rows = logits[live]
        mask = ab_ag_pair_mask(pose.feat_head, antibody_asym_ids, antigen_asym_ids)
        if mask.shape != rows.shape[:2]:
            raise ValueError(f"{pose.pose_id}: mask {tuple(mask.shape)} does not match the live "
                             f"logit grid {tuple(rows.shape[:2])}")
        return PosePrediction(pae_logits=rows, pair_mask=mask, iptm=iptm, pae_ce=cum)

    return predict_one
