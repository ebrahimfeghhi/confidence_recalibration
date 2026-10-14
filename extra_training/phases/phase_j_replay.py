#!/usr/bin/env python
"""PHASE J -- does a CACHED unit give the FT3 loss the uncached path would give?

WHY THIS GATES TRAINING. Cached training replaces two expensive halves of an FT3 step: the trunk
forward (247.2M frozen parameters) and the mini-rollout. If the cached halves do not reproduce what
the live code computes, the finetune optimises something other than FT3 and no comparison afterwards
means anything. This file is the gate: it refuses, loudly, rather than reporting a number.

WHAT IS HELD FIXED, AND WHY THAT IS THE RIGHT TEST. The POSE is taken from the cache on both sides.
It has to be: our poses are full inference samples and the fold is not bit-reproducible on this
hardware, so no live run can regenerate one. The trunk tensors likewise come from inference
featurisation, which is deliberate -- the head is being calibrated on the distribution it is deployed
on. So the question the replay can actually answer, and the one that matters, is:

    given the same trunk and the same pose, is the LABEL that phase_j_build_cache stored exactly the
    label the live code derives, and does the loss computed from it agree term by term?

THE REVIEWER'S CAVEAT, which stands: "a cached-versus-uncached loss match can pass even if both paths
receive the same incorrect labels." That is true of check 1 and check 2 below, so this file does not
rely on them alone. phase_j_build_cache.py carries the independent checks -- the label's atoms must be
the job cif's atoms, and the per-pose label must be a permutation of its siblings -- and neither goes
through the loss. Check 3 here adds the other thing training depends on and the loss cannot see: that
the gradient reaches the confidence head and nothing else.

WHAT IS COMPARED
  1. LABEL: the stored permuted coordinates and mask against a live permutation of the same pose.
  2. LOSS: every FT3 term (plddt, pde, pae, resolved) and the total, from the stored label against
     the live one, on the same head output.
  3. GRADIENT: backward on the cached loss, then the number of parameters outside the confidence
     head carrying a non-zero gradient, which must be zero.
"""
import argparse
import copy
import functools
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(ROOT, "phases"))
print = functools.partial(print, flush=True)

TERMS = ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss", "loss")


def head_prefixes(model):
    """The parameter names FT3 is allowed to move.

    protenix.py's train_confidence_only puts the trunk in eval mode and asserts the diffusion and
    distogram weights are zero; what it does NOT do is tell you which parameters the optimiser may
    touch. The confidence head is its own module, so its name prefix is the honest definition, and
    the check below is against the model's own module tree rather than a list written out by hand.
    """
    names = [n for n, _m in model.named_modules() if n.endswith("confidence_head")]
    if not names:
        raise SystemExit("STOP: the model has no module named *confidence_head; the gradient check "
                         "cannot be defined")
    return tuple(n + "." for n in names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/train_fold")
    ap.add_argument("--labels", default=f"{ROOT}/train_fold/labels")
    ap.add_argument("--unit", default=None, help="<pdb>/<arm>_<seed>; default the first labelled")
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--indices", default=f"{ROOT}/training/prep/job_indices.csv")
    ap.add_argument("--bio-dir", default=f"{ROOT}/training/prep/job_bioassembly")
    ap.add_argument("--mmcif-dir", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa")
    ap.add_argument("--seq-index", default=f"{ROOT}/building_msa/work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--crop-size", type=int, default=3400)
    ap.add_argument("--coord-tol", type=float, default=1e-4,
                    help="max |stored - live| label coordinate, A")
    ap.add_argument("--loss-tol", type=float, default=1e-5,
                    help="max |stored - live| on any loss term")
    ap.add_argument("--pose", type=int, default=-1, help="-1 = every pose in the unit")
    ap.add_argument("--alpha-pae", type=float, default=1.0,
                    help="FT3's value; the shipped default is 0, which cannot train ipTM")
    a = ap.parse_args()

    import torch
    # EXPLICIT, because the submodule is not imported by `import torch`. With grad enabled the
    # confidence head's pairformer takes its checkpointing path (pairformer.py:339 ->
    # model/utils.py:842), which reaches for torch.utils.checkpoint and raises AttributeError if
    # nothing has imported it. The rescorer never hit this: it runs under no_grad.
    import torch.utils.checkpoint                                                 # noqa: F401
    import phase_j_dataset as DS
    import phase_j_build_cache as BC
    import phase_j_rescore_cache as RC
    from protenix.data.pipeline.dataset import get_datasets
    from protenix.model.loss import ProtenixLoss
    from protenix.utils.permutation.permutation import SymmetricPermutation

    unit = a.unit
    if unit is None:
        found = sorted(glob.glob(f"{a.labels}/*/*.npz"))
        if not found:
            raise SystemExit(f"STOP: no built labels under {a.labels}; run phase_j_build_cache.py")
        unit = f"{os.path.basename(os.path.dirname(found[0]))}/{os.path.basename(found[0])[:-4]}"
    pdb, arm_seed = unit.split("/")
    lab_p = os.path.join(a.labels, pdb, f"{arm_seed}.npz")
    trunk_p = os.path.join(a.fold, "trunks", pdb, f"{arm_seed}.npz")
    for p in (lab_p, trunk_p):
        if not os.path.exists(p):
            raise SystemExit(f"STOP: no {p}")
    print(f"replaying {unit}")

    dev = torch.device("cuda")
    # THE SAME MODEL ASSEMBLY THE RESCORER USES, imported rather than copied so the two cannot drift.
    model, configs = RC.build_model(a.ckpt_dir, a.model_name, dev)
    # FT3's own switch. It is what makes the loss skip diffusion and distogram, so the replay must
    # set it or it would be comparing a different objective.
    configs.train_confidence_only = True
    # alpha_pae = 1, WHICH THE SHIPPED DEFAULT IS NOT. configs_base.py:402 reads
    # `"alpha_pae": 0.0,  # or 1 in finetuning stage 3` -- so with the defaults the PAE term is
    # multiplied by zero, and PAE is what ipTM is computed from (sample_confidence.py builds ipTM out
    # of the PAE logits). Training FT3 at the shipped default would leave the one number this project
    # is judged on untouched while pLDDT, PDE and resolved still moved and the total loss still fell.
    # Measured on 12qa: total 0.000356 = 1e-4 x (2.371 + 1.167 + 0.026), the PAE term absent.
    configs.loss.weight.alpha_pae = float(a.alpha_pae)
    loss_fn = ProtenixLoss(configs)
    w = {k: loss_fn.loss_weight[k] for k in ("plddt_loss", "pde_loss", "pae_loss", "resolved_loss")}
    print("  loss weights: " + "  ".join(f"{k.replace('_loss','')} {v:g}" for k, v in w.items()))
    zero = [k for k, v in w.items() if not v]
    if zero:
        raise SystemExit(f"STOP: {', '.join(zero)} carries weight 0, so that term cannot train. "
                         "alpha_confidence and alpha_pae are in configs_base.py:401-403; FT3 wants "
                         "alpha_pae = 1 (their own comment).")

    cfg = DS.build_configs(a.crop_size, a.indices, a.bio_dir, a.mmcif_dir, a.msa_root,
                           a.seq_index, a.msa_root)
    cfg.train_confidence_only = True
    err = os.path.join(ROOT, "train_prep", "dataset_errors")
    train, _ = get_datasets(cfg, err)
    byp = BC.merged_lookup(train)
    if pdb not in byp:
        raise SystemExit(f"STOP: {pdb} has no dataset datapoint")
    item = train[byp[pdb][0]]
    feat_ds, label_dict, label_full = (item["input_feature_dict"], item["label_dict"],
                                       item["label_full_dict"])
    perm = SymmetricPermutation(cfg, error_dir=err)

    lab = np.load(lab_p, allow_pickle=True)
    z, feat_npz, poses = BC.load_trunk(trunk_p)
    BC.check_alignment(feat_npz, feat_ds, pdb)
    print(f"  {poses.shape[0]} pose(s), {poses.shape[1]} atoms, sample indices "
          f"{[int(x) for x in lab['sample_index']]}")

    def to_dev(d):
        return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}

    # THE FEATURE DICT THE LOSS READS. The head-side arrays come from the npz (they are the ones the
    # pose was scored with, and check_alignment has just proved they equal the dataset's); the
    # loss-side arrays come from the dataset item, which is where the label came from.
    feat = to_dev({k: (v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v)))
                   for k, v in feat_ds.items() if torch.is_tensor(v) or
                   isinstance(v, (np.ndarray, int, float))})
    for k, v in feat_ds.items():
        if k not in feat:
            feat[k] = v                       # lists the permutation needs, e.g. atom_perm_list

    feat_head, s_inputs, s_trunk, zt, _x, nc = RC.load_unit(trunk_p, dev)
    poses_t = torch.as_tensor(poses, dtype=torch.float32, device=dev)

    idxs = range(poses.shape[0]) if a.pose < 0 else [a.pose]
    worst_c = worst_m = 0.0
    worst_term, worst_name = 0.0, ""
    for i in idxs:
        x = poses_t[i][None]                   # [1, N_atom, 3], as the loss expects N_sample=1

        # ---- 1. LABEL: stored against a live permutation of the same pose
        ld_live = copy.deepcopy(label_dict)
        ld_live, _ = perm.permute_label_to_match_mini_rollout(
            x.cpu(), feat_ds, ld_live, copy.deepcopy(label_full))
        c_live = ld_live["coordinate"].detach().cpu().numpy().astype(np.float32)
        m_live = ld_live["coordinate_mask"].detach().cpu().numpy().astype(np.uint8)
        c_cache, m_cache = lab["label_coordinate"][i], lab["label_coordinate_mask"][i]
        if c_live.shape != c_cache.shape:
            raise SystemExit(f"STOP: pose {i} label shape {c_cache.shape} cached, {c_live.shape} live")
        dm = float(np.abs(m_live.astype(int) - m_cache.astype(int)).max())
        msk = m_cache.astype(bool)
        dc = float(np.abs(c_live[msk] - c_cache[msk]).max())
        worst_c, worst_m = max(worst_c, dc), max(worst_m, dm)

        # ---- 2. LOSS: the same head output scored against each label
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            plddt, pae, pde, resolved = model.run_confidence_head(
                input_feature_dict=feat_head, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=zt,
                pair_mask=None, x_pred_coords=x,
                triangle_multiplicative=configs.triangle_multiplicative,
                triangle_attention=configs.triangle_attention, inplace_safe=False, chunk_size=None)
        pred = {"plddt": plddt.float(), "pae": pae.float(), "pde": pde.float(),
                "resolved": resolved.float(), "coordinate_mini": x}
        got = {}
        for tag, cm in (("cache", (c_cache, m_cache)), ("live", (c_live, m_live))):
            ld = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ld_live.items()}
            ld["coordinate"] = torch.as_tensor(cm[0], dtype=torch.float32, device=dev)
            ld["coordinate_mask"] = torch.as_tensor(cm[1].astype(np.float32), device=dev)
            _cum, met = loss_fn.calculate_losses(feat, pred, ld, mode="train")
            got[tag] = {k: float(met[k]) for k in TERMS if k in met}
        for k in got["cache"]:
            d = abs(got["cache"][k] - got["live"][k])
            if d > worst_term:
                worst_term, worst_name = d, k
        print(f"  pose {i} (sample {int(lab['sample_index'][i])}): "
              f"label max|dx| {dc:.2e} A, mask diff {dm:.0f} | "
              + "  ".join(f"{k.replace('_loss','')} {got['cache'][k]:.6f}" for k in TERMS
                          if k in got["cache"]))

    # ---- 3. GRADIENT: the cached loss must move the confidence head and nothing else
    pref = head_prefixes(model)
    for p in model.parameters():
        p.requires_grad_(True)
        p.grad = None
    x = poses_t[0][None]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        plddt, pae, pde, resolved = model.run_confidence_head(
            input_feature_dict=feat_head, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=zt,
            pair_mask=None, x_pred_coords=x,
            triangle_multiplicative=configs.triangle_multiplicative,
            triangle_attention=configs.triangle_attention, inplace_safe=False, chunk_size=None)
    ld = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in ld_live.items()}
    ld["coordinate"] = torch.as_tensor(lab["label_coordinate"][0], dtype=torch.float32, device=dev)
    ld["coordinate_mask"] = torch.as_tensor(lab["label_coordinate_mask"][0].astype(np.float32),
                                            device=dev)
    cum, _met = loss_fn.calculate_losses(
        feat, {"plddt": plddt.float(), "pae": pae.float(), "pde": pde.float(),
               "resolved": resolved.float(), "coordinate_mini": x}, ld, mode="train")
    cum.backward()
    inside = outside = 0
    outside_names = []
    for n, p in model.named_parameters():
        if p.grad is None or not bool(p.grad.abs().sum() > 0):
            continue
        if n.startswith(pref):
            inside += 1
        else:
            outside += 1
            if len(outside_names) < 5:
                outside_names.append(n)
    nhead = sum(p.numel() for n, p in model.named_parameters() if n.startswith(pref))
    ntot = sum(p.numel() for p in model.parameters())
    print(f"\ngradient: {inside} confidence-head tensor(s) with a non-zero grad, {outside} outside "
          f"it{(' e.g. ' + ', '.join(outside_names)) if outside_names else ''}")
    print(f"confidence head is {nhead/1e6:.1f}M of {ntot/1e6:.1f}M parameters "
          f"({100*nhead/ntot:.1f}%)")
    print(f"\nlabel: max |stored - live| coordinate {worst_c:.2e} A, mask differences {worst_m:.0f}")
    print(f"loss:  max |stored - live| over {', '.join(TERMS)} = {worst_term:.2e} "
          f"({worst_name or 'none'})")

    fail = []
    if worst_c > a.coord_tol:
        fail.append(f"the stored label differs from the live one by {worst_c:.2e} A "
                    f"(tolerance {a.coord_tol})")
    if worst_m > 0:
        fail.append("the stored coordinate mask differs from the live one")
    if worst_term > a.loss_tol:
        fail.append(f"the FT3 loss differs by {worst_term:.2e} on {worst_name} "
                    f"(tolerance {a.loss_tol})")
    if inside == 0:
        fail.append("no confidence-head parameter received a gradient")
    if outside:
        fail.append(f"{outside} parameter(s) outside the confidence head received a gradient")
    if fail:
        raise SystemExit("STOP: " + "; ".join(fail) + ". The cache does not stand in for the "
                         "uncached FT3 path, so training on it would not be training FT3.")
    print("\nPASS: the cached label is the live label, the FT3 loss agrees term by term, and the "
          "gradient reaches the confidence head only.")
    print("NOTE: this does not establish that the LABEL IS CORRECT -- both paths read the same "
          "stored structure. That is what phase_j_build_cache.py's two independent checks are for: "
          "every resolved label atom must be an atom of the job cif, and each pose's label must be "
          "a permutation of its siblings'.")


if __name__ == "__main__":
    main()
