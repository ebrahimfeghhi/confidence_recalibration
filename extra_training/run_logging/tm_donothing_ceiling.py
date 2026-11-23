#!/usr/bin/env python
"""THE DO-NOTHING CEILING OF THE CENTRED TM LOSS. Label-only, no model.

THE ALGEBRA. A head that predicts the SAME pair score for every pose of a structure has
s_pred(i,j) constant over poses, so

    e      = s_pred - s_true
    e_mean = s_pred - mean_pose(s_true)
    c      = e - e_mean = -(s_true - mean_pose(s_true))

and the loss mean(c^2) collapses to the VARIANCE of the TRUE per-pair TM score across the
structure's poses. No prediction appears in it. So this number is the loss a head gets for free by
carrying no pose-specific information at all, and the trained arm's number is only meaningful
against it: at the ceiling the head's scores say nothing about which pose is which.

NO FORWARD PASS IS NEEDED. s_true = f(true aligned error) and the true aligned error comes from
PAELoss.calculate_label, which aligns the POSE coordinates (in the trunk) to the LABEL coordinates
-- arithmetic on two coordinate arrays, not a network. It runs on the CPU.

AVERAGED THE WAY THE LOSS AVERAGES, which is the whole point of doing it this way rather than
quoting a generic variance: the same per-chain-pair blocks, the same d0 per pair from that pair's
token count, the same "pairs valid in EVERY pose" mask, and the same mean over (pose, pair).
"""
import argparse
import collections
import csv
import glob
import os
import random
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
from protenix.model.loss import PAELoss                             # noqa: E402
import label_index as LI                                            # noqa: E402
import tm_centered_loss as TMC                                      # noqa: E402


def roles_map(path):
    d = {}
    for r in csv.DictReader(open(path)):
        d.setdefault(r["pdb"], {})[r["ref_chain"]] = r["model_chain"]
    return d


def scored_pairs(pool, roles):
    out = collections.defaultdict(set)
    for r in csv.DictReader(open(pool)):
        m1 = roles.get(r["pdb"], {}).get(r["chain1"])
        m2 = roles.get(r["pdb"], {}).get(r["chain2"])
        if m1 is None or m2 is None:
            continue
        a1, a2 = ord(m1) - ord("A"), ord(m2) - ord("A")
        out[r["pdb"]].add((min(a1, a2), max(a1, a2)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="training/slice280/pool.csv")
    ap.add_argument("--roles", default="training/slice280/chain_roles.csv")
    ap.add_argument("--fold", default="training/fold")
    ap.add_argument("--structures", type=int, default=20,
                    help="how many structures to sample; 0 means all")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pae = PAELoss(min_bin=0.0, max_bin=32.0, no_bins=64)
    pairs = scored_pairs(a.pool, roles_map(a.roles))
    pdbs = sorted(pairs)
    if a.structures and a.structures < len(pdbs):
        random.seed(a.seed)
        pdbs = sorted(random.sample(pdbs, a.structures))

    tot_sq, tot_n = 0.0, 0
    per_struct = []
    for p in pdbs:
        trunks = sorted(glob.glob(f"{a.fold}/trunks/{p}/free_*.npz"))
        if not trunks:
            continue
        blocks, S = None, []
        for tp in trunks:
            sd = int(re.search(r"free_(\d+)\.npz$", tp).group(1))
            lp = f"{a.fold}/labels/{p}/free_{sd}.npz"
            if not os.path.exists(lp):
                continue
            lab = np.load(lp, allow_pickle=True)
            t = np.load(tp)
            x = torch.as_tensor(t["ci_coordinate"], dtype=torch.float32)
            asym = torch.as_tensor(np.asarray(t["ci_feat_asym_id"]))
            hf = torch.as_tensor(np.asarray(lab["lf_has_frame"]))
            if blocks is None:
                aa = asym.reshape(-1).long()
                fa = aa[hf.reshape(-1).bool()]
                blocks = []
                for xx, yy in sorted(pairs[p]):
                    blk = (((fa[:, None] == xx) & (aa[None, :] == yy))
                           | ((fa[:, None] == yy) & (aa[None, :] == xx)))
                    if bool(blk.any()):
                        blocks.append((blk, int(((aa == xx) | (aa == yy)).sum())))
                if not blocks:
                    break
            row = LI.pae_target_row(lab)
            for i in range(x.shape[0]):
                with torch.no_grad():
                    sq, _b, _pm = pae.calculate_label(
                        pred_coordinate=x[i:i + 1],
                        true_coordinate=torch.as_tensor(lab["label_coordinate"][row],
                                                        dtype=torch.float32),
                        coordinate_mask=torch.as_tensor(
                            np.asarray(lab["label_coordinate_mask"][row]).astype(np.float32)),
                        rep_atom_mask=torch.as_tensor(np.asarray(lab["lf_pae_rep_atom_mask"])),
                        frame_atom_index=torch.as_tensor(np.asarray(lab["lf_frame_atom_index"])),
                        has_frame=hf)
                sqf = sq.float().reshape(sq.shape[-2], sq.shape[-1])
                acc = torch.zeros_like(sqf)
                m = torch.zeros_like(sqf)
                for blk, npt in blocks:
                    d0 = TMC.d0_for_pair(npt)
                    acc = acc + (1.0 / (1.0 + sqf / (float(d0) ** 2))) * blk.float()
                    m = m + blk.float()
                S.append((acc, m))
        if not S or blocks is None:
            continue
        st = torch.stack([s for s, _ in S])            # [n_pose, N_frame, N_token]
        mk = torch.stack([m for _, m in S])
        keep = (mk > 0).all(dim=0).float()             # pairs valid in EVERY pose
        if float(keep.sum()) == 0:
            continue
        var = ((st - st.mean(dim=0, keepdim=True)) ** 2) * keep
        tot_sq += float(var.sum())
        tot_n += int(keep.sum()) * st.shape[0]
        per_struct.append((p, float(var.sum() / (keep.sum() * st.shape[0])), st.shape[0]))

    print(f"  {len(per_struct)} structure(s), {tot_n:,} (pose, pair) term(s)")
    print(f"  DO-NOTHING CEILING (variance of the true per-pair TM score across poses):")
    print(f"     {tot_sq / max(1, tot_n):.6f}")
    v = sorted(x[1] for x in per_struct)
    if v:
        print(f"     per structure: median {v[len(v)//2]:.6f}  min {v[0]:.6f}  max {v[-1]:.6f}")
    print(f"  poses per structure: {sorted({x[2] for x in per_struct})}")


if __name__ == "__main__":
    main()
