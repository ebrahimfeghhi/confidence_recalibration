#!/usr/bin/env python
"""Check pose_direction.direction_tensor on real cached data. CPU only, no model, no GPU.

Five properties, each of which would be a real bug if it failed:

1. RIGID-MOTION INVARIANCE. Rotate and translate the whole structure and the output must not
   change. This is the symmetry PAE has -- the label is computed in a frame derived from the
   structure itself -- so a pose feature that moved under a global rotation would be feeding the
   head an arbitrary reference the label does not share.
2. UNIT NORM. Every non-zero row must have length 1, because the whole point of normalising is to
   bound the input: an un-normalised component of +/-30 makes the first optimiser update ~30x too
   large even from a zero-init weight.
3. THE SELF-ENTRY IS ZERO. The target atoms are the PAE representative atoms -- the token CENTRE
   atom, CA for standard amino acids -- and the frame origin is the middle frame atom, also CA, so
   [i][i] is the zero vector. This check earned its place: the first version fed the DISTOGRAM
   representative atoms, which are CB, and the self-entry came out at length 1.0 on all three
   structures. That is the CA->CB direction, a near-constant local vector with no pose
   information, and it also meant the feature described a different atom than the PAE label does.
4. ASYMMETRY. out[i][j] must differ from out[j][i]. If it did not, the feature would carry no
   more directional information than cdist already does and C would be pointless.
5. ZERO-INIT IS EXACTLY ZERO. A zero-initialised LinearNoBias(3 -> c_z) must return exactly 0.0
   on this input, not something small -- "step 0 is bit-identical to the trained head" depends on
   it being exact.

Run: /home/ebrahim/Protenix/.venv/bin/python -I phases/phase_k_frame_direction_check.py
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/ebrahim/Protenix")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rand_rigid(seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=torch.float64))[0]
    if torch.det(a) < 0:                     # a ROTATION, not a reflection
        a[:, 0] = -a[:, 0]
    return a.float(), torch.randn(3, generator=g).float() * 25.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=os.path.join(ROOT, "training/fold"))
    ap.add_argument("--pdbs", default="10fj,10gh,10op")
    ap.add_argument("--unit", default="free_500000")
    ap.add_argument("--tol", type=float, default=2e-4,
                    help="invariance tolerance. The basis is built from normalised cross products "
                         "in float32, so a rotation is not expected to be bit-exact.")
    a = ap.parse_args()

    from pose_direction import EPS_LEN, direction_tensor
    from protenix.model.modules.primitives import LinearNoBias

    # A MISSING FILE IS A FAILURE, NOT A SKIP, and so is checking nothing. The first version
    # printed "SKIP" per structure and then "all checks passed" on an empty failure list -- review
    # reproduced all three structures skipping and the script still exiting 0. A checker that
    # cannot tell "the geometry is correct" from "I did not look" is worse than no checker.
    fails, checked, missing = [], 0, []
    wanted = [p for p in a.pdbs.split(",") if p]
    if not wanted:
        raise SystemExit("STOP: --pdbs is empty, so there is nothing to check.")
    for pdb in wanted:
        lp = os.path.join(a.fold, "labels", pdb, a.unit + ".npz")
        tp = os.path.join(a.fold, "trunks", pdb, a.unit + ".npz")
        if not (os.path.exists(lp) and os.path.exists(tp)):
            gone = [q for q in (lp, tp) if not os.path.exists(q)]
            print(f"{pdb}: MISSING {', '.join(gone)}")
            missing.append(pdb)
            continue
        checked += 1
        L = np.load(lp, allow_pickle=True)
        T = np.load(tp, allow_pickle=True)
        fai = torch.as_tensor(L["lf_frame_atom_index"]).long()
        hf = torch.as_tensor(L["lf_has_frame"]).bool()
        # THE PAE REPRESENTATIVE ATOMS, which is what the label is built on -- not
        # ci_feat_distogram_rep_atom_mask, the CB subset the head's cdist uses.
        pae_mask = torch.as_tensor(L["lf_pae_rep_atom_mask"]).bool()
        full = torch.as_tensor(T["ci_coordinate"][0]).float()
        rep = full[pae_mask]
        n_tok = rep.shape[0]

        out = direction_tensor(rep, full, fai, hf)

        # 1. rigid-motion invariance
        R, t = rand_rigid(7)
        full2 = full @ R.T + t
        out2 = direction_tensor(full2[pae_mask], full2, fai, hf)
        dmax = float((out - out2).abs().max())
        ok1 = dmax <= a.tol

        # 2. unit norm on every non-zero row
        ln = torch.linalg.vector_norm(out, dim=-1)
        nz = ln > EPS_LEN
        worst = float((ln[nz] - 1.0).abs().max()) if bool(nz.any()) else 0.0
        ok2 = worst <= 1e-4

        # 3. the self-entry is zero
        diag = float(torch.linalg.vector_norm(out[torch.arange(n_tok), torch.arange(n_tok)],
                                              dim=-1).max())
        ok3 = diag <= EPS_LEN

        # 4. asymmetry, over pairs where BOTH directions exist
        both = nz & nz.T
        asym = float((out - out.transpose(0, 1)).abs()[both].mean()) if bool(both.any()) else 0.0
        ok4 = asym > 1e-3

        # 5. zero-init is exactly zero
        proj = LinearNoBias(in_features=3, out_features=128)
        with torch.no_grad():
            proj.weight.zero_()
            z = proj(out)
        zmax = float(z.abs().max())
        ok5 = zmax == 0.0

        print(f"{pdb}: {n_tok} tokens, {int(hf.sum())} framed, "
              f"{float(nz.float().mean()):.1%} of pairs non-zero")
        print(f"  1 rigid invariance   max|d| {dmax:.2e} (tol {a.tol:.0e})   {'ok' if ok1 else 'FAIL'}")
        print(f"  2 unit norm          max|n-1| {worst:.2e}                  {'ok' if ok2 else 'FAIL'}")
        print(f"  3 self-entry zero    max|diag| {diag:.2e}                  {'ok' if ok3 else 'FAIL'}")
        print(f"  4 asymmetry          mean|out-out^T| {asym:.4f}            {'ok' if ok4 else 'FAIL'}")
        print(f"  5 zero-init exact    max|proj(out)| {zmax:.1e}             {'ok' if ok5 else 'FAIL'}")
        for n, ok in ((1, ok1), (2, ok2), (3, ok3), (4, ok4), (5, ok5)):
            if not ok:
                fails.append(f"{pdb}:{n}")

    if missing:
        raise SystemExit(f"\nSTOP: {len(missing)} requested structure(s) had no cached "
                         f"{a.unit}: {', '.join(missing)}. Point --fold/--pdbs/--unit at data "
                         f"that exists rather than reporting a pass over what was found.")
    if not checked:
        raise SystemExit("\nSTOP: checked 0 structures, so nothing was verified.")
    if fails:
        raise SystemExit(f"\nFAILED: {', '.join(fails)}")
    print(f"\nall checks passed on {checked} structure(s)")


if __name__ == "__main__":
    main()
