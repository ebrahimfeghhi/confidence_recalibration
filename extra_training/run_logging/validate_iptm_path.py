#!/usr/bin/env python
"""DOES OUR AGGREGATION REPRODUCE PROTENIX'S OWN ipTM? The check that settles which iTM store is right.

WHY. run_logging/true_pae_iptm.py and training/fold/itm_labels_853 disagree on 52% of 71,220
shared rows (mean |diff| 0.037, max 0.92), concentrated in 3+-chain structures and only 9.5%
explained by an interface swap -- so the two are computing different quantities, and nothing
measured so far says which is correct. Both are "true error -> TM transform -> pool", so the
disagreement is in the chain mapping or the pooling, NOT in the error.

THE TEST. Those two steps are shared with the PREDICTED score, and the predicted score has an
independent ground truth: the `iptm` column of results_with_trunks.csv, written by the fold itself
from Protenix's own `compute_full_data_and_summary`. So run the confidence head on the cached
pose, take the head's PAE logits, push them through OUR aggregation
(rank_loss.iptm -> sample_confidence.calculate_chain_based_ptm, with rank_loss.parse_iface's
chain join), and compare to that stored column.

  reproduces it   -> our chain mapping and pooling are Protenix's, and itm_labels_853 is the
                     store that is wrong
  does not        -> ours is wrong, and every true-PAE number computed through it is suspect

phase_k_eval is documented as reproducing the stored ipTM to 2e-3 through this same path; this
asserts it on THIS code rather than inheriting the claim.

THE POSE INDEX IS NOT THE SAMPLE INDEX -- the label npz's `sample_index` permutation is
non-identity in 99.3% of training label files, so the stored row is looked up by the mapped
sample id, not by position.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles_all.csv")
    ap.add_argument("--units", type=int, default=120)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--ckpt-dir", default="/home/ebrahim/checkpoint")
    ap.add_argument("--model-name", default="protenix-v2")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    os.environ.setdefault("LAYERNORM_TYPE", "openfold")
    import numpy as np
    import torch
    import phase_j_rescore_cache as RC
    import rank_loss as RKL

    roles = RKL.load_roles(a.roles)
    want = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(a.results) as fh:
        for r in csv.DictReader(fh):
            if r.get("has_trunk") != "1" or r.get("arm") != "free":
                continue
            p = RKL.parse_iface(r["iface"], roles)
            if p is None:
                continue
            try:
                want[(p[0], int(r["seed"]))][r["iface"]][int(r["sample"])] = float(r["iptm"])
            except (ValueError, KeyError, TypeError):
                continue

    dev = torch.device("cuda")
    model, cfg = RC.build_model(a.ckpt_dir, a.model_name, dev)
    units = sorted(want)[:int(a.units)]
    print(f"{len(units)} unit(s)")

    rows = []
    for ui, (pdb, seed) in enumerate(units, 1):
        tp = os.path.join(a.fold, "trunks", pdb, f"free_{seed}.npz")
        lp = os.path.join(a.fold, "labels", pdb, f"free_{seed}.npz")
        if not (os.path.exists(tp) and os.path.exists(lp)):
            continue
        feat, s_inputs, s_trunk, zt, poses, n_cycle = RC.load_unit(tp, dev)
        si = np.asarray(np.load(lp, allow_pickle=True)["sample_index"]).reshape(-1)
        for i in range(min(int(poses.shape[0]), si.shape[0])):
            samp = int(si[i])
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _pl, pae, _pd, _rs = model.run_confidence_head(
                    input_feature_dict=feat, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=zt,
                    pair_mask=None, x_pred_coords=poses[i:i + 1],
                    triangle_multiplicative=cfg.triangle_multiplicative,
                    triangle_attention=cfg.triangle_attention,
                    inplace_safe=False, chunk_size=None)
                logits = pae.detach().float()
                while logits.dim() > 4:
                    logits = logits[0]
                for iface, bys in want[(pdb, seed)].items():
                    if samp not in bys:
                        continue
                    pr = RKL.parse_iface(iface, roles)
                    sc = RKL.iptm(logits, feat, cfg, tuple(int(v) for v in pr[1]))
                    if sc is None:
                        continue
                    rows.append({"pdb": pdb, "iface": iface, "seed": seed, "sample": samp,
                                 "stored": bys[samp], "ours": float(sc)})
            del pae, logits
            torch.cuda.empty_cache()
        del feat, s_inputs, s_trunk, zt, poses
        torch.cuda.empty_cache()
        if ui % 20 == 0:
            print(f"  [{ui}/{len(units)}] {len(rows)} row(s)", flush=True)

    if not rows:
        raise SystemExit("STOP: nothing compared")
    d = np.array([r["ours"] - r["stored"] for r in rows])
    print(f"\nrows {len(rows)}  over {len({r['pdb'] for r in rows})} structure(s), "
          f"{len({r['iface'] for r in rows})} interface(s)")
    print(f"  max |diff| {np.abs(d).max():.6f}   mean |diff| {np.abs(d).mean():.6f}   "
          f"median {np.median(np.abs(d)):.6f}")
    print(f"  within {a.tol}: {(np.abs(d) < a.tol).mean():.4f}")
    nch = collections.Counter()
    with open(a.roles) as fh:
        for r in csv.DictReader(fh):
            nch[r["pdb"]] += 1
    for lab, sel in (("2 chains", lambda r: nch[r["pdb"]] <= 2),
                     ("3+ chains", lambda r: nch[r["pdb"]] > 2)):
        v = np.array([abs(r["ours"] - r["stored"]) for r in rows if sel(r)])
        if len(v):
            print(f"  {lab:10s} n={len(v):5d}  within {a.tol}: {(v < a.tol).mean():.4f}  "
                  f"mean |diff| {v.mean():.6f}")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"  rows -> {a.out}")


if __name__ == "__main__":
    main()
