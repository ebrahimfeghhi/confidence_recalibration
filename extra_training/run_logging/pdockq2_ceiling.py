#!/usr/bin/env python
"""pDockQ2's ceiling: what it would select with PERFECT PAE **and** perfect pLDDT.

THE QUESTION. itm_ceiling.py asks how much oracle headroom a perfect PAE recovers through the
ipTM formula (58.6% max-pooled, 67.7% mean-pooled on the 738 winnable interfaces). pDockQ2 is a
different formula over the same kind of inputs, so it has its own ceiling, and comparing the two
ceilings separates "our predictions are bad" from "our scoring function is lossy".

THE FORMULA, taken from the ipSAE reference implementation rather than from the paper text
(Zhu, Shenoy, Kundrotas & Elofsson 2023; https://github.com/y1zhou/IPSAE src/ipsae/ipsae.py):

    interface  = token pairs of the two chains within PDOCKQ_CUTOFF (8 A) in the PREDICTED structure
    mean_ptm   = mean over those pairs of 1 / (1 + (PAE/10)^2)          # d0 FIXED at 10 A
    mean_plddt = mean pLDDT over the interface tokens of BOTH chains    # 0-100 scale
    x          = mean_plddt * mean_ptm
    pDockQ2    = 1.31 / (1 + exp(-0.075 * (x - 84.733))) + 0.005

THREE STRUCTURAL DIFFERENCES FROM ipTM, each of which could be the one that matters:
  - d0 is FIXED at 10 A; ipTM's is 1.24*(max(N,19)-15)^(1/3) - 1.8 and grows with interface size.
  - it pools by MEAN over interface pairs; ipTM takes the MAX over anchor rows.
  - it scores ONLY pairs within 8 A -- actual contacts -- where ipTM scores every cross-chain pair.
  - and it uses pLDDT at all, which ipTM does not.

WHAT "CEILING" MEANS HERE, and it is a stronger assumption than itm_ceiling's. pDockQ2 reads two
predicted quantities, so this substitutes the truth for BOTH:
  - PAE -> the true aligned error, from PAELoss.calculate_label's one-hot, read as bin centres.
  - pLDDT -> the true lDDT, from Protenix's own calculate_atom_bespoke_lddt, the same function
    PLDDTLoss uses to build its training labels, normalised by its weight and scaled x100.
So this is the score pDockQ2 would report if the model's two confidence heads were both perfect.

THE 8 A CUTOFF USES THE PREDICTED COORDINATES, not the true ones, exactly as pDockQ2 does at
inference -- the predicted structure is what you have. Only PAE and pLDDT are idealised.

TWO APPROXIMATIONS, stated rather than buried:
  - pDockQ2 uses C-beta pLDDT; this uses Protenix's representative atom per token, which for a
    protein residue is not C-beta. The interface SET it induces therefore differs slightly.
  - true PAE is read as the centre of its 0.5 A bin, because that is the resolution the label
    has. The continuous error would be a marginally different number.

TOKENS WITHOUT TRUE COORDINATES ARE DROPPED, not imputed: calculate_atom_bespoke_lddt is only
defined where the truth exists, and a token with no truth has no lDDT to substitute. On 10fj that
is 13 of 625 tokens. Dropping them shrinks the interface slightly; imputing a value would invent
one.

CPU ONLY, NO GPU, NO MODEL. Deterministic: fixed seeds 500000-500009, sorted iteration, stride
sharding. Candidates and DockQ come from the eval's own per-pose rows, so the ceiling is measured
over exactly the poses the arms were compared on.

Usage:
  python -I run_logging/pdockq2_ceiling.py --workers 9
  python -I run_logging/pdockq2_ceiling.py --report-only
  python -I run_logging/pdockq2_ceiling.py --shard 0 --workers 9 --limit-units 20   # timing probe
"""
import argparse
import collections
import csv
import glob
import math
import os
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import provenance as PROV
from phase_k_eval import interface_chain_pairs                       # noqa: E402
from protenix.model import sample_confidence                         # noqa: E402
from protenix.model.loss import PAELoss, calculate_atom_bespoke_lddt  # noqa: E402

SEEDS = tuple(range(500000, 500010))
N_BINS, MIN_BIN, MAX_BIN = 64, 0.0, 32.0
PDOCKQ_CUTOFF = 8.0          # A, between the two chains' representative atoms
PDQ2_D0 = 10.0               # A, FIXED -- not interface-size dependent
PDQ2_L, PDQ2_K, PDQ2_X0, PDQ2_B = 1.31, 0.075, 84.733, 0.005

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=9)
ap.add_argument("--shard", type=int, default=-1)
ap.add_argument("--rows-dir", default="training/runs/rowweight_compare/rowweight_ep55")
ap.add_argument("--pool", default="training/full_train/pool_winnable.csv")
ap.add_argument("--roles", default="training/fold/chain_roles.csv")
ap.add_argument("--fold", default="training/fold")
ap.add_argument("--out", default="training/runs/pdockq2_ceiling")
ap.add_argument("--limit-units", type=int, default=0)
ap.add_argument("--report-only", action="store_true")
a = ap.parse_args()
OUT = os.path.join(ROOT, a.out)


def pdockq2(mean_plddt, mean_ptm):
    x = float(mean_plddt) * float(mean_ptm)
    return PDQ2_L / (1.0 + math.exp(-PDQ2_K * (x - PDQ2_X0))) + PDQ2_B


def unit_rows(pdb, seed, pairs, pl, bc):
    """{(iface, sample): (pdockq2, mean_ptm, mean_plddt, npairs)} for one (structure, seed)."""
    tp = os.path.join(ROOT, a.fold, "trunks", pdb, f"free_{seed}.npz")
    lp = os.path.join(ROOT, a.fold, "labels", pdb, f"free_{seed}.npz")
    if not (os.path.exists(tp) and os.path.exists(lp)):
        return {}
    tr = np.load(tp, allow_pickle=True)
    lb = np.load(lp, allow_pickle=True)
    asym = torch.as_tensor(tr["ci_feat_asym_id"]).long()
    a2t = torch.as_tensor(tr["ci_feat_atom_to_token_idx"]).long()
    hf = torch.as_tensor(lb["lf_has_frame"]).bool()
    fai = torch.as_tensor(lb["lf_frame_atom_index"]).long()
    rep = torch.as_tensor(lb["lf_pae_rep_atom_mask"]).bool()
    isnuc = (torch.as_tensor(lb["lf_is_rna"]) | torch.as_tensor(lb["lf_is_dna"])).bool()
    ispoly = ~torch.as_tensor(lb["lf_is_ligand"]).bool()
    n_tok = int(asym.numel())
    out = {}
    for p in range(int(lb["n_pose"])):
        pred = torch.as_tensor(tr["ci_coordinate"][p]).float()
        true = torch.as_tensor(lb["label_coordinate"][p]).float()
        cmask = torch.as_tensor(lb["label_coordinate_mask"][p]).bool()

        # ---- true PAE in angstroms, from the label's own one-hot ----
        _, bins, pm = pl.calculate_label(
            pred_coordinate=pred.unsqueeze(0), true_coordinate=true,
            coordinate_mask=cmask.float(), rep_atom_mask=rep, frame_atom_index=fai, has_frame=hf)
        pae_A = bc[bins[0].argmax(-1)]                       # [N_tok, N_tok]

        # ---- true pLDDT per token, x100, only where the truth exists ----
        l_at, w_at = calculate_atom_bespoke_lddt(
            pred_coordinate=pred[cmask], true_coordinate=true[cmask],
            is_nucleotide=isnuc[cmask], is_polymer=ispoly[cmask], rep_atom_mask=rep[cmask])
        lddt = (l_at / (w_at + 1e-6)).reshape(-1)
        # map the masked rep atoms back to their token ids
        rep_tok = a2t[rep & cmask]
        rep_sel = rep[cmask]
        plddt = torch.full((n_tok,), float("nan"))
        plddt[rep_tok] = lddt[rep_sel] * 100.0
        have = ~torch.isnan(plddt)

        # ---- token coordinates from the PREDICTED structure, for the 8 A interface ----
        tok_xyz = torch.full((n_tok, 3), float("nan"))
        tok_xyz[rep_tok] = pred[rep & cmask]

        samp = int(lb["sample_index"][p])
        for iface, ax, ay in pairs:
            ia = torch.nonzero(have & (asym == ax)).reshape(-1)
            ib = torch.nonzero(have & (asym == ay)).reshape(-1)
            if ia.numel() == 0 or ib.numel() == 0:
                continue
            d = torch.cdist(tok_xyz[ia], tok_xyz[ib])
            close = d <= PDOCKQ_CUTOFF
            npair = int(close.sum())
            if npair == 0:
                # NO CONTACTS, NO SCORE. ipsae returns 0.0 here; recording it as 0.0 is correct --
                # a pose whose chains do not touch should not win a selection.
                out[(iface, samp)] = (0.0, 0.0, 0.0, 0)
                continue
            sub = pae_A[ia][:, ib][close]
            mean_ptm = float((1.0 / (1.0 + (sub / PDQ2_D0) ** 2)).mean())
            itf = torch.cat([ia[close.any(1)], ib[close.any(0)]])
            mean_plddt = float(plddt[itf].mean())
            out[(iface, samp)] = (pdockq2(mean_plddt, mean_ptm), mean_ptm, mean_plddt, npair)
        del pae_A, bins, pm
    return out


def run_shard(k):
    pairs_by_pdb, unres = interface_chain_pairs(os.path.join(ROOT, a.pool),
                                                os.path.join(ROOT, a.roles))
    units = [(p, s) for p in sorted(pairs_by_pdb) for s in SEEDS]
    mine = units[k::a.workers]
    if a.limit_units:
        mine = mine[:a.limit_units]
    pl = PAELoss()
    bc = sample_confidence.get_bin_centers(min_bin=MIN_BIN, max_bin=MAX_BIN,
                                           no_bins=N_BINS).float()
    rows, t0 = [], time.time()
    for i, (pdb, seed) in enumerate(mine):
        for (iface, samp), v in unit_rows(pdb, seed, pairs_by_pdb[pdb], pl, bc).items():
            rows.append((iface, seed, samp, f"{v[0]:.6f}", f"{v[1]:.6f}",
                         f"{v[2]:.3f}", v[3]))
        if i and i % 25 == 0:
            el = time.time() - t0
            print(f"  shard {k}: {i}/{len(mine)} units, {len(rows)} rows, {el:.0f}s "
                  f"(eta {el/i*(len(mine)-i)/60:.1f} min)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f"pdq2_s{k}.csv")
    with open(f + ".tmp", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iface", "seed", "sample", "pdockq2", "mean_ptm", "mean_plddt", "npairs"])
        w.writerows(rows)
    os.replace(f + ".tmp", f)
    print(f"  shard {k}: {len(rows)} row(s) -> {os.path.relpath(f, ROOT)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if k == 0 and unres:
        print(f"  NOTE: {len(unres)} pool pair(s) unresolved in chain_roles: {unres[:5]}")
    return 0


def report():
    pdq = {}
    for f in sorted(glob.glob(os.path.join(OUT, "pdq2_s*.csv"))):
        for r in csv.DictReader(open(f)):
            pdq[(r["iface"], int(r["seed"]), int(r["sample"]))] = (
                float(r["pdockq2"]), float(r["mean_ptm"]), float(r["mean_plddt"]),
                int(r["npairs"]))
    if not pdq:
        sys.exit(f"  no pdq2_s*.csv under {a.out}")
    dq, ship = {}, {}
    for f in sorted(glob.glob(os.path.join(ROOT, a.rows_dir, "rows_b*_s*.csv"))):
        for r in csv.DictReader(open(f)):
            try:
                k = (r["iface"], int(r["seed"]), int(r["sample"]))
                dq[k] = float(r["dockq"]); ship[k] = float(r["stored_iptm"])
            except (ValueError, TypeError, KeyError):
                continue
    # the iTM ceiling, for a like-for-like comparison on the same poses
    itm = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "training", "runs", "gt_iptm_ceiling",
                                           "gt_iptm_s*.csv"))):
        for r in csv.DictReader(open(f)):
            itm[(r["iface"], int(r["seed"]), int(r["sample"]))] = (
                float(r["gt_iptm_max"]), float(r["gt_iptm_mean"]))

    by = collections.defaultdict(list)
    for k, v in pdq.items():
        if k in dq and k in ship:
            im = itm.get(k, (float("nan"), float("nan")))
            by[k[0]].append((v[0], v[1], ship[k], dq[k], im[0], im[1]))
    ifc = sorted(by)
    full = max(collections.Counter(len(by[i]) for i in ifc).items(), key=lambda kv: kv[1])[0]
    keep = [i for i in ifc if len(by[i]) == full]
    n = len(keep)
    pick = lambda j: np.array([max(by[i], key=lambda t: t[j])[3] for i in keep])  # noqa: E731
    base = pick(2)
    orac = np.array([max(t[3] for t in by[i]) for i in keep])
    hr = orac.mean() - base.mean()
    sel = {"pDockQ2 (true PAE+pLDDT)": pick(0), "  its PAE term alone": pick(1),
           "iTM, MAX over rows": pick(4), "iTM, MEAN over rows": pick(5)}
    print(f"\n  CEILINGS -- {n} interface(s) at {full} candidates each"
          + (f"  ({len(ifc)-n} dropped for a short candidate set)" if len(ifc) != n else ""))
    print(f"\n    {'selector':>26} {'mean DockQ':>11} {'frac':>8} {'vs base':>10} "
          f"{'t':>7} {'% headroom':>11}")
    print(f"    {'shipped ipTM (baseline)':>26} {base.mean():>11.4f} "
          f"{(base >= 0.23).mean():>8.4f} {'--':>10} {'--':>7} {'--':>11}")
    for nm, s in sel.items():
        if np.isnan(s).any():
            print(f"    {nm:>26} {'(no data)':>11}")
            continue
        d = s - base
        se = d.std(ddof=1) / math.sqrt(n)
        print(f"    {nm:>26} {s.mean():>11.4f} {(s >= 0.23).mean():>8.4f} "
              f"{d.mean():>+10.4f} {d.mean()/se:>+7.2f} {100*d.mean()/hr:>10.1f}%")
    print(f"    {'oracle (best DockQ)':>26} {orac.mean():>11.4f} "
          f"{(orac >= 0.23).mean():>8.4f} {hr:>+10.4f} {'--':>7} {'100.0%':>11}")
    p2, im = sel["pDockQ2 (true PAE+pLDDT)"], sel["iTM, MEAN over rows"]
    if not np.isnan(im).any():
        d = p2 - im
        se = d.std(ddof=1) / math.sqrt(n)
        print(f"\n    pDockQ2 minus iTM-mean, paired: {d.mean():+.4f} +- {se:.4f} "
              f"(t={d.mean()/se:+.2f})")
    nz = [v[3] for v in pdq.values()]
    print(f"\n    interface contacts within {PDOCKQ_CUTOFF:.0f} A: median {np.median(nz):.0f}, "
          f"{100*np.mean(np.array(nz) == 0):.1f}% of poses have NONE (scored 0.0)")


if __name__ == "__main__":
    if a.report_only:
        report()
    elif a.shard >= 0:
        sys.exit(run_shard(a.shard))
    else:
        t0 = time.time()
        ps = [subprocess.Popen(
            [sys.executable, "-I", os.path.abspath(__file__), "--shard", str(k),
             "--workers", str(a.workers), "--rows-dir", a.rows_dir, "--pool", a.pool,
             "--roles", a.roles, "--fold", a.fold, "--out", a.out]
            + (["--limit-units", str(a.limit_units)] if a.limit_units else []))
            for k in range(a.workers)]
        rc = [p.wait() for p in ps]
        print(f"\n  {a.workers} worker(s) in {(time.time()-t0)/60:.1f} min, rc={rc}")
        if any(rc):
            sys.exit(f"STOP: shard(s) failed with {rc}")
        # PROVENANCE BEFORE THE REPORT, so a failed report still leaves the record of what ran.
        PROV.write(OUT, ROOT,
                   inputs=[a.pool, a.roles, os.path.join(a.rows_dir, "rows_b10_*_s*.csv"),
                           os.path.join(a.fold, "labels"), os.path.join(a.fold, "trunks")],
                   outputs_glob="pdq2_s*.csv",
                   extra={"workers": a.workers, "seeds": list(SEEDS)})
        report()
