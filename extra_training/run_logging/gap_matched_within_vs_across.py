#!/usr/bin/env python
"""GAP-MATCHED pairwise accuracy: within-seed pose pairs vs across-seed pose pairs.

THE QUESTION. ipTM ranks across seeds well and within a seed barely at all, but the two
comparisons are not equally hard: within a seed the two poses usually differ in DockQ by much
less than two poses from different seeds do. An unconditional accuracy gap could be entirely
that. So condition on the gap: among pairs whose |DockQ difference| falls in the same bin, is
the head still worse at the within-seed pair?

PAIR CONSTRUCTION, both types inside ONE interface, so interface identity is never the
difference between the two populations:
  within  -- the C(5,2) = 10 pairs of samples that share a seed
  across  -- every pair of poses from two DIFFERENT seeds of that interface
Ties in DockQ are dropped (the question is undefined); a tie in ipTM counts as half a hit, the
same convention as run_logging/within_seed_iptm_dockq.py.

TWO MATCHINGS, because binning is coarse:
  binned  -- accuracy per |dDockQ| bin, each population scored on its own members of that bin
  paired  -- for each within-seed pair, the across-seed pairs of the SAME interface whose gap is
             within --tol; the two accuracies are then means over the same gap distribution by
             construction, and each within pair contributes once with its matched set's mean.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

AB_ROLES = {"H", "L"}
EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]


def is_ag(role):
    return bool(role) and role.startswith("ag")


def hit(dq_i, ip_i, dq_j, ip_j):
    """1 correct, 0.5 ipTM tie, 0 wrong. Caller guarantees dq_i != dq_j."""
    if ip_i == ip_j:
        return 0.5
    return 1.0 if (ip_i > ip_j) == (dq_i > dq_j) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles_all.csv")
    ap.add_argument("--train-structures",
                    default=f"{ROOT}/train_val_splits/cache_train_structures.csv")
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--iptm-col", default="iptm")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="gap tolerance for the paired matching")
    ap.add_argument("--abag-only", action="store_true")
    a = ap.parse_args()

    import numpy as np
    import rank_loss as RKL

    roles_join = RKL.load_roles(a.roles)
    role_of = {}
    with open(a.roles) as fh:
        for r in csv.DictReader(fh):
            role_of.setdefault(r["pdb"], {})[r["model_chain"]] = r["role"].strip()

    trainset = set()
    if os.path.exists(a.train_structures):
        with open(a.train_structures) as fh:
            for r in csv.DictReader(fh):
                trainset.add(r["pdb"])

    cell = collections.defaultdict(dict)          # (iface, seed) -> sample -> (dockq, iptm)
    with open(a.results) as fh:
        rd = csv.DictReader(fh)
        col = a.iptm_col if a.iptm_col in (rd.fieldnames or []) else "pair_iptm"
        for r in rd:
            if r.get("has_trunk") not in (None, "", "1"):
                continue
            if "arm" in r and r.get("arm") not in (None, "", "free"):
                continue
            try:
                cell[(r["iface"], int(r["seed"]))][int(r["sample"])] = (
                    float(r["dockq"]), float(r[col]))
            except (ValueError, KeyError, TypeError):
                continue

    by_iface = collections.defaultdict(dict)      # iface -> seed -> [(dockq, iptm) x poses]
    for (iface, seed), d in cell.items():
        if len(d) != int(a.poses):
            continue
        by_iface[iface][seed] = [d[k] for k in sorted(d)]

    kept, dropped_abag, pops = 0, 0, {}
    for tag in ("within", "across"):
        pops[tag] = {"gap": [], "hit": []}
    paired_w, paired_a, paired_n = [], [], 0

    for iface, seeds in by_iface.items():
        if len(seeds) < 2:
            continue
        p = RKL.parse_iface(iface, roles_join)
        if p is None:
            continue
        pdb, pair = p
        ro = role_of.get(pdb) or {}
        ra, rb = ro.get(chr(ord("A") + pair[0])), ro.get(chr(ord("A") + pair[1]))
        abag = (ra in AB_ROLES and is_ag(rb)) or (is_ag(ra) and rb in AB_ROLES)
        if a.abag_only and not abag:
            dropped_abag += 1
            continue
        kept += 1

        order = sorted(seeds)
        poses = [(s, k) for s in order for k in range(int(a.poses))]
        dq = np.array([seeds[s][k][0] for s, k in poses])
        ip = np.array([seeds[s][k][1] for s, k in poses])
        sd = np.array([s for s, _ in poses])

        n = len(poses)
        ii, jj = np.triu_indices(n, k=1)
        gap = np.abs(dq[ii] - dq[jj])
        live = gap > 0
        ii, jj, gap = ii[live], jj[live], gap[live]
        same = sd[ii] == sd[jj]
        correct = np.where(ip[ii] == ip[jj], 0.5,
                           ((ip[ii] > ip[jj]) == (dq[ii] > dq[jj])).astype(float))

        pops["within"]["gap"].append(gap[same])
        pops["within"]["hit"].append(correct[same])
        pops["across"]["gap"].append(gap[~same])
        pops["across"]["hit"].append(correct[~same])

        # paired matching, inside this interface
        wg, wh = gap[same], correct[same]
        ag_, ah = gap[~same], correct[~same]
        if len(ag_) == 0:
            continue
        srt = np.argsort(ag_)
        ag_s, ah_s = ag_[srt], ah[srt]
        lo = np.searchsorted(ag_s, wg - a.tol, side="left")
        hi = np.searchsorted(ag_s, wg + a.tol, side="right")
        cs = np.concatenate([[0.0], np.cumsum(ah_s)])
        cnt = hi - lo
        ok = cnt > 0
        if not ok.any():
            continue
        paired_w.append(wh[ok])
        paired_a.append((cs[hi[ok]] - cs[lo[ok]]) / cnt[ok])
        paired_n += int(ok.sum())

    for tag in pops:
        pops[tag]["gap"] = np.concatenate(pops[tag]["gap"])
        pops[tag]["hit"] = np.concatenate(pops[tag]["hit"])

    print(f"interfaces {kept}"
          + (f"   (non-abAg dropped {dropped_abag})" if a.abag_only else "")
          + f"   within pairs {len(pops['within']['gap'])}"
          + f"   across pairs {len(pops['across']['gap'])}")
    print()
    print(f"{'|dDockQ| bin':>16s} {'within acc':>11s} {'n':>9s} "
          f"{'across acc':>11s} {'n':>10s} {'within-across':>14s}")
    for lo_, hi_ in zip(EDGES[:-1], EDGES[1:]):
        row = [f"[{lo_:.2f},{hi_:.2f})".rjust(16)]
        vals = {}
        for tag in ("within", "across"):
            g, h = pops[tag]["gap"], pops[tag]["hit"]
            m = (g >= lo_) & (g < hi_)
            vals[tag] = (float(h[m].mean()) if m.any() else float("nan"), int(m.sum()))
            row.append(f"{vals[tag][0]:11.4f} {vals[tag][1]:9d}")
        d = vals["within"][0] - vals["across"][0]
        row.append(f"{d:+14.4f}")
        print(" ".join(row))
    print(f"{'ALL':>16s} "
          f"{pops['within']['hit'].mean():11.4f} {len(pops['within']['hit']):9d} "
          f"{pops['across']['hit'].mean():11.4f} {len(pops['across']['hit']):9d} "
          f"{pops['within']['hit'].mean() - pops['across']['hit'].mean():+14.4f}")
    print(f"{'median gap':>16s} {np.median(pops['within']['gap']):11.4f} "
          f"{'':9s} {np.median(pops['across']['gap']):11.4f}")

    if paired_n:
        from scipy import stats as sps
        w = np.concatenate(paired_w)
        c = np.concatenate(paired_a)
        t = sps.ttest_rel(w, c)
        print()
        print(f"PAIRED on gap within +-{a.tol} inside the same interface, "
              f"{paired_n} within-seed pair(s) matched")
        print(f"  within-seed accuracy          {w.mean():.4f}")
        print(f"  matched across-seed accuracy  {c.mean():.4f}")
        print(f"  difference                    {w.mean() - c.mean():+.4f}   "
              f"t {t.statistic:+.2f}  p {t.pvalue:.2e}")


if __name__ == "__main__":
    main()
