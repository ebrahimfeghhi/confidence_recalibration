#!/usr/bin/env python
"""WITHIN-SEED ipTM vs DockQ. Does ipTM's pose-to-pose movement point at better docking?

WHY THIS AND NOT THE VARIANCE SHARE. run_logging/pose_pair_signal_sites.py showed the pair
representation's pose-dependent energy survives all four Pairformer blocks and is AMPLIFIED 1.72x
by the PAE readout, with ipTM still moving 1.43x the injection level. That says the head responds
to the pose. It does not say the response is USEFUL: the network can respond strongly to geometry
that has nothing to do with docking accuracy. This measures the sign and the strength of the
association instead of its magnitude.

MEAN-CENTRED WITHIN EACH SEED, which is the whole point. ipTM is effectively a per-seed quantity
-- 2.6% of within-interface variance sits among a seed's 5 samples against DockQ's 49% -- so an
uncentred correlation is dominated by between-seed differences that selection inside a seed cannot
use. Centring both quantities inside each (interface, seed) group removes exactly the part of the
signal that Success@1 cannot exploit.

THREE STATISTICS, because they fail differently:
  pooled Pearson    on the centred values, all groups together. Sensitive to magnitude, so a few
                    groups with large swings can carry it.
  mean within-group Spearman, one coefficient per group then averaged. Scale-free and equally
                    weighted per group, which is what a per-seed selector actually faces.
  pairwise ranking accuracy  over all within-seed pose pairs with distinct DockQ: the fraction
                    where ipTM orders the pair the way DockQ does. 0.5 is chance. This is the
                    quantity that converts directly into selection.

Reported for every group and again for AB x AG interfaces alone, and split train / held-out by
train_val_splits/cache_train_structures.csv, because a statistic measured only where the model was
fit is not evidence about selection.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

AB_ROLES = {"H", "L"}


def is_ag(role):
    return bool(role) and role.startswith("ag")


def stats(groups):
    """(n_groups, n_poses, pooled_pearson, mean_spearman, pair_acc, n_pairs) for [(d[], s[])]."""
    import numpy as np
    from scipy import stats as sps
    cd, cs, rhos, hit, tot = [], [], [], 0, 0
    for d, s in groups:
        d = np.asarray(d, dtype=float)
        s = np.asarray(s, dtype=float)
        cd.append(d - d.mean())
        cs.append(s - s.mean())
        if len(set(d.tolist())) > 1 and len(set(s.tolist())) > 1:
            rhos.append(float(sps.spearmanr(d, s).statistic))
        for i in range(len(d)):
            for j in range(i + 1, len(d)):
                if d[i] == d[j]:
                    continue
                tot += 1
                # A TIE IN ipTM IS HALF A HIT, not a hit. Scoring it as correct would report the
                # head as better than chance exactly where it is saying nothing at all.
                if s[i] == s[j]:
                    hit += 0.5
                elif (s[i] > s[j]) == (d[i] > d[j]):
                    hit += 1
    cd = np.concatenate(cd)
    cs = np.concatenate(cs)
    pr = sps.pearsonr(cd, cs)
    return (len(groups), len(cd), float(pr.statistic), float(pr.pvalue),
            (float(np.mean(rhos)) if rhos else float("nan")), len(rhos),
            (hit / tot if tot else float("nan")), tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles_all.csv")
    ap.add_argument("--train-structures",
                    default=f"{ROOT}/train_val_splits/cache_train_structures.csv")
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--min-spread", type=float, default=0.0,
                    help="restrict to groups whose ground-truth DockQ spans at least this much")
    a = ap.parse_args()

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

    by = collections.defaultdict(dict)
    with open(a.results) as fh:
        for r in csv.DictReader(fh):
            if r.get("has_trunk") != "1" or r.get("arm") != "free":
                continue
            try:
                by[(r["iface"], int(r["seed"]))][int(r["sample"])] = (
                    float(r["dockq"]), float(r["iptm"]))
            except (ValueError, KeyError, TypeError):
                continue

    buckets = collections.defaultdict(list)
    for (iface, _seed), d in by.items():
        if len(d) != int(a.poses):
            continue
        dq = [d[k][0] for k in sorted(d)]
        ip = [d[k][1] for k in sorted(d)]
        if max(dq) - min(dq) < float(a.min_spread):
            continue
        p = RKL.parse_iface(iface, roles_join)
        if p is None:
            continue
        pdb, pair = p
        ro = role_of.get(pdb) or {}
        ra = ro.get(chr(ord("A") + pair[0]))
        rb = ro.get(chr(ord("A") + pair[1]))
        abag = (ra in AB_ROLES and is_ag(rb)) or (is_ag(ra) and rb in AB_ROLES)
        split = "train" if pdb in trainset else "heldout"
        buckets[("all", "both")].append((dq, ip))
        buckets[("all", split)].append((dq, ip))
        if abag:
            buckets[("abag", "both")].append((dq, ip))
            buckets[("abag", split)].append((dq, ip))

    print(f"groups with {a.poses} poses and a trunk: {len(buckets[('all','both')])}"
          + (f"   (DockQ spread >= {a.min_spread})" if a.min_spread else ""))
    print(f"{'subset':14s} {'groups':>7s} {'poses':>7s} {'pearson':>9s} {'p':>10s} "
          f"{'spearman':>9s} {'n_rho':>6s} {'pair_acc':>9s} {'pairs':>8s}")
    for key in (("all", "both"), ("all", "train"), ("all", "heldout"),
                ("abag", "both"), ("abag", "train"), ("abag", "heldout")):
        g = buckets.get(key)
        if not g:
            continue
        n, np_, pr, pv, rho, nrho, acc, npair = stats(g)
        print(f"{key[0] + '/' + key[1]:14s} {n:7d} {np_:7d} {pr:+9.4f} {pv:10.2e} "
              f"{rho:+9.4f} {nrho:6d} {acc:9.4f} {npair:8d}")


if __name__ == "__main__":
    main()
