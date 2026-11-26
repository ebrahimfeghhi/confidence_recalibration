#!/usr/bin/env python
"""How much does ipTM move per unit of REAL quality improvement, within a seed vs across seeds?

THE MODEL. Every pair of poses from the same chain pair is oriented so the DockQ difference is
positive: d = DockQ(better) - DockQ(worse) > 0. The response is the SIGNED ipTM difference,
y = ipTM(better) - ipTM(worse), so a positive y means ipTM preferred the genuinely better pose.

    y  ~  1 + d + d^2  +  same * (1 + d + d^2)

`same` is 1 when the two poses come from one seed (one trunk) and 0 when they come from two.
The d slope is ipTM's SENSITIVITY: how much score it spends per unit of true improvement. The
interaction is how much of that sensitivity is lost inside a seed, which is the question.

WHY SIGNED, AND WHY NOT |dipTM|. The absolute difference measures how far the score moved, and
we already know it moves ~13x further across seeds at a matched gap. But movement that does not
track quality is noise, and |dipTM| cannot tell the two apart: a pair whose ipTM swings 0.3 the
WRONG way scores the same as one that swings 0.3 the right way. The signed slope separates them
-- a population with a huge |dipTM| and a flat slope is being told apart by something that is
not pose quality.

A SECOND, DECISION-RELEVANT FIT. The slope is in ipTM units, but selection only cares whether
the ordering is right. So the same design matrix is fitted as a logistic model on

    correct = 1[y > 0]

which is the binned accuracy table with the gap as a CONTINUOUS control instead of a bin. That
is what answers the residual-skew objection to the binned version: inside the top bin, which
spans 0.40 to 1.01, cross-seed pairs could simply sit nearer the upper edge. Here d is in the
model, so that cannot be mistaken for a seed effect.

d^2 IS IN BOTH because the binned table shows the effect concentrating above a 0.2 gap; a single
linear term would average a steep high-gap region together with a flat low-gap one.

STANDARD ERRORS ARE CLUSTERED ON THE CHAIN PAIR. The pairs are massively non-independent: five
poses of one seed generate 10 pairs that share three poses on average, and every pair of an
interface shares its structure, its trunks and its reference. Plain OLS errors would be far too
small. The sandwich treats each chain pair as one cluster, which is the level the sampling
actually varies at.

TIES. Pairs with d == 0 are dropped -- the question is undefined. For the logistic fit an exact
ipTM tie is dropped too and counted, since 1[y > 0] cannot represent a half-hit.

SEED SUBSAMPLING. Enumerating every pair of a PXM chain pair with 600 poses is 180k pairs, 53M
over the dataset. So at most --max-seeds seeds are kept per chain pair, chosen at random, with
all of each kept seed's poses. This preserves the within/across structure exactly rather than
thinning one population relative to the other.
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ols_cluster(X, y, g):
    """OLS with cluster-robust covariance. Returns (beta, se)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    n, k = X.shape
    meat = np.zeros((k, k))
    for gi in np.unique(g):
        m = g == gi
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    G = len(np.unique(g))
    c = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = XtX_inv @ meat @ XtX_inv * c
    return beta, np.sqrt(np.maximum(np.diag(V), 0.0))


def logit_cluster(X, y, g, iters=60, tol=1e-9):
    """Logistic regression by IRLS with cluster-robust covariance. Returns (beta, se)."""
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = np.maximum(p * (1 - p), 1e-10)
        H = X.T @ (X * W[:, None])
        step = np.linalg.pinv(H) @ (X.T @ (y - p))
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    eta = np.clip(X @ beta, -30, 30)
    p = 1.0 / (1.0 + np.exp(-eta))
    W = np.maximum(p * (1 - p), 1e-10)
    H_inv = np.linalg.pinv(X.T @ (X * W[:, None]))
    meat = np.zeros((k, k))
    for gi in np.unique(g):
        m = g == gi
        s = X[m].T @ (y[m] - p[m])
        meat += np.outer(s, s)
    G = len(np.unique(g))
    c = G / max(G - 1, 1)
    V = H_inv @ meat @ H_inv * c
    return beta, np.sqrt(np.maximum(np.diag(V), 0.0))


def build_pairs(by_iface, max_seeds, rng):
    """-> d, y, same, cluster index. Pairs are formed INSIDE one chain pair only."""
    D, Y, S, C = [], [], [], []
    for ci, (iface, seeds) in enumerate(sorted(by_iface.items())):
        keys = sorted(seeds)
        if len(keys) > max_seeds:
            keys = [keys[i] for i in sorted(rng.choice(len(keys), max_seeds, replace=False))]
        dq, ip, sd = [], [], []
        for k in keys:
            for q, t in seeds[k]:
                dq.append(q)
                ip.append(t)
                sd.append(k)
        if len(dq) < 2:
            continue
        dq = np.asarray(dq, dtype=np.float64)
        ip = np.asarray(ip, dtype=np.float64)
        sd = np.asarray(sd)
        ii, jj = np.triu_indices(len(dq), k=1)
        d = dq[ii] - dq[jj]
        live = d != 0.0
        ii, jj, d = ii[live], jj[live], d[live]
        if ii.size == 0:
            continue
        # orient so the DockQ difference is positive
        flip = d < 0
        hi = np.where(flip, jj, ii)
        lo = np.where(flip, ii, jj)
        D.append(dq[hi] - dq[lo])
        Y.append(ip[hi] - ip[lo])
        S.append((sd[hi] == sd[lo]).astype(np.float64))
        C.append(np.full(ii.size, ci))
    if not D:
        return (np.empty(0),) * 4
    return (np.concatenate(D), np.concatenate(Y),
            np.concatenate(S), np.concatenate(C))


def report(tag, d, y, same, clus, args, extra=None, extra_names=()):
    """extra: additional design columns (dataset fixed effects for the pooled fit)."""
    n = d.size
    if n < 50:
        print(f"\n=== {tag}: only {n} pair(s), skipped")
        return
    X = np.column_stack([np.ones(n), d, d ** 2, same, same * d, same * d ** 2])
    names = ["const", "d", "d^2", "same", "same:d", "same:d^2"]
    if extra is not None and extra.size:
        X = np.column_stack([X, extra])
        names = names + list(extra_names)

    print(f"\n=== {tag}")
    print(f"    pairs {n}   within-seed {int(same.sum())}   across-seed {int(n - same.sum())}"
          f"   chain pairs (clusters) {len(np.unique(clus))}")

    beta, se = ols_cluster(X, y, clus)
    print("\n    SIGNED dipTM  ~  d + d^2, by seed sharing   (SEs clustered on the chain pair)")
    print(f"      {'term':<10} {'coef':>10} {'SE':>9} {'t':>8}")
    for nm, b, s in zip(names, beta, se):
        t = b / s if s > 0 else float("nan")
        print(f"      {nm:<10} {b:>10.4f} {s:>9.4f} {t:>8.2f}")

    # sensitivity = d(dipTM)/dd at a given gap, per population
    print(f"\n      sensitivity d(dipTM)/d(dDockQ), i.e. ipTM spent per unit of real gain")
    print(f"      {'gap':>6} {'across':>10} {'within':>10} {'within/across':>14}")
    for gg in (0.05, 0.10, 0.20, 0.40, 0.60):
        a = beta[1] + 2 * beta[2] * gg
        w = a + beta[4] + 2 * beta[5] * gg
        r = w / a if a != 0 else float("nan")
        print(f"      {gg:>6.2f} {a:>10.4f} {w:>10.4f} {r:>14.3f}")

    ok = y != 0.0
    dropped = int((~ok).sum())
    yb = (y[ok] > 0).astype(np.float64)
    Xb, cb = X[ok], clus[ok]
    lb, lse = logit_cluster(Xb, yb, cb)
    print(f"\n    ORDERED CORRECTLY (logistic), {dropped} exact ipTM tie(s) dropped")
    print(f"      {'term':<10} {'coef':>10} {'SE':>9} {'z':>8}")
    for nm, b, s in zip(names, lb, lse):
        z = b / s if s > 0 else float("nan")
        print(f"      {nm:<10} {b:>10.4f} {s:>9.4f} {z:>8.2f}")

    print(f"\n      fitted P(correct) at a given DockQ gap")
    print(f"      {'gap':>6} {'across':>10} {'within':>10} {'difference':>12}")
    for gg in (0.02, 0.05, 0.10, 0.20, 0.40, 0.60):
        # Zeros for any dataset dummies, i.e. predicted at the OMITTED dataset level.
        pad = [0.0] * (lb.size - 6)
        xa = np.array([1, gg, gg ** 2, 0, 0, 0] + pad, dtype=np.float64)
        xw = np.array([1, gg, gg ** 2, 1, gg, gg ** 2] + pad, dtype=np.float64)
        pa = 1 / (1 + np.exp(-xa @ lb))
        pw = 1 / (1 + np.exp(-xw @ lb))
        print(f"      {gg:>6.2f} {pa:>10.4f} {pw:>10.4f} {pw - pa:>12.4f}")

    # joint test that the three `same` terms are all zero, cluster-robust Wald
    for lbl, b, s in (("signed dipTM", beta, se), ("ordering", lb, lse)):
        z = [b[i] / s[i] if s[i] > 0 else 0.0 for i in (3, 4, 5)]
        print(f"      [{lbl}] z on same, same:d, same:d^2 = "
              f"{z[0]:.2f}, {z[1]:.2f}, {z[2]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=f"{HERE}/training/fold/abag_all_results.csv")
    ap.add_argument("--max-seeds", type=int, default=20)
    ap.add_argument("--iptm-col", default="iptm", choices=["iptm", "iptm_mean"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by", default="dataset", choices=["split", "dataset"],
                    help="one fit per dataset (default), or per dataset/split/pose_set")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # GROUPS. --by split keeps every (dataset, split, pose_set) apart, which is what the first
    # version did. --by dataset is the per-benchmark view: one fit per dataset, and sabdab twice
    # -- once on the original five poses per seed, once with the regenerated five added.
    #
    # Adding the regenerated poses does more than add rows: a seed then holds TEN poses off one
    # trunk, so its within-seed pairs go from C(5,2)=10 to C(10,2)=45, and 25 of those 45 pair an
    # original pose against a regenerated one. Those are still within-seed pairs by the
    # definition that matters here -- one trunk, one pair rep, different diffusion noise.
    GROUPS = {
        "split": None,
        # (label, row filter, counts toward the pooled fit). The two sabdab groups OVERLAP --
        # the second contains the first -- so only one of them may enter the pool or sabdab is
        # counted twice. The superset is the one that enters.
        "dataset": [
            ("foldbench", lambda r: r["dataset"] == "foldbench", True),
            ("pxm", lambda r: r["dataset"] == "pxm", True),
            ("sabdab  original 5 poses per seed",
             lambda r: r["dataset"] == "sabdab" and r["pose_set"] == "orig", False),
            ("sabdab  + the regenerated 5 poses (10 per seed)",
             lambda r: r["dataset"] == "sabdab", True),
        ],
    }

    groups = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))
    specs = GROUPS[args.by]
    with open(args.master) as fh:
        for r in csv.DictReader(fh):
            v = r.get(args.iptm_col) or ""
            if not v:
                continue
            try:
                rec = (float(r["dockq"]), float(v))
                seed = int(r["seed"])
            except (TypeError, ValueError):
                continue
            if specs is None:
                keys = [(r["dataset"], r["split"], r["pose_set"])]
            else:
                keys = [(lbl,) for lbl, pred, _ in specs if pred(r)]
            for k in keys:
                groups[k][r["interface"]][seed].append(rec)

    print(f"master {os.path.relpath(args.master, os.path.dirname(HERE))}   "
          f"ipTM column {args.iptm_col}   max seeds per chain pair {args.max_seeds}"
          f"   grouped by {args.by}")

    poolable = (None if specs is None
                else {(lbl,) for lbl, _, inpool in specs if inpool})
    pool = {"d": [], "y": [], "same": [], "clus": [], "ds": []}
    cluster_id = {}
    for key in sorted(groups):
        d, y, same, clus = build_pairs(groups[key], args.max_seeds, rng)
        report("  ".join(x for x in key if x), d, y, same, clus, args)
        if d.size == 0 or (poolable is not None and key not in poolable):
            continue
        # CLUSTER ACROSS DATASETS ON (dataset, chain pair). sabdab train orig and regen are the
        # SAME folds off the SAME trunks with fresh diffusion noise, so they are not independent
        # observations and must land in one cluster, not two.
        ifaces = sorted(groups[key])
        gid = np.empty(clus.size, dtype=np.int64)
        for local, iface in enumerate(ifaces):
            key_g = (key[0].split()[0], iface)
            if key_g not in cluster_id:
                cluster_id[key_g] = len(cluster_id)
            gid[clus == local] = cluster_id[key_g]
        pool["d"].append(d)
        pool["y"].append(y)
        pool["same"].append(same)
        pool["clus"].append(gid)
        pool["ds"].append(np.full(d.size, key[0].split()[0]))

    if pool["d"]:
        d = np.concatenate(pool["d"])
        y = np.concatenate(pool["y"])
        same = np.concatenate(pool["same"])
        clus = np.concatenate(pool["clus"])
        ds = np.concatenate(pool["ds"])

        print("\n" + "=" * 78)
        print("POOLED over every dataset. Clusters are (dataset, chain pair), so sabdab train's")
        print("original and regenerated poses -- the same folds off the same trunks -- count as")
        print("one cluster rather than two independent ones.")
        report("POOLED  (no dataset effects)", d, y, same, clus, args)

        # dataset fixed effects, first level dropped
        levels = sorted(set(ds.tolist()))[1:]
        dum = np.column_stack([(ds == lv).astype(np.float64) for lv in levels])
        print("\nThe same pooled fit with DATASET FIXED EFFECTS. Without them the pooled slope")
        print("also absorbs the fact that the datasets have different DockQ gap distributions")
        print(f"and difficulty; the omitted level is {sorted(set(ds.tolist()))[0]}.")
        report("POOLED  (+ dataset fixed effects)", d, y, same, clus, args,
               extra=dum, extra_names=[f"ds:{lv}" for lv in levels])
    return 0


if __name__ == "__main__":
    sys.exit(main())
