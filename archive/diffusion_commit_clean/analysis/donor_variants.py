#!/usr/bin/env python
"""Compare redundancy tests for donor selection, on the quantity that predicts rescue.

THE CHOICE BEING TESTED. Every variant sorts poses by pair_iptm and walks down the list keeping a pose
only if it is not redundant with what is already kept. They differ ONLY in what "redundant with a kept
pose" means on a complex with several interfaces, where a pair of poses has one DockQ per interface:

  own     DockQ at the interface being sorted            (what the running sweep does)
  max     the largest DockQ over all interfaces          -- must differ EVERYWHERE to be kept
  min     the smallest DockQ over all interfaces         -- differing ANYWHERE is enough
  mean    the average over interfaces                    (build_simple_queue.py's test)

`own` is the only one that needs the per-interface split for the FILTER; max/min/mean reduce a pair of
poses to one number. All four still sort per interface, so none of them is reference-free yet.

WHAT IS REPORTED. Not donor-set overlap -- that says how much the sets differ, not whether the
difference is good. Coverage: does the capped donor set contain at least one pose with GT DockQ >= 0.23
FOR THAT INTERFACE. A donor set with no good pose for an interface cannot rescue it, so coverage is
the ceiling on rescue, and it is measured per interface while selection happens per complex.

Single-interface complexes are reported separately: all four tests are identical there by
construction, so pooling them would dilute the comparison with rows that cannot differ.

    donor_variants.py [--cap 50]
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/run_exp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_complex_queue import MATRICES, T, round_robin  # noqa: E402
from fmt import table  # noqa: E402
from separation import pool_classes  # noqa: E402

ACC = 0.23


def stack(ifaces):
    """Per-complex arrays on a common pose index. NaN marks a pose absent from an interface."""
    keys = []
    raw = []
    for f in ifaces:
        z = np.load(f"{MATRICES}/{f}.npz")
        k = [(str(s), int(x)) for s, x in zip(z["seed"], z["sample"])]
        raw.append((k, z["pair_iptm"].astype(float), z["dockq"].astype(float),
                    z["dockq_gt"].astype(float)))
        keys += k
    poses = sorted(set(keys))
    idx = {k: i for i, k in enumerate(poses)}
    n, m = len(poses), len(ifaces)
    ip = np.full((m, n), np.nan)
    gt = np.full((m, n), np.nan)
    M = np.full((m, n, n), np.nan)
    for a, (k, i_, d_, g_) in enumerate(raw):
        g = np.array([idx[x] for x in k])
        ip[a, g] = i_
        gt[a, g] = g_
        M[a][np.ix_(g, g)] = d_
    return poses, ip, gt, M


def walk(order, sim, cap):
    """Greedy keep-if-distinct walk. `sim(p, kept)` returns the redundancy score, NaN = unknown."""
    L = []
    for p in order:
        p = int(p)
        if all(np.isnan(s) or s < T for s in (sim(p, j) for j in L)):
            L.append(p)
            if len(L) >= cap:
                break
    return L


def reduce_fn(kind):
    if kind == "max":
        return np.nanmax
    if kind == "min":
        return np.nanmin
    return np.nanmean


def select(ifaces, cap, kind):
    """Donor indices for one complex under one redundancy test, plus the per-interface lists."""
    poses, ip, gt, M = stack(ifaces)
    per = {}
    for a, f in enumerate(ifaces):
        order = np.argsort(-np.nan_to_num(ip[a], nan=-1.0))
        order = [p for p in order if not np.isnan(ip[a, p])]
        if kind == "own":
            def sim(p, j, a=a):
                return M[a, p, j]
        else:
            red = reduce_fn(kind)

            def sim(p, j, red=red):
                col = M[:, p, j]
                return np.nan if np.all(np.isnan(col)) else red(col)
        per[f] = walk(order, sim, cap)
    sel = [p for p, _ in round_robin(per, cap)]
    return poses, gt, ifaces, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=50)
    A = ap.parse_args()

    bypdb = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{MATRICES}/*.npz")):
        i = os.path.basename(f)[:-4]
        bypdb[i.split("_")[0]].append(i)
    cls = pool_classes()

    kinds = ["own", "max", "min", "mean"]
    cov = collections.defaultdict(lambda: collections.defaultdict(list))
    ndon = collections.defaultdict(list)
    for pdb, ifs in sorted(bypdb.items()):
        multi = "multi-interface" if len(ifs) > 1 else "single-interface"
        for kind in kinds:
            poses, gt, ifaces, sel = select(ifs, A.cap, kind)
            ndon[(kind, multi)].append(len(sel))
            for a, f in enumerate(ifaces):
                c = cls.get(f)
                if c is None:
                    continue
                good = [p for p in sel if gt[a, p] >= ACC]
                cov[(kind, multi)][c].append(bool(good))

    for multi in ("multi-interface", "single-interface"):
        rows = []
        for kind in kinds:
            d = cov[(kind, multi)]
            r = [kind, f"{np.mean(ndon[(kind, multi)]):.1f}"]
            for c in ("decoy", "solved", "nocov"):
                v = d.get(c, [])
                r.append(f"{sum(v)}/{len(v)}" if v else "-")
            rows.append(r)
        print(f"\n=== {multi} complexes   (cap {A.cap})")
        print(table(["redundancy test", "mean donors per complex",
                     "decoy covered", "solved covered", "nocov covered"],
                    rows, align="l r r r r"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
