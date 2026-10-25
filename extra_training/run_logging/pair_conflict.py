#!/usr/bin/env python
"""Do the ranking loss's own pairs pull against each other? CPU only, no GPU, no checkpoint.

    python -I run_logging/pair_conflict.py

WHAT THIS IS, AND WHAT IT IS NOT. The gradient-alignment experiment that produced the ~0.107
figure measured ||sum g_i|| / sum ||g_i|| over PER-EXAMPLE PARAMETER gradients, and needed a
forward and a backward through the confidence head for every pose. That is GPU work and is not
what this does.

This measures the same ratio one level earlier, in SCORE space, over the pairs INSIDE one
interface. Every pair (i,j) contributes a coefficient vector on that interface's ipTM scores
with exactly two non-zero entries:

    dL/ds_i = -(2 w sgn / den) * sigmoid(-m/tau)        and the negative of it on s_j

so the pair's direction is fixed by which pose it favours and its length by w * sigmoid(-m/tau).
`den` is shared by every pair in the interface and cancels out of the ratio, so the number is
scale-free and does not depend on tau's effect on the overall loss magnitude -- only on how the
pairs are oriented relative to one another.

WHAT IT ANSWERS. Whether contradictory COMPARISONS cancel: pair (a,b) wanting a up while pair
(a,c) wants a down. If the straddle restriction raises this ratio, the pairs it removed were
genuinely pulling against the ones it kept.

WHAT IT CANNOT ANSWER. Whether gradients cancel ACROSS interfaces, or in parameter space, where
two pairs that agree about their own poses can still disagree about a shared weight. The
parameter-space measurement remains the real test and still needs GPUs.

THE BASELINE IS 1/sqrt(N) per interface, N the pair count: that is what random pair orientations
would give. Reported alongside, because a ratio of 0.1 means nothing until it is read against it.
"""
import argparse
import csv
import math
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, "phases")
import rank_loss as RKL                                             # noqa: E402

DQ_TOL = 1e-6


def load(path, keep=None):
    seen = defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if keep is not None and r["iface"].split("_")[0].lower() not in keep:
                continue
            try:
                seen[r["iface"]][(r["seed"], r["sample"])] = (float(r["iptm"]), float(r["dockq"]))
            except (KeyError, ValueError):
                continue
    return seen


def conflict(poses, pairs, tau, straddle, weight_by_gap=True):
    """(alignment, n_pairs) for one interface, or (None, 0) when it carries under two pairs."""
    keys = sorted(poses)
    acc = defaultdict(float)     # pose -> summed coefficient
    tot = 0.0                    # sum of per-pair vector norms
    n = 0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ki, kj = keys[a], keys[b]
            if not RKL.pair_included(ki, kj, pairs):
                continue
            si, di = poses[ki]
            sj, dj = poses[kj]
            d = di - dj
            if abs(d) <= DQ_TOL:
                continue
            if not RKL.pair_straddles(di, dj, straddle):
                continue
            w = abs(d) if weight_by_gap else 1.0
            sgn = 1.0 if d > 0 else -1.0
            m = (si - sj) * sgn
            g = w * (1.0 / (1.0 + math.exp(m / tau)))       # w * sigmoid(-m/tau)
            acc[ki] += -g * sgn
            acc[kj] += g * sgn
            tot += g * math.sqrt(2.0)                        # ||(+g, -g)||
            n += 1
    if n < 2 or tot <= 0:
        return None, n
    net = math.sqrt(sum(v * v for v in acc.values()))
    return net / tot, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--restrict-to", default=None)
    a = ap.parse_args()

    keep = None
    if a.restrict_to:
        keep = set()
        with open(a.restrict_to) as fh:
            for r in csv.DictReader(fh):
                keep.add((r.get("pdb") or r.get("pdb_id") or next(iter(r.values()))).lower())

    data = load(a.results, keep)
    arms = (("baseline  cross-seed, tau 0.0305", "cross", 0.0305, 0.0),
            ("straddle  all pairs, tau 0.1429", "all", 0.1429, 0.23),
            ("control   all pairs, tau 0.1429, NO straddle", "all", 0.1429, 0.0),
            ("control   cross-seed, tau 0.0305, straddle", "cross", 0.0305, 0.23))
    print(f"source {a.results}   {len(data)} interface(s)\n")
    print(f"{'arm':46s} {'ifaces':>7} {'pairs/iface':>12} {'alignment':>10} {'1/sqrt(N)':>10} {'ratio':>7}")
    for name, pairs, tau, straddle in arms:
        al, ns = [], []
        for ifc, poses in data.items():
            v, n = conflict(poses, pairs, tau, straddle)
            if v is not None:
                al.append(v)
                ns.append(n)
        if not al:
            print(f"{name:46s} {'--':>7}")
            continue
        base = st.mean(1.0 / math.sqrt(n) for n in ns)
        m = st.mean(al)
        print(f"{name:46s} {len(al):7d} {st.mean(ns):12.0f} {m:10.4f} {base:10.4f} {m/base:7.2f}x")


if __name__ == "__main__":
    main()
