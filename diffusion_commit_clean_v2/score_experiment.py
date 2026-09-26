#!/usr/bin/env python
"""Score a self-templating experiment: how many confident decoys became acceptable, and what broke.

SUMMARY
    The only step that reads ground truth. Everything before it is reference-free.

    For each interface it compares two selections, both by top pooled pair_iptm:

      before:  the best-confidence pose among the FREE poses alone
      after:   the best-confidence pose once the templated poses are added

    rescued = was below 0.23 before and is at or above it after. broken = the reverse.

    TWO POOLING RULES, both reported, because which one is right depends on the class. "templated only"
    selects among the templated poses; "free + templated" keeps the free pool in the running. On decoys
    the free pool has nothing acceptable by definition, so pooling it back in can only let a confident
    free decoy outrank a rescued pose -- measured at 4 lost rescues. On solved interfaces the free pool
    holds the good pose and is worth keeping as a floor.

    CLASSES come from the free pool alone: solved if its most confident pose is acceptable, decoy if not
    but some pose is, nocov if none is. Pooled across seeds, because a confident decoy is defined by the
    most confident pose in the whole pool being wrong while a good pose exists.

    PARTIAL INTERFACES ARE EXCLUDED. An interface is only counted once every donor the queue selected
    for it has been folded, so a half-finished sweep does not read as a weak result.

INPUTS
    --exp:       the experiment directory: queue.tsv and results.csv. Required.
    --pool-seeds: how many seeds per complex define the free baseline and the classes. Default 50.
    --pool:      the pool csv. Default paths.PXM2025_POOL_CSV.

    THE BASELINE COMES FROM THE POOL CSV, not the selection matrices. The matrices' pair_iptm is read
    at the anchor-frame chains (what selection needs) while their dockq_gt is at the reference-matched
    chains, so pairing them mixes two interfaces; they also omit poses with no contact. Both reporters
    now share pool_by_seed/classes_at/load_scored.
    --classes:   which classes to report. Default decoy,solved,nocov.
    --out:       write the table to this markdown file as well as stdout. Optional.

OUTPUTS
    stdout: one row per (class, pooling rule) -- interfaces, mean DockQ before and after, how many were
            acceptable before and after, rescued, broken
    --out:  the same table as markdown

EXAMPLE
    score_experiment.py --exp experimental_data/reffree_50 \\
        --out experimental_results/reffree_50.md

FUNCTIONS
    pool_by_seed():   the pool csv -> its rows and each complex's first-n seeds
    classes_at():     the first-n pool -> {interface: solved | decoy | nocov}
    load_scored():    an experiment -> the templated (ipTM, DockQ) per interface, and which are unfinished
    score():          those -> the table rows
    main():           read, score, print, optionally write markdown
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
import pool_index  # noqa: E402
from fmt import table  # noqa: E402

ACCEPTABLE = 0.23          # CAPRI "acceptable"


def pool_by_seed(n_seeds, pool_csv=None):
    """in: how many seeds define the pool, the pool csv.  out: (pool rows, {complex: first-n seed ids})

    The baseline and the classes both use the FIRST n seeds per complex, never the whole pool: the pool
    holds up to 95 seeds on some complexes, and labelling on those lets extra free seeds decide what
    counts as a decoy before any method runs. Both reporters (this file and compare_arms.py) use this.
    """
    pool = pool_index.load_pool(pool_csv)
    seen = collections.defaultdict(set)
    for iface, rows in pool.items():
        for s, _, _, _ in rows:
            seen[iface.split("_")[0]].add(s)
    return pool, {p: set(sorted(v, key=int)[:n_seeds]) for p, v in seen.items()}


def classes_at(pool, first):
    """in: pool rows and the first-n seed sets.  out: {interface: solved | decoy | nocov}

    Evaluation, so it reads ground truth: pair_iptm and dockq here are both measured at the chains
    matched to the reference, which keeps each (confidence, correctness) pair about one interface.
    """
    out = {}
    for iface, rows in pool.items():
        keep = [r for r in rows if r[0] in first[iface.split("_")[0]]]
        if not keep:
            continue
        best = max(keep, key=lambda r: r[3])
        out[iface] = ("solved" if best[2] >= ACCEPTABLE
                      else "decoy" if any(r[2] >= ACCEPTABLE for r in keep) else "nocov")
    return out


def load_scored(exp):
    """in: the experiment directory.
    out: ({interface: [(iptm, dockq)]} templated rows, {interfaces not yet complete})

    COMPLETE means every fold the queue lists for the COMPLEX has rows on the interface. A fold scores
    every interface of its complex (2,794 of 2,794 folds did, 5 samples each), so an interface's
    templated set is only whole once all its complex's folds are in. The previous test counted only
    the folds MOTIVATED by the interface, so an interface passed while its complex's other folds were
    unscored, and one that motivated no fold always passed.
    """
    tags = collections.defaultdict(set)
    for line in open(f"{exp}/queue.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6:
            tags[f[1]].add(f[3])
    made, folds = collections.defaultdict(list), collections.defaultdict(set)
    for r in csv.DictReader(open(f"{exp}/results.csv")):
        if r["dockq"] and r["iptm"]:
            made[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
            folds[r["iface"]].add(r["arm"])
    unfinished = {i for i in made if not tags[i.split("_")[0]] <= folds[i]}
    return made, unfinished


def score(pool, first, made, unfinished, classes, cls):
    """in: pool rows, first-n seeds, the loaded experiment, the classes to report, the class labels.
    out: (table rows, how many interfaces were excluded as unfinished)
    """
    acc = collections.defaultdict(lambda: [[], [], []])
    partial = 0
    for iface, rows in sorted(pool.items()):
        if cls.get(iface) not in classes or iface not in made:
            continue
        if iface in unfinished:
            partial += 1
            continue
        free = [(ip, dq) for s, _, dq, ip in rows if s in first[iface.split("_")[0]]]
        before = max(free)[1]
        pooled = free + made[iface]
        after = max(pooled)[1]
        tonly = max(made[iface])[1]
        k = cls.get(iface, "?")
        acc[k][0].append(before)
        acc[k][1].append(after)
        acc[k][2].append(tonly)

    rows = []
    for k in classes:
        b, a, t = (np.array(x) for x in acc[k])
        if not len(b):
            continue
        for name, v in (("templated only", t), ("free + templated", a)):
            rows.append([k, name, len(b), f"{b.mean():.3f}", f"{v.mean():.3f}",
                         f"{int((b >= ACCEPTABLE).sum())}/{len(b)}",
                         f"{int((v >= ACCEPTABLE).sum())}/{len(v)}",
                         int(((b < ACCEPTABLE) & (v >= ACCEPTABLE)).sum()),
                         int(((b >= ACCEPTABLE) & (v < ACCEPTABLE)).sum())])
    return rows, partial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--pool-seeds", type=int, default=50)
    ap.add_argument("--classes", default="decoy,solved,nocov")
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    pool, first = pool_by_seed(A.pool_seeds, A.pool)
    cls = classes_at(pool, first)
    made, unfinished = load_scored(A.exp)
    rows, partial = score(pool, first, made, unfinished, A.classes.split(","), cls)
    head = ["class", "pooling", "interfaces", "mean before", "mean after",
            "acceptable before", "acceptable after", "rescued", "broken"]
    out = table(head, rows, align="l l r r r r r r r")
    # each class contributes two rows, one per pooling rule, so halve the interface total
    n_scored = sum(int(r[2]) for r in rows) // 2
    n_q = sum(1 for line in open(f"{A.exp}/queue.tsv") if line.strip())
    tail = (f"\n  experiment: {A.exp}\n"
            f"  interfaces scored: {n_scored}   excluded as unfinished: {partial}\n"
            f"  queue length: {n_q} folds")
    print(out)
    print(tail)
    if A.out:
        os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
        open(A.out, "w").write(f"# {os.path.basename(A.exp)}\n\n{out}\n{tail}\n")
        print(f"\n  wrote {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
