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
    --matrices:  one or more directories of .npz files, colon-separated. Earlier entries win. Two are
                 needed in practice: the anchor frame for multi-interface complexes and the crystal
                 frame for the single-interface ones, which were never rebuilt because for one chain
                 pair the frames are identical. Reading only one silently drops 50 complexes.
    --classes:   which classes to report. Default decoy,solved,nocov.
    --out:       write the table to this markdown file as well as stdout. Optional.

OUTPUTS
    stdout: one row per (class, pooling rule) -- interfaces, mean DockQ before and after, how many were
            acceptable before and after, rescued, broken
    --out:  the same table as markdown

EXAMPLE
    score_experiment.py --exp experimental_data/reffree_50 \\
        --matrices experimental_data/matrices_anchor_all:experimental_data/matrices \\
        --out experimental_results/reffree_50.md

FUNCTIONS
    pool_classes():   the free pool -> {interface: solved | decoy | nocov}
    load_scored():    an experiment -> the templated (ipTM, DockQ) per interface, and which are complete
    score():          those -> the table rows
    main():           read, score, print, optionally write markdown
"""
import argparse
import collections
import csv
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
import pool_index  # noqa: E402
from fmt import table  # noqa: E402

ACCEPTABLE = 0.23          # CAPRI "acceptable"


def pool_classes(pool_csv=None):
    """in: the pool csv.  out: {interface: solved | decoy | nocov}"""
    best = collections.defaultdict(lambda: (-1.0, 0.0))
    anygood = collections.defaultdict(bool)
    for iface, rows in pool_index.load_pool(pool_csv).items():
        for _, _, dq, ip in rows:
            if ip > best[iface][0]:
                best[iface] = (ip, dq)
            anygood[iface] |= dq >= ACCEPTABLE
    return {k: ("solved" if v[1] >= ACCEPTABLE else "decoy" if anygood[k] else "nocov")
            for k, v in best.items()}


def load_scored(exp):
    """in: the experiment directory.
    out: (templated (iptm, dockq) per interface, tags the queue asked for, tags that have rows)
    """
    expected, qtags = collections.Counter(), collections.defaultdict(set)
    for line in open(f"{exp}/queue.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6:
            expected[f[2]] += 1
            qtags[f[2]].add(f[3])
    made, folds = collections.defaultdict(list), collections.defaultdict(set)
    for r in csv.DictReader(open(f"{exp}/results.csv")):
        if r["dockq"] and r["iptm"]:
            made[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
            folds[r["iface"]].add(r["arm"])
    return made, expected, qtags, folds


def score(made, expected, qtags, folds, matrices, classes, cls):
    """in: the loaded experiment, the matrix directories, the classes to report.
    out: (table rows, how many interfaces were excluded as unfinished)
    """
    # earlier directories win, so the anchor frame takes precedence where both hold an interface
    files = {}
    for d in [d for d in matrices.split(":") if d]:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            files.setdefault(os.path.basename(f)[:-4], f)

    acc = collections.defaultdict(lambda: [[], [], []])
    partial = 0
    for iface, f in sorted(files.items()):
        if iface not in made:
            continue
        # the cap varies per interface -- the diversity filter exhausts the pool before 50 on most of
        # them -- so completeness is measured against what the queue asked for, not a constant
        if len(folds[iface] & qtags.get(iface, set())) < expected.get(iface, 0):
            partial += 1
            continue
        z = np.load(f)
        free = list(zip(z["pair_iptm"].astype(float), z["dockq_gt"].astype(float)))
        before = float(max(free)[1])
        pooled = free + made[iface]
        after = float(pooled[int(np.argmax([x for x, _ in pooled]))][1])
        tmpl = made[iface]
        tonly = float(tmpl[int(np.argmax([x for x, _ in tmpl]))][1]) if tmpl else before
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
    ap.add_argument("--matrices",
                    default=f"{paths.ANCHOR_MATRICES}:{paths.XTAL_MATRICES}")
    ap.add_argument("--classes", default="decoy,solved,nocov")
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    cls = pool_classes(A.pool)
    made, expected, qtags, folds = load_scored(A.exp)
    rows, partial = score(made, expected, qtags, folds, A.matrices,
                          A.classes.split(","), cls)
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
