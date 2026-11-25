#!/usr/bin/env python
"""The three within-seed statistics for PREDICTED ipTM and GROUND-TRUTH-PAE ipTM, side by side.

Reads run_logging/true_pae_iptm.py's per-pose csv and reports, for each score, the pooled Pearson
on within-seed-centred values, the mean per-group Spearman and the within-seed pairwise ranking
accuracy -- computed by within_seed_iptm_dockq.stats, the same function that produced the numbers
for the predicted score, so the two columns are not two implementations of one definition.

Also reports the oracle: the fraction of groups where the pose each score picks is the group's own
best by DockQ, against 1/5 = 0.200 for a blind pick. Selection, not correlation, is what the arm
has to beat.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))


def top1(groups, which):
    """Fraction of groups whose argmax by `which` is the group's best DockQ, and the mean DockQ
    that choice realises. Ties on the score go to the LOWEST sample index, which is what
    phase_k_eval.py:795 does, so the number is comparable with every Success@1 in the repo."""
    hit, dq = 0, []
    for g in groups:
        order = sorted(range(len(g["dockq"])), key=lambda i: (-g[which][i], i))
        pick = order[0]
        dq.append(g["dockq"][pick])
        if g["dockq"][pick] >= max(g["dockq"]) - 1e-12:
            hit += 1
    return hit / len(groups), sum(dq) / len(dq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--train-structures",
                    default=f"{ROOT}/train_val_splits/cache_train_structures.csv")
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--min-spread", type=float, default=0.0)
    a = ap.parse_args()

    from within_seed_iptm_dockq import stats

    trainset = set()
    if os.path.exists(a.train_structures):
        with open(a.train_structures) as fh:
            for r in csv.DictReader(fh):
                trainset.add(r["pdb"])

    g = collections.defaultdict(lambda: {"dockq": [], "iptm_pred": [], "iptm_true": [],
                                         "sample": []})
    with open(a.rows) as fh:
        for r in csv.DictReader(fh):
            k = (r["pdb"], r["iface"], int(r["seed"]))
            d = g[k]
            d["dockq"].append(float(r["dockq"]))
            d["iptm_pred"].append(float(r["iptm_pred"]))
            d["iptm_true"].append(float(r["iptm_true"]))
            d["sample"].append(int(r["sample"]))

    buckets = collections.defaultdict(list)
    for (pdb, _iface, _seed), d in g.items():
        if len(d["dockq"]) != int(a.poses):
            continue
        if max(d["dockq"]) - min(d["dockq"]) < float(a.min_spread):
            continue
        split = "train" if pdb in trainset else "heldout"
        buckets["both"].append(d)
        buckets[split].append(d)

    print(f"groups: " + "  ".join(f"{k} {len(v)}" for k, v in sorted(buckets.items()))
          + (f"   (DockQ spread >= {a.min_spread})" if a.min_spread else ""))
    print(f"{'subset':10s} {'score':10s} {'pearson':>9s} {'spearman':>9s} {'pair_acc':>9s} "
          f"{'top1':>7s} {'mean_dq':>8s}")
    for sub in ("both", "train", "heldout"):
        gs = buckets.get(sub)
        if not gs:
            continue
        for which in ("iptm_pred", "iptm_true"):
            pairs = [(d["dockq"], d[which]) for d in gs]
            _n, _np, pr, _pv, rho, _nr, acc, _npair = stats(pairs)
            t1, mdq = top1(gs, which)
            print(f"{sub:10s} {which:10s} {pr:+9.4f} {rho:+9.4f} {acc:9.4f} "
                  f"{t1:7.4f} {mdq:8.4f}")
        # ORACLE AND FLOOR for the same groups, so the two scores can be read against the range
        # they are competing in rather than against each other alone.
        orc = sum(max(d["dockq"]) for d in gs) / len(gs)
        flr = sum(sum(d["dockq"]) / len(d["dockq"]) for d in gs) / len(gs)
        print(f"{sub:10s} {'oracle':10s} {'':>9s} {'':>9s} {'':>9s} {1.0:7.4f} {orc:8.4f}")
        print(f"{sub:10s} {'random':10s} {'':>9s} {'':>9s} {0.5:9.4f} {0.2:7.4f} {flr:8.4f}")


if __name__ == "__main__":
    main()
