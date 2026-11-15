#!/usr/bin/env python
"""WHERE THE ORACLE GAP COMES FROM on the lists the model gets wrong.

A list is WINNABLE when at least one candidate is acceptable (DockQ >= 0.23), and FAILED when it is
winnable and the model's argmax pose is not acceptable. The gap those failures carry is
oracle - selected, summed over the pool, and the question is which of two stories explains it:

  MINORITY HIT   the acceptable poses are a small, scattered minority -- the generator got lucky in
                 one seed and the list is near-unwinnable in practice.
  UNDERCONFIDENT the acceptable poses are plentiful, and the model still ranks a bad one first.

So every failed list is reported with: how many of its candidates are acceptable, how concentrated
they are (the within-seed share of the seed holding the best acceptable pose), where the best
acceptable pose ranks by score, and the score gap it would have to close to win.
"""
import argparse
import collections
import csv
import glob
import statistics

ACC = 0.23


def load(paths, score_col):
    rows = collections.defaultdict(list)
    for p in paths:
        for r in csv.DictReader(open(p)):
            try:
                rows[r["iface"]].append((float(r[score_col]), float(r["dockq"]), r["seed"],
                                         int(r["sample"])))
            except (TypeError, ValueError, KeyError):
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="shard csv glob")
    ap.add_argument("--score", default="iptm", help="iptm (ours) or stored_iptm (shipped)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    rows = load(sorted(glob.glob(a.glob)), a.score)
    win, fail = [], []
    for iface, cand in rows.items():
        if not any(d >= ACC for _, d, _, _ in cand):
            continue
        top = max(cand, key=lambda t: t[0])
        win.append(iface)
        if top[1] >= ACC:
            continue
        acc = [c for c in cand if c[1] >= ACC]
        best = max(acc, key=lambda t: t[1])
        by_seed = collections.Counter(s for _, _, s, _ in cand)
        acc_in_seed = sum(1 for _, _, s, _ in acc if s == best[2])
        order = sorted(cand, key=lambda t: -t[0])
        rank = min(order.index(c) for c in acc) + 1
        fail.append(dict(iface=iface, n=len(cand), n_acc=len(acc),
                         acc_seed_share=acc_in_seed / by_seed[best[2]],
                         n_seeds_with_acc=len({s for _, _, s, _ in acc}),
                         n_seeds=len(by_seed), rank=rank,
                         score_gap=top[0] - max(c[0] for c in acc),
                         gap=max(d for _, d, _, _ in cand) - top[1]))

    print(f"=== {a.label or a.glob} [{a.score}] ===")
    print(f"winnable lists: {len(win)}   failed: {len(fail)} ({len(fail)/max(1,len(win)):.1%})")
    if not fail:
        return
    tot = sum(f["gap"] for f in fail)
    print(f"DockQ they leave on the table: {tot:.2f} total, {tot/len(win):.4f} per winnable list")
    # THE SPLIT THE QUESTION ASKS FOR: a lucky hit is a minority inside its own seed AND rare
    # overall; anything else the model simply ranked wrong.
    lucky = [f for f in fail if f["n_acc"] <= 0.1 * f["n"] and f["acc_seed_share"] <= 0.4]
    plenty = [f for f in fail if f["n_acc"] >= 0.3 * f["n"]]
    mid = [f for f in fail if f not in lucky and f not in plenty]
    for name, g in (("MINORITY HIT (<=10% acceptable, <=2 of 5 in its seed)", lucky),
                    ("MIDDLE", mid),
                    ("PLENTY ACCEPTABLE (>=30% of candidates)", plenty)):
        if not g:
            print(f"  {name}: 0")
            continue
        print(f"  {name}: {len(g)} list(s), {sum(f['gap'] for f in g):.2f} DockQ "
              f"({sum(f['gap'] for f in g)/tot:.0%} of the gap)")
        print(f"      median acceptable {statistics.median(f['n_acc'] for f in g):.0f} of "
              f"{statistics.median(f['n'] for f in g):.0f} candidates, "
              f"in {statistics.median(f['n_seeds_with_acc'] for f in g):.0f} of "
              f"{statistics.median(f['n_seeds'] for f in g):.0f} seed(s)")
        print(f"      best acceptable ranks {statistics.median(f['rank'] for f in g):.0f} by score; "
              f"score gap to close {statistics.median(f['score_gap'] for f in g):.4f}")
    print(f"  overall: best acceptable pose already in the top 5 by score in "
          f"{sum(1 for f in fail if f['rank'] <= 5)} of {len(fail)} failed list(s)")


if __name__ == "__main__":
    main()
