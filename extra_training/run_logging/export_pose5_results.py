#!/usr/bin/env python
"""The live arm's eval passes as a CSV, for review outside this repo.

ONE ROW PER COMPLETED PASS, val and slice, carrying BOTH ipTM poolings and -- the point of the
file -- the margin beside the score spread it is measured in. Margin is in ipTM units, so it
inflates or deflates with the head's score scale whether or not the ranking changed; the
margin_over_spread column is what makes the two comparable across epochs and across arms.
"""
import collections
import csv
import glob
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_fullset as P                                            # noqa: E402

COLS = ["side", "epoch", "q_max", "top1_max", "rho_max", "q_mean", "top1_mean", "rho_mean",
        "margin", "score_spread", "margin_over_spread",
        "shipped_q", "shipped_top1", "shipped_rho", "shipped_margin"]


def _complete(pat):
    """epoch -> dir, for passes with all three shard markers. A dir read between a shard's rows
    and its marker is short by a third of the pool, which moves the mean without looking wrong."""
    w = {}
    for d in glob.glob(pat):
        m = re.search(r"ep(\d+)$", d)
        if m and len(glob.glob(os.path.join(d, "rows_*.csv.meta.json"))) == 3:
            w[int(m.group(1))] = d
    return w


def _spread(d):
    by = collections.defaultdict(list)
    for f in glob.glob(os.path.join(d, "rows_*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                by[r["iface"]].append(float(r["iptm"]))
            except (TypeError, ValueError, KeyError):
                pass
    return st.median([max(v) - min(v) for v in by.values()]) if by else float("nan")


def main():
    rid = sys.argv[1] if len(sys.argv) > 1 else open("training/runs/pose5_arm_rid").read().strip()
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/pose5_results/jyz3g2_val_slice.csv"
    sides = (("val", f"training/runs/{rid}/valfull83_eval_shards_ep*"),
             ("slice", f"training/runs/{rid}/slice280_eval_shards_ep*"))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows, seen = [], {}
    for side, pat in sides:
        dirs = _complete(pat)
        pts = P.score_points(dirs)
        seen[side] = sorted(pts)
        for e in sorted(pts):
            d, sp = pts[e], _spread(dirs[e])
            rows.append([side, e, f"{d['q']:.4f}", f"{d['top1']:.4f}", f"{d['rho']:.4f}",
                         f"{d['q_mean']:.4f}", f"{d['top1_mean']:.4f}", f"{d['rho_mean']:.4f}",
                         f"{d['margin']:.4f}", f"{sp:.4f}", f"{d['margin'] / sp:.4f}",
                         f"{d['s_q']:.4f}", f"{d['s_top1']:.4f}", f"{d['s_rho']:.4f}",
                         f"{d['s_margin']:.4f}"])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} row(s); val {seen['val']}, slice {seen['slice']}")


if __name__ == "__main__":
    main()
