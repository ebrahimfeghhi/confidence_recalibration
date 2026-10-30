#!/usr/bin/env python
"""The 2x2: {in-sample, held out} x {this head, shipped}, all four scored on the SAME crop.

READS THE ROWS, NOT THE JSONS. The per-pass json carries mean DockQ and frac acceptable, but the
comparison this answers is PAIRED -- the same interfaces, the same candidate poses, two scoring
functions -- so the deltas and their t statistics have to be built from the rows. Spearman is only
in the rows at all.

THE SHIPPED COLUMN IS A CROPPED PASS, not the stored_iptm column. stored_iptm is the shipped
head's recorded ipTM from the ordinary full-structure pipeline; using it here would compare a
cropped head against an uncropped baseline, which is the asymmetry section 8c is about.

PAIRING IS ON (iface, seed, sample). A pass that dropped a unit -- a failed pose join, say --
would otherwise shift one column's population and turn a difference in coverage into a difference
in DockQ. Only interfaces complete in BOTH passes are reported, and the count is printed.
"""
import csv
import glob
import math
import os
import statistics as st
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "training", "runs", "cropped_eval")
ACC = 0.23


def spearman(xs, ys):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                out[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    dx = sum((p - mx) ** 2 for p in rx) ** 0.5
    dy = sum((q - my) ** 2 for q in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def load(tag):
    """{iface: {(seed, sample): (iptm_max, iptm_mean, dockq)}} for one pass."""
    g = defaultdict(dict)
    for f in glob.glob(os.path.join(D, tag, "rows_*_s*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                g[r["iface"]][(int(r["seed"]), int(r["sample"]))] = (
                    float(r["iptm"]), float(r["iptm_mean"]), float(r["dockq"]))
            except (TypeError, ValueError, KeyError):
                continue
    return g


def paired_t(d):
    n = len(d)
    if n < 2:
        return float("nan"), float("nan")
    m = sum(d) / n
    sd = st.stdev(d)
    se = sd / math.sqrt(n) if sd else 0.0
    return m, (m / se if se else float("inf") if m else 0.0)


for tag_pool, ours_tag, ship_tag, label in (
        ("insample", "insample_ours_ep30", "insample_shipped", "IN-SAMPLE  738 interfaces, budget 10 (all 50 training poses)"),
        ("valfull", "valfull_ours_ep30", "valfull_shipped", "HELD OUT  valfull, budget 30 (150 candidates)")):
    A, B = load(ours_tag), load(ship_tag)
    if not A or not B:
        print(f"\n  {label}\n    not evaluated yet ({ours_tag} / {ship_tag})")
        continue
    full = max(len(v) for v in A.values())
    keep = sorted(k for k in set(A) & set(B)
                  if len(A[k]) == full and set(A[k]) == set(B[k]))
    print(f"\n  {label}")
    print(f"    {len(keep)} interface(s) complete and paired in BOTH passes, {full} candidates each")
    print(f"      {'selector':>22}  {'meanDockQ':>10} {'fracAcc':>8} {'rho':>8}   {'vs shipped':>11} {'t':>7}")
    base = {}
    for who, G in (("shipped", B), ("ours", A)):
        for pool, j in (("MAX", 0), ("MEAN", 1)):
            sel, acc, rho = [], [], []
            for k in keep:
                v = G[k]
                ks = sorted(v)
                s = [v[q][j] for q in ks]
                d = [v[q][2] for q in ks]
                b = max(range(len(ks)), key=lambda i: s[i])
                sel.append(d[b])
                acc.append(1.0 if d[b] >= ACC else 0.0)
                rho.append(spearman(s, d))
            name = f"{who}, {pool}-pooled"
            key = (who, pool)
            base[key] = sel
            if who == "shipped":
                print(f"      {name:>22}  {sum(sel)/len(sel):>10.4f} {sum(acc)/len(acc):>8.4f} "
                      f"{st.median(rho):>+8.4f}")
            else:
                ref = base[("shipped", pool)]
                m, t = paired_t([x - y for x, y in zip(sel, ref)])
                print(f"      {name:>22}  {sum(sel)/len(sel):>10.4f} {sum(acc)/len(acc):>8.4f} "
                      f"{st.median(rho):>+8.4f}   {m:>+11.4f} {t:>+7.2f}")
    # THE HEADLINE: held-out Spearman under matched cropping, ours minus shipped, same pooling.
    for pool, j in (("MEAN", 1), ("MAX", 0)):
        da, db = [], []
        for k in keep:
            ks = sorted(A[k])
            da.append(spearman([A[k][q][j] for q in ks], [A[k][q][2] for q in ks]))
            db.append(spearman([B[k][q][j] for q in ks], [B[k][q][2] for q in ks]))
        m, t = paired_t([x - y for x, y in zip(da, db)])
        print(f"      Spearman {pool}, ours minus shipped (paired): {m:+.4f}  t={t:+.2f}  "
              f"median {st.median(da) - st.median(db):+.4f}")
