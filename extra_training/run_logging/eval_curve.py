#!/usr/bin/env python
"""Per-epoch held-out numbers for one arm on the 83-structure fold, and the shipped reference.

ONE PLACE THAT DEFINES THE HELD-OUT SERIES, so the plot cannot drift from what the evaluator wrote.

ipTM COMES IN TWO POOLINGS AND THEY ARE NOT INTERCHANGEABLE. An interface's confidence pools a
chain-pair matrix over anchor rows: MAX is what shipped ipTM is and what every prior baseline used,
MEAN is what this arm trains and selects on (every metric json here carries select_by "mean").
The row files carry both -- `iptm` is max-pooled, `iptm_mean` is mean-pooled -- and reading the
wrong one is not a cosmetic error: the arm's within-interface rho is +0.152 on `iptm` and +0.174 on
`iptm_mean`, so plotting `iptm` understated the arm 30-fold while the title claimed it was showing
the quantity selection uses. results.csv has no mean-pooled column at all, so the shipped reference
is max-pooled only and is labelled as such rather than quietly compared across poolings.

SPEARMAN NEEDS AVERAGE RANKS. argsort(argsort(x)) gives ORDINAL ranks, which differ whenever two
poses tie -- and shipped's ipTM is stored at 4 dp, so 25.7% of its per-interface values tie against
0.05% of the arm's recomputed ones. Under ordinal ranks shipped's median moved over a 0.0103 range
across random row orders, eleven times the difference being reported. Ties are averaged here.
"""
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KEYS = ("mean_dockq", "mean_dockq_select_max", "frac_acc", "frac_acc_select_max",
        "oracle_mean_dockq", "oracle_frac_acc")


def _comb(a, b, k):
    na, nb = a["n_interfaces"], b["n_interfaces"]
    return (na * a[k] + nb * b[k]) / (na + nb)


def shipped():
    """The shipped head over the same 141 interfaces, or None."""
    f1 = os.path.join(ROOT, "training/valfull/metric_shipped_kept30.json")
    f2 = os.path.join(ROOT, "training/valfull83/metric_shipped_val30.json")
    if not (os.path.exists(f1) and os.path.exists(f2)):
        return None
    A, B = json.load(open(f1)), json.load(open(f2))
    return {k: _comb(A, B, k) for k in KEYS if k in A and k in B}


def series(chain):
    """{epoch: metric dict} over the arm's whole resume chain, newest run winning."""
    out = {}
    for rid in chain:
        for f in sorted(glob.glob(os.path.join(ROOT, "training/runs", rid,
                                               "metric_valfull83_ep*.json"))):
            m = re.search(r"_ep(\d+)\.json$", f)
            d = json.load(open(f))
            if m and d.get("n_interfaces") == 141:
                out[int(m.group(1))] = d
    # epoch 5 predates the 83 pool and was scored as two passes. Combining them is exact, not an
    # approximation: aggregate() picks a pose by argmax WITHIN one interface and holds no pool-wide
    # normalisation or threshold, so adding interfaces cannot change an already-scored pick.
    a = os.path.join(ROOT, "training/runs/5dgrsu/metric_valfull_ep5.json")
    b = os.path.join(ROOT, "training/runs/5dgrsu/metric_ep5_val30.json")
    if 5 not in out and os.path.exists(a) and os.path.exists(b):
        A, B = json.load(open(a)), json.load(open(b))
        out[5] = {k: _comb(A, B, k) for k in KEYS} | {"n_interfaces": 141, "_combined": True}
    return out


def spearman(x, y):
    """Tie-aware Spearman. nan if undefined."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")

    def avg_rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        s = v[o]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and s[j + 1] == s[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    rx, ry = avg_rank(x), avg_rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum())
    return float((rx * ry).sum() / d) if d else float("nan")


def median_within(by_iface, col):
    """Median Spearman computed INSIDE each interface -- the comparison the argmax makes."""
    v = [spearman([p[col] for p in u], [p["dockq"] for p in u])
         for u in by_iface.values() if len(u) >= 8]
    v = [r for r in v if r == r]
    return float(np.median(v)) if v else float("nan")


def rows_for_epoch(rid_dirs, epoch):
    """{iface: [{iptm, iptm_mean, dockq, stored_iptm}]} from an epoch's shard row files."""
    by = collections.defaultdict(list)
    for d in rid_dirs:
        for f in glob.glob(os.path.join(ROOT, "training/runs", d, f"*_ep{epoch}", "rows_*.csv")):
            for r in csv.DictReader(open(f)):
                try:
                    by[r["iface"]].append({k: float(r[k]) for k in
                                           ("iptm", "iptm_mean", "dockq", "stored_iptm")})
                except (KeyError, TypeError, ValueError):
                    pass
    return by


def rho_series(chain):
    """{epoch: {"mean": rho, "max": rho}} wherever row files exist, plus the shipped reference."""
    out = {}
    for ep in range(1, 200):
        by = rows_for_epoch(chain, ep)
        if by:
            out[ep] = {"mean": median_within(by, "iptm_mean"),
                       "max": median_within(by, "iptm")}
    # shipped from the SAME rows' stored_iptm column: it is results.csv's own recorded value,
    # verified identical to the fold's at 0.0 max difference over all 21150 rows.
    shp = float("nan")
    for ep in sorted(out, reverse=True):
        by = rows_for_epoch(chain, ep)
        if by:
            shp = median_within(by, "stored_iptm")
            break
    return out, shp
