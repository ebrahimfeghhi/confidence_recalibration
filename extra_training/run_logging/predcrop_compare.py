#!/usr/bin/env python
"""OURS vs SHIPPED on the SAME label-free crop, paired per interface.

WHY THIS EXISTS. The predicted-interface eval ran four passes: {ours, shipped} x {5 A, 2.5 A}.
The only control that isolates the head is the SHIPPED head scored on the IDENTICAL crop -- not
the stored uncropped ipTM, which differs from both by the crop itself. Quoting `baseline_mean_dockq`
from a cropped pass would silently compare across two different scoring inputs.

Pairs on iface, so the t statistic is over the 92 interfaces both passes scored. Refuses if the
two passes do not cover the same interfaces or the same pose set, because a partial shard would
otherwise read as a head difference.

    python -I run_logging/predcrop_compare.py training/runs/predcrop_eval 5 2.5
"""
import csv, glob, math, os, statistics as st, sys
from collections import defaultdict


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "rows_b30_*.csv")))
    if not fs:
        raise SystemExit(f"STOP: no rows_b30_*.csv under {d}")
    by = defaultdict(dict)
    for f in fs:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                key = (r["seed"], r["sample"])
                if key in by[r["iface"]]:
                    raise SystemExit(f"STOP: duplicate pose {r['iface']} {key} in {d}")
                by[r["iface"]][key] = (float(r["iptm"]), float(r["iptm_mean"]),
                                       float(r["dockq"]), float(r["stored_iptm"]))
    return by


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
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def paired_t(d):
    n = len(d)
    m = sum(d) / n
    if n < 2:
        return m, float("nan")
    s = st.stdev(d)
    return m, (m / (s / math.sqrt(n)) if s > 0 else float("nan"))


root = sys.argv[1] if len(sys.argv) > 1 else "training/runs/predcrop_eval"
margins = sys.argv[2:] or ["5", "2.5"]

print(f"  OURS vs SHIPPED on the same label-free union crop, paired per interface")
print(f"  {'margin':>7} {'pooling':>8} {'ours':>8} {'shipped':>8} {'delta':>8} {'t':>7}"
      f" {'rho ours':>9} {'rho ship':>9} {'d rho':>8} {'t':>7}")
for mg in margins:
    o = load(os.path.join(root, f"ours_m{mg}"))
    s = load(os.path.join(root, f"shipped_m{mg}"))
    ifs = sorted(set(o) & set(s))
    if set(o) != set(s):
        raise SystemExit(f"STOP: margin {mg} interface sets differ "
                         f"(ours {len(o)}, shipped {len(s)}, shared {len(ifs)})")
    for pool, idx in (("MEAN", 1), ("MAX", 0)):
        do, ds, ro, rs = [], [], [], []
        for i in ifs:
            po, ps = o[i], s[i]
            if set(po) != set(ps):
                raise SystemExit(f"STOP: {i} at margin {mg} has different pose sets")
            ks = sorted(po)
            do.append(max((po[k][idx], po[k][2]) for k in ks)[1])
            ds.append(max((ps[k][idx], ps[k][2]) for k in ks)[1])
            dq = [po[k][2] for k in ks]
            ro.append(spearman([po[k][idx] for k in ks], dq))
            rs.append(spearman([ps[k][idx] for k in ks], dq))
        md, t = paired_t([a - b for a, b in zip(do, ds)])
        mr, tr = paired_t([a - b for a, b in zip(ro, rs)])
        print(f"  {mg+' A':>7} {pool:>8} {sum(do)/len(do):8.4f} {sum(ds)/len(ds):8.4f}"
              f" {md:+8.4f} {t:+7.2f} {st.median(ro):+9.4f} {st.median(rs):+9.4f}"
              f" {mr:+8.4f} {tr:+7.2f}")
    print(f"          n={len(ifs)} interface(s), {len(o[ifs[0]])} pose(s) each")
