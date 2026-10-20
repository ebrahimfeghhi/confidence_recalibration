#!/usr/bin/env python
"""Spearman(ipTM, DockQ) per interface, for a trained head against the shipped one.

WHY THIS EXISTS. val10's mean DockQ was flat -- epoch 133 read 0.2251 against the shipped head's
0.2626, inside the 0.2098-0.3045 spread of ten earlier passes -- while the training PAE had fallen
37%. "Flat" invites the reading that nothing happened. It is wrong: the mean hides large offsetting
per-interface moves, and this is what showed that.

WHAT IT MEASURES, AND WHY NOT PAE. The shard row files an evaluation leaves behind already carry
everything needed: the trained head's recomputed ipTM, results.csv's stored ipTM (which IS the
shipped head's selection statistic), and the true DockQ, for every pose. So ranking quality is
measurable directly, with no labels, no GPU and no second pass -- unlike the held-out PAE loss,
which needed labels built for the validation fold first.

Rank correlation rather than the selected pose's DockQ: selection is one pick per interface and
therefore noisy, while the correlation uses all 500 poses. The selected value is printed beside it
as corroboration, not as independent evidence.

WHAT IT FOUND on training/runs/2u8tf4/manual_eval_ep133:
  mean rho trained 0.1032, shipped 0.1994 -- the trained head ranks WORSE on average
  but 9 of 18 interfaces improved, with swings of +0.93 to -1.16
  9qrv_AB +0.495 and 9bx7_AC -0.415 on selected DockQ, nearly cancelling -- and those are two of
  the three interfaces that carried 85-88% of every earlier arm's validation gain
  both heads are LOW in absolute terms, 0.10 and 0.20: ipTM barely orders poses within an
  interface at all, which bounds any ipTM-based selection method

  python -I analysis/oneoff/05_val10_rank_correlation.py --rows-dir training/runs/2u8tf4/manual_eval_ep133
"""
import argparse
import collections
import csv
import glob
import os
import statistics

ACCEPTABLE = 0.23


def spearman(xs, ys):
    """Rank correlation with ties averaged. No scipy in this venv."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for t in range(i, j + 1):
                rk[order[t]] = avg
            i = j + 1
        return rk
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True,
                    help="an evaluation's shard work directory, holding rows_*_s?.csv")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.rows_dir, "rows_*_s?.csv")))
    if not paths:
        raise SystemExit(f"STOP: no rows_*_s?.csv under {a.rows_dir}. This reads the shard rows an "
                         f"evaluation leaves behind, not a metric json.")
    rows = collections.defaultdict(list)
    n_blank = 0
    for p in paths:
        for r in csv.DictReader(open(p)):
            if r["stored_iptm"] == "" or r["dockq"] == "":
                n_blank += 1
                continue
            rows[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]), float(r["dockq"])))
    if n_blank:
        # NOT SILENTLY. A pose without a stored ipTM has no shipped-head side, so dropping it
        # compares the two heads over different pose sets.
        raise SystemExit(f"STOP: {n_blank} row(s) lack a stored ipTM or a DockQ, so the trained and "
                         f"shipped heads would be compared over different poses.")

    print(f"{len(paths)} shard file(s), {len(rows)} interface(s), "
          f"{sum(len(v) for v in rows.values())} pose(s)")
    print("%-10s %7s %7s %8s   %7s %7s" % ("iface", "rho_tr", "rho_sh", "d_rho", "sel_tr", "sel_sh"))
    T, S, sel_t, sel_s = [], [], [], []
    for ifc in sorted(rows):
        v = rows[ifc]
        tr, sh, dq = [x[0] for x in v], [x[1] for x in v], [x[2] for x in v]
        rt, rs = spearman(tr, dq), spearman(sh, dq)
        st, ss = max(zip(tr, dq))[1], max(zip(sh, dq))[1]
        T.append(rt); S.append(rs); sel_t.append(st); sel_s.append(ss)
        print("%-10s %7.3f %7.3f %+8.3f   %7.3f %7.3f" % (ifc, rt, rs, rt - rs, st, ss))

    n = len(T)
    print(f"\nmean spearman(ipTM, DockQ):  trained {statistics.mean(T):.4f}   "
          f"shipped {statistics.mean(S):.4f}   delta {statistics.mean(T) - statistics.mean(S):+.4f}")
    print(f"interfaces where trained ranks better: {sum(1 for x, y in zip(T, S) if x > y)} of {n}")
    print(f"mean selected DockQ:  trained {statistics.mean(sel_t):.4f}   "
          f"shipped {statistics.mean(sel_s):.4f}")
    print(f"frac acceptable:      trained {sum(1 for d in sel_t if d >= ACCEPTABLE) / n:.4f}   "
          f"shipped {sum(1 for d in sel_s if d >= ACCEPTABLE) / n:.4f}")
    # THE BIGGEST MOVERS BOTH WAYS, because the mean is the thing that misleads here.
    moves = sorted(((t - s, i) for t, s, i in zip(sel_t, sel_s, sorted(rows))), reverse=True)
    print("\nlargest selected-DockQ moves:")
    for d, ifc in moves[:3] + moves[-3:]:
        print(f"  {ifc:10s} {d:+.4f}")


if __name__ == "__main__":
    main()
