#!/usr/bin/env python
"""THE TOP STRATUM, WITH THE GAP MATCHED PAIR BY PAIR INSTEAD OF BINNED.

THE OBJECTION THIS ANSWERS. run_logging/paired_interface_slopes.py compares the two cells inside a
|dDockQ| stratum, and its top stratum [0.40, 1.01) is 0.61 WIDE. If across-seed pairs sit higher
inside that band than within-seed pairs do, the comparison is confounded, and neither statistic is
immune to it:

  P(correct)  has no gap normalisation at all -- a bigger gap is simply easier, so a cell sitting
              higher in the band scores higher mechanically.
  the slope   b = sum(d*y)/sum(d^2) is immune only if the relation is LINEAR through the origin.
              Under y = B*d + G*d^2 it evaluates to b = B + G * sum(d^3)/sum(d^2), which depends on
              where in the band the cell's gaps lie. The pooled fit put a large positive d^2 on
              this relation, so G is not zero and the leak is real.

WHAT THIS DOES INSTEAD. For each within-seed pair with gap g, take the across-seed pairs OF THE
SAME CHAIN PAIR whose gap is within --tol of g, and compare against the mean of that matched set.
The two cells then share the gap distribution to +-tol by construction, so curvature and within-band
location cannot differ between them. With d matched there is nothing left to normalise, so the
statistic is the RAW signed ipTM gap y, not a slope -- the quantity a selector actually sees.

Each within-seed pair contributes once, with its matched set's mean; each chain pair contributes
one within number and one across number per stratum; the test is paired across chain pairs, the
same design as paired_interface_slopes.py. The printed mean gap per cell is the audit: if the two
gap columns are not equal to three decimals the matching did not work and nothing below it means
anything.
"""
import argparse
import collections
import csv
import functools
import os

print = functools.partial(print, flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=f"{ROOT}/training/fold/abag_all_results.csv")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--min-matched", type=int, default=3,
                    help="a chain pair needs this many matched within pairs in a stratum to count")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import numpy as np
    from scipy import stats as sps

    poses = collections.defaultdict(list)
    for r in csv.DictReader(open(a.master)):
        if r["pose_set"] != "orig":
            continue
        try:
            poses[(f'{r["dataset"]} {r["split"]}'.strip(), r["interface"])].append(
                (float(r["dockq"]), float(r["iptm"]), r["seed"]))
        except (ValueError, KeyError, TypeError):
            continue

    # (dataset, stratum) -> chain pair -> (gap_w, gap_a, y_w, y_a, c_w, c_a, n)
    rec = collections.defaultdict(dict)
    raw = collections.defaultdict(lambda: ([], []))   # (dataset, stratum) -> unmatched gaps
    for key, v in poses.items():
        if len({s for _, _, s in v}) < 2:
            continue
        dq = np.array([x[0] for x in v])
        ip = np.array([x[1] for x in v])
        sd = np.array([x[2] for x in v])
        ii, jj = np.triu_indices(len(v), k=1)
        g0 = dq[ii] - dq[jj]
        live = g0 != 0.0
        ii, jj = ii[live], jj[live]
        g0 = g0[live]
        if ii.size == 0:
            continue
        flip = g0 < 0
        hi = np.where(flip, jj, ii)
        lo = np.where(flip, ii, jj)
        d = dq[hi] - dq[lo]
        y = ip[hi] - ip[lo]
        same = sd[hi] == sd[lo]
        corr = np.where(y > 0, 1.0, np.where(y == 0, 0.5, 0.0))

        for b0, b1 in zip(EDGES[:-1], EDGES[1:]):
            m = (d >= b0) & (d < b1)
            dw, yw, cw = d[m & same], y[m & same], corr[m & same]
            da, ya, ca = d[m & ~same], y[m & ~same], corr[m & ~same]
            if dw.size == 0 or da.size == 0:
                continue
            raw[(key[0], (b0, b1))][0].append(dw)
            raw[(key[0], (b0, b1))][1].append(da)
            srt = np.argsort(da)
            ds, ys, cs_ = da[srt], ya[srt], ca[srt]
            cy = np.concatenate([[0.0], np.cumsum(ys)])
            cc = np.concatenate([[0.0], np.cumsum(cs_)])
            cd = np.concatenate([[0.0], np.cumsum(ds)])
            left = np.searchsorted(ds, dw - a.tol, side="left")
            right = np.searchsorted(ds, dw + a.tol, side="right")
            cnt = right - left
            ok = cnt > 0
            if ok.sum() < a.min_matched:
                continue
            rec[(key[0], (b0, b1))][key] = (
                float(dw[ok].mean()),
                float(((cd[right[ok]] - cd[left[ok]]) / cnt[ok]).mean()),
                float(yw[ok].mean()),
                float(((cy[right[ok]] - cy[left[ok]]) / cnt[ok]).mean()),
                float(cw[ok].mean()),
                float(((cc[right[ok]] - cc[left[ok]]) / cnt[ok]).mean()),
                int(ok.sum()))

    print(f"GAP MATCHED PAIR BY PAIR at +-{a.tol} inside each chain pair, "
          f"then paired across chain pairs")
    print("the two 'mean gap' columns ARE the audit -- they must agree, or the match failed")
    print()
    rows_out = []
    for ds in sorted({k[0] for k in rec}):
        print(f"=== {ds}")
        print(f"{'|dDockQ| stratum':>17} {'chain pairs':>11} {'matched w pairs':>15} "
              f"{'mean gap w':>10} {'mean gap a':>10} | "
              f"{'unmatched med gap w':>19} {'a':>7} | "
              f"{'mean dipTM w':>12} {'a':>9} {'diff':>9} {'t':>7} {'p':>9} | "
              f"{'P(ok) w':>8} {'a':>8} {'diff':>9} {'t':>7} {'p':>9}")
        for b in zip(EDGES[:-1], EDGES[1:]):
            v = rec.get((ds, b))
            if not v or len(v) < 3:
                continue
            A = np.array(list(v.values()))
            gw, ga, yw, ya, cw, ca, nn = (A[:, i] for i in range(7))
            ty = sps.ttest_rel(yw, ya)
            tc = sps.ttest_rel(cw, ca)
            rw = np.concatenate(raw[(ds, b)][0])
            ra = np.concatenate(raw[(ds, b)][1])
            lab = f"[{b[0]:.2f},{b[1]:.2f})"
            print(f"{lab:>17} {len(v):11d} {int(nn.sum()):15d} "
                  f"{gw.mean():10.4f} {ga.mean():10.4f} | "
                  f"{np.median(rw):19.4f} {np.median(ra):7.4f} | "
                  f"{yw.mean():12.4f} {ya.mean():9.4f} {yw.mean() - ya.mean():+9.4f} "
                  f"{ty.statistic:+7.2f} {ty.pvalue:9.2e} | "
                  f"{cw.mean():8.4f} {ca.mean():8.4f} {cw.mean() - ca.mean():+9.4f} "
                  f"{tc.statistic:+7.2f} {tc.pvalue:9.2e}")
            rows_out.append({"dataset": ds, "stratum": lab, "chain_pairs": len(v),
                             "matched_within_pairs": int(nn.sum()),
                             "mean_gap_within": gw.mean(), "mean_gap_across": ga.mean(),
                             "unmatched_med_gap_within": float(np.median(rw)),
                             "unmatched_med_gap_across": float(np.median(ra)),
                             "diptm_within": yw.mean(), "diptm_across": ya.mean(),
                             "diptm_diff": yw.mean() - ya.mean(),
                             "diptm_t": ty.statistic, "diptm_p": ty.pvalue,
                             "pok_within": cw.mean(), "pok_across": ca.mean(),
                             "pok_diff": cw.mean() - ca.mean(),
                             "pok_t": tc.statistic, "pok_p": tc.pvalue})
        print()

    if a.out and rows_out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        print(f"rows -> {a.out}")


if __name__ == "__main__":
    main()
