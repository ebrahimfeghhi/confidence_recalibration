#!/usr/bin/env python
"""BOTH POOLINGS, SIDE BY SIDE, FOR ONE ARM'S VAL PASSES -- with the paired error bars.

    val_two_tables.py <rid> [<rid2> ...] [--pool training/valfull83/pool.csv]
                      [--restrict training/valfull83/pool_noleak.csv] [--md]

WHY IT EXISTS. The owner asked for max-pooled and mean-pooled reported separately for every arm,
and the single-table habit hid two things for a day: Arm S's mean-pooled column never beat shipped
at ANY epoch while its max-pooled column did at epochs 4-5, and a `select_max` field that the
trainer's in-run passes omit (only eval_saved_ckpts writes it) made it look as though max-pooling
had not been measured at all. Both poolings are in every rows_*.csv as `iptm` and `iptm_mean`, so
nothing needs re-running; this derives them the same way for every epoch and every arm.

THE DELTAS ARE PAIRED AND CARRY A t. A difference of 0.004 on 139 interfaces is not a result: the
per-interface spread of top-1 DockQ is ~0.075 because the metric jumps discretely when the pick
changes, so the paired SE is ~0.006 and anything under about |0.012| cannot be resolved. Printing
the t next to the delta is what stops a 0.004 from being read as an improvement.
"""
import argparse, csv, glob, collections, statistics, math, os, sys

ACC = 0.23

def load(rid, ep):
    d = collections.defaultdict(list)
    for f in glob.glob(f"training/runs/{rid}/valfull83_eval_shards_ep{ep}/rows*_s*.csv"):
        if "paeloss" in f:
            continue
        for r in csv.DictReader(open(f)):
            try:
                d[r["iface"]].append((float(r["stored_iptm"]), float(r["iptm"]),
                                      float(r["iptm_mean"]), float(r["dockq"])))
            except (KeyError, ValueError):
                continue
    return d

def epochs_of(rid):
    out = []
    for p in glob.glob(f"training/runs/{rid}/valfull83_eval_shards_ep*"):
        m = os.path.basename(p).rsplit("ep", 1)[-1]
        if m.isdigit() and glob.glob(f"{p}/rows*_s*.csv"):
            out.append(int(m))
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rids", nargs="+")
    ap.add_argument("--restrict", default="training/valfull83/pool_noleak.csv",
                    help="interfaces the tables are limited to; the default drops 8z7j, the one "
                         "val structure that shares an antibody cluster with the training pool.")
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()

    keep = None
    if a.restrict and os.path.exists(a.restrict):
        keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'
                for r in csv.DictReader(open(a.restrict))}

    eps = {}
    for rid in a.rids:
        for e in epochs_of(rid):
            d = load(rid, e)
            if d:
                eps[e] = d
    if not eps:
        sys.exit("no val row csvs found for " + ", ".join(a.rids))

    keys = sorted(set.intersection(*[set(d) for d in eps.values()]))
    if keep is not None:
        keys = [k for k in keys if k in keep]
    n = len(keys)
    if not n:
        sys.exit("no interfaces survive the restrict set")

    def col(d, idx):
        t1 = [max(d[k], key=lambda t: t[idx])[3] for k in keys]
        b5 = [max(sorted(d[k], key=lambda t: -t[idx])[:5], key=lambda t: t[3])[3] for k in keys]
        return t1, b5

    # RANK OF THE FIRST ACCEPTABLE POSE, the graded version of the acc column. top-1 and acc only
    # move when the pick crosses DockQ 0.23, so a reordering that lifts an acceptable pose from
    # rank 40 to rank 5 is invisible in both; this is its 1-based position in the arm's own
    # ranking. In-sample it is the only statistic in this project to clear significance
    # (t = -3.68 at n=885) while every DockQ column stayed inside the noise. LOWER IS BETTER, so
    # a NEGATIVE delta is an improvement -- the opposite sign convention from the DockQ columns,
    # which is why the header says so.
    #
    # WINNABLE INTERFACES ONLY. Rank is undefined where no candidate is acceptable, and those are
    # dropped rather than censored at n+1: censoring makes the mean a function of the candidate
    # count and so incomparable between pools of different depth.
    win = [k for k in keys if any(t[3] >= ACC for t in eps[min(eps)][k])]
    nw = len(win)

    def rank(d, idx):
        out = []
        for k in win:
            order = sorted(d[k], key=lambda t: -t[idx])
            out.append(next(j for j, t in enumerate(order, 1) if t[3] >= ACC))
        return out

    br = rank(eps[min(eps)], 0)
    bt, bb = col(eps[min(eps)], 0)
    f = lambda L: (sum(L) / n, 100.0 * sum(1 for x in L if x >= ACC) / n)

    for idx, title in ((1, "MAX-POOLED ipTM"), (2, "MEAN-POOLED ipTM")):
        print(f"\n**{title}** -- val83{' leak-free' if keep else ''}, n={n}"
              f" ({nw} winnable, for the rank columns)\n")
        if a.md:
            print("| epoch | top-1 | acc | best@5 | acc@5 | Δtop1 | t | Δbest@5 | t | "
                  "mean rank 1st acc | Δrank (neg=better) | t |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|")
            x, y = f(bt); c, d2 = f(bb)
            print(f"| shipped | {x:.4f} | {y:.1f}% | {c:.4f} | {d2:.1f}% | — | | — | | "
                  f"{statistics.fmean(br):.2f} | — | |")
        else:
            print(f"  {'epoch':>6s} {'top-1':>8s} {'acc':>7s} {'best@5':>8s} {'acc@5':>7s}"
                  f" {'Δtop1':>9s} {'t':>6s} {'Δb@5':>9s} {'t':>6s}")
            x, y = f(bt); c, d2 = f(bb)
            print(f"  {'shipped':>6s} {x:8.4f} {y:6.1f}% {c:8.4f} {d2:6.1f}%")
        for e in sorted(eps):
            t1, b5 = col(eps[e], idx)
            d1 = [p - q for p, q in zip(t1, bt)]
            d5 = [p - q for p, q in zip(b5, bb)]
            m1 = sum(d1) / n; s1 = statistics.stdev(d1) / math.sqrt(n) if n > 1 else float("nan")
            m5 = sum(d5) / n; s5 = statistics.stdev(d5) / math.sqrt(n) if n > 1 else float("nan")
            rk = rank(eps[e], idx)
            dr = [p - q for p, q in zip(rk, br)]
            mr = sum(dr) / nw
            sr = statistics.stdev(dr) / math.sqrt(nw) if nw > 1 else float("nan")
            x, y = f(t1); c, d2 = f(b5)
            if a.md:
                print(f"| {e} | {x:.4f} | {y:.1f}% | {c:.4f} | {d2:.1f}% | "
                      f"{m1:+.4f} | {m1/s1:+.2f} | {m5:+.4f} | {m5/s5:+.2f} | "
                      f"{statistics.fmean(rk):.2f} | {mr:+.2f} | {mr/sr:+.2f} |")
            else:
                print(f"  {e:6d} {x:8.4f} {y:6.1f}% {c:8.4f} {d2:6.1f}% "
                      f"{m1:+9.4f} {m1/s1:+6.2f} {m5:+9.4f} {m5/s5:+6.2f}")

if __name__ == "__main__":
    main()
