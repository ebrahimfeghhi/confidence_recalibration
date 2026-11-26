#!/usr/bin/env python
"""SEEDS x POSES, THE WHOLE GRID, so an equal-budget comparison can be read off rather than
argued about.

WHY THIS REPLACES THE EARLIER PAIR. seed_count_curve.py reported 5->10 seeds converting 54% of
its oracle gain and depth_5_vs_10.py reported 5->10 poses converting none. That contrast is real
but it is NOT a fair budget comparison: 5->10 seeds at 5 poses each adds 25 poses, while 5->10
poses inside one seed adds 5. Any statement about where to spend has to hold the spend fixed.

TWO WAYS TO HOLD IT FIXED, because they disagree and the disagreement is the interesting part:

  EQUAL POSES   m x p, the number of structures decoded. The fair comparison the pose count makes
                is (10 seeds x 5 poses) against (5 seeds x 10 poses) -- both 50 poses, both grown
                from the same 5 x 5 = 25.

  EQUAL COMPUTE a seed costs a trunk and a pose does not. Measured on the fold logs and the
                rediffuse harness, one diffusion sample costs about one recycle iteration (1.94s
                vs 1.34s at 536 tokens, 11.08 vs 13.52 at 1,499), and the trunk is 10 recycles.
                So cost ~ m x (10 + p) in recycle units, and 5 seeds x 10 poses costs 100 while
                10 seeds x 5 poses costs 150. The pose option is CHEAPER at equal pose count.
                The ratio drifts with token count, so this column is a guide, not a price.

Restricted to interfaces carrying all --seeds seeds at all --poses poses, so every cell is drawn
from one pool. Random subsets of seeds and of poses, the same subsets for the oracle and for the
selection, averaged over interfaces then over reps. Ties in ipTM go to the lowest flat index,
which runs seed-major then sample.
"""
import argparse
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC, MED = 0.23, 0.49


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="a 10-pose table from run_logging/merge_10pose_results.py --which both")
    ap.add_argument("--poses", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--iptm-col", default="iptm")
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()

    import numpy as np

    cell = collections.defaultdict(dict)
    with open(a.results) as fh:
        rd = csv.DictReader(fh)
        col = a.iptm_col if a.iptm_col in (rd.fieldnames or []) else "pair_iptm"
        for r in rd:
            try:
                cell[(r["iface"], int(r["seed"]))][int(r["sample"])] = (
                    float(r["dockq"]), float(r[col]))
            except (ValueError, KeyError, TypeError):
                continue

    by = collections.defaultdict(dict)
    for (iface, seed), d in cell.items():
        if len(d) == int(a.poses):
            by[iface][seed] = [d[k] for k in sorted(d)]
    ifaces = sorted(i for i, s in by.items() if len(s) >= int(a.seeds))
    if not ifaces:
        raise SystemExit("no interface carries the full seeds x poses block")

    S, P, N = int(a.seeds), int(a.poses), len(ifaces)
    DQ = np.empty((N, S, P))
    IP = np.empty_like(DQ)
    for n, iface in enumerate(ifaces):
        for m, s in enumerate(sorted(by[iface])[:S]):
            for k, (d, i_) in enumerate(by[iface][s]):
                DQ[n, m, k] = d
                IP[n, m, k] = i_
    print(f"interfaces {N}   block {S} seeds x {P} poses   poses {DQ.size}   reps {a.reps}")

    rng = np.random.default_rng(a.rng)
    idx = np.arange(N)[:, None]

    def run(m, p):
        o_, s_, oa, sa, om, sm = [], [], [], [], [], []
        for _ in range(int(a.reps)):
            ps = np.array([rng.choice(S, size=m, replace=False) for _ in range(N)])
            pp = np.array([rng.choice(P, size=p, replace=False) for _ in range(N)])
            dq = DQ[idx, ps][np.arange(N)[:, None, None],
                             np.arange(m)[None, :, None], pp[:, None, :]]
            ip = IP[idx, ps][np.arange(N)[:, None, None],
                             np.arange(m)[None, :, None], pp[:, None, :]]
            dq = dq.reshape(N, -1)
            ip = ip.reshape(N, -1)
            sel = dq[np.arange(N), np.argmax(ip, axis=1)]
            orc = dq.max(axis=1)
            o_.append(orc.mean()); s_.append(sel.mean())
            oa.append((orc >= ACC).mean()); sa.append((sel >= ACC).mean())
            om.append((orc >= MED).mean()); sm.append((sel >= MED).mean())
        return tuple(float(np.mean(x)) for x in (o_, s_, oa, sa, om, sm))

    MS = [m for m in (1, 2, 3, 5, 10, 15) if m <= S]
    PS = [p for p in (1, 2, 5, 10) if p <= P]
    grid = {}
    print()
    print(f"{'seeds':>6s} {'poses/seed':>11s} {'total poses':>12s} {'cost~m(10+p)':>13s} "
          f"{'oracle':>8s} {'selected':>9s} {'gap':>7s} {'sel>=.23':>9s} {'sel>=.49':>9s}")
    for m in MS:
        for p in PS:
            grid[(m, p)] = run(m, p)
            o, s, _a1, s1, _a2, s2 = grid[(m, p)]
            print(f"{m:6d} {p:11d} {m * p:12d} {m * (10 + p):13d} "
                  f"{o:8.4f} {s:9.4f} {o - s:7.4f} {s1:9.4f} {s2:9.4f}")

    print()
    print("EQUAL TOTAL POSES -- which allocation wins, and what it costs")
    bud = collections.defaultdict(list)
    for (m, p) in grid:
        bud[m * p].append((m, p))
    print(f"{'poses':>6s} {'allocation':>14s} {'cost':>6s} {'oracle':>8s} {'selected':>9s} "
          f"{'sel>=.49':>9s}")
    for tot in sorted(bud):
        opts = sorted(bud[tot])
        if len(opts) < 2:
            continue
        best = max(opts, key=lambda mp: grid[mp][1])
        for (m, p) in opts:
            o, s, _a1, _s1, _a2, s2 = grid[(m, p)]
            star = "  <-- best selected" if (m, p) == best else ""
            print(f"{tot:6d} {f'{m} seed x {p} pose':>14s} {m * (10 + p):6d} "
                  f"{o:8.4f} {s:9.4f} {s2:9.4f}{star}")
        print()

    # EQUAL COMPUTE, which is the budget that actually binds. Allocations are banded by
    # m*(10+p) because no two cells cost exactly the same, and the best SELECTED DockQ in each
    # band is what a fixed GPU allowance would buy.
    print("EQUAL COMPUTE -- best selected DockQ per cost band, cost ~ m x (10 + p)")
    print(f"{'cost band':>14s} {'allocation':>14s} {'poses':>6s} {'oracle':>8s} "
          f"{'selected':>9s} {'sel>=.49':>9s}")
    bands = [(0, 25), (25, 50), (50, 80), (80, 125), (125, 175), (175, 240), (240, 1000)]
    for lo, hi in bands:
        inb = [(m, p) for (m, p) in grid if lo <= m * (10 + p) < hi]
        if not inb:
            continue
        best = max(inb, key=lambda mp: grid[mp][1])
        worst_p = max(inb, key=lambda mp: mp[1])
        for (m, p) in sorted({best, worst_p}):
            o, s, _a1, _s1, _a2, s2 = grid[(m, p)]
            tag = "  <-- best" if (m, p) == best else "  (most poses/seed in band)"
            print(f"{f'{lo}-{hi}':>14s} {f'{m} seed x {p} pose':>14s} {m * p:6d} "
                  f"{o:8.4f} {s:9.4f} {s2:9.4f}{tag}")
        print()

    # THE POSE AXIS AT FIXED SEEDS, stated plainly: what does 10x the diffusion depth buy once
    # the seed count is held?
    print("POSES PER SEED, at fixed seed count (selected DockQ)")
    print(f"{'seeds':>6s} " + " ".join(f"{f'p={p}':>8s}" for p in PS) + f" {'p10-p1':>8s}")
    for m in MS:
        row = [grid[(m, p)][1] for p in PS]
        print(f"{m:6d} " + " ".join(f"{v:8.4f}" for v in row)
              + f" {row[-1] - row[0]:+8.4f}")


if __name__ == "__main__":
    main()
