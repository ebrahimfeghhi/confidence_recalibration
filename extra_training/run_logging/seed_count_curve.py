#!/usr/bin/env python
"""ORACLE AND SELECTED DockQ AS THE NUMBER OF SEEDS GROWS, at a fixed 5 poses per seed.

THE POINT OF SEPARATING THIS FROM depth_5_vs_10.py. Doubling the diffusion samples inside a seed
raises the oracle by +0.0314 and delivers -0.0003 to the selection -- nothing. The seed axis is a
different question, and it has to be asked the same way before "seeds convert and samples do not"
means anything: subsample m of an interface's 15 seeds, take the oracle over the 5m poses and
what ipTM picks from them, and read the two curves side by side.

WHY SUBSAMPLE RATHER THAN COMPARE THE FIRST m SEEDS. The fold order is not random with respect to
anything, and an interface's seeds differ in quality, so the first 5 would carry whatever
ordering the queue had. R independent random subsets per m, the same subsets for the oracle and
for the selection, averaged over interfaces and then over reps.

RESTRICTED TO INTERFACES WITH ALL 15 SEEDS x 5 SAMPLES so that every m is drawn from the same
pool and the m=15 point is the whole set rather than a different population.

Tie-break as everywhere else: highest ipTM, then lowest seed, then lowest sample.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
ACC, MED = 0.23, 0.49


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--iptm-col", default="iptm")
    ap.add_argument("--rng", type=int, default=0)
    a = ap.parse_args()

    import numpy as np

    cell = collections.defaultdict(dict)
    with open(a.results) as fh:
        rd = csv.DictReader(fh)
        col = a.iptm_col if a.iptm_col in (rd.fieldnames or []) else "pair_iptm"
        for r in rd:
            if r.get("has_trunk") not in (None, "", "1"):
                continue
            if "arm" in r and r.get("arm") not in (None, "", "free"):
                continue
            try:
                cell[(r["iface"], int(r["seed"]))][int(r["sample"])] = (
                    float(r["dockq"]), float(r[col]))
            except (ValueError, KeyError, TypeError):
                continue

    by = collections.defaultdict(dict)
    for (iface, seed), d in cell.items():
        if len(d) == int(a.poses):
            by[iface][seed] = [d[k] for k in sorted(d)]
    ifaces = [i for i, s in by.items() if len(s) >= int(a.seeds)]
    if not ifaces:
        raise SystemExit("no interface carries the full seed set")

    # [n_iface, seeds, poses] for DockQ and ipTM, seeds in sorted order
    DQ = np.empty((len(ifaces), int(a.seeds), int(a.poses)))
    IP = np.empty_like(DQ)
    for n, iface in enumerate(sorted(ifaces)):
        sd = sorted(by[iface])[:int(a.seeds)]
        for m, s in enumerate(sd):
            for k, (d, i_) in enumerate(by[iface][s]):
                DQ[n, m, k] = d
                IP[n, m, k] = i_
    print(f"interfaces {len(ifaces)}   seeds {a.seeds}   poses/seed {a.poses}   "
          f"reps {a.reps}   poses {DQ.size}")

    rng = np.random.default_rng(a.rng)
    ms = [m for m in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15) if m <= int(a.seeds)]
    print()
    print(f"{'seeds':>6s} {'poses':>6s} {'oracle':>8s} {'selected':>9s} {'gap':>7s} "
          f"{'orc>=.23':>9s} {'sel>=.23':>9s} {'orc>=.49':>9s} {'sel>=.49':>9s}")
    prev = None
    out = {}
    for m in ms:
        o_, s_, oa, sa, om, sm = [], [], [], [], [], []
        for _ in range(int(a.reps)):
            pick = np.array([rng.choice(int(a.seeds), size=m, replace=False)
                             for _ in range(len(ifaces))])
            # gather the chosen seeds: [n_iface, m, poses] -> flatten the pose axis
            idx = np.arange(len(ifaces))[:, None]
            dq = DQ[idx, pick].reshape(len(ifaces), -1)
            ip = IP[idx, pick].reshape(len(ifaces), -1)
            # LOWEST FLAT INDEX WINS A TIE, and the flat index runs seed-major then sample, which
            # is the (seed, sample) order the rest of the repo breaks ties by.
            sel = dq[np.arange(len(ifaces)), np.argmax(ip, axis=1)]
            orc = dq.max(axis=1)
            o_.append(orc.mean())
            s_.append(sel.mean())
            oa.append((orc >= ACC).mean())
            sa.append((sel >= ACC).mean())
            om.append((orc >= MED).mean())
            sm.append((sel >= MED).mean())
        out[m] = (np.mean(o_), np.mean(s_), np.mean(oa), np.mean(sa), np.mean(om), np.mean(sm))
        o, s, a1, s1, a2, s2 = out[m]
        print(f"{m:6d} {m * int(a.poses):6d} {o:8.4f} {s:9.4f} {o - s:7.4f} "
              f"{a1:9.4f} {s1:9.4f} {a2:9.4f} {s2:9.4f}")
        prev = m

    print()
    print(f"{'step':>12s} {'d oracle':>9s} {'d selected':>11s} {'converted':>10s}")
    for lo, hi in [(1, 5), (5, 10), (1, 10), (5, 15), (10, 15)]:
        if lo in out and hi in out:
            do = out[hi][0] - out[lo][0]
            ds = out[hi][1] - out[lo][1]
            print(f"{f'{lo}->{hi} seeds':>12s} {do:+9.4f} {ds:+11.4f} "
                  f"{(ds / do * 100 if do else float('nan')):9.1f}%")


if __name__ == "__main__":
    main()
