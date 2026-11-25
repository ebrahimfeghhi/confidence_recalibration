#!/usr/bin/env python
"""5 POSES PER SEED VS 10. Does doubling the diffusion depth raise the oracle without raising
what the confidence head delivers?

THE SETUP. Each (interface, seed) unit now has the fold's original 5 poses and 5 more drawn from
the SAME cached trunk. That makes the comparison purely about depth: the trunk, the MSA, the
conditioning and the seed are all held fixed, and only the number of diffusion samples changes.
Figure 1 asked this at 1 vs 5 by subsampling; this asks it at 5 vs 10 with poses that did not
exist before.

THE TWO ipTM COLUMNS ARE COMPARABLE, which is what makes the 10-pose argmax meaningful: the new
poses were scored through rank_loss.iptm, and run_logging/validate_iptm_path.py showed that path
reproduces the fold's own stored ipTM on 99.34% of rows to 2e-3.

Reported per unit and averaged: the oracle (best DockQ available), what ipTM selects, and the
two CAPRI rates. Ties in ipTM go to the lowest sample id, as everywhere else.
"""
import argparse
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC, MED = 0.23, 0.49


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--new-dockq", default=f"{ROOT}/training/fold/poses_extra/dockq_extra.csv")
    ap.add_argument("--new-iptm", default=f"{ROOT}/training/fold/poses_extra/results_extra.csv")
    ap.add_argument("--nsample", type=int, default=5)
    a = ap.parse_args()

    import numpy as np
    from scipy import stats as sps

    old = collections.defaultdict(dict)
    with open(a.old) as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") != "free" or r.get("has_trunk") != "1":
                continue
            try:
                old[(r["iface"], int(r["seed"]))][int(r["sample"])] = (
                    float(r["dockq"]), float(r["iptm"]))
            except (ValueError, KeyError):
                continue
    ndq = collections.defaultdict(dict)
    with open(a.new_dockq) as fh:
        for r in csv.DictReader(fh):
            try:
                ndq[(r["iface"], int(r["seed"]))][int(r["sample"])] = float(r["dockq"])
            except (ValueError, KeyError):
                continue
    nip = collections.defaultdict(dict)
    with open(a.new_iptm) as fh:
        for r in csv.DictReader(fh):
            try:
                nip[(r["iface"], int(r["seed"]))][int(r["sample"])] = float(r["iptm"])
            except (ValueError, KeyError):
                continue

    five, ten = [], []
    for k, d in ndq.items():
        if k not in old or len(old[k]) != a.nsample or len(d) != a.nsample:
            continue
        ip = nip.get(k)
        if not ip or len(ip) != a.nsample:
            continue
        o = [(s, old[k][s][0], old[k][s][1]) for s in sorted(old[k])]
        n = [(s, d[s], ip[s]) for s in sorted(d)]
        for name, cand in (("5", o), ("10", o + n)):
            pick = sorted(cand, key=lambda t: (-t[2], t[0]))[0]
            rec = dict(sel=pick[1], orc=max(c[1] for c in cand))
            (five if name == "5" else ten).append(rec)

    n = len(five)
    print(f"paired on {n} (interface, seed) unit(s)\n")
    print(f"{'quantity':26s} {'5 poses':>9s} {'10 poses':>9s} {'delta':>9s} {'t':>8s} {'p':>10s}")
    for lab, key in (("oracle DockQ", "orc"), ("DockQ of ipTM's pick", "sel")):
        f = np.array([r[key] for r in five])
        t_ = np.array([r[key] for r in ten])
        st = sps.ttest_rel(t_, f)
        print(f"{lab:26s} {f.mean():9.4f} {t_.mean():9.4f} {t_.mean() - f.mean():+9.4f} "
              f"{st.statistic:+8.2f} {st.pvalue:10.2e}")
    for lab, key, thr in (("oracle frac >= 0.23", "orc", ACC), ("oracle frac >= 0.49", "orc", MED),
                          ("selected frac >= 0.23", "sel", ACC),
                          ("selected frac >= 0.49", "sel", MED)):
        f = np.array([r[key] >= thr for r in five], dtype=float)
        t_ = np.array([r[key] >= thr for r in ten], dtype=float)
        st = sps.ttest_rel(t_, f)
        print(f"{lab:26s} {f.mean():9.4f} {t_.mean():9.4f} {t_.mean() - f.mean():+9.4f} "
              f"{st.statistic:+8.2f} {st.pvalue:10.2e}")
    orc_d = np.mean([r["orc"] for r in ten]) - np.mean([r["orc"] for r in five])
    sel_d = np.mean([r["sel"] for r in ten]) - np.mean([r["sel"] for r in five])
    print(f"\nconverted: {sel_d / orc_d:.3f} of the oracle gain reaches the selection"
          if orc_d else "\noracle did not move")


if __name__ == "__main__":
    main()
