#!/usr/bin/env python
"""DOES REUSING A CACHED TRUNK COST POSE QUALITY? The paired test.

THE WORRY. A normal fold builds ONE input feature dict and hands it to both halves of the model:
InputFeatureEmbedder reads `ref_pos` to build `s_inputs` (which feeds the trunk), and the
diffusion module reads `ref_pos` again directly. The featuriser applies an independent random
rotation per ref_space_uid to that reference template, so the two reads normally share one draw.

run_logging/rediffuse_batch.py pairs a CACHED trunk -- built back then from draw A, which was
never saved -- with a freshly built dict carrying draw B. The two halves therefore disagree about
the template's frame, a combination the model was never trained on. Reading the code cannot say
whether it matters, and the draw cannot be recovered: `ref_pos` is not in the npz, and the
rotation happens before the trunk.

THE TEST. Every unit already has 5 ORIGINAL poses made with matched halves. The new 5 come from
the same trunk with mismatched halves. Pairing the two sets per (interface, seed) and testing
across units isolates the mismatch: the noise draw differs too, but that is independent of the
condition and averages out, while a systematic cost would show as a consistent shift.

WHAT IS COMPARED, per (interface, seed): mean DockQ over the 5, the oracle (max), the ipTM of the
pose ipTM would pick, and mean ipTM. Paired t-test and the sign split across units, because a
mean shift with a 50/50 sign split is noise and one with a lopsided split is not.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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

    rows = []
    for k, d in ndq.items():
        if k not in old or len(old[k]) != a.nsample or len(d) != a.nsample:
            continue
        ip = nip.get(k)
        if not ip or len(ip) != a.nsample:
            continue
        o_dq = np.array([old[k][s][0] for s in sorted(old[k])])
        o_ip = np.array([old[k][s][1] for s in sorted(old[k])])
        n_dq = np.array([d[s] for s in sorted(d)])
        n_ip = np.array([ip[s] for s in sorted(ip)])
        rows.append(dict(
            k=k,
            o_mean=o_dq.mean(), n_mean=n_dq.mean(),
            o_max=o_dq.max(), n_max=n_dq.max(),
            o_sel=o_dq[int(np.argmax(o_ip))], n_sel=n_dq[int(np.argmax(n_ip))],
            o_ip=o_ip.mean(), n_ip=n_ip.mean()))

    if not rows:
        raise SystemExit("STOP: no (interface, seed) has both sets complete")
    print(f"paired on {len(rows)} (interface, seed) unit(s) over "
          f"{len({r['k'][0].split('_')[0] for r in rows})} structure(s)\n")
    print(f"{'quantity':24s} {'original':>10s} {'new':>10s} {'delta':>9s} {'t':>8s} "
          f"{'p':>10s} {'new worse':>10s}")
    for lab, ko, kn in (("mean DockQ over the 5", "o_mean", "n_mean"),
                        ("oracle DockQ (max)", "o_max", "n_max"),
                        ("DockQ of ipTM's pick", "o_sel", "n_sel"),
                        ("mean ipTM", "o_ip", "n_ip")):
        o = np.array([r[ko] for r in rows])
        n = np.array([r[kn] for r in rows])
        t = sps.ttest_rel(n, o)
        worse = int((n < o - 1e-9).sum())
        print(f"{lab:24s} {o.mean():10.4f} {n.mean():10.4f} {n.mean() - o.mean():+9.4f} "
              f"{t.statistic:+8.2f} {t.pvalue:10.2e} {worse:5d}/{len(rows):<4d}")
    print("\nA systematic cost from the frame mismatch would be a NEGATIVE delta on DockQ with a "
          "lopsided sign split. A null is a delta near zero, a small t and a ~50/50 split.")


if __name__ == "__main__":
    main()
