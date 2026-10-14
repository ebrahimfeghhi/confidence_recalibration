#!/usr/bin/env python
"""Complex-level donor selection: the version with no per-interface split.

WHY. The running selection (build_complex_queue.py) sorts poses SEPARATELY for each interface, which
means it must know which model chain pair is "interface A-B" for every pose. That identity comes from
chain_map.py, which gets it by aligning each pose to the CRYSTAL -- so the selection cannot be run at
test time as written.

Splitting by interface is the only thing that needs those labels. This version does not split:

  sort    mean pair_iptm across the complex's interfaces       -- one number per pose
  filter  mean pose-pose DockQ across interfaces, keep if < 0.23 to everything kept
  cap     50

WHAT THIS DOES AND DOES NOT REMOVE. It removes the per-interface split. It does NOT by itself remove
the crystal: the matrices it reads were built on crystal-pinned chains, so the per-interface DockQ
being averaged here is still crystal-labelled. Isolating one change at a time is the point -- this
script measures what the SPLIT was worth. Whether the labels themselves can be replaced (anchor pose
instead of crystal) is a separate question and a separate measurement.

Poses are not present in every interface's matrix -- build_matrices drops poses with no interface at
the 5.0 A contact cutoff -- so both means are over the interfaces where the pose survives, and a pair
of poses is compared only over interfaces where BOTH survive. A pair sharing no interface cannot be
called redundant and is treated as distinct.

    build_simple_queue.py --out run_exp/queue_simple.tsv [--cap 50] [--diff]
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths  # noqa: E402
from build_complex_queue import MATRICES, T, leaders, round_robin  # noqa: E402


def load_complex(ifaces):
    """Union pose list for one complex, plus mean ipTM and mean pose-pose DockQ over interfaces.

    Returns (poses, ip, M, nseen) where poses are (seed, sample) keys, ip[i] is the mean pair_iptm,
    M[i,j] the mean pose-pose DockQ, and nseen[i,j] how many interfaces that mean is over (0 = never
    co-occur, so not comparable).
    """
    keys, per = [], []
    for f in ifaces:
        z = np.load(f"{MATRICES}/{f}.npz")
        k = [(str(s), int(x)) for s, x in zip(z["seed"], z["sample"])]
        per.append((k, z["pair_iptm"].astype(float), z["dockq"].astype(float)))
        keys += k
    poses = sorted(set(keys))
    idx = {k: i for i, k in enumerate(poses)}
    n = len(poses)
    ip_s, ip_n = np.zeros(n), np.zeros(n)
    M_s, M_n = np.zeros((n, n)), np.zeros((n, n))
    for k, ip, M in per:
        g = np.array([idx[x] for x in k])
        np.add.at(ip_s, g, ip)
        np.add.at(ip_n, g, 1.0)
        M_s[np.ix_(g, g)] += M
        M_n[np.ix_(g, g)] += 1.0
    return poses, ip_s / np.maximum(ip_n, 1), M_s / np.maximum(M_n, 1), M_n


def simple_leaders(ifaces, cap):
    """ipTM-ordered greedy walk at the COMPLEX level. Returns (seed, sample) keys in order."""
    poses, ip, M, nseen = load_complex(ifaces)
    L = []
    for p in np.argsort(-ip):
        p = int(p)
        # never co-occurring means never shown to be redundant -> keep
        if all(nseen[p, j] == 0 or M[p, j] < T for j in L):
            L.append(p)
            if len(L) >= cap:
                break
    return [poses[j] for j in L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--seed", type=int, default=430000)
    ap.add_argument("--diff", action="store_true",
                    help="compare against build_complex_queue's selection at the same cap")
    A = ap.parse_args()

    bypdb = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{MATRICES}/*.npz")):
        i = os.path.basename(f)[:-4]
        bypdb[i.split("_")[0]].append(i)

    lines, rows = [], []
    for pdb, ifs in sorted(bypdb.items()):
        sel = simple_leaders(ifs, A.cap)
        for n, (s, x) in enumerate(sel):
            pose = paths.pool_pose(pdb, s, x)
            if os.path.exists(pose):
                lines.append("\t".join(["tmpl", pdb, ifs[0], f"S{n:02d}", str(A.seed), pose]))
        if A.diff:
            # the running selection, regenerated in full (no reuse skipping)
            cur = round_robin({i: leaders(i, A.cap) for i in ifs}, A.cap)
            curk = {os.path.basename(p) + os.path.dirname(p) for p, _ in cur}
            newk = {os.path.basename(paths.pool_pose(pdb, s, x))
                    + os.path.dirname(paths.pool_pose(pdb, s, x)) for s, x in sel}
            rows.append([pdb, len(ifs), len(curk), len(newk),
                         len(curk & newk), 100.0 * len(curk & newk) / max(len(curk | newk), 1)])

    if A.out:
        os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
        open(A.out, "w").write("\n".join(lines) + "\n")
        print(f"  donors written: {len(lines)}  ->  {A.out}")

    if A.diff:
        import csv
        w = csv.writer(open("simple_vs_running.csv", "w"))
        w.writerow(["pdb", "n_interfaces", "n_running", "n_simple", "n_shared", "jaccard_pct"])
        w.writerows(rows)
        multi = [r for r in rows if r[1] > 1]
        single = [r for r in rows if r[1] == 1]
        for name, g in (("single-interface", single), ("multi-interface", multi)):
            if not g:
                continue
            j = np.array([r[5] for r in g])
            print(f"  {name:17s} n={len(g):3d}  donor-set overlap: "
                  f"median {np.median(j):5.1f}%  mean {j.mean():5.1f}%  "
                  f"identical {int((j == 100).sum())}/{len(g)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
