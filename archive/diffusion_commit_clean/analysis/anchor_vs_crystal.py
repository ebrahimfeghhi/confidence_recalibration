#!/usr/bin/env python
"""Does the reference-free labelling pick the same donors as the crystal-anchored one?

THE QUESTION. Donor selection sorts poses per interface, which requires knowing which model chain pair
IS that interface. The running sweep gets that by aligning every pose to the crystal, so it cannot be
run at test time. anchor_relabel.py answers the same question using the complex's highest-mean-ipTM
POSE as the reference. If the donors barely move, the method is reference-free at no cost.

THREE NUMBERS, because they fail independently:

  labelling agreement  fraction of poses the two frames assign the SAME chain pair. Reported per class
                       because the nocov class is the adversarial case -- no pose there is acceptable,
                       so its anchor is guaranteed to be a bad pose. If a bad anchor broke the
                       alignment, nocov is where it shows.
  donor-set overlap    Jaccard of the capped donor sets. Says how much the difference propagates.
  coverage             does the donor set contain a pose with GT DockQ >= 0.23 for that interface.
                       This is the one that matters: overlap says the sets DIFFER, coverage says
                       whether the difference is worse. Coverage is the ceiling on rescue.

GT IS ALWAYS THE CRYSTAL, in both arms. `dockq_gt` is read from the pool CSV keyed by the CRYSTAL
interface, so it measures the pose against the native structure no matter which frame selected it.
Evaluating the anchor arm in its own frame would be scoring it against its own opinion.

Row sets differ between frames: build_matrices drops a pose with zero contacts at its assigned chain
pair, and the frames assign different pairs. So agreement is computed on the poses present in BOTH,
and the row counts are reported so a donor difference caused by poses entering/leaving the pool is not
mistaken for one caused by reordering.

    anchor_vs_crystal.py [--cap 50]
"""
import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/run_exp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_complex_queue import T, round_robin  # noqa: E402
from fmt import table  # noqa: E402
from separation import pool_classes  # noqa: E402

XTAL = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices"
ANCH = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices_anchor30"
ACC = 0.23


def load(d, iface):
    z = np.load(f"{d}/{iface}.npz")
    keys = [(str(s), int(x)) for s, x in zip(z["seed"], z["sample"])]
    return keys, z["pair_iptm"].astype(float), z["dockq"].astype(float), \
        z["dockq_gt"].astype(float), z["model_ab"], z["model_ag"]


def leaders(keys, ip, M, cap):
    L = []
    for p in np.argsort(-ip):
        p = int(p)
        if all(M[p, j] < T for j in L):
            L.append(p)
            if len(L) >= cap:
                break
    return [keys[j] for j in L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--anchor-dir", default=ANCH, help="anchor-frame matrices")
    A = ap.parse_args()
    globals()["ANCH"] = A.anchor_dir

    cls = pool_classes()
    bypdb = collections.defaultdict(list)
    for f in sorted(os.listdir(ANCH)):
        if f.endswith(".npz") and os.path.exists(f"{XTAL}/{f}"):
            i = f[:-4]
            bypdb[i.split("_")[0]].append(i)

    agree = collections.defaultdict(lambda: [0, 0])
    ov, nd, rows_n = collections.defaultdict(list), collections.defaultdict(list), \
        collections.defaultdict(list)
    cov = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))

    for pdb, ifs in sorted(bypdb.items()):
        cs = [cls[i] for i in ifs if i in cls]
        if not cs:
            continue
        klass = "decoy" if "decoy" in cs else ("solved" if "solved" in cs else "nocov")
        px, pa, gtx = {}, {}, {}
        for i in ifs:
            kx, ipx, Mx, gx, abx, agx = load(XTAL, i)
            ka, ipa, Ma, ga, aba, aga = load(ANCH, i)
            # labelling agreement on the poses both frames kept
            xa = {k: (str(a), str(b)) for k, a, b in zip(kx, abx, agx)}
            aa = {k: (str(a), str(b)) for k, a, b in zip(ka, aba, aga)}
            both = set(xa) & set(aa)
            agree[klass][0] += sum(xa[k] == aa[k] for k in both)
            agree[klass][1] += len(both)
            rows_n[klass].append((len(kx), len(ka)))
            px[i] = leaders(kx, ipx, Mx, A.cap)
            pa[i] = leaders(ka, ipa, Ma, A.cap)
            gtx[i] = dict(zip(kx, gx)) | {k: v for k, v in zip(ka, ga)}
        sx = {p for p, _ in round_robin(px, A.cap)}
        sa = {p for p, _ in round_robin(pa, A.cap)}
        nd[klass].append((len(sx), len(sa)))
        ov[klass].append(100.0 * len(sx & sa) / max(len(sx | sa), 1))
        for i in ifs:
            if i not in cls:
                continue
            c = cls[i]
            for name, sel in (("crystal", sx), ("anchor", sa)):
                good = any(gtx[i].get(k, 0.0) >= ACC for k in sel)
                cov[name][c][0] += good
                cov[name][c][1] += 1

    r1 = [[k, f"{agree[k][0]}/{agree[k][1]}", f"{100.0*agree[k][0]/max(agree[k][1],1):.1f}%"]
          for k in ("decoy", "solved", "nocov") if agree[k][1]]
    print("\n=== labelling agreement: same chain pair as the crystal frame")
    print(table(["class", "poses agreeing / compared", "agreement"], r1, align="l r r"))

    r2 = []
    for k in ("decoy", "solved", "nocov"):
        if not ov[k]:
            continue
        rx = np.array([a for a, _ in rows_n[k]]), np.array([b for _, b in rows_n[k]])
        r2.append([k, len(ov[k]), f"{np.median(ov[k]):.0f}%",
                   f"{np.mean([a for a, _ in nd[k]]):.1f}", f"{np.mean([b for _, b in nd[k]]):.1f}",
                   f"{rx[0].mean():.0f}", f"{rx[1].mean():.0f}"])
    print("\n=== donor sets, per complex (cap %d)" % A.cap)
    print(table(["class", "complexes", "median donor overlap",
                 "donors crystal", "donors anchor", "matrix rows crystal", "matrix rows anchor"],
                r2, align="l r r r r r r"))

    r3 = []
    for c in ("decoy", "solved", "nocov"):
        a, b = cov["crystal"][c], cov["anchor"][c]
        if b[1]:
            r3.append([c, b[1], f"{a[0]}/{a[1]}", f"{b[0]}/{b[1]}"])
    print("\n=== coverage: donor set holds a pose with GT DockQ >= 0.23 for that interface")
    print(table(["interface class", "interfaces", "crystal frame", "anchor frame"],
                r3, align="l r r r"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
