#!/usr/bin/env python
"""Write the fold queue for one experiment design. One place, one file per design.

WHY THIS EXISTS. fold.py reads a plain TSV where each line is one GPU job:

    tmpl  <pdb>  <interface>  <tag>  <seed>  <pose.cif> [<pose.cif> ...]

`tmpl` is the job KIND: hand the model every pose on this line as one template set. A line with one
pose is a single-template fold. Older queue files spell it `all5`, from when every design used five
poses; fold.py still accepts that spelling, so those files keep working.

The only real work here is CHOOSING the donor poses, and that choice is the experiment. Three of the
four designs below were originally built by commands typed at a shell and never saved, so the runs
existed but the logic that produced them did not. Each design is reproducible from here now, and
--verify diffs a regenerated queue against the one the workers actually ran.

DESIGNS
  top5     5 donors per arm: oracle (best by GT DockQ), conf (best by ipTM), ipmatch (wrong poses
           whose ipTM matches the oracle donors -- the control for "it only helps when ipTM is low")
  single   1 donor per arm: best1 (single best by GT DockQ), conf1 (single best by ipTM)
  method   diversity walk ordered by DEGREE, K=20, each leader paired with its nearest unused pose
  k50      diversity walk ordered by ipTM, K=50, leaders only, one template per fold

    build_queue.py --design k50 --out run_exp/queue_k50.tsv [--verify <live queue.tsv>]
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")

import paths  # noqa: E402
from separation import pool_classes  # noqa: E402

MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices"
T = 0.23          # CAPRI-acceptable, used both as "distinct enough" and as "similar enough"
SEEDS = {"top5": 400000, "single": 410000, "method": 420000, "k50": 430000}


def leaders(M, order, k, anchor=None):
    """Greedy diversity walk: keep a pose only if it is below T similarity to everything kept."""
    L = [] if anchor is None else [int(anchor)]
    for p in order:
        p = int(p)
        if p in L:
            continue
        if all(M[p, j] < T for j in L):
            L.append(p)
            if len(L) >= k:
                break
    return L


def nearest_unused(M, p, used):
    for q in np.argsort(-M[p]):
        q = int(q)
        if q != p and q not in used:
            return q
    return None


def build(design):
    cls = pool_classes()
    seed = SEEDS[design]
    lines = []
    for f in sorted(glob.glob(f"{MATRICES}/*.npz")):
        iface = os.path.basename(f)[:-4]
        pdb = iface.split("_")[0]
        z = np.load(f)
        M = z["dockq"].astype(float)
        ip = z["pair_iptm"].astype(float)
        dq = z["dockq_gt"].astype(float)
        pose = lambda i: paths.pool_pose(pdb, z["seed"][i], int(z["sample"][i]))  # noqa: E731

        if design == "k50":
            L = leaders(M, np.argsort(-ip), 50)
            for n, p in enumerate(L):
                lines.append(("tmpl", pdb, iface, f"K{n:02d}", seed, [pose(p)]))

        elif design == "method":
            L = leaders(M, np.argsort(-(M >= T).sum(1)), 20, anchor=int(np.argmax(ip)))
            used = set(L)
            for n, p in enumerate(L):
                q = nearest_unused(M, p, used)
                if q is not None:
                    used.add(q)
                ps = [pose(p)] + ([pose(q)] if q is not None else [])
                lines.append(("tmpl", pdb, iface, f"L{n:02d}", seed, ps))

        elif design == "single":
            ib, ic = int(np.argmax(dq)), int(np.argmax(ip))
            picks = {"best1": ib} if ib == ic else {"best1": ib, "conf1": ic}
            for arm, i in picks.items():
                lines.append(("tmpl", pdb, iface, arm, seed, [pose(i)]))

        elif design == "top5":
            o = list(np.argsort(-dq)[:5])
            c = list(np.argsort(-ip)[:5])
            arms = [("oracle", o), ("conf", c)]
            m = ipmatched(dq, ip, o, c, 5)
            if m is not None:
                arms.append(("ipmatch", m))
            for arm, idx in arms:
                for k in range(5):
                    lines.append(("tmpl", pdb, iface, arm, seed + k, [pose(i) for i in idx]))
    return [f"{k}\t{p}\t{i}\t{t}\t{s}\t{' '.join(ps)}" for k, p, i, t, s, ps in lines
            if all(os.path.exists(x) for x in ps)]


def ipmatched(dq, ip, o_idx, c_idx, k, acc=0.23):
    """k WRONG poses whose ipTM matches the oracle donors', disjoint from both other arms."""
    used = set(o_idx) | set(c_idx)
    pool = [i for i in range(len(dq)) if i not in used and dq[i] < acc]
    out = []
    for t in ip[o_idx]:
        cand = [i for i in pool if i not in out]
        if not cand:
            return None
        out.append(min(cand, key=lambda i: abs(ip[i] - t)))
    if len(out) != k or float(np.abs(ip[out] - ip[o_idx]).mean()) > 0.05:
        return None
    return out


def interleave(lines, pattern, last=()):
    """Order interfaces so a partial run stays balanced, spreading complexes so early results are
    not all one pdb. Classes in `last` go entirely at the end."""
    cls = pool_classes()
    byif = collections.OrderedDict()
    for l in lines:
        byif.setdefault(l.split("\t")[2], []).append(l)

    def spread(ifs):
        bypdb = collections.OrderedDict()
        for i in ifs:
            bypdb.setdefault(i.split("_")[0], []).append(i)
        out = []
        while any(bypdb.values()):
            for p in list(bypdb):
                if bypdb[p]:
                    out.append(bypdb[p].pop(0))
        return out

    pools = {k: spread([i for i in byif if cls.get(i) == k]) for k in set(pattern) | set(last)}
    order, idx = [], {k: 0 for k in pools}
    while any(idx[k] < len(pools[k]) for k in pattern):
        for k in pattern:
            if idx[k] < len(pools[k]):
                order.append(pools[k][idx[k]])
                idx[k] += 1
    for k in last:
        order += pools[k][idx.get(k, 0):]
    return [l for i in order for l in byif[i]]


ORDER = {"k50": (["decoy", "decoy", "solved"], ["nocov"]),
         "method": (["decoy", "decoy", "decoy", "solved", "nocov"], []),
         "top5": (["decoy", "decoy", "decoy", "solved", "nocov"], []),
         "single": (None, None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=sorted(SEEDS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", help="diff against the queue the workers actually ran")
    A = ap.parse_args()

    lines = build(A.design)
    pat, last = ORDER[A.design]
    if pat:
        lines = interleave(lines, pat, last)
    os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
    open(A.out, "w").write("\n".join(lines) + "\n")
    print(f"  {A.design}: {len(lines)} folds -> {A.out}")

    if A.verify:
        live = [l.rstrip("\n") for l in open(A.verify) if l.strip()]
        same = live == lines
        print(f"  verify against {A.verify}: {'IDENTICAL' if same else 'DIFFERS'}")
        if not same:
            print(f"    live {len(live)} lines, regenerated {len(lines)}")
            sl, sn = set(live), set(lines)
            print(f"    only in live: {len(sl - sn)}   only in regenerated: {len(sn - sl)}")
            for x in list(sn - sl)[:2]:
                print(f"      + {x[:110]}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
