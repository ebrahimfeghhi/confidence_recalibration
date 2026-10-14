#!/usr/bin/env python
"""Queue the top-5-pose self-templating experiment: oracle templates versus confident templates.

THE COMPARISON. For each interface take its 5 best poses two ways -- by ground-truth DockQ (oracle)
and by ipTM (what a user could actually pick) -- and self-template on each set. The oracle arm is an
UPPER BOUND, not a method: it answers "if we could identify the good poses, would self-templating
exploit them?" If it does not beat the confidence arm, no amount of better clustering helps, because
clustering can at best find what the oracle already had.

The two sets are nearly disjoint and that is the point: on 87 of 91 decoy interfaces they share ZERO
poses, and the oracle set averages DockQ 0.369 against the confidence set's 0.061.

Emits fold.py's queue format, which takes explicit pose paths:

    all5  <pdb>  <interface>  <tag>  <seed>  <pose1> ... <pose5>

`all5` means "these N poses are one template set, averaged by TemplateEmbedder" -- it does NOT
require them to come from one seed, despite the name.

FRESH SEEDS (400000+) so nothing collides with the pool's 300000-300049 run directories.

    make_queue.py --out queue.tsv [--classes decoy,solved,nocov] [--nseeds 5] [--ntmpl 5]
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths  # noqa: E402
from separation import pool_classes  # noqa: E402

SEED0 = 400000


MAX_IP_DEV = 0.05   # a control that is 0.285 ipTM away from what it controls for is not a control


def ipmatched(dq, ip, o_idx, c_idx, k, acc=0.23):
    """k WRONG poses whose ipTM matches the oracle donors', disjoint from both other arms.

    THE CONTROL THIS EXISTS FOR. On 9h01_AB the oracle donors gained +0.454 ipTM and the confident
    ones +0.096, but the oracle donors STARTED at 0.495 and the confident ones at 0.837 -- so
    "self-templating helps more when ipTM is low" explains that without pose quality mattering at all.
    Matching ipTM while holding DockQ wrong removes the headroom explanation: any remaining gap is
    attributable to the donors being correct.

    Greedy nearest-ipTM without replacement, one match per oracle donor. Returns None when the
    interface cannot supply k wrong poses at those confidences -- which is not a failure, it is an
    interface where the control is impossible, and it must be recorded rather than approximated with
    whatever poses happen to be left.
    """
    used = set(o_idx) | set(c_idx)
    pool = [i for i in range(len(dq)) if i not in used and dq[i] < acc]
    out = []
    for t in ip[o_idx]:
        cand = [i for i in pool if i not in out]
        if not cand:
            return None
        out.append(min(cand, key=lambda i: abs(ip[i] - t)))
    if len(out) != k:
        return None
    # REFUSE A BAD MATCH. On solved interfaces the high-ipTM poses ARE the good ones, so there is no
    # wrong pose at matching confidence and the nearest available is 0.285 ipTM away -- emitting that
    # would label a different experiment as a control. Better to have no arm on those interfaces and
    # say so than to have one that cannot support the claim.
    if float(np.abs(ip[out] - ip[o_idx]).mean()) > MAX_IP_DEV:
        return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--classes", default="decoy,solved,nocov")
    ap.add_argument("--nseeds", type=int, default=5)
    ap.add_argument("--ntmpl", type=int, default=5)
    A = ap.parse_args()

    keep = set(A.classes.split(","))
    cls = pool_classes()
    lines, stats = [], collections.Counter()
    for f in sorted(glob.glob(f"{A.matrices}/*.npz")):
        iface = os.path.basename(f)[:-4]
        if cls.get(iface) not in keep:
            continue
        pdb = iface.split("_")[0]
        z = np.load(f)
        if len(z["seed"]) < A.ntmpl:
            continue
        paths_by_i = [paths.pool_pose(pdb, s, int(x)) for s, x in zip(z["seed"], z["sample"])]
        dqv, ipv = z["dockq_gt"].astype(float), z["pair_iptm"].astype(float)
        o_idx = list(np.argsort(-dqv)[:A.ntmpl])
        c_idx = list(np.argsort(-ipv)[:A.ntmpl])
        arms = [("oracle", o_idx), ("conf", c_idx)]
        m_idx = ipmatched(dqv, ipv, o_idx, c_idx, A.ntmpl)
        if m_idx is not None:
            arms.append(("ipmatch", m_idx))
        for arm, order in arms:
            sel = [paths_by_i[i] for i in order[:A.ntmpl]]
            missing = [p for p in sel if not os.path.exists(p)]
            if missing:
                print(f"  skip {iface} {arm}: {len(missing)} pose files absent", flush=True)
                continue
            for k in range(A.nseeds):
                lines.append("\t".join(["all5", pdb, iface, arm, str(SEED0 + k), " ".join(sel)]))
            stats[arm] += 1

    with open(A.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {A.out}: {len(lines)} folds  " +
          "  ".join(f"{k}={v} interfaces" for k, v in sorted(stats.items())) +
          f"  x {A.nseeds} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
