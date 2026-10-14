#!/usr/bin/env python
"""The four-way comparison: oracle-1 vs the K=50 method, each with and without the free poses.

MATCHED INTERFACES ONLY. Every row is computed on the interfaces that are COMPLETE in both runs, so
the comparison is never between different interface sets -- the oracle run covers 188 interfaces and
the method run is still filling in, and comparing their full sets would flatter whichever happens to
have the easier remainder.

WHY BOTH POOLING RULES. Selecting over templated poses alone discards the free pool's 250 poses and
its prevalence signal; selecting over both restores the baseline as a floor. Which is better depends
on donor quality, and the two runs sit on opposite sides of that: the oracle templates the pool's best
pose and its output is often beaten by the free pool's confident decoy, while the method's weaker
donors benefit from the free poses as a safety net.
"""
import collections, csv, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths                                     # noqa: E402
from fmt import table                            # noqa: E402
from separation import pool_classes              # noqa: E402

MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices"
ACC = 0.23
cls = pool_classes()
expected = collections.Counter()
for line in open(f"{paths.DATA_DIR}/k50/queue.tsv"):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 6:
        expected[f[2]] += 1


def load(path, arm_only):
    made, units = collections.defaultdict(list), collections.defaultdict(set)
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            made[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
            units[r["iface"]].add(r["arm"] if arm_only else (r["arm"], r["seed"]))
    return made, units


k_made, k_units = load(f"{paths.DATA_DIR}/k50/results.csv", True)
s_made, s_units = load(f"{paths.DATA_DIR}/single/results.csv", False)
pick = lambda X: float(X[int(np.argmax([a for a, _ in X]))][1])   # noqa: E731

res = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f"{MATRICES}/*.npz")):
    i = os.path.basename(f)[:-4]
    k = cls.get(i, "?")
    # complete in BOTH runs, or the row is not comparable
    if len(k_units.get(i, ())) < expected.get(i, 10 ** 9) or len(s_units.get(i, ())) < 2:
        continue
    z = np.load(f)
    free = list(zip(z["pair_iptm"].astype(float), z["dockq_gt"].astype(float)))
    res[k]["free only"].append(pick(free))
    res[k]["oracle-1, templated only"].append(pick(s_made[i]))
    res[k]["oracle-1, free+templated"].append(pick(free + s_made[i]))
    res[k]["K=50, templated only"].append(pick(k_made[i]))
    res[k]["K=50, free+templated"].append(pick(free + k_made[i]))

ARMS = ("oracle-1, templated only", "oracle-1, free+templated",
        "K=50, templated only", "K=50, free+templated")
rows = []
nd = len(res["decoy"]["free only"])
ns = len(res["solved"]["free only"])
for arm in ARMS:
    r = [arm]
    for k, n in (("decoy", nd), ("solved", ns)):
        if not n:
            r += ["-", "-"]
            continue
        b, a = np.array(res[k]["free only"]), np.array(res[k][arm])
        r.append(f"{int(((b < ACC) & (a >= ACC)).sum())}/{n}")
        r.append(int(((b >= ACC) & (a < ACC)).sum()))
    rows.append(r)
print(table(["arm / pooling", "decoys rescued", "decoys broken", "solved rescued", "solved broken"],
            rows, align="l r r r r"))
print(f"\n  matched interfaces: {nd} decoy, {ns} solved (complete in both runs)")
print(f"  free-pool baseline: decoys 0/{nd} acceptable, solved {int((np.array(res['solved']['free only']) >= ACC).sum())}/{ns}")
