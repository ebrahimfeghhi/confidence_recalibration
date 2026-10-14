#!/usr/bin/env python
"""Is the method better than just running more seeds? Fixed 70-seed baseline.

THE CONTROL EVERY OTHER TABLE DUCKS. Our "before" is the 50-seed pool, and the method then spends a
median of 20 more folds (31 on decoys). Comparing the two rewards the method for extra compute. So:

    baseline-50      the first 50 free seeds                     (what "before" means everywhere else)
    baseline-70      the first 70 free seeds                     (spend the extra folds on sampling)
    method           the first 50 free seeds + its K templated   (spend them on self-templating)

70 is chosen because the method adds a median of 20 folds across all 292 interfaces. It is uniform,
which a per-interface match is not: the method self-allocates 5 folds to solved interfaces and 40 to
no-coverage ones, so matching its allocation exactly would hand the baseline the method's own
targeting for free.

Seeds are taken in ascending order, never at random, so the arms are reproducible. Scores come from
BOTH score files -- pxm2025/dockq_pxm2025.csv and free/results.csv -- because neither alone covers the
pool plus the sweep, and reading one at a time is how earlier counts of "how many seeds do we have"
came out wrong three different ways.
"""
import collections, csv, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")
import paths                                   # noqa: E402
from fmt import table                          # noqa: E402
from separation import pool_classes            # noqa: E402

ACC, BASE, MATCH = 0.23, 50, 70
MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices"
cls = pool_classes()

seeds = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(str(paths.PXM2025_POOL_CSV))):
    seeds[f"{r['pdb']}_{r['chain1']}{r['chain2']}"][r["seed"]].append(
        (float(r["pair_iptm"]), float(r["dockq"])))
if os.path.exists(f"{paths.FREE_DIR}/results.csv"):
    for r in csv.DictReader(open(f"{paths.FREE_DIR}/results.csv")):
        seeds[r["iface"]][r["seed"]].append((float(r["iptm"]), float(r["dockq"])))

expected = collections.Counter()
for line in open(f"{paths.DATA_DIR}/k50/queue.tsv"):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 6:
        expected[f[2]] += 1
tmpl, folds = collections.defaultdict(list), collections.defaultdict(set)
for r in csv.DictReader(open(f"{paths.DATA_DIR}/k50/results.csv")):
    tmpl[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
    folds[r["iface"]].add(r["arm"])

pick = lambda X: float(X[int(np.argmax([a for a, _ in X]))][1])   # noqa: E731
rows = collections.defaultdict(lambda: collections.defaultdict(list))
ks = collections.defaultdict(list)
for f in sorted(glob.glob(f"{MATRICES}/*.npz")):
    i = os.path.basename(f)[:-4]
    K = expected.get(i, 0)
    if not K or len(folds.get(i, ())) < K:
        continue
    order = sorted(seeds[i], key=lambda s: int(s))
    if len(order) < MATCH:
        continue
    k = cls.get(i, "?")
    ks[k].append(K)
    b50 = [p for s in order[:BASE] for p in seeds[i][s]]
    b70 = [p for s in order[:MATCH] for p in seeds[i][s]]
    rows[k][f"{BASE} free seeds (before)"].append(pick(b50))
    rows[k][f"{MATCH} free seeds (matched)"].append(pick(b70))
    rows[k][f"{BASE} free + K templated"].append(pick(b50 + tmpl[i]))

ARMS = (f"{BASE} free seeds (before)", f"{MATCH} free seeds (matched)", f"{BASE} free + K templated")
for k in ("decoy", "solved", "nocov"):
    if not rows[k][ARMS[0]]:
        continue
    b = np.array(rows[k][ARMS[0]])
    out = []
    for arm in ARMS:
        a = np.array(rows[k][arm])
        out.append([arm, len(a), round(float(a.mean()), 3), f"{int((a >= ACC).sum())}/{len(a)}",
                    int(((b < ACC) & (a >= ACC)).sum()) if arm != ARMS[0] else "-",
                    int(((b >= ACC) & (a < ACC)).sum()) if arm != ARMS[0] else "-"])
    print(f"\n{k.upper()}  (n={len(b)}, median K={int(np.median(ks[k]))} templated folds)")
    print(table(["arm", "interfaces", "mean shipped", "acceptable", "rescued", "broken"],
                out, align="l r r r r r"))
