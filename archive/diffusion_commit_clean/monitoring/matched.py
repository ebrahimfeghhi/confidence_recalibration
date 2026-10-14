#!/usr/bin/env python
"""Compute-matched: 50 free seeds + K templated folds versus simply running 70 free seeds.

THE QUESTION EVERY OTHER TABLE DUCKS. All our comparisons pit the method against a FIXED 50-seed pool,
so the method gets K extra folds the baseline never had. That flatters it. The honest question is
whether those K folds are better spent self-templating on diverse leaders or just sampling more seeds.

The free sweep kept running past the original pool, leaving ~50 additional seeds (300050+) per
interface in experimental_data/free/results.csv -- a ready-made matched baseline, no new compute.

Three arms, all selected the same way (highest ipTM over whatever poses the arm has):
    pool            the original 50-seed pool               (what we call "before")
    pool + K free   plus K extra FREE seeds                 (compute-matched control)
    pool + K tmpl   plus the K self-templated folds         (the method)

Extra seeds are taken in ascending seed order, not at random, so the arm is reproducible.

CAVEAT worth carrying: interfaces that have enough extra seeds are the ones the sweep revisited most,
which correlates with being small and fast to fold. The matched subset may be easier than the whole.
"""
import collections, csv, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")
import paths                                   # noqa: E402
from fmt import table                          # noqa: E402
from separation import pool_classes            # noqa: E402

ACC = 0.23
MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices"
cls = pool_classes()

expected = collections.Counter()
for line in open(f"{paths.DATA_DIR}/k50/queue.tsv"):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 6:
        expected[f[2]] += 1

tmpl, folds = collections.defaultdict(list), collections.defaultdict(set)
for r in csv.DictReader(open(f"{paths.DATA_DIR}/k50/results.csv")):
    tmpl[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
    folds[r["iface"]].add(r["arm"])

# extra free seeds the sweep produced beyond the scored pool
extra = collections.defaultdict(lambda: collections.defaultdict(list))
if os.path.exists(f"{paths.FREE_DIR}/results.csv"):
    for r in csv.DictReader(open(f"{paths.FREE_DIR}/results.csv")):
        extra[r["iface"]][r["seed"]].append((float(r["iptm"]), float(r["dockq"])))

pick = lambda X: float(X[int(np.argmax([a for a, _ in X]))][1])   # noqa: E731
rows = collections.defaultdict(lambda: collections.defaultdict(list))
nmatched = collections.Counter()
for f in sorted(glob.glob(f"{MATRICES}/*.npz")):
    i = os.path.basename(f)[:-4]
    K = expected.get(i, 0)
    if not K or len(folds.get(i, ())) < K:
        continue
    z = np.load(f)
    pool = list(zip(z["pair_iptm"].astype(float), z["dockq_gt"].astype(float)))
    seeds_in_pool = {str(s) for s in z["seed"]}
    avail = [s for s in sorted(extra[i], key=lambda x: int(x)) if s not in seeds_in_pool]
    if len(avail) < K:
        continue                      # cannot match this interface's budget: exclude, do not fudge
    nmatched[cls.get(i, "?")] += 1
    more = [p for s in avail[:K] for p in extra[i][s]]
    k = cls.get(i, "?")
    rows[k]["pool only"].append(pick(pool))
    rows[k][f"pool + K free seeds"].append(pick(pool + more))
    rows[k][f"pool + K templated"].append(pick(pool + tmpl[i]))

ARMS = ("pool only", "pool + K free seeds", "pool + K templated")
for k in ("decoy", "solved", "nocov"):
    if not rows[k]["pool only"]:
        continue
    b = np.array(rows[k]["pool only"])
    out = []
    for arm in ARMS:
        a = np.array(rows[k][arm])
        out.append([arm, len(a), round(float(a.mean()), 3), f"{int((a >= ACC).sum())}/{len(a)}",
                    int(((b < ACC) & (a >= ACC)).sum()) if arm != "pool only" else "-",
                    int(((b >= ACC) & (a < ACC)).sum()) if arm != "pool only" else "-"])
    print(f"\n{k.upper()}  (n={len(b)} interfaces with enough extra free seeds to match)")
    print(table(["arm", "interfaces", "mean shipped", "acceptable", "rescued", "broken"],
                out, align="l r r r r r"))
print(f"\n  matched interfaces: {dict(nmatched)}")
