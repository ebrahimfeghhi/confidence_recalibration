#!/usr/bin/env python
"""Does the label-free selection method rescue confident decoys?

SELECTION IS POOLED OVER THE FREE POSES AND THE TEMPLATED ONES TOGETHER, which is what a user has
in hand. Templated-only selection scores slightly better on decoys (4/39 vs 3/40 when measured) but
broke a solved interface; pooling with free costs one rescue and preserved 24/24.

WHAT IS BEING SCORED. For each interface the method proposed 20 donor pairs with no access to ground
truth: anchor on the top-ipTM pose, order the rest by DEGREE (how many poses lie within pose-pose
DockQ 0.23), walk that order keeping only poses dissimilar to everything kept, then pair each leader
with its nearest unused pose. Every pair was self-templated. Here we pool everything the 20 folds
produced, pick the single most confident pose, and compare its DockQ to what the free pool would have
shipped. Ground truth enters ONLY at this scoring step.

POOLED, NEVER WITHIN-FOLD. In use you cannot tell which of the 20 folds came from a good donor, so
selection has to range over all of them at once. Scoring the best fold instead would be a ceiling, and
reporting one earlier in this project inflated 5 of 32 into 8 of 24.

PARTIAL INTERFACES ARE EXCLUDED. An interface with 3 of its 20 folds done has had only a fraction of
its shortlist explored, and counting it would understate the method early and drift upward as folds
land -- which looks like a trend and is an artefact.
"""
import collections, csv, glob, os, sys
import numpy as np
sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths                                    # noqa: E402
from fmt import table                           # noqa: E402
from separation import pool_classes             # noqa: E402

MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices"
ACC = 0.23
cls = pool_classes()
R = f"{paths.DATA_DIR}/k50/results.csv"
expected = collections.Counter()
for _l in open(f"{paths.DATA_DIR}/k50/queue.tsv"):
    _f = _l.rstrip("\n").split("\t")
    if len(_f) >= 6:
        expected[_f[2]] += 1
made, folds = collections.defaultdict(list), collections.defaultdict(set)
if os.path.exists(R):
    for r in csv.DictReader(open(R)):
        made[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
        folds[r["iface"]].add(r["arm"])

rows = collections.defaultdict(lambda: [[], []])
partial = 0
for f in sorted(glob.glob(f"{MATRICES}/*.npz")):
    i = os.path.basename(f)[:-4]
    if i not in made:
        continue
    # K VARIES PER INTERFACE: the diversity filter exhausts the pool before 50 on 228 of 292
    # interfaces, so completeness is measured against what the QUEUE asked for, not a constant.
    if len(folds[i]) < expected.get(i, 10**9):
        partial += 1
        continue
    z = np.load(f)
    free = list(zip(z["pair_iptm"].astype(float), z["dockq_gt"].astype(float)))
    before = float(max(free)[1])
    # POOL THE FREE POSES BACK IN. Selecting over templated poses alone discards 250 poses of
    # evidence and, worse, discards the PREVALENCE signal: leader selection gives a basin holding 150
    # poses the same single representative as one holding 5. On 8rwb_AC that let a pose templated from
    # the pool's best decoy (ipTM 0.9376 -> 0.9387) outrank 30 genuinely good free poses and ship
    # DockQ 0.076 in place of 0.718. Pooling with the free poses restores the baseline as a floor.
    P = free + made[i]
    after = float(P[int(np.argmax([x for x, _ in P]))][1])
    rows[cls.get(i, "?")][0].append(before)
    rows[cls.get(i, "?")][1].append(after)

out = []
for k in ("decoy", "solved", "nocov"):
    b, a = (np.array(x) for x in rows[k])
    if not len(b):
        continue
    out.append([k, len(b), round(b.mean(), 3), round(a.mean(), 3),
                f"{int((b >= ACC).sum())}/{len(b)}", f"{int((a >= ACC).sum())}/{len(a)}",
                int(((b < ACC) & (a >= ACC)).sum()), int(((b >= ACC) & (a < ACC)).sum())])
print(table(["class", "interfaces complete", "mean before", "mean after",
             "acceptable before", "acceptable after", "rescued", "broken"], out, align="l r r r r r r r"))
print(f"\n  complete interfaces scored: {sum(len(v[0]) for v in rows.values())}   "
      f"partial (excluded): {partial}   folds done: {sum(len(v) for v in folds.values())}/{sum(expected.values())}")
