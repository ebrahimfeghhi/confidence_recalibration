#!/usr/bin/env python
"""Bar chart: how many extra free seeds the matched baseline runs per complex.

    ./plot_topup_seeds.py [--out <png>]

x  = complex (PDB id), ordered by total extra seeds then by id
y  = number of extra free seeds from block 151-200, split into
       DONE      already scored (free_compute_matched + free_matched_topup)
       TO RUN    still in free_matched_topup/queue.tsv
One bar = one complex. Bar height = the method's own donor count for that complex (capped at 50),
which is exactly the compute the matched baseline must spend there. Bars are coloured by the
complex's class: decoy-containing, else solved-containing, else nocov.

Complexes where EVERY interface already has free top-ipTM >= 0.8 are omitted: the method is not
applied there, so the baseline runs no extra seeds for them.
"""
import argparse, collections, csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/ebrahim/tts_abag")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import compare_arms as C
from score_experiment import classes_at, pool_by_seed

FB = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"
M, N = f"{FB}/free_matched_topup", f"{FB}/selftmpl_mask_on_nc10"
T, BLOCK = 0.8, (151, 200)

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/home/ebrahim/tts_abag/experimental_results/topup_extra_seeds.png")
A = ap.parse_args()
ds, _, _ = C.arm_defaults("foldbench")
pool, first = pool_by_seed(50, str(ds.pool_csv)); cls = classes_at(pool, first)
per, _ = C.donors_used(N, first)

ipf = collections.defaultdict(list)
for i in pool:
    p = i.split("_")[0]
    r = [(x, d) for s, _, d, x in pool[i] if s in first[p]]
    if r: ipf[p].append((i, max(r)[0]))
need = {p for p, v in ipf.items() if any(x[1] < T for x in v)}

have = collections.defaultdict(set)
for sub in ("free_compute_matched", "free_matched_topup"):
    try:
        for r in csv.DictReader(open(f"{FB}/{sub}/results.csv")):
            if r.get("dockq") and BLOCK[0] <= int(r["seed"]) <= BLOCK[1]:
                have[r["iface"].split("_")[0]].add(int(r["seed"]))
    except Exception: pass
queued = collections.Counter()
for l in open(f"{M}/queue.tsv"):
    f = l.rstrip("\n").split("\t")
    if len(f) >= 5: queued[f[1]] += 1

def klass(p):
    cs = {cls.get(i) for i, _ in ipf[p]}
    return "decoy" if "decoy" in cs else ("solved" if "solved" in cs else "nocov")

rows = []
for p in need:
    want = min(len(per.get(p, ())), 50)
    done = min(len(have[p]), want)
    # TO RUN is what is LEFT, not the raw queue line count. The queue was written before these folds
    # started, so a fold finished since then appears in BOTH have[] and queued[] -- stacking those
    # pushed 8slb to 90 and 8av2 to 66 on a scale capped at 50.
    todo = max(0, want - done)
    rows.append((want, p, done, todo, klass(p)))
rows.sort(key=lambda r: (-r[0], r[1]))

COL = {"decoy": "#c0392b", "solved": "#2980b9", "nocov": "#7f8c8d"}
fig, ax = plt.subplots(figsize=(max(9, .3 * len(rows)), 5.2))
x = range(len(rows))
ax.bar(x, [r[2] for r in rows], color=[COL[r[4]] for r in rows], label="done")
ax.bar(x, [r[3] for r in rows], bottom=[r[2] for r in rows],
       color=[COL[r[4]] for r in rows], alpha=.35, hatch="///", label="to run")
ax.set_xticks(list(x)); ax.set_xticklabels([r[1] for r in rows], rotation=90, fontsize=7)
ax.set_xlabel("complex (one bar per complex)")
ax.set_ylabel(f"extra free seeds from block {BLOCK[0]}-{BLOCK[1]}")
ax.set_title(f"Matched-baseline extra seeds per complex  (= the method's donor count, capped at 50)\n"
             f"solid = already scored, hatched = still queued; only complexes with an interface below "
             f"free top-ipTM {T}", fontsize=9)
h = [plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ("decoy", "solved", "nocov")]
h.append(plt.Rectangle((0, 0), 1, 1, color="k", alpha=.35, hatch="///"))
ax.legend(h, ["decoy-containing", "solved-containing", "nocov", "still to run"], fontsize=8)
ax.margins(x=.01); fig.tight_layout(); fig.savefig(A.out, dpi=150)
print(f"wrote {A.out}")
print(f"  complexes {len(rows)}   total extra seeds {sum(r[0] for r in rows)}"
      f"   done {sum(r[2] for r in rows)}   queued {sum(r[3] for r in rows)}")
for k in ("decoy", "solved", "nocov"):
    sub = [r for r in rows if r[4] == k]
    if sub: print(f"  {k:<7} {len(sub):>3} complexes, seeds min {min(r[0] for r in sub)} "
                  f"median {sorted(r[0] for r in sub)[len(sub)//2]} max {max(r[0] for r in sub)}")
