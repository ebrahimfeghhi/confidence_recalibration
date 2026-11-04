#!/usr/bin/env python
"""Choose 30 of 100 validation seed units per structure, to make the dev set INFORMATIVE.

WHAT THIS IS, SAID PLAINLY. This is deliberate outcome-dependent construction of a DEVELOPMENT
set. It reads DockQ -- a label -- to decide which candidate poses survive. That is not a simulation
of 30 random seeds and must never be reported as one, and no aggregate from the selected pool is
comparable to the old first-30-seed numbers. The final test set stays independent and uses all 100
seeds.

WHY BOTHER. The dev set exists to answer one question: can the head tell a good pose from a bad
one? An interface whose candidates all score the same DockQ answers nothing, however many of them
there are. Measured on the current first-30 pool: 47 of 92 interfaces have within-interface DockQ
sd below 0.05 and 20 are below 0.02, and on those 20 the shipped head already picks an acceptable
pose on 13 of the 13 that have one. Half the benchmark cannot move a result in either direction.

WHAT IT MUST NOT DO. Maximising raw variance is the obvious objective and the wrong one: it
selects the extremes of each interface and throws away the middle, which is where selection is
actually hard. So the objective here is COVERAGE of the full 100-seed quality distribution, with
the bins capped -- a bin already at its target earns nothing more -- and extra weight on the bins
either side of the DockQ 0.23 acceptability threshold.

THE UNIT OF SELECTION IS A SEED, NOT A POSE. A seed unit holds all 5 poses of every interface of
its structure, and the trunk tensors are per unit. So a structure with several interfaces gets ONE
subset, scored jointly across its interfaces with equal weight per interface -- not the best subset
for its first interface.

DETERMINISM. Greedy with a fixed bin layout, fixed weights, and ties broken on the lowest seed
number. No RNG anywhere. Degenerate structures, where every candidate is identical, fall through
to seeds 400000-400029 by that tie-break rather than erroring.

Usage:
  python -I run_logging/select_val_seeds.py --compare          # method comparison, writes nothing
  python -I run_logging/select_val_seeds.py --write-manifest training/valfull/seed_manifest.json
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACC = 0.23
# BIN EDGES, with 0.23 ON a boundary so no bin straddles the threshold and the two bins either
# side of it can be weighted. The top bin is wide because near-native poses are interchangeable
# for selection purposes; the bins around the threshold are narrow because that is where a wrong
# pick costs an acceptable answer.
EDGES = [0.0, 0.05, 0.15, 0.23, 0.35, 0.50, 0.70, 1.01]
WEIGHTS = [1.0, 1.5, 3.0, 3.0, 1.5, 1.0, 1.0]   # per bin, highest either side of 0.23
ALGO_VERSION = "anchored-coverage-v2"


def bin_of(d):
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= d < EDGES[i + 1]:
            return i
    return len(EDGES) - 2


def load(results, seed_lo, seed_hi):
    """structure -> interface -> seed -> [dockq per pose]; plus the same for iptm."""
    dq = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    it = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    with open(results) as f:
        for r in csv.DictReader(f):
            ifc = (r.get("iface") or "").strip()
            if not ifc:
                continue
            try:
                s = int(r["seed"])
                d = float(r["dockq"])
            except (ValueError, TypeError, KeyError):
                continue
            if not (seed_lo <= s <= seed_hi):
                continue
            pdb = ifc.split("_")[0]
            dq[pdb][ifc][s].append(d)
            try:
                it[pdb][ifc][s].append(float(r["iptm"]))
            except (ValueError, TypeError, KeyError):
                pass
    return dq, it


def targets(per_if, k, n_total):
    """Per interface, the capped per-bin quota the selected subset should reach."""
    out = {}
    for ifc, byseed in per_if.items():
        cnt = collections.Counter()
        for s, ds in byseed.items():
            for d in ds:
                cnt[bin_of(d)] += 1
        tot = sum(cnt.values()) or 1
        out[ifc] = {b: max(1, round(c * k / len(byseed))) if c else 0 for b, c in cnt.items()}
    return out


def score(sel_counts, tgt):
    """Capped, weighted coverage in [0,1]. A bin at quota earns nothing more."""
    num = den = 0.0
    for b, t in tgt.items():
        if not t:
            continue
        w = WEIGHTS[b]
        num += w * min(sel_counts.get(b, 0), t)
        den += w * t
    return num / den if den else 1.0


def greedy(per_if, k):
    seeds = sorted({s for byseed in per_if.values() for s in byseed})
    tgt = targets(per_if, k, len(seeds))
    chosen = []
    counts = {ifc: collections.Counter() for ifc in per_if}
    while len(chosen) < k and len(chosen) < len(seeds):
        best = None
        base = st.mean([score(counts[i], tgt[i]) for i in per_if])
        for s in seeds:
            if s in chosen:
                continue
            tot = 0.0
            for ifc, byseed in per_if.items():
                c = counts[ifc].copy()
                for d in byseed.get(s, []):
                    c[bin_of(d)] += 1
                tot += score(c, tgt[ifc])
            gain = tot / len(per_if) - base
            # TIE-BREAK ON THE LOWEST SEED, so the result is identical on every machine.
            if best is None or gain > best[0] + 1e-12:
                best = (gain, s)
        _, s = best
        chosen.append(s)
        for ifc, byseed in per_if.items():
            for d in byseed.get(s, []):
                counts[ifc][bin_of(d)] += 1
    return sorted(chosen)


def anchored(per_if, k):
    """ORACLE-ANCHORED COVERAGE GREEDY -- the method this file settles on.

    Plain coverage greedy reproduces the shape of the quality distribution well but drops 0.0218
    of oracle DockQ, because the single best pose of an interface sits in a bin that one other
    seed can also fill. The ceiling of the selection task is the oracle, so losing it lowers the
    headroom the head is being measured against for no gain in informativeness.

    Fix: force in the seed carrying each interface's best pose BEFORE the greedy runs, then let
    coverage spend the remaining slots. Costs at most one slot per interface -- structures here
    carry 1 to 3 -- and makes oracle retention exact by construction rather than incidental.

    A second anchor goes in for every interface that has any pose on each side of 0.23: the seed
    holding the best individual POSE below the threshold. Without it a threshold-spanning
    interface can be reduced to all-acceptable, which removes precisely the decision the dev set
    exists to test. If that seed is the oracle seed -- a seed often holds poses on both sides --
    the dedup absorbs it and no extra slot is spent, which is correct: the pool still spans.
    """
    anchors = []
    for ifc, byseed in sorted(per_if.items()):
        best = max(((max(ds), s) for s, ds in byseed.items() if ds), default=None)
        if best:
            anchors.append(best[1])
        # THE SUBTHRESHOLD ANCHOR IS PER POSE, NOT PER SEED. Raised in review. The first version
        # filtered to seeds whose BEST pose was below 0.23, i.e. seeds that are entirely
        # unacceptable -- which skips the seed holding the best individual subthreshold pose
        # whenever that seed also holds an acceptable one, and a seed routinely holds both.
        # The quantity wanted is max over POSES with d < ACC, and the seed carrying it.
        sub = [(d, s) for s, ds in byseed.items() for d in ds if d < ACC]
        if sub and best and best[0] >= ACC:
            anchors.append(max(sub)[1])
    seeds = sorted({s for b in per_if.values() for s in b})
    chosen = []
    for s in anchors:                       # dedup, keep first occurrence, deterministic order
        if s not in chosen and len(chosen) < k:
            chosen.append(s)
    tgt = targets(per_if, k, len(seeds))
    counts = {ifc: collections.Counter() for ifc in per_if}
    for s in chosen:
        for ifc, byseed in per_if.items():
            for d in byseed.get(s, []):
                counts[ifc][bin_of(d)] += 1
    while len(chosen) < k and len(chosen) < len(seeds):
        base = st.mean([score(counts[i], tgt[i]) for i in per_if])
        best = None
        for s in seeds:
            if s in chosen:
                continue
            tot = 0.0
            for ifc, byseed in per_if.items():
                c = counts[ifc].copy()
                for d in byseed.get(s, []):
                    c[bin_of(d)] += 1
                tot += score(c, tgt[ifc])
            gain = tot / len(per_if) - base
            if best is None or gain > best[0] + 1e-12:
                best = (gain, s)
        chosen.append(best[1])
        for ifc, byseed in per_if.items():
            for d in byseed.get(best[1], []):
                counts[ifc][bin_of(d)] += 1
    return sorted(chosen)


def first_k(per_if, k):
    return sorted({s for b in per_if.values() for s in b})[:k]


def maxvar(per_if, k):
    """THE METHOD THIS FILE EXISTS TO REJECT -- kept so the comparison is honest."""
    seeds = sorted({s for b in per_if.values() for s in b})
    chosen = []
    while len(chosen) < k:
        best = None
        for s in seeds:
            if s in chosen:
                continue
            tot = 0.0
            for ifc, byseed in per_if.items():
                vals = [d for q in chosen + [s] for d in byseed.get(q, [])]
                tot += st.pstdev(vals) if len(vals) > 1 else 0.0
            v = tot / len(per_if)
            if best is None or v > best[0] + 1e-12:
                best = (v, s)
        chosen.append(best[1])
    return sorted(chosen)


def stratified(per_if, k):
    """Rank seeds by each interface's mean DockQ, take an even spread of ranks."""
    seeds = sorted({s for b in per_if.values() for s in b})
    key = {}
    for s in seeds:
        ms = [st.mean(byseed[s]) for byseed in per_if.values() if byseed.get(s)]
        key[s] = st.mean(ms) if ms else 0.0
    order = sorted(seeds, key=lambda s: (key[s], s))
    idx = [round(i * (len(order) - 1) / (k - 1)) for i in range(k)]
    out, seen = [], set()
    for i in idx:
        j = i
        while order[j] in seen:
            j = (j + 1) % len(order)
        seen.add(order[j]); out.append(order[j])
    return sorted(out)


def metrics(per_if, sel):
    """Coverage of the FULL distribution, retained oracle, acceptability, spread, threshold span."""
    out = []
    for ifc, byseed in per_if.items():
        full = [d for s in byseed for d in byseed[s]]
        sub = [d for s in sel for d in byseed.get(s, [])]
        if not sub:
            continue
        fb = collections.Counter(bin_of(d) for d in full)
        sb = collections.Counter(bin_of(d) for d in sub)
        occupied = sum(1 for b in fb if sb.get(b))
        # L1 distance between the two bin HISTOGRAMS, normalised -> 0 is identical shape
        fn = {b: c / len(full) for b, c in fb.items()}
        sn = {b: c / len(sub) for b, c in sb.items()}
        l1 = sum(abs(fn.get(b, 0) - sn.get(b, 0)) for b in set(fn) | set(sn)) / 2
        out.append({
            "iface": ifc,
            "bins_full": len(fb), "bins_kept": occupied,
            "bin_cover": occupied / len(fb),
            "hist_l1": l1,
            "oracle_full": max(full), "oracle_sub": max(sub),
            "sd_full": st.pstdev(full), "sd_sub": st.pstdev(sub),
            "acc_full": max(full) >= ACC, "acc_sub": max(sub) >= ACC,
            "spans_full": (max(full) >= ACC) and (min(full) < ACC),
            "spans_sub": (max(sub) >= ACC) and (min(sub) < ACC),
        })
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--results", default="validation/fold/results.csv")
ap.add_argument("--k", type=int, default=30)
ap.add_argument("--seed-lo", type=int, default=400000)
ap.add_argument("--seed-hi", type=int, default=400099)
ap.add_argument("--compare", action="store_true")
ap.add_argument("--write-manifest", default="")
a = ap.parse_args()

DQ, IT = load(os.path.join(ROOT, a.results), a.seed_lo, a.seed_hi)
print(f"loaded {len(DQ)} structure(s), {sum(len(v) for v in DQ.values())} interface(s), "
      f"seeds {a.seed_lo}-{a.seed_hi}")

METHODS = [("first30", first_k), ("stratified", stratified), ("maxvar", maxvar),
           ("coverage-greedy", greedy), ("anchored-coverage", anchored)]
if not a.compare:
    METHODS = [("anchored-coverage", anchored)]

results = {}
for name, fn in METHODS:
    rows = []
    picks = {}
    for pdb, per_if in sorted(DQ.items()):
        sel = fn(per_if, a.k)
        picks[pdb] = sel
        rows += metrics(per_if, sel)
    results[name] = (rows, picks)
    n = len(rows)
    print(f"\n=== {name} ===")
    print(f"  bin coverage of the full distribution: {st.mean([r['bin_cover'] for r in rows]):.3f} "
          f"(1.0 = every occupied quality bin still represented)")
    print(f"  histogram L1 distance to full:         {st.mean([r['hist_l1'] for r in rows]):.4f} "
          f"(0 = identical shape)")
    print(f"  oracle DockQ retained:                 {st.mean([r['oracle_sub'] for r in rows]):.4f} "
          f"of {st.mean([r['oracle_full'] for r in rows]):.4f} "
          f"(lost {st.mean([r['oracle_full']-r['oracle_sub'] for r in rows]):.4f})")
    print(f"  within-interface sd:                   {st.mean([r['sd_sub'] for r in rows]):.4f} "
          f"vs {st.mean([r['sd_full'] for r in rows]):.4f} full")
    print(f"  interfaces with >=1 acceptable pose:   {sum(r['acc_sub'] for r in rows)} "
          f"of {sum(r['acc_full'] for r in rows)} available")
    print(f"  interfaces SPANNING 0.23:              {sum(r['spans_sub'] for r in rows)} "
          f"of {sum(r['spans_full'] for r in rows)} available")
    print(f"  interfaces with sd < 0.05:             {sum(1 for r in rows if r['sd_sub']<0.05)} of {n}")

if a.write_manifest:
    rows, picks = results["anchored-coverage"]
    byif = {r["iface"]: r for r in rows}
    man = {"algo": ALGO_VERSION, "k": a.k, "edges": EDGES, "weights": WEIGHTS,
           "acceptable": ACC, "results_csv": a.results,
           "seed_range": [a.seed_lo, a.seed_hi], "rng": "none (deterministic greedy)",
           "structures": {}}
    for pdb, sel in picks.items():
        man["structures"][pdb] = {
            "seeds": sel,
            "interfaces": {r["iface"]: {k: (round(v, 6) if isinstance(v, float) else v)
                                        for k, v in r.items() if k != "iface"}
                           for r in rows if r["iface"].split("_")[0] == pdb},
        }
    blob = json.dumps(man, sort_keys=True).encode()
    man["digest"] = hashlib.sha256(blob).hexdigest()[:16]
    p = os.path.join(ROOT, a.write_manifest)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    print(f"\nwrote {a.write_manifest}  digest {man['digest']}  "
          f"{len(man['structures'])} structure(s)")
