#!/usr/bin/env python
"""free 50 seeds vs free + matched extra seeds, drawn from ONE clean seed block.

    ./free_vs_matched.py [--dataset foldbench] [--n-extra 50] [--iptm-max 0.8]

THE SANITY CHECK THIS EXISTS FOR. Adding free seeds should make the answer BETTER. It did not: with
the old baseline, "free + matched" lost to plain "free" on 23 of 131 interfaces and was 0.013 worse
on mean DockQ. Two candidate causes were ruled out --

  not a scaling law   top-ipTM accuracy rises monotonically with seeds WITHIN the pool
                      (67.5% at 5 seeds -> 81.0% at 50, 40 random subsets per point)
  not a settings gap  free_compute_matched records n_cycle 10, n_sample 5, bf16, no templates, the
                      same as the pool's inferred defaults

-- leaving the seed block itself. The old matched arm drew from the UNION of free_compute_matched
(the purpose-built 151-200 block) and the free pool's own results.csv, which carries 697 seeds in
ragged leftovers (0-100, 151-339, 538-572, 739-861, 1327-1453, 1915-2036) whose provenance was never
recorded -- the FoldBench free pool has no run.json at all, every directory is *-legacy-no-provenance.
matched_seeds() takes the lowest-numbered available, so it drew mostly from 0-100. Measured over 113
interfaces, those drawn extras were 0.060 worse than the pool on mean DockQ, 0.061 on mean ipTM and
0.063 on oracle, and held a better pose than the pool on only 41% of interfaces (chance ~50%).

So this script reads the matched extras from free_compute_matched ONLY. Interfaces whose complex has
fewer than --n-extra seeds in that block are reported as NOT YET COVERED rather than silently folded
into the mean -- the earlier attempt to restrict the block dropped every decoy interface without
saying so, because free_compute_matched covered only 74 complexes and none of the decoys.

The free_matched_topup run is folding the missing 151-200 seeds for the 38 complexes that have an
interface below --iptm-max, which is where the method is applied and therefore where the baseline
needs matching.
"""
import argparse
import collections
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/ebrahim/tts_abag")

import compare_arms as C  # noqa: E402
from score_experiment import ACCEPTABLE, classes_at, pool_by_seed  # noqa: E402

# ONE directory per arm, as of the 2026-10-06 consolidation, and DERIVED FROM THE DATASET rather than
# hardcoded. The method briefly lived in two foldbench directories and this file read only the first,
# so the decoy row printed as empty for hours while the folds existed on disk; it was then also a
# literal foldbench path, so --dataset pxm2025 would have silently scored foldbench's arm.
BLOCK_BY_DATASET = {"foldbench": (151, 200), "pxm2025": (300050, 300149)}


def method_arms(ds):
    """out: [the dataset's method arm], the mask-ON self-templating run"""
    return [f"{ds.data}/selftmpl_mask_on_nc10"]

# the clean, provenance-recorded extra-seed block per dataset; main() picks by --dataset. The pool
# itself is 101-150 on foldbench and 300000-300049 on pxm2025.


def matched_rows(ds, lo, hi):
    """out: {iface: {seed: [(iptm, dockq)]}} from free_compute_matched and the top-up, block only."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for sub in ("free",):   # free/results.csv IS the matched block since the 2026-10-06 split
        # ds.data is ALREADY the dataset's own directory (for FoldBench that is
        # experimental_data/protenix/foldbench), so appending protenix/foldbench again -- as an
        # earlier version did -- silently read nothing and printed an empty table while 44
        # interfaces already had their full 50 seeds.
        p = f"{ds.data}/{sub}/results.csv"
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if not (r.get("dockq") and r.get("iptm")):
                continue
            try:
                s = int(r["seed"])
            except ValueError:
                continue
            if lo <= s <= hi:
                out[r["iface"]][s].append((float(r["iptm"]), float(r["dockq"])))
    return out


def method_rows(exps, lo, hi):
    """out: ({iface: [(iptm, dockq)]} templated poses, {complex: in-block and fully scored?})

    SEED DISCIPLINE. Every number in this file has to come from seeds 101-200: the free pool is
    101-150 on all 104 complexes and the matched extras are 151-200. The method arm is NOT uniformly
    in that block -- 29 of its 104 complexes were queued with the contaminated `matched_seeds` draw
    (7ucx got seeds 0-3), the same contamination that made the old baseline look wrong. So a complex
    qualifies only if EVERY queued method fold has a seed in [lo, hi] AND every one of them is
    scored; a partially scored complex would quietly compare a full baseline against a short method.
    """
    want = collections.defaultdict(list)
    got = collections.defaultdict(set)
    rows = collections.defaultdict(list)
    for exp in ([exps] if isinstance(exps, str) else exps):
        # A refolded complex appears in BOTH queues. Its parent-arm folds are all outside the block
        # by construction -- that is why it was refolded -- so taking the in-block seeds as the queue
        # for such a complex keeps exactly the refold and drops the contaminated original.
        if not os.path.exists(f"{exp}/queue.tsv"):
            continue
        for line in open(f"{exp}/queue.tsv"):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 6 and lo <= int(f[4]) <= hi:
                want[f[1]].append(int(f[4]))
        if not os.path.exists(f"{exp}/results.csv"):
            continue
        for r in csv.DictReader(open(f"{exp}/results.csv")):
            if not (r.get("dockq") and r.get("iptm")):
                continue
            s = int(r["seed"])
            if not (lo <= s <= hi):
                continue
            got[r["iface"].split("_")[0]].add(s)
            rows[r["iface"]].append((float(r["iptm"]), float(r["dockq"])))
    ok = {p: len(got[p]) >= len(set(v)) for p, v in want.items()}
    return rows, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="foldbench")
    ap.add_argument("--n-extra", type=int, default=50)
    ap.add_argument("--iptm-max", type=float, default=0.8)
    ap.add_argument("--oracle", action="store_true", help="also print the oracle table (off by default)")
    # ON by default (user, 2026-10-06), with --no-method-only to drop it. It is a DIAGNOSTIC, not a
    # deployable arm: selecting from templated poses alone means discarding the free pool, which you
    # could only justify by knowing the answer in advance. It is shown because the two columns
    # disagree in an informative way -- on decoys method-only reaches 4/13 while the pool reaches
    # 2/13, so pooling LOSES rescues the method found, a more confident wrong free pose outranking a
    # correct templated one. Quote `free 50 + method` as the result; read this column to see whether
    # a gap is coverage or selection.
    ap.add_argument("--method-only", action=argparse.BooleanOptionalAction, default=True,
                    help="show the TEMPLATED-POSES-ONLY diagnostic column (default: on)")
    A = ap.parse_args()
    ds, _, _ = C.arm_defaults(A.dataset)
    global BLOCK, METHOD_ARM, METHOD_ARMS
    BLOCK = BLOCK_BY_DATASET[A.dataset]
    METHOD_ARMS = method_arms(ds)
    METHOD_ARM = METHOD_ARMS[0]
    pool, first = pool_by_seed(50, str(ds.pool_csv))
    cls = classes_at(pool, first)
    mc = matched_rows(ds, *BLOCK)
    # COVERAGE THRESHOLD = the method's own donor count for that complex, capped at 50 -- not a flat
    # 50. Self-templating picks donors PER INTERFACE: a complex with N interfaces contributes the M_i
    # donors of each, capped at 50 per complex. Verified on all 104 complexes that distinct donors
    # equals queue folds, so this count is exactly the folds the method spends and therefore exactly
    # what the matched baseline must spend. Demanding 50 everywhere would leave small-donor complexes
    # permanently "not yet covered" (7ucx needs 4, 8aok 8) while their baseline was already complete.
    per, _ = C.donors_used(METHOD_ARM, first)

    free_ip = {}
    for i in pool:
        pdb = i.split("_")[0]
        r = [(p, d) for s, _, d, p in pool[i] if s in first[pdb]]
        if r:
            free_ip[i] = max(r)[0]

    # EVERY interface appears in the table. Where free top-ipTM >= --iptm-max the method is not
    # applied, so no extra seeds are run and the "free 50 + extra" cell is simply the free 50 cell --
    # those interfaces are never "pending", they are complete by construction. Only interfaces BELOW
    # the threshold need extras, and only they can be pending.
    rows, pending, skipped = [], collections.Counter(), collections.Counter()
    for iface in sorted(pool):
        k = cls.get(iface)
        if k not in ("decoy", "solved", "nocov") or iface not in free_ip:
            continue
        base = [(ip, dq) for s, _, dq, ip in pool[iface] if s in first[iface.split("_")[0]]]
        if not base:
            continue
        if free_ip[iface] >= A.iptm_max:
            skipped[k] += 1
            rows.append((k, free_ip[iface], base, base, iface.split("_")[0], iface))      # no extra compute spent here
            continue
        want = min(len(per.get(iface.split("_")[0], ())), A.n_extra) or A.n_extra
        seeds = sorted(mc.get(iface, {}))
        if len(seeds) < want:
            pending[k] += 1
            continue
        add = [v for s in seeds[:want] for v in mc[iface][s]]
        rows.append((k, free_ip[iface], base, base + add, iface.split("_")[0], iface))

    # ---- ONE table, on interfaces where BOTH arms have finished -----------------------------------
    # The comparison is only honest where the same interface carries all three columns, so the table
    # is the INTERSECTION: the matched baseline has its full extra-seed count AND the method's folds
    # are inside the block and fully scored. An interface missing either is pending, never averaged in
    # -- a half-finished method arm against a complete baseline would read as the method losing.
    mrows, mok = method_rows(METHOD_ARMS, *BLOCK)
    # An interface above the threshold needs NOTHING from either arm -- all three columns are its
    # free 50 -- so it is complete on arrival and must not wait on the method arm's queue. Only
    # sub-threshold interfaces require both arms to have finished.
    both = [r for r in rows if r[1] >= A.iptm_max or mok.get(r[4])]
    pend = collections.Counter(pending)
    for k in ("decoy", "solved", "nocov"):
        pend[k] += sum(1 for r in rows
                       if r[0] == k and r[1] < A.iptm_max and not mok.get(r[4]))

    print(f"free 50 vs free 50 + extra vs free 50 + method, dataset {A.dataset}, "
          f"seeds {BLOCK[0]}-{BLOCK[1]} only")
    print(f"{len(both)} interfaces over {len({r[4] for r in both})} complexes where BOTH the matched "
          f"baseline and the method have finished.")
    # BOTH ARMS SKIP AN INTERFACE ALREADY ABOVE --iptm-max, so every row is compute-matched.
    # The decision is PER INTERFACE, not per complex: an interface whose free top-ipTM clears the
    # threshold is left alone even when a sibling interface of the same complex is below it.
    #
    # The FoldBench method arm was queued before this rule and physically folded everywhere, so its
    # poses exist above the threshold -- they are DISCARDED here rather than used. That is not a
    # handicap, it is the deployment: 3 of the method's 4 solved-interface breakages sat above 0.8
    # (8fdd_AC 0.579->0.221, 8fdd_BC 0.799->0.216, 8fxv_AB 0.251->0.059), against 29 small gains
    # there averaging +0.047, and dropping both groups nets BETTER on acceptable count and on mean.
    # It also means every row spends alike, which the earlier "method folds above 0.8, baseline does
    # not" version never did.
    print(f"Both arms skip an interface whose free top-ipTM is >= {A.iptm_max} (per interface, not "
          f"per complex),\nso all three columns are identical there and every row is "
          f"compute-matched.\n")
    # No method-alone column: the method is never deployed on its own -- its poses join the free pool
    # and ONE top-ipTM selection is made over the union.
    picks = [(C.top, "top-ipTM")] + ([(C.oracle, "oracle")] if A.oracle else [])
    for pick, name in picks:
        print(f"=== {name} ===")
        mo = " method only |" if A.method_only else ""
        mos = "---:|" if A.method_only else ""
        print(f"| class | interfaces (n) | free 50 | free 50 + extra |{mo} free 50 + method | delta |")
        print(f"|:---|---:|---:|---:|{mos}---:|---:|")
        for k in ("decoy", "solved", "nocov", "ALL"):
            sub = [r for r in both if k == "ALL" or r[0] == k]
            if not sub:
                # printed rather than skipped: a silently absent class reads as "no effect there"
                print(f"| {k} | 0 | -- | -- |" + (" -- |" if A.method_only else "") + " -- | -- |")
                continue
            cols, means = [], []
            # method-alone falls back to the free pool where the method is not applied (>= iptm_max)
            # and where no templated pose exists -- otherwise the cell would be an empty selection.
            gs = [lambda r: r[2], lambda r: r[3]]
            if A.method_only:
                gs.append(lambda r: (mrows.get(r[5], []) if r[1] < A.iptm_max else []) or r[2])
            gs.append(lambda r: r[2] + (mrows.get(r[5], []) if r[1] < A.iptm_max else []))
            for g in gs:
                v = [pick(g(r)) for r in sub]
                means.append(st.mean(v))
                cols.append(f"{sum(1 for x in v if x >= ACCEPTABLE)}/{len(v)} ({st.mean(v):.3f})")
            # delta is METHOD against the compute-matched baseline, not against plain free 50: the
            # baseline is what the method has to beat at equal compute.
            cols.append(f"{means[-1] - means[1]:+.3f}")
            print(f"| {k} | {len(sub)} | " + " | ".join(cols) + " |")
        print()
    tot_sk = sum(1 for r in both if r[1] >= A.iptm_max)
    print(f"of those {len(both)}, {tot_sk} are above {A.iptm_max}: neither arm spends, all three "
          f"columns equal")
    if sum(pend.values()):
        print("STILL AWAITING FOLDS (one or both arms unfinished): "
              + ", ".join(f"{v} {k}" for k, v in sorted(pend.items()) if v))
    lo = [r for r in both if r[1] < A.iptm_max]
    if lo:
        print(f"\non the {len(lo)} interfaces below {A.iptm_max} (where both arms actually spend):")
        for pick, name in picks:
            f = [pick(r[2]) for r in lo]
            m = [pick(r[3]) for r in lo]
            t = [pick(r[2] + mrows.get(r[5], [])) for r in lo]   # all of `lo` is below the threshold
            print(f"  {name:<9} free {st.mean(f):.3f} | + extra {st.mean(m):.3f} "
                  f"({st.mean(m) - st.mean(f):+.3f}) | + method {st.mean(t):.3f} "
                  f"({st.mean(t) - st.mean(f):+.3f})")


if __name__ == "__main__":
    main()
