#!/usr/bin/env python
"""When extra FREE seeds rescue a decoy, is it because oracle DockQ improved?

SUMMARY
    A decoy is an interface where, over the first 50 free seeds, an acceptable pose (DockQ >= 0.23)
    exists but the top-pair_iptm pose is not one. Some complexes have been folded for more free seeds
    than that. Pool every free pose available for the interface, re-run the SAME selector (argmax
    pair_iptm, never an oracle), and ask which decoys become correct -- "saved".

    Then the question this file exists for: for a saved decoy, did the pool's ORACLE DockQ go up?

      yes  -> the rescue came with better poses, consistent with "ipTM recognises exceptional poses"
      no   -> the same poses were there all along and the argmax simply landed differently, which is
              a lottery win and says nothing about recognition

    The NOT-saved decoys are reported beside them as the control. Without that column a big oracle
    gain among the saved is uninterpretable: extra seeds raise oracle DockQ for everyone, so the
    comparison that matters is saved vs not-saved, not saved vs its own baseline.

FREE ONLY, AND NO DOUBLE COUNTING
    Sources are the dataset's pool csv and, beside the free pose tree, results.csv rows whose `arm`
    is exactly "free". pxm2025's results.csv also carries free100 (seeds 300050-300129) and the
    bf16/fp32 precision replicates; free100's seed range OVERLAPS both the pool csv's extra seeds and
    the free block, so the same fold can appear under two labels. Those arms are dropped and counted
    in the stdout header rather than silently merged. Everything kept is deduplicated on
    (iface, seed, sample).

    Both sources report PAIR ipTM for the row's own interface -- the pool csv as `pair_iptm`, the
    results.csv as `iptm` (scorer.py:87, `ip = v.get("pair_iptm")`). They are the same quantity.

INPUTS
    --dataset:  pxm2025 or foldbench, repeatable. Default: both.
    --seeds:    how many seeds define the baseline and the classes. Default 50, as every reporter.
    --out:      per-interface csv. Optional.

OUTPUTS
    stdout: the saved decoys one per row with their oracle change, then saved vs not-saved summary
    --out:  dataset, iface, n_base_poses, n_extra_poses, sel_base, sel_ext, oracle_base, oracle_ext,
            d_oracle, saved

EXAMPLE
    analysis/extra_seeds_rescue.py --out ../experimental_results/extra_seeds_rescue.csv

FUNCTIONS
    free_rows():  a dataset -> {iface: {(seed, sample): (pair_iptm, dockq)}} split into base and extra
    main():       per dataset, classify, pool, tabulate
"""
import argparse
import collections
import csv
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import paths  # noqa: E402
from score_experiment import ACCEPTABLE, classes_at, pool_by_seed  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))


def free_rows(dataset, n_seeds):
    """out: (base, extra, dropped_arms)

    base  {iface: {(seed, sample): (pair_iptm, dockq)}} the first `n_seeds` seeds -- the baseline the
          classes are defined on
    extra the same shape, every OTHER free pose found, deduplicated against base and against itself
    """
    ds = paths.DATASETS[dataset]
    pool, first = pool_by_seed(n_seeds, str(ds.pool_csv))
    base = collections.defaultdict(dict)
    extra = collections.defaultdict(dict)

    # the pool csv itself: its first `n_seeds` seeds are the baseline, anything beyond is extra
    for r in csv.DictReader(open(str(ds.pool_csv))):
        if not r.get("dockq") or not r.get("pair_iptm"):
            continue
        iface = f"{r['pdb']}_{r['chain1']}{r['chain2']}"
        key = (r["seed"], r["sample"])
        v = (float(r["pair_iptm"]), float(r["dockq"]))
        (base if r["seed"] in first.get(r["pdb"], ()) else extra)[iface][key] = v

    # the free pose tree's own results.csv, free arm only
    dropped = collections.Counter()
    rc = f"{ds.free}/results.csv"
    if os.path.exists(rc):
        for r in csv.DictReader(open(rc)):
            if r.get("arm") != "free":
                dropped[r.get("arm")] += 1
                continue
            if not r.get("dockq") or not r.get("iptm"):
                continue
            iface, key = r["iface"], (r["seed"], r["sample"])
            if key in base.get(iface, {}) or key in extra.get(iface, {}):
                continue                                   # same fold already counted
            extra[iface][key] = (float(r["iptm"]), float(r["dockq"]))
    return base, extra, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", choices=paths.DATASET_NAMES, default=None)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    datasets = A.dataset or list(paths.DATASET_NAMES)

    allrows = []
    for dataset in datasets:
        ds = paths.DATASETS[dataset]
        pool, first = pool_by_seed(A.seeds, str(ds.pool_csv))
        cls = classes_at(pool, first)
        base, extra, dropped = free_rows(dataset, A.seeds)

        print(f"\n=== {dataset} ===")
        if dropped:
            print("  arms dropped (not plain free): "
                  + ", ".join(f"{k} {v} rows" for k, v in sorted(dropped.items())))

        rows = []
        for iface, c in sorted(cls.items()):
            if c != "decoy" or iface not in base:
                continue
            b = list(base[iface].values())
            e = list(extra.get(iface, {}).values())
            if not e:
                continue                                   # no extra free seeds for this interface
            sel_b = max(b, key=lambda x: x[0])[1]
            sel_e = max(b + e, key=lambda x: x[0])[1]
            orc_b, orc_e = max(x[1] for x in b), max(x[1] for x in b + e)
            saved = sel_e >= ACCEPTABLE > sel_b
            rows.append((iface, len(b), len(e), sel_b, sel_e, orc_b, orc_e, orc_e - orc_b, saved))
            allrows.append((dataset,) + rows[-1])

        if not rows:
            print("  no decoy interface has extra free seeds")
            continue
        sv = [r for r in rows if r[8]]
        ns = [r for r in rows if not r[8]]
        print(f"  {len(rows)} decoy interfaces with extra free seeds; {len(sv)} saved, {len(ns)} not")
        if sv:
            print(f"\n  SAVED decoys\n  {'interface':14}{'base poses':>11}{'extra':>7}"
                  f"{'sel base':>10}{'sel ext':>9}{'oracle base':>13}{'oracle ext':>12}{'d oracle':>10}")
            for r in sorted(sv, key=lambda r: -r[7]):
                print(f"  {r[0]:14}{r[1]:>11}{r[2]:>7}{r[3]:>10.3f}{r[4]:>9.3f}"
                      f"{r[5]:>13.3f}{r[6]:>12.3f}{r[7]:>+10.3f}")
        print(f"\n  mean change in oracle DockQ from the extra free seeds")
        for lbl, g in (("saved", sv), ("not saved", ns)):
            if not g:
                continue
            d = [r[7] for r in g]
            print(f"    {lbl:10} n={len(g):3}  mean {st.mean(d):+.3f}  median {st.median(d):+.3f}  "
                  f"oracle unchanged on {sum(1 for x in d if x < 1e-9)}/{len(g)}")

    if A.out:
        os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
        with open(A.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "iface", "n_base_poses", "n_extra_poses", "sel_base", "sel_ext",
                        "oracle_base", "oracle_ext", "d_oracle", "saved"])
            w.writerows(allrows)
        print(f"\n  wrote {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
