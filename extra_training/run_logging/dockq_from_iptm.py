#!/usr/bin/env python
"""Rebuild the DockQ selection metric from emitted ipTM rows, and check it against the eval.

    python -I run_logging/dockq_from_iptm.py <iptm_rows.csv> --results validation/fold/results.csv
                                             [--compare training/runs/<rid>/train_eval_shards_epN]

WHY THIS EXISTS. The held-out PAE pass and the DockQ eval forward the SAME poses through the SAME
head every epoch -- verified identical as sets of (pdb, seed, sample) on j583i8 -- and ipTM comes
from the PAE logits the loss pass already holds. So one of the two forwards is redundant. This is
the arithmetic half of removing it, kept SEPARATE from the run so the merge can be judged against
the eval's own output before anything is deleted.

THE ARITHMETIC IS PROVEN, THE JOIN IS NOT. Recomputing the metric from eval rows reproduces the
reported figure exactly (0.586783 / 0.750000 against a recorded 0.5868 / 0.7500 on j583i8 epoch 1).
What is NOT proven in general is the join: phase_k_eval maps poses to DockQ by COORDINATE IDENTITY
against the _sample_<k>.cif files, because sample ordering is not guaranteed, and this module uses
a naive (iface, seed, sample) key. Measured on j583i8 epoch 1 the two agree on all 13,800 rows --
one pool, one day. --compare is how that is re-established rather than assumed, and it is why
nothing here removes the eval.
"""
import argparse, csv, glob, os
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("rows")
ap.add_argument("--results", required=True)
ap.add_argument("--compare", default="", help="an eval shard dir to check this against")
ap.add_argument("--thresh", type=float, default=0.23)
ap.add_argument("--budget", type=int, default=30,
                help="seeds per interface, as phase_k_eval means it. Rows outside "
                     "[--seed-base, --seed-base + budget) are DROPPED rather than silently "
                     "widening the population: 70 extra seeds move mean_dockq by 0.0052 and "
                     "frac_acc by 0.011 on this pool, the same order as the signal being read.")
ap.add_argument("--seed-base", type=int, default=400000)
ap.add_argument("--iptm-tol", type=float, default=1e-6,
                help="the largest per-pose |d ipTM| against the eval's rows that still counts as "
                     "the same forward. NOT ZERO, and the reason matters: 'any difference is the "
                     "join' is too strong, because two code paths can select different cuDNN or "
                     "TF32 kernels and an uncropped pass presents different batch shapes, so a "
                     "perfectly correct join can still differ in the last bits. 1e-6 is far below "
                     "the ipTM gaps that decide a selection -- within-seed std is ~0.006 and the "
                     "top1-top2 gap ~0.002 -- and far above float32 round-off, so it separates "
                     "numerical noise from a row lined up against the wrong pose. Raise it only "
                     "with a measurement, never to make a comparison pass.")
a = ap.parse_args()

dq = {}
with open(a.results) as fh:
    for r in csv.DictReader(fh):
        dq[(r["iface"], r["seed"], r["sample"])] = float(r["dockq"])

by, missing, out_of_range, dup = defaultdict(list), 0, 0, 0
seen = set()
with open(a.rows) as fh:
    for r in csv.DictReader(fh):
        if not (a.seed_base <= int(r["seed"]) < a.seed_base + a.budget):
            out_of_range += 1
            continue
        k = (r["iface"], r["seed"], r["sample"])
        if k not in dq:
            missing += 1
            continue
        if k in seen:
            dup += 1
        seen.add(k)
        by[r["iface"]].append((float(r["iptm"]), dq[k], int(r["seed"]), int(r["sample"])))
if not by:
    raise SystemExit(f"STOP: no rows of {a.rows} joined to {a.results}")
# THE EVAL'S OWN TIE-BREAK, (iptm, -seed, -sample). phase_k_eval adopted it because the
# row-order version "selected different DockQ depending on how the work was split"; picking by
# file order here reintroduces exactly that. It matters more in emitted rows than in the eval's,
# because the emitter writes ipTM at 6 dp and ties are therefore MORE likely: 1 of 92 interfaces
# on j583i8 epoch 1 ties at that precision with two different DockQ values behind it.
sel = [max(v, key=lambda t: (t[0], -t[2], -t[3]))[1] for v in by.values()]
n = len(by)
mean = sum(sel) / n
frac = sum(1 for d in sel if d >= a.thresh) / n
# PHASE_K_EVAL REFUSES ALL OF THESE; a rebuilt metric that does not is a metric over
# "whatever showed up". The emitter flushes per unit by design, so a pass that dies at 60% leaves
# a file this would otherwise summarise as if it were complete.
if dup:
    raise SystemExit(f"STOP: {dup} duplicate (iface, seed, sample) row(s) in {a.rows}. The "
                     f"emitter truncates per process, so this is what a sharded or multi-rank "
                     f"writer clobbering one file looks like.")
if missing:
    raise SystemExit(f"STOP: {missing} row(s) of {a.rows} have no DockQ in {a.results}. A metric "
                     f"over the rows that happened to join is not the metric.")
depths = {len(v) for v in by.values()}
if len(depths) != 1:
    raise SystemExit(f"STOP: interfaces have unequal pose depth {sorted(depths)}; the population "
                     f"is not the cross product phase_k_eval asserts.")
print(f"from ipTM rows: mean_dockq {mean:.6f}  frac_acc {frac:.6f}  ({n} interfaces)"
      + f"  [depth {depths.pop()}/iface"
      + (f", {out_of_range} row(s) outside the seed budget dropped" if out_of_range else "")
      + "]")

if a.compare:
    rows = []
    for f in glob.glob(os.path.join(a.compare, f"rows_b{a.budget}_*.csv")):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"STOP: no rows_b{a.budget}_*.csv under {a.compare}")
    eb = defaultdict(list)
    for r in rows:
        eb[r["iface"]].append((float(r["iptm"]), float(r["dockq"]), int(r["seed"]),
                               int(r["sample"])))
    es = [max(v, key=lambda t: (t[0], -t[2], -t[3]))[1] for v in eb.values()]
    em, ef = sum(es) / len(eb), sum(1 for d in es if d >= a.thresh) / len(eb)
    print(f"from the eval : mean_dockq {em:.6f}  frac_acc {ef:.6f}  ({len(eb)} interfaces)")
    # PER-POSE, NOT JUST THE AGGREGATE. Two different ipTM vectors can select the same pose on
    # every interface and still disagree about the statistic; a metric that matches by luck is the
    # failure this is meant to catch.
    ev = {(r["iface"], r["seed"], r["sample"]): float(r["iptm"]) for r in rows}
    mine = {}
    with open(a.rows) as fh:
        for r in csv.DictReader(fh):
            mine[(r["iface"], r["seed"], r["sample"])] = float(r["iptm"])
    both = set(ev) & set(mine)
    worst = max((abs(ev[k] - mine[k]) for k in both), default=float("nan"))
    print(f"  pose overlap {len(both):,} of {len(ev):,} eval / {len(mine):,} emitted; "
          f"worst |d ipTM| {worst:.6f}")
    # THE VERDICT IS THE EXIT STATUS, not a printed boolean. This module printed its findings and
    # returned 0 either way, and verify_iptm_join.sh ended on an echo, so a FAILED comparison was
    # indistinguishable from a verified one to any automated caller. Review blocker.
    _ifaces_ok = set(by) == set(eb)
    _metric_ok = abs(mean - em) < 1e-6 and abs(frac - ef) < 1e-9
    _iptm_ok = bool(both) and worst == worst and worst <= a.iptm_tol
    # PER-POSE SELECTION, not just the aggregate: two different ipTM vectors can agree on the
    # statistic while selecting different poses on offsetting interfaces.
    #
    # THE IDENTITY OF THE SELECTED POSE, NOT ITS DockQ. The first version of this check compared
    # element [1] of the tuple, which is DockQ -- so two DIFFERENT poses that happen to share a
    # DockQ value counted as agreement, and a near-tie whose ordering the two paths reverse would
    # have passed while the selector itself disagreed. The whole point of the merge is to
    # reproduce the SELECTOR, not its aggregate performance. Caught in review of 8bcf8202.
    # Tuples are (iptm, dockq, seed, sample), so [2:4] is the pose key and [1] is DockQ.
    _tb = lambda t: (t[0], -t[2], -t[3])
    _mine_arg = {k: max(v, key=_tb) for k, v in by.items()}
    _eval_arg = {k: max(v, key=_tb) for k, v in eb.items()}
    _both_if = set(_mine_arg) & set(_eval_arg)
    _sel_diff = [k for k in _both_if
                 if _mine_arg[k][2:4] != _eval_arg[k][2:4]]
    # DockQ equality stays as its OWN check: the same pose must also carry the same label, which
    # is what catches a correct selection joined to the wrong row of results.csv.
    _dq_diff = [k for k in _both_if
                if abs(_mine_arg[k][1] - _eval_arg[k][1]) > 1e-12]
    _cov_ok = set(ev) == set(mine)
    print(f"  interfaces match: {_ifaces_ok}   metric match: {_metric_ok}")
    print(f"  pose-key sets identical: {_cov_ok}   "
          f"interfaces selecting a different pose: {len(_sel_diff)}   "
          f"same pose but different DockQ: {len(_dq_diff)}")
    _fail = []
    if not _cov_ok:
        _fail.append(f"pose-key sets differ: {len(set(ev) - set(mine))} only in the eval, "
                     f"{len(set(mine) - set(ev))} only in the emitted rows")
    if not _ifaces_ok:
        _fail.append(f"interface sets differ ({len(set(by))} vs {len(set(eb))})")
    if not _iptm_ok:
        _fail.append(f"worst |d ipTM| {worst:.3e} exceeds --iptm-tol {a.iptm_tol:.3e}")
    if _sel_diff:
        _fail.append(f"{len(_sel_diff)} interface(s) select a different (seed, sample), e.g. "
                     + ", ".join(f"{k}: mine {_mine_arg[k][2:4]} vs eval {_eval_arg[k][2:4]}"
                                 for k in _sel_diff[:3]))
    if _dq_diff:
        _fail.append(f"{len(_dq_diff)} interface(s) select the SAME pose with a different DockQ, "
                     f"e.g. " + ", ".join(f"{k}: {_mine_arg[k][1]:.4f} vs {_eval_arg[k][1]:.4f}"
                                          for k in _dq_diff[:3])
                     + " -- that is the (iface, seed, sample) key joining to the wrong "
                       "results.csv row")
    if not _metric_ok:
        _fail.append(f"aggregate differs: {mean:.6f}/{frac:.6f} vs {em:.6f}/{ef:.6f}")
    if _fail:
        print("\nJOIN NOT VERIFIED:")
        for _f in _fail:
            print("  - " + _f)
        print("The separate DockQ eval must stay. A rebuilt metric that disagrees with the eval "
              "on identical inputs is lining rows up against the wrong poses.")
        raise SystemExit(1)
    print("\nJOIN VERIFIED: identical pose keys, ipTM within tolerance, the same (seed, sample) "
          "selected on every interface carrying the same DockQ, same aggregate.")
