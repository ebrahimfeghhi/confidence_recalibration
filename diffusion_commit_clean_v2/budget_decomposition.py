#!/usr/bin/env python
"""Why does DockQ improve from 10 seeds to 50? Coverage (a new pose) or confidence (a better rank)?

SUMMARY
    Seed scaling helps. This asks what it buys. Two causes are possible and they have opposite
    implications for what to build next:

      COVERAGE    the 10-seed subset contained nothing as good as what 50 seeds selected. The extra
                  compute found structure that was genuinely not there before.
      CONFIDENCE  the 10-seed subset ALREADY contained a pose that good; pair_iptm ranked something
                  else top. The extra compute bought a confident instance of a pose already reachable.

    A pose's pair_iptm never "improves" -- a pose arrives with the seed that made it and its score is
    fixed. So CONFIDENCE means: the good basin was reachable at 10 seeds but only via seeds that scored
    it low, and by 50 seeds a seed in that same basin arrives that happens to be confident. That is the
    commitment re-roll measured on 9n6e (one seed, repeated: 0.022 -> 0.644).

    WHICH 10 SEEDS matters, so the 10-seed arm is not "the first 10". Every interface is measured over
    --draws random 10-subsets of its 50 and the attribution is averaged, so no result rests on a lucky
    subset. The 50-seed arm is the whole pool and is deterministic.

    PROXY, NOT PROOF. orc10 >= sel50 shows a pose that good existed at 10 seeds, not that it is the
    same pose in the same basin. The basin-level version needs the pose-to-pose DockQ matrices
    (paths.DATASETS[ds].matrices) and is not done here.

    THE HEADLINE is `orc10>=sel50`: the share of interfaces where a perfect selector on a 5x cheaper
    budget already matches real pair_iptm at full budget. Where that is high, scaling is paying for
    confidence rather than structure, and the ceiling of better scoring is worth more than more seeds.

    SELECTION is argmax pooled pair_iptm over every pose of the chosen seeds (5 per seed), which is the
    project's selection rule. CLASSES come from the 50-seed pool, same definition as score_experiment.
    On a decoy interface sel50 is below ACCEPTABLE by construction, so classes are reported separately
    rather than pooled.

    INTERFACES ABOVE pair_iptm 0.8 ARE SKIPPED, by the pool max, since one pool serves both arms.

INPUTS
    --dataset:  pxm2025, foldbench, or both (default both)
    --low:      the small budget in seeds. Default 10.
    --high:     the full budget in seeds. Default 50.
    --draws:    random low-budget subsets per interface. Default 200.
    --eps:      DockQ change below this is "no change". Default 0.05.
    --max-iptm: skip an interface whose pool max pair_iptm exceeds this. Default 0.8.
    --seed:     RNG seed, so the table reproduces. Default 0.
    --out:      write the markdown table here as well as stdout. Optional.
    --per-interface-csv: write the per-interface rows here. Optional.

OUTPUTS
    stdout: a table per dataset -- one row per class, plus ALL -- with n, mean sel10, mean sel50, the
            gain, how that gain splits across coverage/confidence, and the orc10>=sel50 share
    --out / --per-interface-csv: the same numbers as markdown / csv

EXAMPLE
    budget_decomposition.py --out ../experimental_results/budget_decomposition.md \\
        --per-interface-csv ../experimental_results/budget_decomposition_per_iface.csv

FUNCTIONS
    load():     a dataset -> {interface: {seed: [(dockq, pair_iptm)]}} over the canonical free seeds
    classify(): a 50-seed interface -> solved | decoy | nocov
    decompose(): one interface -> its averaged attribution over the random low-budget draws
    table():    the per-interface rows -> the per-class table
    main():     read, decompose, print, optionally write
"""
import argparse
import collections
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths            # noqa: E402
import pool_index       # noqa: E402
from score_experiment import ACCEPTABLE  # noqa: E402  -- one definition of "acceptable"


def load(ds_name):
    """in: a dataset name.  out: {interface: {seed: [(dockq, pair_iptm), ...]}}

    Restricted to Dataset.pool_seeds -- the canonical 50 FREE seeds. Any other seed in the pool csv
    (top-ups, repeats, precision arms) is dropped, so the budget axis is free seeds only.
    """
    d = paths.DATASETS[ds_name]
    want = set(d.pool_seeds)
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for iface, rows in pool_index.load_pool(str(d.pool_csv)).items():
        for seed, _sample, dockq, iptm in rows:
            if str(seed) in want:
                out[iface][str(seed)].append((dockq, iptm))
    return out


def classify(by_seed):
    """in: {seed: [(dockq, pair_iptm)]} for one interface at the full budget.  out: the class label

    Same rule as score_experiment.classes_at: the pool's most confident pose decides solved, and a
    decoy is an interface whose confident pick is wrong while some pose is acceptable.
    """
    poses = [p for v in by_seed.values() for p in v]
    best = max(poses, key=lambda p: p[1])
    if best[0] >= ACCEPTABLE:
        return "solved"
    return "decoy" if any(p[0] >= ACCEPTABLE for p in poses) else "nocov"


def decompose(by_seed, low, draws, eps, rng):
    """in: one interface's {seed: [(dockq, pair_iptm)]}, the low budget, draws, eps, an RNG.
    out: a dict of that interface's averaged attribution, or None if it has too few seeds

    Per draw: select by argmax pair_iptm at the low budget and at the full budget, then attribute the
    change. COVERAGE requires that nothing at the low budget was within eps of what the full budget
    picked; otherwise an improvement is CONFIDENCE, because the structure was already in hand.
    """
    seeds = sorted(by_seed)
    if len(seeds) < low:
        return None
    allp = [p for s in seeds for p in by_seed[s]]
    sel_hi = max(allp, key=lambda p: p[1])[0]
    orc_hi = max(p[0] for p in allp)

    sel_hi_iptm = max(allp, key=lambda p: p[1])[1]
    sel_lo, orc_lo, sel_lo_iptm, buckets, gains, beat = [], [], [], collections.Counter(), [], []
    for _ in range(draws):
        pick = rng.choice(len(seeds), size=low, replace=False)
        poses = [p for i in pick for p in by_seed[seeds[i]]]
        best_lo = max(poses, key=lambda p: p[1])
        s_lo = best_lo[0]
        o_lo = max(p[0] for p in poses)
        sel_lo.append(s_lo)
        orc_lo.append(o_lo)
        sel_lo_iptm.append(best_lo[1])
        gains.append(sel_hi - s_lo)
        beat.append(o_lo >= sel_hi - eps)
        if sel_hi - s_lo <= -eps:
            buckets["regression"] += 1
        elif sel_hi - s_lo < eps:
            buckets["no change"] += 1
        elif o_lo >= sel_hi - eps:
            buckets["confidence"] += 1
        else:
            buckets["coverage"] += 1

    gains = np.array(gains)
    pos = gains > eps
    # how the realised gain splits: DockQ actually gained under each cause, per draw
    share = {}
    for k in ("coverage", "confidence"):
        m = np.array([b == k for b in _labels(gains, orc_lo, sel_hi, eps)])
        share[k] = float(gains[m].sum() / draws) if m.any() else 0.0
    return dict(
        n_seeds=len(seeds), sel_lo=float(np.mean(sel_lo)), orc_lo=float(np.mean(orc_lo)),
        sel_hi=sel_hi, orc_hi=orc_hi, gain=float(gains.mean()),
        iptm_lo=float(np.mean(sel_lo_iptm)), iptm_hi=sel_hi_iptm,
        p_coverage=buckets["coverage"] / draws, p_confidence=buckets["confidence"] / draws,
        p_nochange=buckets["no change"] / draws, p_regression=buckets["regression"] / draws,
        gain_coverage=share["coverage"], gain_confidence=share["confidence"],
        p_improved=float(pos.mean()), p_orc_lo_beats_sel_hi=float(np.mean(beat)),
        acc_lo=float(np.mean([s >= ACCEPTABLE for s in sel_lo])),
        acc_hi=float(sel_hi >= ACCEPTABLE),
        acc_orc_lo=float(np.mean([o >= ACCEPTABLE for o in orc_lo])),
        acc_orc_hi=float(orc_hi >= ACCEPTABLE))


def _labels(gains, orc_lo, sel_hi, eps):
    """in: per-draw gains, per-draw low-budget oracles, the full-budget pick, eps.
    out: the per-draw cause label -- the same branch as decompose(), reused for the gain shares
    """
    out = []
    for g, o in zip(gains, orc_lo):
        if g <= -eps:
            out.append("regression")
        elif g < eps:
            out.append("no change")
        elif o >= sel_hi - eps:
            out.append("confidence")
        else:
            out.append("coverage")
    return out


def table(rows, low, high, classes=("decoy", "solved", "nocov")):
    """in: [(interface, class, the decompose dict)], the two budgets, the classes to report.
    out: markdown lines -- two tables

    Every row carries n. The DockQ columns are reported per class because on a decoy sel_hi is below
    ACCEPTABLE by construction, so a pooled mean would read as a method failure.

    TABLE 1 attributes the realised gain. "% cov/conf/no chg/regr" are shares of the random low-budget
    DRAWS; "gain from" columns are the DockQ actually gained under each cause, so a rare cause with a
    large jump is not hidden behind a common cause with a small one.

    TABLE 2 is the compute question: orc@lo is a perfect selector on the cheap budget, sel@hi is real
    pair_iptm on the full one. Columns are DockQ except "iptm of pick", which is the pooled pair_iptm
    of the pose selection actually returned, and the "acceptable" columns, which are shares of
    interfaces at or above ACCEPTABLE.
    """
    def block(head, cells):
        out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for cls in list(classes) + ["ALL"]:
            sub = [r for _i, c, r in rows if cls == "ALL" or c == cls]
            if sub:
                out.append("| " + " | ".join(cells(cls, sub)) + " |")
        return out

    def mk(sub):
        return lambda k: float(np.mean([s[k] for s in sub]))

    t1 = block(["class", "n ifaces", f"sel@{low}", f"sel@{high}", "gain", "% cov", "% conf",
                "% no chg", "% regr", "gain from cov", "gain from conf"],
               lambda cls, sub: _t1(cls, sub, mk(sub)))
    t2 = block(["class", "n ifaces", f"sel@{low}", f"orc@{low}", f"sel@{high}", f"orc@{high}",
                f"orc@{low} >= sel@{high}", f"iptm of pick@{low}", f"iptm of pick@{high}",
                f"acceptable sel@{low}", f"acceptable sel@{high}", f"acceptable orc@{low}"],
               lambda cls, sub: _t2(cls, sub, mk(sub)))
    return (["**Where the gain comes from** (DockQ; shares over random low-budget draws)", ""] + t1
            + ["", "**Compute equivalence** (a perfect selector on the cheap budget vs real pair_iptm "
               "on the full one)", ""] + t2)


def _t1(cls, sub, m):
    gc, gf = m("gain_coverage"), m("gain_confidence")
    tot = gc + gf
    pct = lambda v: f"{v:+.3f} ({100*v/tot:.0f}%)" if tot > 0 else f"{v:+.3f}"  # noqa: E731
    return [cls, str(len(sub)), f"{m('sel_lo'):.3f}", f"{m('sel_hi'):.3f}", f"{m('gain'):+.3f}",
            f"{100*m('p_coverage'):.0f}%", f"{100*m('p_confidence'):.0f}%",
            f"{100*m('p_nochange'):.0f}%", f"{100*m('p_regression'):.0f}%", pct(gc), pct(gf)]


def _t2(cls, sub, m):
    return [cls, str(len(sub)), f"{m('sel_lo'):.3f}", f"{m('orc_lo'):.3f}", f"{m('sel_hi'):.3f}",
            f"{m('orc_hi'):.3f}", f"{100*m('p_orc_lo_beats_sel_hi'):.0f}%",
            f"{m('iptm_lo'):.3f}", f"{m('iptm_hi'):.3f}",
            f"{100*m('acc_lo'):.0f}%", f"{100*m('acc_hi'):.0f}%", f"{100*m('acc_orc_lo'):.0f}%"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="both", choices=list(paths.DATASET_NAMES) + ["both"])
    ap.add_argument("--low", type=int, default=10)
    ap.add_argument("--high", type=int, default=50)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--max-iptm", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-interface-csv", default=None)
    a = ap.parse_args()

    names = list(paths.DATASET_NAMES) if a.dataset == "both" else [a.dataset]
    lines, per_iface = [], []
    for ds in names:
        pool = load(ds)
        rng = np.random.default_rng(a.seed)
        rows, skip_depth, skip_hi = [], 0, 0
        for iface in sorted(pool):
            by_seed = pool[iface]
            if len(by_seed) < a.high:
                skip_depth += 1
                continue
            if max(p[1] for v in by_seed.values() for p in v) > a.max_iptm:
                skip_hi += 1
                continue
            r = decompose(by_seed, a.low, a.draws, a.eps, rng)
            if r is None:
                skip_depth += 1
                continue
            cls = classify(by_seed)
            rows.append((iface, cls, r))
            per_iface.append(dict(dataset=ds, iface=iface, cls=cls, **r))
        lines += [f"### {ds}  ({a.low} -> {a.high} seeds, {a.draws} draws, eps {a.eps})",
                  f"skipped: {skip_depth} below {a.high} seeds, {skip_hi} with pool max "
                  f"pair_iptm > {a.max_iptm}; analyzed {len(rows)}", ""]
        lines += table(rows, a.low, a.high) + [""]

    text = "\n".join(lines)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        open(a.out, "w").write(text + "\n")
        print(f"wrote {a.out}")
    if a.per_interface_csv and per_iface:
        os.makedirs(os.path.dirname(os.path.abspath(a.per_interface_csv)), exist_ok=True)
        with open(a.per_interface_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_iface[0]))
            w.writeheader()
            w.writerows(per_iface)
        print(f"wrote {a.per_interface_csv} ({len(per_iface)} interfaces)")


if __name__ == "__main__":
    main()
