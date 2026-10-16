"""THE VALIDATION METRIC for the confidence-head finetune, and the baseline to beat.

WHAT IS OPTIMISED, decided 2026-10-15: of the poses a free seed budget produces, pick the one with
the highest PER-INTERFACE pair_iptm, then report

    mean_dockq  -- the mean over interfaces of the SELECTED pose's DockQ
    frac_acc    -- the fraction of interfaces whose SELECTED pose has DockQ >= 0.23 (CAPRI acceptable)

This needs no FT3 validation labels. It is computed from the scores the fold already writes, which is
why validation label-building is NOT on the critical path -- labels would only be needed for a
validation LOSS curve, and checkpoints are selected on this metric instead.

pair_iptm, NEVER GLOBAL ipTM. The `iptm` column of results.csv is per-interface, verified rather than
assumed: of 19,000 (pdb, seed, sample) groups that span more than one interface, 18,964 carry
DIFFERENT values per interface.

THE ORACLE AND RANDOM COLUMNS ARE THE POINT OF THE TABLE. `oracle` is argmax DockQ over the same
pool -- the best the sampler produced, whether or not confidence found it -- so oracle minus selected
is the headroom a better ranker can win at ZERO extra compute. `random` is the mean over all poses,
which is what selection beats. Measured 2026-10-16 at N=100 over 92 interfaces: selected 0.5779,
oracle 0.7171, random ~0.4975. So selection is worth ~0.08 and leaves ~0.14 on the table.

THESE NUMBERS MOVED ON 2026-10-16 and the old ones are not comparable. Two different baselines had
been quoted in the tree, 0.5776 and 0.5783, from runs either side of an earlier fix; both predate the
score-row repair, which replaced 2,147 of 46,000 rows where validation/fold held the refold's trunks
beside the ORIGINAL fold's DockQ for 7tzh, 8v8f and 9cn2 (see docs/pose_provenance.md). Only 6 of
the 92 interfaces could move, and the effect is small -- 0.5783 -> 0.5779 -- but one number now
appears everywhere rather than two, and it is the one this file prints.

ipTM SELECTION PEAKS AROUND 25 SEEDS AND THEN DECAYS -- 0.5855 at 25, 0.5797 at 50, 0.5779 at 100 --
while the oracle keeps climbing, 0.6888 -> 0.7067 -> 0.7171. More sampling produces better poses AND
a greater chance that confidence prefers a confidently wrong one. That is the confident-decoy effect
measured on this validation set, and it is the thing the finetune is meant to fix.

EVERY POOL INTERFACE MUST BE PRESENT, which is why --pool is required rather than optional. 7shy_AH
was absent from results.csv entirely -- OST drops an interface when either target chain is missing
from its chain mapping (ost/mol/alg/scoring.py:1121) -- so the metric was quietly computed over 91 of
92 interfaces. It was a well-predicted interface, DockQ 0.766-0.898, so its absence made the baseline
PESSIMISTIC; the error could as easily have gone the other way. A missing interface is now fatal.
Recover one with phase_j_fill_scores.py --targets <pdb>:<IFACE>.
"""
import argparse
import collections
import csv
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCEPTABLE = 0.23          # CAPRI 'acceptable'; not a tunable of this project


def pool_interfaces(path):
    """{iface_id} from a pool csv of (pdb, chain1, chain2), in both chain orders."""
    out = set()
    with open(path) as fh:
        for r in csv.DictReader(fh):
            p, c1, c2 = r["pdb"], r["chain1"], r["chain2"]
            out.add((p, frozenset((c1, c2))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{ROOT}/validation/fold/results.csv")
    ap.add_argument("--pool", default=f"{ROOT}/validation/fold/pool.csv",
                    help="REQUIRED coverage contract: every interface here must appear in --results, "
                         "or the metric is being computed over a subset. See the docstring on 7shy.")
    ap.add_argument("--seed-base", type=int, default=400000)
    ap.add_argument("--budgets", default="1,5,10,25,50,100",
                    help="seed budgets N; selection is over the first N seeds x all samples")
    ap.add_argument("--arm", default="free")
    ap.add_argument("--allow-missing", default="",
                    help="comma-separated iface ids permitted to be absent, each a deliberate and "
                         "documented exclusion. Empty means none are.")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.results)))
    byif = collections.defaultdict(list)
    bad = 0
    for r in rows:
        if a.arm and r.get("arm") != a.arm:
            continue
        try:
            byif[r["iface"]].append((int(r["seed"]), float(r["iptm"]), float(r["dockq"])))
        except (TypeError, ValueError):
            bad += 1
    if bad:
        print(f"NOTE: {bad} row(s) had an unparseable iptm or dockq and were dropped")

    # COVERAGE FIRST. A metric over an unknown subset is worse than no metric.
    want = pool_interfaces(a.pool)
    have = {(i.split("_")[0], frozenset(i.split("_", 1)[1])) for i in byif}
    allow = {x.strip() for x in a.allow_missing.split(",") if x.strip()}
    missing = sorted(f"{p}_{''.join(sorted(cs))}" for p, cs in (want - have))
    missing = [m for m in missing if m not in allow]
    print(f"coverage: {len(have & want)} of {len(want)} pool interface(s) scored in {a.results}")
    if missing:
        raise SystemExit(
            f"STOP: {len(missing)} pool interface(s) have NO scored rows: {missing}\n"
            f"  The metric would be computed over a SUBSET. Recover them with\n"
            f"  phases/phase_j_fill_scores.py --targets <pdb>:<IFACE> --seeds <range>\n"
            f"  or name them in --allow-missing with a documented reason.")

    # seed-completeness per interface, reported rather than enforced: a budget larger than what an
    # interface actually has would otherwise silently compare different budgets across interfaces
    depth = {i: len({s for s, _it, _d in v}) for i, v in byif.items()}
    print(f"seed depth: min {min(depth.values())}, max {max(depth.values())} over "
          f"{len(depth)} interface(s)")

    print()
    print(f"{'N':>5} {'n':>4} {'mean_dockq':>11} {'frac_acc':>9} "
          f"{'oracle_mean':>12} {'oracle_acc':>11} {'random_mean':>12} {'headroom':>9}")
    for N in [int(x) for x in a.budgets.split(",")]:
        sel, orc, rnd, acc, oacc, short = [], [], [], 0, 0, 0
        for ifc, v in byif.items():
            sub = [x for x in v if x[0] < a.seed_base + N]
            if not sub:
                continue
            if depth[ifc] < N:
                short += 1
            best = max(sub, key=lambda x: x[1])
            sel.append(best[2]); acc += best[2] >= ACCEPTABLE
            om = max(x[2] for x in sub); orc.append(om); oacc += om >= ACCEPTABLE
            rnd.append(statistics.mean(x[2] for x in sub))
        n = len(sel)
        if not n:
            continue
        ms, mo = statistics.mean(sel), statistics.mean(orc)
        print(f"{N:>5} {n:>4} {ms:>11.4f} {acc/n:>9.4f} {mo:>12.4f} {oacc/n:>11.4f} "
              f"{statistics.mean(rnd):>12.4f} {mo-ms:>9.4f}"
              + (f"   ({short} iface(s) shallower than N)" if short else ""))
    print()
    print(f"selection = argmax pair_iptm over the first N seeds x all samples, per interface.")
    print(f"headroom  = oracle_mean - mean_dockq: what a better ranker wins at zero extra compute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
