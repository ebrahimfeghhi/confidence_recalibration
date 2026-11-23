#!/usr/bin/env python
"""EVERY ARM'S BEST HELD-OUT EPOCH, ON ONE SET, SCORED THE SAME WAY.

    rank_arms_val.py [--metric top1|best5] [--pool max|mean|either] [--md]

WHY IT EXISTS. Each arm's numbers have been read in its own session against whatever baseline was
handy, and the comparison across arms was never made on one set with one rule. This walks every
run that has valfull83 eval rows, restricts to the 139 leak-free interfaces (dropping 8z7j, the
one val structure sharing an antibody cluster with training), recomputes shipped from each run's
OWN rows via stored_iptm, and reports the paired delta with its t.

THE BASELINE IS PER-ARM AND THAT IS DELIBERATE. stored_iptm is the shipped head's own ipTM for
the same poses, so each arm is compared against shipped on exactly the poses it scored. The
recomputed baseline differs in the fourth decimal between runs (0.5882 vs 0.5886) because the
shards differ, not because there are two baselines.

BEST-EPOCH SELECTION IS A BIASED STATISTIC AND IS LABELLED AS ONE. Picking the best of 18 epochs
inflates the delta; `n_eps` is printed so a one-epoch arm is not read as equal to an 18-epoch one,
and `n_sig` counts how many of the arm's epochs clear |t| >= 1.977 -- an arm significant at 16 of
18 epochs is a different claim from one significant at 1 of 18.
"""
import argparse, collections, csv, glob, math, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from val_two_tables import load, epochs_of, ACC

RESTRICT = "training/valfull83/pool_noleak.csv"
TCRIT = 1.977


def paired_t(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return 0.0, 0.0
    m = statistics.fmean(d)
    sd = statistics.stdev(d)
    return m, (m / (sd / math.sqrt(n)) if sd > 0 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="top1", choices=("top1", "best5"))
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--min-eps", type=int, default=1)
    a = ap.parse_args()

    keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'
            for r in csv.DictReader(open(RESTRICT))}

    rids = sorted({os.path.basename(os.path.dirname(d))
                   for d in glob.glob("training/runs/*/valfull83_eval_shards_ep*")
                   if glob.glob(d + "/rows*_s*.csv")})
    rids = [r for r in rids if r not in ("latest", "archive")]

    rows = []
    for rid in rids:
        eps = {}
        for e in epochs_of(rid):
            d = load(rid, e)
            if d:
                eps[e] = d
        if not eps:
            continue
        keys = sorted(set.intersection(*[set(d) for d in eps.values()]) & keep)
        if len(keys) < 100:          # a partial pass is not comparable with a full one
            continue

        def col(d, idx):
            t1 = [max(d[k], key=lambda t: t[idx])[3] for k in keys]
            b5 = [max(sorted(d[k], key=lambda t: -t[idx])[:5], key=lambda t: t[3])[3] for k in keys]
            return (t1, b5)[0 if a.metric == "top1" else 1]

        base = col(eps[min(eps)], 0)
        best = None
        n_sig = 0
        for e in sorted(eps):
            for pname, idx in (("max", 1), ("mean", 2)):
                v = col(eps[e], idx)
                m, t = paired_t(v, base)
                if t >= TCRIT:
                    n_sig += 1
                if best is None or m > best[0]:
                    best = (m, t, e, pname, statistics.fmean(v), statistics.fmean(base),
                            100.0 * sum(1 for x in v if x >= ACC) / len(keys))
        if best and len(eps) >= a.min_eps:
            rows.append((rid, len(keys), len(eps), n_sig) + best)

    rows.sort(key=lambda r: -r[4])
    name = lambda r: (open(f"training/runs/{r}_name").read().strip()
                      if os.path.exists(f"training/runs/{r}_name") else "")
    hdr = ("rid", "arm", "n", "n_eps", "n_sig", "best ep", "pool",
           f"{a.metric}", "shipped", "delta", "t", "acc")
    if a.md:
        print(f"**BEST {a.metric.upper()} EPOCH PER ARM** -- val83 leak-free, paired vs each run's "
              f"own shipped; |t| >= {TCRIT} is p<0.05\n")
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
    for rid, n, neps, nsig, m, t, e, pool, val, ship, acc in rows:
        c = [rid, name(rid)[:34], n, neps, nsig, e, pool,
             f"{val:.4f}", f"{ship:.4f}", f"{m:+.4f}", f"{t:+.2f}", f"{acc:.1f}%"]
        print(("| " + " | ".join(str(x) for x in c) + " |") if a.md
              else "  ".join(str(x).ljust(9) for x in c))


if __name__ == "__main__":
    main()
