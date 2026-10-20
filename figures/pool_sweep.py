#!/usr/bin/env python
"""One pass over a benchmark's poses, producing per-interface curves and cluster-aware error bars.

WHY ONE MODULE. Both figures -- the thresholded acceptable-fraction one and the continuous mean
DockQ one -- need the same pool definition, the same selection rule and the same bootstrap. Two
implementations of that drift, and they already did once when one validated a metric json's budget
and the other did not.

TWO SOURCES OF VARIABILITY, KEPT APART. They answer different questions and conflating them is how
an error bar ends up meaning nothing in particular:

  SEED SUBSAMPLING -- "if I had drawn a different N seeds for these same interfaces, how much would
  the number move?" Measured as the spread across bootstrap draws WITHIN an interface, then
  averaged. It shrinks as N approaches the pool depth and vanishes at N = 100, because then the
  subset IS the pool. It is a property of the budget, not of the benchmark.

  TEST-SET UNCERTAINTY -- "if I had a different sample of complexes from the same distribution,
  how much would the number move?" This is the one that belongs on a plot as an error bar, and it
  is what a binomial SEM over interfaces gets wrong.

WHY A BINOMIAL SEM OVER INTERFACES IS OPTIMISTIC. It assumes interfaces are independent draws.
They are not: 39 of SAbDab2's 53 complexes, 52 of FoldBench's 104 and 73 of PXM's 123 contribute
more than one interface, and PXM has complexes contributing ten. Interfaces of one complex share a
structure, a fold difficulty and an MSA, so they fail together. Resampling COMPLEXES with
replacement and recomputing the statistic prices that in.

NO HOMOLOGY CLUSTERS AVAILABLE. None of the three registries carries a cluster column, so the
resampling unit is the complex. That is a LOWER BOUND on the true correlation -- homologous
complexes under different PDB ids are still correlated and this does not capture it -- so the
bands below are conservative relative to a binomial SEM and still optimistic relative to truth.
"""
import collections
import csv
import random

ACCEPTABLE = 0.23


def sweep(path, ic, sc, smc, dc, qc, max_budget, draws, rng_seed=0, pdb_col=None):
    """Per-interface curves for all four quantities, plus the seed-subsampling spread.

    Returns a dict with, per interface: the complex it belongs to, and for each budget the
    fraction of draws whose ipTM pick was acceptable, the same for the oracle, and the mean
    selected and oracle DockQ.
    """
    by = collections.defaultdict(list)
    pdb_of = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                sd, sm = int(r[sc]), int(r[smc])
                dq, ip = float(r[dc]), float(r[qc])
            except (ValueError, KeyError, TypeError):
                continue
            ifc = r[ic]
            by[ifc].append((sd, sm, ip, dq))
            # THE COMPLEX, from its own column where there is one. Falling back to the interface
            # name's prefix is right for every id in these three benchmarks ("7qu2_AC" -> "7qu2")
            # but it is a convention, so a real column wins.
            pdb_of[ifc] = r[pdb_col] if (pdb_col and pdb_col in r) else ifc.split("_")[0]
    kept = {i: v for i, v in by.items() if len({t[0] for t in v}) >= max_budget}

    out = {"ifaces": [], "pdb": [], "sel_acc": [], "orc_acc": [], "sel_dq": [], "orc_dq": [],
           "seed_sd_acc": [], "seed_sd_dq": [], "budgets": list(range(1, max_budget + 1))}
    rng = random.Random(rng_seed)
    for ifc, rows in kept.items():
        byseed = collections.defaultdict(list)
        for t in rows:
            byseed[t[0]].append(t)
        seeds = sorted(byseed)
        sel_hits = [0.0] * max_budget
        orc_hits = [0.0] * max_budget
        sel_sum = [0.0] * max_budget
        orc_sum = [0.0] * max_budget
        sel_sq = [0.0] * max_budget
        selhit_sq = [0.0] * max_budget
        for _ in range(draws):
            order = seeds[:]
            rng.shuffle(order)
            best_dq = -1.0
            sel = None
            for n in range(1, max_budget + 1):
                for sd, sm, ip, dq in byseed[order[n - 1]]:
                    if dq > best_dq:
                        best_dq = dq
                    cand = (ip, -sd, -sm, dq)
                    if sel is None or cand[:3] > sel[:3]:
                        sel = cand
                j = n - 1
                hit = 1.0 if sel[3] >= ACCEPTABLE else 0.0
                sel_hits[j] += hit
                selhit_sq[j] += hit
                orc_hits[j] += 1.0 if best_dq >= ACCEPTABLE else 0.0
                sel_sum[j] += sel[3]
                sel_sq[j] += sel[3] * sel[3]
                orc_sum[j] += best_dq
        out["ifaces"].append(ifc)
        out["pdb"].append(pdb_of[ifc])
        out["sel_acc"].append([h / draws for h in sel_hits])
        out["orc_acc"].append([h / draws for h in orc_hits])
        out["sel_dq"].append([v / draws for v in sel_sum])
        out["orc_dq"].append([v / draws for v in orc_sum])
        # SPREAD ACROSS DRAWS, within this interface -- the seed-subsampling term.
        out["seed_sd_acc"].append([max(0.0, (selhit_sq[j] / draws) - (sel_hits[j] / draws) ** 2) ** 0.5
                                   for j in range(max_budget)])
        out["seed_sd_dq"].append([max(0.0, (sel_sq[j] / draws) - (sel_sum[j] / draws) ** 2) ** 0.5
                                  for j in range(max_budget)])
    return out


def add_gap(sw):
    """Per-interface selection gap, so the GAP gets its own bootstrap rather than a hand-made
    combination of two curves' error bars. The gap is the claim; it should carry the interval."""
    sw["gap_acc"] = [[o - s_ for o, s_ in zip(orow, srow)]
                     for orow, srow in zip(sw["orc_acc"], sw["sel_acc"])]
    sw["gap_dq"] = [[o - s_ for o, s_ in zip(orow, srow)]
                    for orow, srow in zip(sw["orc_dq"], sw["sel_dq"])]
    return sw


def mean_curve(sw, key):
    rows = sw[key]
    n = len(rows)
    return [sum(r[j] for r in rows) / n for j in range(len(sw["budgets"]))]


def cluster_sem(sw, key, n_boot=400, rng_seed=1):
    """SEM of the mean, resampling COMPLEXES with replacement.

    The statistic is recomputed on each replicate from whichever interfaces the resampled
    complexes carry, so a complex contributing ten interfaces moves the replicate ten times -- the
    correlation a per-interface binomial SEM assumes away.
    """
    rows = sw[key]
    nb = len(sw["budgets"])
    groups = collections.defaultdict(list)
    for i, pdb in enumerate(sw["pdb"]):
        groups[pdb].append(i)
    keys = list(groups)
    rng = random.Random(rng_seed)
    acc = [[] for _ in range(nb)]
    for _ in range(n_boot):
        idx = []
        for _ in range(len(keys)):
            idx.extend(groups[keys[rng.randrange(len(keys))]])
        m = len(idx)
        for j in range(nb):
            acc[j].append(sum(rows[i][j] for i in idx) / m)
    out = []
    for j in range(nb):
        vals = acc[j]
        mu = sum(vals) / len(vals)
        out.append((sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5)
    return out


def binomial_sem(sw, key):
    """The naive band, kept so the two can be compared rather than asserted to differ."""
    rows = sw[key]
    n = len(rows)
    ms = mean_curve(sw, key)
    return [(p * (1 - p) / n) ** 0.5 for p in ms]


def mean_seed_sd(sw, key):
    """Seed-subsampling spread, averaged over interfaces. NOT an error bar on the mean."""
    rows = sw[key]
    n = len(rows)
    return [sum(r[j] for r in rows) / n for j in range(len(sw["budgets"]))]
