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


def _complex_sums(sw, key):
    """Per-complex summed curves and interface counts, so a bootstrap replicate is a sum over
    ~100 complexes instead of ~300 interfaces. Makes 2,000 replicates as cheap as 400 were."""
    rows = sw[key]
    nb = len(sw["budgets"])
    sums, counts = {}, {}
    for i, pdb in enumerate(sw["pdb"]):
        if pdb not in sums:
            sums[pdb] = [0.0] * nb
            counts[pdb] = 0
        acc = sums[pdb]
        r = rows[i]
        for j in range(nb):
            acc[j] += r[j]
        counts[pdb] += 1
    keys = list(sums)
    return keys, sums, counts, nb


def _replicates(sw, key, n_boot, rng_seed):
    """n_boot interface-weighted means, each over complexes resampled with replacement.

    THE ESTIMAND IS UNCHANGED. A replicate sums the interfaces of its sampled complexes and
    divides by how many interfaces that is, so it stays the interface-weighted mean the point
    estimate uses -- resampling changes the UNCERTAINTY model, not what is being measured.
    """
    keys, sums, counts, nb = _complex_sums(sw, key)
    rng = random.Random(rng_seed)
    n_k = len(keys)
    out = [[0.0] * n_boot for _ in range(nb)]
    for b in range(n_boot):
        tot = [0.0] * nb
        m = 0
        for _ in range(n_k):
            k = keys[rng.randrange(n_k)]
            sk = sums[k]
            for j in range(nb):
                tot[j] += sk[j]
            m += counts[k]
        for j in range(nb):
            out[j][b] = tot[j] / m
    return out


def cluster_ci(sw, key, n_boot=2000, rng_seed=1, alpha=0.05):
    """95% percentile confidence interval from the cluster bootstrap, as (lo, hi) per budget.

    PERCENTILES, NOT +/- SEM. A +/-1 standard error band is about 68% and is not the convention
    these curves are read against; the 2.5th/97.5th percentiles of the replicate distribution are.
    They also do not assume the distribution is symmetric, which matters near the ends of the
    range where a proportion is bounded -- the oracle curve runs above 90% on FoldBench and a
    symmetric band there would extend past 100%.

    n_boot defaults to 2,000: percentile estimates need far more replicates than a standard
    deviation does, because they depend on the tails rather than the spread.
    """
    reps = _replicates(sw, key, n_boot, rng_seed)
    lo_i = max(0, int((alpha / 2) * n_boot) - 1)
    hi_i = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    lo, hi = [], []
    for j in range(len(sw["budgets"])):
        v = sorted(reps[j])
        lo.append(v[lo_i])
        hi.append(v[hi_i])
    return lo, hi


def cluster_sem(sw, key, n_boot=400, rng_seed=1):
    """SD across cluster-bootstrap replicates. Kept for the printed table and for comparing
    against the naive binomial SEM; the FIGURE draws cluster_ci instead."""
    reps = _replicates(sw, key, n_boot, rng_seed)
    out = []
    for j in range(len(sw["budgets"])):
        v = reps[j]
        mu = sum(v) / len(v)
        out.append((sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** 0.5)
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
