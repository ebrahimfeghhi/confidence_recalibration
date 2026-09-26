#!/usr/bin/env python
"""Self-templating vs free + compute-matched sampling, per interface and per complex, with paired tests.

SUMMARY
    compare_arms.py counts acceptable selected poses per class. This reports the same comparison two
    ways and asks whether the difference is beyond chance:

      per interface   the field's usual unit. Acceptable counts and mean selected-pose DockQ, and an
                      exact McNemar test on the discordant interfaces (acceptable under one arm only).
      per complex     each complex weighted equally: per class, a complex's fraction of acceptable
                      interfaces, averaged over complexes. Interfaces of one complex share its donors
                      and folds, so they are not independent; a cluster bootstrap resamples whole
                      complexes for a 95% interval on the difference in acceptable rate and mean DockQ.

    The arms are built by compare_arms.py's own functions (the same interfaces, classes, compute
    matching and selected poses), so every count here equals its table.

INPUTS
    --exp:        the templated experiment. Default experimental_data/self_template_fixed_50_seed.
    --extra-csv:  free results csvs, colon-separated. Default as compare_arms.py.
    --boot:       bootstrap resamples. Default 10000.
    --seed:       bootstrap RNG seed. Default 0.
    --out:        also write the report as markdown here. Optional.

OUTPUTS
    stdout (and --out): per-interface and per-complex tables for decoy / solved / nocov, the McNemar
    p-values and the bootstrap intervals

EXAMPLE
    analysis/arm_statistics.py --out diffusion_commit_clean_v2/analysis/arm_statistics.md

FUNCTIONS
    selected():  the arms, per interface -> {interface: (class, free + matched DockQ, method DockQ)}
    mcnemar():   discordant counts -> exact two-sided p-value
    per_complex(): per-interface values -> per-complex means, per class
    bootstrap(): complexes resampled -> 95% interval on the method minus free + matched difference
    main():      assemble, test, print
"""
import argparse
import collections
import math
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, V2)
sys.path.insert(0, os.path.dirname(V2))

import paths  # noqa: E402
import compare_arms as C  # noqa: E402
from fmt import table  # noqa: E402
from score_experiment import ACCEPTABLE, classes_at, load_scored, pool_by_seed  # noqa: E402

CLASSES = ("decoy", "solved", "nocov")


def selected(exp, extra_csv):
    """in: the experiment, free results csvs.  out: ({interface: (class, matched DockQ, method DockQ)},
    interfaces excluded as unfinished, complexes not compute-matched) -- compare_arms.arms' own rules
    """
    pool, first = pool_by_seed(50)
    cls = classes_at(pool, first)
    per, _ = C.donors_used(exp, first)
    extra = C.extra_free(extra_csv, first)
    tmpl = C.tmpl_by_donor(exp)
    made, unfinished = load_scored(exp)
    by_pdb = collections.defaultdict(list)
    for i in pool:
        by_pdb[i.split("_")[0]].append(i)
    out, partial, unmatched = {}, 0, set()
    for iface, rows in sorted(pool.items()):
        pdb = iface.split("_")[0]
        if cls.get(iface) not in CLASSES or iface not in made:
            continue
        if iface in unfinished:
            partial += 1
            continue
        n = len(per.get(pdb, ()))
        seeds = C.matched_seeds(extra, by_pdb[pdb], n)
        if len(seeds) < n:
            unmatched.add(pdb)
            continue
        base = [(ip, dq) for s, _, dq, ip in rows if s in first[pdb]]
        matched = C.top(base + [v for s in seeds for v in extra[iface][s]])
        method = C.top(tmpl[iface]) if tmpl.get(iface) else C.top(base)
        out[iface] = (cls[iface], matched, method)
    return out, partial, len(unmatched)


def mcnemar(b, c):
    """in: interfaces acceptable under the method only (b) and under free + matched only (c).
    out: exact two-sided McNemar p-value (binomial on the discordant pairs)"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def per_complex(vals):
    """in: [(interface, matched, method)] of one class.  out: {complex: (acc matched, acc method,
    mean matched, mean method)} -- each a mean over that complex's interfaces"""
    by = collections.defaultdict(list)
    for i, m, t in vals:
        by[i.split("_")[0]].append((m, t))
    return {p: (statistics.mean(m >= ACCEPTABLE for m, _ in v),
                statistics.mean(t >= ACCEPTABLE for _, t in v),
                statistics.mean(m for m, _ in v), statistics.mean(t for _, t in v))
            for p, v in by.items()}


def bootstrap(pc, boot, rng):
    """in: per-complex values, resamples, an RNG.  out: 95% intervals (acceptable rate, mean DockQ) of
    method minus free + matched, resampling whole complexes"""
    keys = list(pc)
    da, dm = [], []
    for _ in range(boot):
        s = [pc[rng.choice(keys)] for _ in keys]
        da.append(statistics.mean(x[1] - x[0] for x in s))
        dm.append(statistics.mean(x[3] - x[2] for x in s))
    q = lambda v: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1])  # noqa: E731
    return q(da), q(dm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=f"{paths.DATA_DIR}/self_template_fixed_50_seed")
    ap.add_argument("--extra-csv",
                    default=f"{paths.FREE_DIR}/results.csv:{paths.DATA_DIR}/free_compute_matched/results.csv")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    sel, partial, unmatched = selected(A.exp, A.extra_csv)
    rng = random.Random(A.seed)
    lines = [f"# Self-templating vs free + compute-matched", "",
             f"experiment `{os.path.basename(A.exp)}` -- {len(sel)} interfaces counted; "
             f"{partial} excluded as unfinished; {unmatched} complexes not yet compute-matched. "
             f"Acceptable = DockQ >= {ACCEPTABLE} for the selected (top-ipTM) pose.", ""]

    rows = []
    for k in CLASSES:
        v = [(i, m, t) for i, (c, m, t) in sel.items() if c == k]
        if not v:
            continue
        n = len(v)
        b = sum(t >= ACCEPTABLE > m for _, m, t in v)
        c = sum(m >= ACCEPTABLE > t for _, m, t in v)
        rows.append([k, n, f"{sum(m >= ACCEPTABLE for _, m, _ in v)}/{n}",
                     f"{sum(t >= ACCEPTABLE for *_, t in v)}/{n}",
                     f"{statistics.mean(m for _, m, _ in v):.3f}",
                     f"{statistics.mean(t for *_, t in v):.3f}", b, c, f"{mcnemar(b, c):.3g}"])
    lines += ["## Per interface", "",
              table(["class", "interfaces", "free + matched acceptable", "self-template acceptable",
                     "free + matched mean DockQ", "self-template mean DockQ",
                     "self-template only", "free + matched only", "McNemar p"], rows,
                    align="l r r r r r r r r"), ""]

    rows = []
    for k in CLASSES:
        v = [(i, m, t) for i, (c, m, t) in sel.items() if c == k]
        if not v:
            continue
        pc = per_complex(v)
        (lo_a, hi_a), (lo_m, hi_m) = bootstrap(pc, A.boot, rng)
        x = list(pc.values())
        rows.append([k, len(pc), f"{statistics.mean(y[0] for y in x):.3f}",
                     f"{statistics.mean(y[1] for y in x):.3f}", f"[{lo_a:+.3f}, {hi_a:+.3f}]",
                     f"{statistics.mean(y[2] for y in x):.3f}", f"{statistics.mean(y[3] for y in x):.3f}",
                     f"[{lo_m:+.3f}, {hi_m:+.3f}]"])
    lines += ["## Per complex (each complex weighted equally)", "",
              table(["class", "complexes", "free + matched acceptable rate",
                     "self-template acceptable rate", "difference, 95% CI",
                     "free + matched mean DockQ", "self-template mean DockQ", "difference, 95% CI"],
                    rows, align="l r r r r r r r"), "",
              f"95% intervals: cluster bootstrap over complexes, {A.boot} resamples, seed {A.seed}. "
              f"McNemar: exact, two-sided, on discordant interfaces (treats interfaces as independent).",
              ""]
    text = "\n".join(lines)
    print(text)
    if A.out:
        open(A.out, "w").write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
