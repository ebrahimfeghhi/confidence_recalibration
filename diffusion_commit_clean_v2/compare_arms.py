#!/usr/bin/env python
"""Compare the self-templating method against free-sampling baselines at matched compute.

SUMMARY
    score_experiment.py answers "did templating help, given the pool". This answers the harder
    question: spend the same number of extra folds on MORE FREE SEEDS instead, and which wins.

    Four arms, all selected the same way -- top pooled pair_iptm, DockQ read off the selected pose:

      free 50           the complex's 50 lowest seed ids, and nothing else. The honest starting point.
      free + matched    those 50 plus as many extra free seeds as the method spent donor folds.
      method (50)       the templated poses, counting only donors drawn from the first 50 seeds.
      method (all)      every templated pose, including donors from the deeper pool.

    WHY TWO METHOD ARMS. The pool grew after the matrices were built: 74 interfaces hold up to 75
    seeds, and 302 of 3,132 donor poses came from seeds past the first 50. So the method as run had a
    deeper candidate pool than "50 free seeds" on some complexes. "method (50)" removes that advantage
    and is the arm to quote against "free 50"; "method (all)" is what actually ran.

    CLASSES come from the 50-seed pool, not the grown one. Labelling on the grown pool lets extra free
    seeds decide what counts as a decoy before any method runs. Measured effect: 2 of 292 labels move,
    both nocov -> decoy, so this is a small correction -- but it is the correct basis.

    COMPUTE MATCHING is per complex: a complex is matched once it has at least as many extra scored
    free seeds as the method spent donor folds on it. Unmatched complexes are counted and excluded,
    never silently folded in.

INPUTS
    --exp:        the templated experiment. Default experimental_data/self_template_fixed_50_seed.
    --extra-csv:  results csvs holding extra free folds, colon-separated. Default the free pool and
                  the compute-matched baseline.
    --classes:    which classes to report. Default decoy,solved,nocov.
    --pool-seeds: how many seeds define the baseline pool. Default 50.
    --out:        write the table as markdown here. Optional.

OUTPUTS
    stdout: one row per class -- n, then acceptable-at-0.23 under each of the four arms
    --out:  the same table as markdown

EXAMPLE
    compare_arms.py --out experimental_results/arms_compare.md

FUNCTIONS
    donors_used():    the experiment -> {complex: {donor pose}} and which came from the first N seeds
    extra_free():     extra free folds -> {interface: {seed: [(iptm, dockq)]}}
    tmpl_by_donor():  the experiment's scored rows -> {interface: [(iptm, dockq, donor)]}
    top():            rows -> the DockQ of the highest-ipTM one
    arms():           assemble the four arms per interface
    main():           read, compare, print, optionally write markdown
"""
import argparse
import collections
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
import pool_index  # noqa: E402
from fmt import table  # noqa: E402
from score_experiment import ACCEPTABLE, classes_at, load_scored, pool_by_seed  # noqa: E402

DONOR_SEED = re.compile(r"/free/([0-9a-z]{4})/(\d+)-")


def donors_used(exp, first):
    """in: the experiment and the first-n seed sets.
    out: ({complex: {donor}}, {donor: whether its seed is in the first n})
    """
    srcs = set()
    for line in open(f"{exp}/queue.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6:
            srcs.add(f[5])
    for r in csv.DictReader(open(f"{exp}/results.csv")):
        d = r.get("donor_pose", "")
        if d and d != "-1":
            srcs.add(d)
    per, shallow = collections.defaultdict(set), {}
    for d in srcs:
        m = DONOR_SEED.search(d)
        if not m:
            continue
        pdb, seed = m.group(1), m.group(2)
        per[pdb].add(d)
        shallow[d] = seed in first.get(pdb, set())
    return per, shallow


def extra_free(csvs, first):
    """in: free results csvs and the first-n seed sets.
    out: {interface: {seed: [(iptm, dockq)]}} for seeds OUTSIDE the first n
    """
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for src in [c for c in csvs.split(":") if c and os.path.exists(c)]:
        for r in csv.DictReader(open(src)):
            if not r.get("dockq") or not r.get("iptm"):
                continue
            pdb = r["iface"].split("_")[0]
            if r["seed"] in first.get(pdb, set()):
                continue
            out[r["iface"]][r["seed"]].append((float(r["iptm"]), float(r["dockq"])))
    return out


def tmpl_by_donor(exp):
    """in: the experiment.  out: {interface: [(iptm, dockq, donor)]}"""
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{exp}/results.csv")):
        if r["dockq"] and r["iptm"]:
            out[r["iface"]].append((float(r["iptm"]), float(r["dockq"]),
                                    r.get("donor_pose", "")))
    return out


def top(rows):
    """in: [(iptm, dockq, ...)].  out: the DockQ of the highest-iptm pose, or None"""
    return max(rows, key=lambda r: r[0])[1] if rows else None


def arms(pool, first, cls, per, shallow, extra, tmpl, made, unfinished, classes,
         require_match=True):
    """in: everything loaded.  out: (rows, n unmatched complexes, n partial interfaces)"""
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    unmatched, partial = set(), 0
    for iface, rows in sorted(pool.items()):
        pdb = iface.split("_")[0]
        if cls.get(iface) not in classes or iface not in made:
            continue
        if iface in unfinished:
            partial += 1
            continue
        n_donor = len(per.get(pdb, ()))
        avail = sorted(extra.get(iface, {}), key=int)
        if len(avail) < n_donor:
            unmatched.add(pdb)
            if require_match:
                continue
            avail = []
        base = [(ip, dq) for s, _, dq, ip in rows if s in first[pdb]]
        if not base:
            continue
        add = [v for s in avail[:n_donor] for v in extra[iface][s]]
        t_all = tmpl.get(iface, [])
        t_50 = [r for r in t_all if shallow.get(r[2], False)]
        k = cls[iface]
        acc[k]["free50"].append(top(base))
        acc[k]["matched"].append(top(base + add))
        acc[k]["m50"].append(top(t_50) if t_50 else top(base))
        acc[k]["mall"].append(top(t_all) if t_all else top(base))
    rows = []
    for k in classes:
        d = acc[k]
        if not d:
            continue
        n = len(d["free50"])
        hit = lambda v: f"{sum(1 for x in v if x >= ACCEPTABLE)}/{n}"  # noqa: E731
        rows.append([k, n, hit(d["free50"]), hit(d["matched"]), hit(d["m50"]), hit(d["mall"])])
    return rows, len(unmatched), partial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=f"{paths.DATA_DIR}/self_template_fixed_50_seed")
    ap.add_argument("--extra-csv",
                    default=f"{paths.FREE_DIR}/results.csv:{paths.DATA_DIR}/free_compute_matched/results.csv")
    ap.add_argument("--classes", default="decoy,solved,nocov")
    ap.add_argument("--pool-seeds", type=int, default=50)
    ap.add_argument("--no-require-match", action="store_true",
                    help="report the method arms over every finished interface, matched or not; the free+matched column is then not comparable")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()

    pool, first = pool_by_seed(A.pool_seeds)
    cls = classes_at(pool, first)
    per, shallow = donors_used(A.exp, first)
    extra = extra_free(A.extra_csv, first)
    tmpl = tmpl_by_donor(A.exp)
    made, unfinished = load_scored(A.exp)
    rows, unmatched, partial = arms(pool, first, cls, per, shallow, extra, tmpl,
                                    made, unfinished, A.classes.split(","),
                                    require_match=not A.no_require_match)
    head = ["class", "interfaces (n)", f"free {A.pool_seeds}", "free + matched",
            f"method ({A.pool_seeds}-seed donors)", "method (all donors)"]
    out = table(head, rows, align="l r r r r r")
    tail = (f"\n  experiment: {A.exp}\n"
            f"  classes computed on the first {A.pool_seeds} seeds per complex\n"
            f"  complexes excluded as not yet compute-matched: {unmatched}\n"
            f"  interfaces excluded as unfinished: {partial}")
    print(out)
    print(tail)
    if A.out:
        os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
        open(A.out, "w").write(f"# arm comparison at matched compute\n\n{out}\n{tail}\n")
        print(f"\n  wrote {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
