#!/usr/bin/env python
"""How much does defining interfaces from the crystal change the donors, and their quality?

SUMMARY
    Two donor selections that differ ONLY in how interfaces and chains are defined:

      A  crystal-free, the experiment as run: interfaces are the antibody-antigen chain pairs in contact
         in >= 1 predicted pose, chains named after a predicted anchor pose, ranked by the pose's own
         ipTM at those chains (build_experiment.py).
      B  crystal-defined: the crystal's interfaces only, each pose's chains matched to the crystal's,
         ranked by the pose's ipTM at those matched chains.

    Both use the same pool (seeds 300000-300049), the same pose-to-pose DockQ, the same diversity filter,
    round-robin and cap -- B is built by the pipeline's own pose_to_pose_dockq.py and
    structural_diversity_filter.py with the crystal's maps and interfaces swapped in.

    DONOR QUALITY is measured on the crystal's interfaces, as evaluation: for each crystal interface,
    the DockQ of every donor of its complex (a template is a whole complex and a fold is scored on every
    interface), from the crystal-scored pool csv. Per class (solved / decoy / nocov, from the first 50
    seeds, as the reporters define them): how many interfaces have at least one acceptable donor
    (DockQ >= 0.23), the median best donor DockQ, the mean donor DockQ, and donors per complex.

INPUTS
    --exp:    the crystal-free experiment (A). Required. Its poses.csv defines the pool; its queue.tsv
              is selection A.
    --old:    an earlier experiment whose queue.tsv is reported alongside for donor counts. Optional.
    --work:   directory for B's interfaces, matrices and queue. Required.
    --report: markdown file for the tables. Required.
    --jobs:   CPU processes for B's DockQ matrices. Default 8. No GPU is used.

OUTPUTS
    <work>/interfaces_crystal.csv, <work>/pose_dockq_matrices/, <work>/queue.tsv   selection B
    <report>                                                                       the tables

EXAMPLE
    analysis/crystal_frame_selection.py --exp experimental_data/self_template_fixed_50_seed \\
        --old experimental_data/self_template_fixed_50_seed_old_crystal_iptm \\
        --work experimental_data/analysis/crystal_frame_selection \\
        --report diffusion_commit_clean_v2/analysis/crystal_frame_selection.md

FUNCTIONS
    build_b():         selection B with the pipeline's own scripts
    donors():          a queue -> {pdb: {(seed, sample)}}
    donor_quality():   a selection -> per crystal interface: n donors, any acceptable, best, mean
    summarise():       per class, the numbers above
    main():            build B, compare A and B, write the report
"""
import argparse
import collections
import csv
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.dirname(HERE)
sys.path.insert(0, V2)
sys.path.insert(0, os.path.dirname(V2))

import paths  # noqa: E402
from fmt import table  # noqa: E402
from score_experiment import ACCEPTABLE, classes_at, pool_by_seed  # noqa: E402

PY = str(paths.PROTENIX_PYTHON)


def build_b(exp, work, jobs, log):
    """in: experiment A, the work directory, CPU processes, a logger.  out: selection B's queue path"""
    os.makedirs(work, exist_ok=True)
    pdbs = {r["pdb"] for r in csv.DictReader(open(f"{exp}/poses.csv"))}
    ifaces = f"{work}/interfaces_crystal.csv"
    with open(ifaces, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain_1", "chain_2", "ab_chain", "ag_chain"])
        for r in csv.DictReader(open(str(paths.PXM2025_CHAIN_ROLES))):
            if r["pdb"] in pdbs:
                w.writerow([r["pdb"], r["chain_1"], r["chain_2"], r["ab_chain"], r["ag_chain"]])
    mats = f"{work}/pose_dockq_matrices"
    cmd = [PY, f"{V2}/pose_to_pose_dockq.py", "--poses", f"{exp}/poses.csv",
           "--maps", f"{paths.PXM2025_DIR}/chain_maps.csv", "--roles", ifaces, "--out", mats,
           "--device", "cpu"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
    log(f"  B: DockQ matrices on {jobs} CPU processes -> {mats}")
    procs = [subprocess.Popen(cmd + ["--shard", f"{i}/{jobs}"], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, text=True) for i in range(jobs)]
    for p in procs:
        err = p.communicate()[1]
        if p.returncode:
            raise SystemExit(f"B's DockQ shard failed:\n{err[-2000:]}")
    queue = f"{work}/queue.tsv"
    subprocess.run([PY, f"{V2}/structural_diversity_filter.py", "--matrices", mats, "--out", queue],
                   check=True)
    return queue


def donors(queue):
    """in: a queue path.  out: {pdb: {(seed, sample)}}"""
    out = collections.defaultdict(set)
    for l in open(queue):
        f = l.rstrip("\n").split("\t")
        m = re.search(r"/(\d+)-[^/]+/predictions/[^/]+_sample_(\d+)\.cif$", f[5])
        out[f[1]].add((m.group(1), int(m.group(2))))
    return out


def donor_quality(sel, pool):
    """in: {pdb: donor keys} and the crystal-scored pool.
    out: ({crystal interface: (n donors, any acceptable, best DockQ, mean DockQ)},
          donor-interface pairs with no crystal score -- a pose whose crystal mapping failed)
    """
    out, unscored = {}, 0
    for iface, rows in pool.items():
        d = sel.get(iface.split("_")[0], set())
        dq = [q for s, x, q, _ in rows if (s, x) in d]
        unscored += len(d) - len(dq)
        out[iface] = (len(d), bool(dq) and max(dq) >= ACCEPTABLE,
                      max(dq) if dq else float("nan"), statistics.mean(dq) if dq else float("nan"))
    return out, unscored


def summarise(q, cls, classes):
    """in: donor quality per interface, the classes.  out: table rows per class"""
    rows = []
    for k in classes:
        v = [q[i] for i in q if cls.get(i) == k]
        if v:
            rows.append([k, len(v), f"{sum(1 for x in v if x[1])}/{len(v)}",
                         f"{statistics.median(x[2] for x in v):.3f}",
                         f"{statistics.mean(x[3] for x in v):.3f}"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--old", default=None)
    ap.add_argument("--work", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    A = ap.parse_args()

    lines = []
    log = lambda m: (print(m, flush=True), lines.append(m))  # noqa: E731
    qb = build_b(A.exp, A.work, A.jobs, log)
    sel = {"A crystal-free": donors(f"{A.exp}/queue.tsv"), "B crystal-defined": donors(qb)}
    if A.old:
        sel = {"old arm (crystal-mapped ipTM)": donors(f"{A.old}/queue.tsv"), **sel}

    pool, first = pool_by_seed(50)
    pool = {i: [r for r in rows if r[0] in first[i.split("_")[0]]] for i, rows in pool.items()}
    cls = classes_at(pool, first)
    classes = ["decoy", "solved", "nocov"]

    out = ["# Donor selection: crystal-free (A) vs crystal-defined interfaces (B)", "",
           "Same pool (seeds 300000-300049), pose-to-pose DockQ, diversity filter, round-robin and "
           "cap; only the interface and chain definition differs. Donor DockQ is against the crystal "
           "at each crystal interface (evaluation). Produced by `analysis/crystal_frame_selection.py`.",
           ""]
    out += ["## Donors per complex", ""]
    bins = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 49), (50, 50)]
    rows = []
    for name, s in sel.items():
        v = sorted(len(x) for x in s.values())
        rows.append([name, len(s), sum(v), statistics.median(v), f"{statistics.mean(v):.1f}"]
                    + [sum(1 for x in v if a <= x <= b) for a, b in bins])
    out += [table(["selection", "complexes", "donors", "median", "mean"]
                  + [f"{a}-{b}" if a != b else f"{a} (cap)" for a, b in bins], rows,
                  align="l" + " r" * 10), ""]
    out += ["## Donor quality per class (crystal interfaces)", ""]
    for name, s in sel.items():
        q, unscored = donor_quality(s, pool)
        out += [f"### {name}", "",
                table(["class", "interfaces", "any acceptable donor", "median best donor DockQ",
                       "mean donor DockQ"], summarise(q, cls, classes), align="l r r r r"), "",
                f"donor-interface pairs without a crystal score (mapping failed): {unscored}", ""]
    a, b = sel["A crystal-free"], sel["B crystal-defined"]
    same = sum(1 for p in a if a[p] == b.get(p))
    shared = sum(len(a[p] & b.get(p, set())) for p in a)
    out += ["## Overlap of A and B", "",
            f"- complexes with identical donor sets: {same} / {len(a)}",
            f"- donors in both: {shared} (A has {sum(len(x) for x in a.values())}, "
            f"B has {sum(len(x) for x in b.values())})", ""]
    os.makedirs(os.path.dirname(os.path.abspath(A.report)), exist_ok=True)
    open(A.report, "w").write("\n".join(out))
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
