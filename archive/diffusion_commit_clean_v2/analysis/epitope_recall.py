#!/usr/bin/env python
"""When a confident-decoy interface DOES hold an acceptable pose, how much of the epitope does it get?

SUMMARY
    A decoy interface is one where the top-pair_iptm pose of the first 50 free seeds is wrong and an
    acceptable pose (DockQ >= 0.23) exists somewhere in those seeds. Selection failed, but the pose was
    there. This asks what was there to select: of the antigen residues the antibody touches in the
    crystal, how many does that pose touch.

    ONE POSE PER INTERFACE, the HIGHEST-DockQ acceptable one. That is the best case -- the ceiling a
    perfect selector could reach -- so a low recall here cannot be blamed on having picked a poor
    representative. Its DockQ is reported beside its recall, because recall and DockQ move together and
    a recall of 0.4 at DockQ 0.25 says something different from 0.4 at DockQ 0.60.

    WHY RECALL, NOT JACCARD. The question is how much of the true epitope is covered. Jaccard would
    also charge the pose for touching extra antigen residues, which is a different complaint.

    The epitope itself comes from OST (eval/ost_epitope.py): its chain mapping, its 5 A all-atom
    contacts, its residue alignment. Interfaces whose chains never touch in the crystal, or that OST's
    chain mapping does not cover, are reported as excluded rather than scored 0.

INPUTS
    --dataset:  pxm2025 or foldbench, repeatable. Default: both.
    --seeds:    how many free seeds define the pool and the classes. Default 50, as every reporter.
    --work-dir: where the OST worklist and its raw output go. Default a temp directory.
    --out:      per-interface csv. Optional.

OUTPUTS
    stdout: one row per dataset -- n, median recall, quartiles, and the counts above 0.75, 0.50-0.75
            and below 0.50; then the same split with each band's median pose DockQ
    --out:  dataset, iface, seed, sample, dockq, recall, epitope residues, recovered, pose residues

EXAMPLE
    analysis/epitope_recall.py --out experimental_results/epitope_recall_decoys.csv

FUNCTIONS
    worklist():  a dataset -> the rows to score, one per decoy interface
    run_ost():   a worklist -> {key: (recall, n_crystal, n_both, n_pose)} via the OST binary
    summarise(): recalls and DockQs -> the stdout rows
    main():      per dataset, build, score, tabulate
"""
import argparse
import collections
import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import paths  # noqa: E402
import pool_index  # noqa: E402
from fmt import table  # noqa: E402
from score_experiment import ACCEPTABLE, classes_at, pool_by_seed  # noqa: E402

sys.path.insert(0, str(paths.EVAL_DIR))
import ost_dockq  # noqa: E402

BANDS = (("recovers >= 0.75", 0.75, 1.01), ("0.50 - 0.75", 0.50, 0.75), ("below 0.50", -0.01, 0.50))


def worklist(dataset, n_seeds, want):
    """in: a dataset name, how many seeds define the pool, which classes to score.
    out: ([(key, pose, ref cif, reference antibody chain, reference antigen chain, chain map json)],
         {key: (iface, seed, sample, dockq)}, counts of what was dropped)

    The chain map handed to OST is the pose's own, from the dataset's chain_maps.csv -- the same one
    items_for_interface() just used to name the chains, so the epitope and the DockQ beside it are
    read at one correspondence.

    Every row is one interface's best acceptable pose, for each class asked for. DECOY and SOLVED are
    the comparable pair: both are defined to hold an acceptable pose, and taking the best one in each
    makes the comparison "the best thing a selector could have picked", not "what ipTM picked". NOCOV
    has no acceptable pose by definition and yields nothing.

    A pose with no chain mapping is dropped here rather than sent to OST, so the two drop reasons stay
    distinguishable.
    """
    ds = paths.DATASETS[dataset]
    pool, first = pool_by_seed(n_seeds, str(ds.pool_csv))
    cls = classes_at(pool, first)
    maps = pool_index.load_chain_maps(str(ds.chain_maps))
    roles = pool_index.load_roles(str(ds.chain_roles))
    work, meta, dropped = [], {}, collections.Counter()
    for iface in sorted(i for i, k in cls.items() if k in want):
        pdb = iface.split("_")[0]
        good = [r for r in pool[iface] if r[0] in first[pdb] and r[2] >= ACCEPTABLE]
        if not good:                        # cannot happen: the class is defined by one existing
            dropped["no acceptable pose"] += 1
            continue
        seed, sample, dockq, _ = max(good, key=lambda r: r[2])
        klass = cls[iface]
        items, _ = pool_index.items_for_interface(iface, [(seed, sample)], maps, roles,
                                                  free_dir=str(ds.free))
        if not items:
            dropped["no chain map for the pose"] += 1
            continue
        pose, _, _ = items[0]
        if not os.path.exists(pose):
            dropped["pose file missing"] += 1
            continue
        ref_ab, ref_ag = roles[iface]
        key = f"{dataset}:{iface}"
        chain_map = maps.get((pdb, str(seed), int(sample))) or {}
        work.append((key, pose, ds.ref(pdb), ref_ab, ref_ag, json.dumps(chain_map)))
        meta[key] = (iface, klass, seed, sample, dockq)
    # sorted by pose so the worker reuses one Scorer across interfaces of the same pose
    work.sort(key=lambda r: (r[1], r[0]))
    return work, meta, dropped


def run_ost(work, work_dir, tag):
    """in: the worklist, a directory to write into, a name for the files.
    out: ({key: (recall, n_crystal, n_both, n_pose, chain mapping used)},
         {key: note for the ones with no recall})

    One OST process for the whole list: Scorer construction dominates, and the worker caches it per
    pose, which a per-row process could not do.
    """
    wl = os.path.join(work_dir, f"{tag}_work.tsv")
    out = os.path.join(work_dir, f"{tag}_recall.tsv")
    with open(wl, "w") as fh:
        csv.writer(fh, delimiter="\t").writerows(work)
    cmd = []
    if ost_dockq.TASKSET:
        cmd += ["taskset", "-c", ost_dockq.TASKSET]
    cmd += [ost_dockq.OST, os.path.join(str(paths.EVAL_DIR), "ost_epitope.py"),
            "--work", wl, "--out", out]
    r = subprocess.run(cmd, env=ost_dockq.ENV, capture_output=True, text=True)
    if not os.path.exists(out):
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"OST wrote no output for {tag} (exit {r.returncode})")
    got, notes = {}, {}
    for row in csv.reader(open(out), delimiter="\t"):
        key, recall, n_crystal, n_both, n_pose, note, used = row
        if recall:
            got[key] = (float(recall), int(n_crystal), int(n_both), int(n_pose), used)
        else:
            notes[key] = note
    return got, notes


def summarise(pairs):
    """in: [(recall, dockq, crystal epitope size)].  out: a stdout row -- n, quartiles, band counts

    Quartiles need four points; below that only the median is defined, so the quartile cells are left
    empty rather than filled with the median.

    The epitope size is carried into the table because it sets recall's GRANULARITY: a crystal epitope
    of 3 residues can only return 0, 1/3, 2/3 or 1, and reading such a value as a percentage invites a
    precision it does not have.
    """
    rec = sorted(p[0] for p in pairs)
    q = statistics.quantiles(rec, n=4) if len(rec) >= 4 else None
    row = [len(rec), f"{statistics.median(rec):.2f}",
           f"{q[0]:.2f}" if q else "", f"{q[2]:.2f}" if q else "",
           f"{statistics.median(p[2] for p in pairs):.0f}"]
    for _, lo, hi in BANDS:
        row.append(sum(lo <= r < hi for r in rec))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", choices=paths.DATASET_NAMES, default=None)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--classes", action="append", choices=("decoy", "solved"), default=None,
                    help="default: both, so the decoy-vs-solved comparison is the default output")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    datasets = A.dataset or list(paths.DATASET_NAMES)
    want = tuple(A.classes or ("decoy", "solved"))
    work_dir = A.work_dir or tempfile.mkdtemp(prefix="epitope_recall_")
    os.makedirs(work_dir, exist_ok=True)

    overall, bands, matched, rows, excluded = [], [], [], [], []
    mapping_used = collections.defaultdict(collections.Counter)
    for name in datasets:
        work, meta, dropped = worklist(name, A.seeds, want)
        got, notes = run_ost(work, work_dir, name)
        pairs = collections.defaultdict(list)
        for key, *_ in work:
            iface, klass, seed, sample, dockq = meta[key]
            if key not in got:
                dropped[notes.get(key, "no result")] += 1
                continue
            recall, n_crystal, n_both, n_pose, used = got[key]
            pairs[klass].append((recall, dockq, n_crystal))
            mapping_used[name][used] += 1
            rows.append([name, iface, klass, seed, sample, f"{dockq:.4f}", f"{recall:.4f}",
                         n_crystal, n_both, n_pose, used])
        for klass in want:
            if pairs[klass]:
                overall.append([name, klass] + summarise(pairs[klass]))
        for label, lo, hi in BANDS:
            for klass in want:
                band = [p for p in pairs[klass] if lo <= p[0] < hi]
                if band:
                    bands.append([name, label, klass, len(band),
                                  f"{statistics.median(p[0] for p in band):.2f}",
                                  f"{statistics.median(p[1] for p in band):.2f}",
                                  f"{statistics.median(p[2] for p in band):.0f}"])
        # THE DECOMPOSITION. If decoy poses score lower only because they sit on less of the epitope,
        # then inside a recall band the two classes should score the same and the gap column goes to
        # ~0. A gap that survives is something recall does not explain -- orientation or register on
        # a surface both classes found.
        if "decoy" in pairs and "solved" in pairs:
            for label, lo, hi in BANDS:
                d = [p[1] for p in pairs["decoy"] if lo <= p[0] < hi]
                s = [p[1] for p in pairs["solved"] if lo <= p[0] < hi]
                if d and s:
                    matched.append([name, label, len(d), len(s),
                                    f"{statistics.median(d):.2f}", f"{statistics.median(s):.2f}",
                                    f"{statistics.median(s) - statistics.median(d):+.2f}"])
        for reason, n in sorted(dropped.items()):
            excluded.append([name, reason, n])

    print(f"**Epitope recall of the best acceptable pose** (first {A.seeds} free seeds; recall = "
          f"share of the crystal epitope's antigen residues the pose also contacts)\n")
    print(table(["dataset", "class", "interfaces", "median recall", "lower quartile",
                 "upper quartile", "median epitope residues"] + [b[0] for b in BANDS], overall,
                align="l l r r r r r r r r"))
    print("\n**By band**, with how good those poses are\n")
    print(table(["dataset", "band", "class", "interfaces", "median recall", "median pose DockQ",
                 "median epitope residues"], bands, align="l l l r r r r"))
    if matched:
        print("\n**Pose DockQ at MATCHED epitope recall** -- does recall explain the quality gap?\n")
        print(table(["dataset", "recall band", "decoy n", "solved n", "median DockQ decoy",
                     "median DockQ solved", "gap"], matched, align="l l r r r r r",
                    signed=("gap",)))
    if excluded:
        print("\n**Excluded** (never scored 0 -- there is no epitope to recover, or no mapping)\n")
        print(table(["dataset", "reason", "interfaces"], excluded, align="l l r"))
    for name, c in mapping_used.items():
        print(f"\n  {name}: chain mapping -- the project's on {c['given']}, OST's own search on "
              f"{c['ost']}")
    print(f"  work dir: {work_dir}")
    if A.out:
        with open(A.out, "w") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "iface", "class", "seed", "sample", "dockq", "epitope_recall",
                        "epitope_residues_crystal", "epitope_residues_recovered",
                        "epitope_residues_pose", "chain_mapping"])
            w.writerows(rows)
        print(f"  -> {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
