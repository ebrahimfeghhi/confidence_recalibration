#!/usr/bin/env python
"""Write a queue of untemplated folds -- the pool of candidate poses everything else selects from.

SUMMARY
    Decides WHAT to fold and writes it down. Nothing is launched here; call_folder.py runs the queue,
    and runs the templated queues too, so both kinds of run take the identical launch path.

INPUTS
    --out:         where the queue goes, and where the folds will land. Default the dataset's free pool.
    --dataset:     pxm2025 or foldbench. Default pxm2025. Picks the job jsons, the pool csv and the free
                   pool whose seeds count as taken.
    --complexes:   comma-separated subset. Default: every complex the dataset has a job json for, 131
                   for pxm2025. Anything named here and not found is reported, not skipped silently.
    --seeds:       folds per complex. Default 50. Ignored when --per-complex is given.
    --per-complex: csv of pdb,n_seeds giving a DIFFERENT count per complex, for a compute-matched
                   baseline where each complex gets exactly as many extra free folds as it used donors.
                   Any column named n_seeds is read; other columns are ignored.
    --start-seed:  first seed. Default the dataset's first pool seed (pxm2025 300000, foldbench 101).
    --force:       queue folds that already exist on disk. Default off, and it refuses instead.
    --queue-name:  the queue's file name under --out. Default queue_free.tsv.

    Files read:
    job jsons: one per complex, matched as `*-final-updated.json` (the MSA-attached input). A complex
               whose json is named otherwise is absent from the queue rather than an error.

OUTPUTS
    <out>/<queue-name>: one line per fold, tab separated:

        kind   pdb    target   tag    seed
        free   8ror   *        free   300000

    `target` is `*` because a free fold is not aimed at one interface -- it is scored against every
    interface of its complex. `tag` names the arm in the results csv.

EXAMPLE
    build_free_queue.py --out experimental_data/free_v2 --seeds 50
    build_free_queue.py --out /tmp/try --complexes 8ror,9k94 --seeds 3

FUNCTIONS
    complexes():           which complexes the dataset has a job json for
    queue_lines():         one queue line per (complex, seed)
    read_per_complex():    a csv of pdb,n_seeds -> {pdb: n_seeds}
    seeds_in_use():        the pool, results csvs and run dirs -> every seed already used, per complex
    existing_seeds():      which of those already have a fold on disk
    unreadable_warning():  whether later steps will be able to FIND what this writes
    main():                write the queue, or refuse if the folds already exist
"""
import argparse
import collections
import glob
import os
import sys

# the repo root is this file's parent, so nothing here carries an absolute path; paths.py owns
# every location from that point on
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402


def complexes(dataset, want=None):
    """in: dataset name, optional set of complexes.  out: (found, not found)

    Matches `*-final-updated.json`, the MSA-attached input. An input named anything else is absent
    from the queue rather than an error, so the caller reports what it asked for and did not get.
    """
    jobs = str(paths.DATASETS[dataset].jobs)
    found = {os.path.basename(p).split("-")[0]
             for p in glob.glob(f"{jobs}/*-final-updated.json")}
    return sorted(found if not want else (found & want)), sorted((want or set()) - found)


def queue_lines(pdbs, seeds, start_seed, per_complex=None, taken=None):
    """in: complexes, seeds each, first seed, optionally {pdb: n_seeds} and {pdb: seeds to avoid}.
    out: one queue line per (complex, seed)

    Seeds are allocated PER COMPLEX, skipping any already in `taken`. A single global range is not
    safe: one complex's earlier folds reached 300129 while the range started at 300100, so its new
    seed collided with a seed that already had a score and would have been counted twice.
    """
    out = []
    for pdb in pdbs:
        n = per_complex.get(pdb, 0) if per_complex is not None else seeds
        used = set(taken.get(pdb, ())) if taken else set()
        s = start_seed
        for _ in range(n):
            while str(s) in used:
                s += 1
            out.append(f"free\t{pdb}\t*\tfree\t{s}")
            used.add(str(s))
            s += 1
    return out


def seeds_in_use(pool_csv, results_csvs, run_dirs):
    """in: the pool csv, results csvs, run directories.  out: {pdb: every seed already folded or scored}

    Includes the run being extended (its results.csv and its <pdb>/<seed>-* directories), so a
    top-up never reuses a seed that run already folded -- the free pool alone does not know them.
    """
    import csv as _csv
    used = collections.defaultdict(set)
    if os.path.exists(pool_csv):
        for r in _csv.DictReader(open(pool_csv)):
            used[r["pdb"]].add(r["seed"])
    for src in results_csvs:
        if os.path.exists(src):
            for r in _csv.DictReader(open(src)):
                used[r["iface"].split("_")[0]].add(r["seed"])
    for d in run_dirs:
        for f in glob.glob(f"{d}/*/*-*"):
            used[os.path.basename(os.path.dirname(f))].add(os.path.basename(f).split("-")[0])
    return used


def read_per_complex(path):
    """in: a csv with pdb and n_seeds columns.  out: {pdb: n_seeds}"""
    import csv as _csv
    with open(path) as fh:
        return {r["pdb"]: int(r["n_seeds"]) for r in _csv.DictReader(fh)}


def existing_seeds(out, lines):
    """in: output dir and the queue lines about to be written.  out: (pdb, seed) already on disk

    Checks the seeds actually queued. With --per-complex or skipped seeds those are not
    start_seed..start_seed+seeds, which the previous version checked instead.
    """
    folds = [(f.split("\t")[1], f.split("\t")[4]) for f in lines]
    return [(pdb, seed) for pdb, seed in folds if glob.glob(f"{out}/{pdb}/{seed}-*")]


def unreadable_warning(out):
    """in: output dir.  out: a warning if later steps will not be able to find what lands there.

    paths.pool_pose() builds '<seed>-legacy-no-provenance', while new folds get a random run id. Poses
    outside the legacy naming are on disk and invisible to selection -- already true of 25 seeds per
    complex in the existing pool.
    """
    return (f"NOTE: poses here get a random run id, but paths.pool_pose() only resolves "
            f"'<seed>-{paths.LEGACY_RUN}'. Until it resolves by glob, poses under {out} will not be "
            f"selected by later steps.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--complexes", default=None, help="comma-separated; default is every complex")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="default the dataset's free pool")
    ap.add_argument("--per-complex", default=None,
                    help="csv of pdb,n_seeds -- a different count per complex")
    ap.add_argument("--force", action="store_true", help="queue folds that already exist on disk")
    ap.add_argument("--queue-name", default="queue_free.tsv",
                    help="file name under --out, so a top-up does not replace the run's first queue")
    A = ap.parse_args()

    want = set(A.complexes.split(",")) if A.complexes else None
    pdbs, missing = complexes(A.dataset, want)
    if missing:
        print(f"  NOT FOUND in {A.dataset}: {','.join(missing)}")
    if not pdbs:
        raise SystemExit("no complexes matched")

    ds = paths.DATASETS[A.dataset]
    A.start_seed = A.start_seed if A.start_seed is not None else int(ds.pool_seeds[0])
    out = A.out or str(ds.free)
    per = read_per_complex(A.per_complex) if A.per_complex else None
    if per:
        pdbs = [p for p in pdbs if per.get(p, 0) > 0]
    taken = seeds_in_use(str(ds.pool_csv), [f"{ds.free}/results.csv", f"{out}/results.csv"],
                         [str(ds.free), out])
    lines = queue_lines(pdbs, A.seeds, A.start_seed, per, taken)
    qpath = f"{out}/{A.queue_name}"
    text = "\n".join(lines) + "\n"
    # REFUSE BEFORE WRITING: both checks used to run after the queue was already written
    seen = existing_seeds(out, lines)
    if seen and not A.force:
        hit = sorted({p for p, _ in seen})
        print(f"  {len(seen)} of {len(lines)} folds already exist under {out} "
              f"({len(hit)} complexes, e.g. {', '.join(hit[:4])}).")
        raise SystemExit("  refusing. --out <new dir> for a fresh pool, or --force to queue anyway.")
    if os.path.exists(qpath) and open(qpath).read() != text and not A.force:
        raise SystemExit(f"  refusing to overwrite a different {qpath}; --force to replace it.")
    os.makedirs(out, exist_ok=True)
    open(qpath, "w").write(text)

    print(f"  dataset {A.dataset}   complexes {len(pdbs)}   seeds {A.seeds} from {A.start_seed}")
    print(f"  folds {len(lines)}   queue -> {qpath}")
    print(f"\n  {unreadable_warning(out)}")
    print(f"\n  next: call_folder.py --queue {qpath} --out {out} --gpus <ids> --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
