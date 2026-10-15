"""Is a fold stage REALLY finished? Trunk, poses and every expected score row -- not just the trunk.

WHY THIS EXISTS. The handoff watcher counted trunk npz files and declared a stage complete when the
count reached the queue length, then killed the workers and switched stages. But the trunk is
written DURING INFERENCE -- folder.py:1034 passes PROTENIX_DUMP_TRUNK into the fold itself -- while
the poses are finalised and the interfaces scored afterwards, and under --overlap-scoring the
scoring runs in a BACKGROUND thread after the fold returns. So the trunk of the last unit appears
while that unit is still being written and scored. Counting trunks reports COMPLETE early and the
stage switch then kills the workers mid-unit, leaving a trunk with no poses and no score rows --
reproduced by the reviewer, and the exact shape of a unit that is silently useless later: the cache
builder needs the poses that trunk was computed from.

A FIXED DELAY WOULD NOT FIX IT. "Wait N minutes after the last trunk" is a guess about how long
scoring takes on a structure whose size varies 2-3x, and it is wrong in both directions: too short
truncates, too long idles three cards.

WHAT COMPLETE MEANS HERE, per queued unit:
    1. the trunk npz exists                                  <fold>/trunks/<pdb>/<tag>_<seed>.npz
    2. EXACTLY ONE finished pose directory exists            <fold>/poses/<pdb>/<seed>-*/run.json
    3. every interface of the complex carries every expected sample's score

(2) is exactly what phase_j_build_cache.py:84 requires -- it refuses 0 or 2 rather than guessing
which is the finished one -- so a stage that passes here is a stage the cache builder can read.

(3) reuses folder.py's OWN iface_scored and interfaces_by_pdb rather than reimplementing them. A
watcher that defined "scored" differently from the worker would disagree with it about when to stop,
which is the class of bug this file exists to close.
"""
import argparse, glob, os, re, sys

sys.path.insert(0, "/home/ebrahim/confidence_recalibration")
from run_model.folder import interfaces_by_pdb, open_csv, iface_scored   # ONE definition of scored
import paths                                                          # repo root, not run_model/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="the fold root: trunks/, poses/, results.csv")
    ap.add_argument("--queue", required=True)
    ap.add_argument("--dataset", required=True, choices=paths.DATASET_NAMES)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--pool", default="", help="interface registry; default the dataset's")
    ap.add_argument("--no-trunk", action="store_true",
                    help="do not require a trunk npz. FoldBench folds dump no trunks, so for that "
                         "stage complete means poses plus every score row. The ordering problem is "
                         "the same there: folder.py:1074 writes run.json BEFORE finish_unit scores "
                         "the unit, and under --overlap-scoring the scoring is deferred to a "
                         "background thread, so a pose count reaches the queue length while the "
                         "last folds are still being scored.")
    ap.add_argument("--strict-scores", action="store_true",
                    help="require EVERY interface scored. Off by default: see TERMINAL below.")
    ap.add_argument("--partial-out", default="", help="write the partially scored units to a csv")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    ds = paths.DATASETS[a.dataset]
    # EXACTLY folder.py:595 -- `A.pool = A.pool or str(ds.pool_csv)`. Not chain_roles: for
    # sabdab_train that file is a role table (pdb, model_chain, ref_chain, auth_chain, role), not an
    # interface registry, and interfaces_by_pdb refuses it. The watcher must read the same registry
    # the worker scored against or the two disagree about what "scored" means.
    pool = a.pool or str(ds.pool_csv)
    ifaces = interfaces_by_pdb(pool)
    csv_path = os.path.join(a.fold, "results.csv")
    rows_done, _units = open_csv(csv_path) if os.path.exists(csv_path) else ({}, {})

    # TERMINAL UNITS: the worker tried to score them, could not finish, SAID SO, and moved on.
    #
    # Some interfaces simply never score. 8vy9's pool lists AC, AD, BC and BD; every one of its ten
    # seeds scores AC and writes zero rows for the other three, and the same shape repeats for
    # 9nw7, 9o4n, 9o4o, 9o55, 9q5m, 9q5v and 9q71 -- 82 units across 8 structures here, and 30 of
    # 200 sampled units in the previous run (10op_FN has zero samples on every seed). The FOLD is
    # fine in every case: 8vy9 reports N_asym 4, N_token 705, a 40 s forward pass and a written
    # trunk. It is the scorer that cannot produce DockQ for those chain pairs.
    #
    # A unit like that is never retried -- folder.py keeps its claim by design -- so requiring every
    # interface would make the stage wait forever for rows nobody will write, never confirm
    # completion, and never hand the cards to the next stage. That is what this gate was supposed
    # to prevent, in the opposite direction.
    #
    # What the training cache needs is the trunk and the poses it was computed from, and those are
    # intact. The DockQ scores are this project's evaluation metric, not the training label. So a
    # unit the worker has finished with counts as done, and the partial ones are COUNTED AND NAMED
    # rather than quietly folded into the total. --strict-scores restores the old behaviour.
    term = set()
    pat = re.compile(r"^\s+(\S+) \S+ (\S+) seed (\d+): SCORING INCOMPLETE")
    for lf in sorted(glob.glob(os.path.join(a.fold, "w*.log"))):
        try:
            with open(lf, errors="replace") as fh:
                for line in fh:
                    m = pat.match(line)
                    if m:
                        term.add((m.group(1), m.group(2), m.group(3)))
        except OSError:
            pass

    units = [l.split("\t") for l in open(a.queue).read().splitlines() if l.strip()]
    n = len(units)
    done = no_trunk = no_pose = bad_pose = unscored = partial = 0
    pending = []
    partial_rows = []
    for tag, pdb, _target, arm, seed in units:
        seed = seed.strip()
        trunk = os.path.join(a.fold, "trunks", pdb, f"{tag}_{seed}.npz")
        pdirs = [d for d in glob.glob(os.path.join(a.fold, "poses", pdb, f"{seed}-*"))
                 if os.path.exists(os.path.join(d, "run.json"))]
        want = sorted(ifaces.get(pdb, ()))
        scored = bool(want) and all(iface_scored(rows_done, i, seed, arm, a.nsample) for i in want)

        # The trunk requirement is OPTIONAL, the rest is not. Written as one if/elif chain with
        # `if a.no_trunk: pass` in front, the chain short-circuited on that first branch and skipped
        # every remaining check: FoldBench came back 0 of 612 complete with 0 in every failure
        # bucket, which cannot be true of any unit. The trunk test is now a separate statement.
        why = None
        if not a.no_trunk and not os.path.exists(trunk):
            no_trunk += 1; why = "no trunk"
        elif len(pdirs) == 0:
            no_pose += 1
            why = "NO finished pose dir" if a.no_trunk else "trunk but NO finished pose dir"
        elif len(pdirs) > 1:
            bad_pose += 1; why = f"{len(pdirs)} finished pose dirs (build_cache will refuse)"
        elif not scored:
            if not a.strict_scores and (pdb, arm, seed) in term:
                partial += 1          # the worker finished with it; see TERMINAL above
                done += 1
                why = None
                partial_rows.append((pdb, seed, ";".join(
                    i for i in want if not iface_scored(rows_done, i, seed, arm, a.nsample))))
            else:
                unscored += 1; why = "poses but score rows incomplete"
        else:
            done += 1
        if why:
            pending.append((pdb, seed, why))

    allsc = "all score rows" if a.strict_scores else "scores done or terminal"
    what = f"one finished pose dir + {allsc}" if a.no_trunk else \
           f"trunk + one finished pose dir + {allsc}"
    print(f"{a.fold}: {done}/{n} units COMPLETE ({what})")
    print(f"  missing trunk                 : {no_trunk}")
    print(f"  {'no finished pose dir         ' if a.no_trunk else 'trunk, no finished pose dir  '} : {no_pose}")
    print(f"  more than one pose dir        : {bad_pose}")
    print(f"  poses, scores incomplete      : {unscored}")
    if partial:
        print(f"  of the {done} complete, {partial} have PARTIAL scores (worker reported SCORING")
        print(f"     INCOMPLETE and moved on; trunk and poses intact, some DockQ missing)")
    if a.partial_out and partial_rows:
        with open(a.partial_out, "w") as fh:
            fh.write("pdb,seed,unscored_interfaces\n")
            for pdb, seed, ifs in sorted(partial_rows):
                fh.write(f"{pdb},{seed},{ifs}\n")
        print(f"  wrote {a.partial_out} ({len(partial_rows)} rows)")
    if a.verbose:
        for pdb, seed, why in pending[:40]:
            print(f"     {pdb} {seed}: {why}")
    # exit 0 ONLY when every unit is complete: the watcher stops workers on this and nothing else
    return 0 if done == n else 1


if __name__ == "__main__":
    sys.exit(main())
