#!/usr/bin/env python
"""Is a fold queue ACTUALLY drained? Exit 0 if yes, 1 if not, printing what is outstanding.

WHY A LOG STRING IS NOT ENOUGH. folder.py prints "queue complete: N folds run, M already scored,
K failed" with K>0 included, and THEN runs a second pass over whatever is still unscored. So the
message appears while the worker is still folding, and appears when folds have failed. An arming
script that waits for that string launches onto cards the fold still holds -- the exact failure it
was written to prevent.

WHY COMPLETION IS PER SAMPLE. The first version of this file tracked which interface NAMES had
appeared in results.csv and compared the count against the pool. A unit with one of five poses
scored therefore passed, as long as every interface showed up at least once -- which is precisely
the state the 240 recovered units are in. Raised in review, and it matters most during recovery.

SO IT REUSES THE SCORER'S OWN DEFINITION rather than reimplementing it: open_csv() builds
{(iface, seed, arm) -> {sample}} and drops any row whose dockq or iptm is missing, unparseable or
non-finite, and iface_scored() requires the explicit set {0 .. nsample-1}. folder.py decides a unit
is finished with exactly that call, so this cannot drift from what the worker believes. Interface
names come from interfaces_by_pdb(), the same pool reader the worker uses.

A unit is outstanding when its trunk is missing, or when any interface of its complex is short of
nsample finite samples for that unit's own arm.

Usage:
  python -I run_logging/training_queue_drained.py
  python -I run_logging/training_queue_drained.py --queue validation/fold/queue.tsv \
      --trunks validation/fold/trunks --csv validation/fold/results.csv \
      --pool validation/fold/pool.csv
"""
import argparse, glob, os, sys

sys.path.insert(0, "/home/ebrahim/confidence_recalibration/run_model")
_argv, sys.argv = sys.argv, ["folder.py"]          # folder.py parses argv only under __main__
import folder                                                              # noqa: E402
sys.argv = _argv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_sets                                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="training/fold/queue.tsv")
ap.add_argument("--trunks", default="training/fold/trunks")
ap.add_argument("--csv", default="training/fold/results.csv")
ap.add_argument("--pool", default="training/fold/pool.csv")
ap.add_argument("--nsample", type=int, default=5)
ap.add_argument("--show", type=int, default=5)
ap.add_argument("--poses", default="training/fold/poses")
ap.add_argument("--gpu-only", action="store_true",
                help="exit 0 once no OUTSTANDING UNIT NEEDS A GPU, ignoring units that are folded "
                     "and merely unscored with their poses still on disk. That is the right "
                     "question for releasing the cards to the next fold; the rescore is CPU work "
                     "that runs alongside it.")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

# READ THE QUEUE WITH folder.read_queue, NOT A SECOND PARSER. Raised in review: this split the
# TSV itself and took field 5 literally, but the queue format allows a compact seed RANGE
# ("500000-500009") that read_queue expands through parse_seeds, and it normalises kind aliases
# and rejects malformed lines too. A range would have been read as the literal string, every glob
# for its trunk would have missed, and the gate would have reported thousands of missing trunks
# and held validation forever. The live queues happen to use literal seeds -- both parsers agree
# at 3,560 and 3,000 units -- so this was latent, not active.
units = [(u[1], u[3], str(u[4])) for u in folder.read_queue(a.queue)]   # pdb, tag/arm, seed

ifaces = folder.interfaces_by_pdb(a.pool)
rows_done, _units_done = folder.open_csv(a.csv)

# THREE CATEGORIES, BECAUSE ONLY SOME OF THEM NEED A GPU. The gate exists to stop a validation
# fold launching onto cards the training fold still needs. An unscored unit whose poses are on
# disk needs ost on a CPU, not a card, so holding three idle GPUs for it is pure waste -- but an
# unscored unit whose poses are GONE needs a refold, which is GPU work and must hold. Collapsing
# the two would either idle the cards or launch while training still has folding to do.
#   gpu_blocking : no trunk, or folded-but-unscored with no poses to rescore from
#   cpu_backlog  : folded, unscored, poses on disk -- rescoreable, does not need a card
no_trunk, rescoreable, needs_refold, ambiguous, no_ifaces = [], [], [], [], []
for pdb, arm, seed in units:
    want = sorted(ifaces.get(pdb, ()))
    if not want:
        no_ifaces.append((pdb, seed)); continue
    if not glob.glob(f"{a.trunks}/{pdb}/*_{seed}.npz"):
        no_trunk.append((pdb, seed)); continue
    short = [i for i in want
             if not folder.iface_scored(rows_done, i, seed, arm, a.nsample)]
    if not short:
        continue
    got = [len(rows_done.get((i, str(seed), arm), set())) for i in short]
    # COMPLETENESS IS PER SAVED RUN AND PER SAMPLE INDEX, not "a cif exists". Raised in review: a
    # unit with its trunk, two of five poses and incomplete scores was being called CPU-rescoreable,
    # but CPU scoring cannot recover the three missing poses -- so the gate could hand the cards
    # over while training still held units that need refolding. pose_sets.classify requires ONE
    # run directory to carry every index 0..nsample-1 with its summary_confidence json (iptm is
    # read from it), and reports ambiguity rather than picking a run.
    st, run, detail = pose_sets.classify(a.poses, pdb, seed, a.nsample)
    rec = (pdb, seed, short, got, len(want), st, detail)
    if st == "complete":
        rescoreable.append(rec)
    elif st == "ambiguous":
        ambiguous.append(rec)
    else:
        needs_refold.append(rec)

# ambiguity blocks too: two complete runs are different folds, and nothing here may
# choose between them on its own.
gpu_blocking = len(no_trunk) + len(needs_refold) + len(ambiguous) + len(no_ifaces)
cpu_backlog = len(rescoreable)
if not a.quiet:
    print(f"  queued {len(units)} unit(s), nsample {a.nsample}")
    print(f"    GPU work left -- no trunk:                 {len(no_trunk)}")
    print(f"    GPU work left -- no complete pose set:     {len(needs_refold)}")
    print(f"    GPU work left -- two complete runs:        {len(ambiguous)}")
    print(f"    GPU work left -- absent from the pool:     {len(no_ifaces)}")
    print(f"    CPU only      -- unscored, poses on disk:  {cpu_backlog}  (rescore, no refold)")
    for pdb, seed in no_trunk[:a.show]:
        print(f"       no trunk:   {pdb} seed {seed}")
    for pdb, seed, short, got, nw, st, detail in needs_refold[:a.show]:
        print(f"       refold:     {pdb} seed {seed}: {st} -- {detail}")
    for pdb, seed, short, got, nw, st, detail in ambiguous[:a.show]:
        print(f"       ambiguous:  {pdb} seed {seed}: {detail}")
    for pdb, seed, short, got, nw, st, detail in rescoreable[:a.show]:
        d = ", ".join(f"{i} {g}/{a.nsample}" for i, g in zip(short[:3], got[:3]))
        print(f"       rescore:    {pdb} seed {seed}: {d}")
    for pdb, seed in no_ifaces[:a.show]:
        print(f"       unpooled:   {pdb} seed {seed}")
    if a.gpu_only and cpu_backlog:
        print(f"    --gpu-only: the {cpu_backlog} rescoreable unit(s) do NOT hold the cards, "
              f"but they must still be rescored or that fold work is wasted")

# --gpu-only: is the training fold finished WITH THE CARDS? Everything else is CPU work that can
# run alongside the next GPU fold. Without it: is the queue complete in every respect?
sys.exit(0 if (gpu_blocking == 0 if a.gpu_only else gpu_blocking == 0 and cpu_backlog == 0) else 1)
