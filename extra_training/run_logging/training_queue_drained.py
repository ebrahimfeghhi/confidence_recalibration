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

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="training/fold/queue.tsv")
ap.add_argument("--trunks", default="training/fold/trunks")
ap.add_argument("--csv", default="training/fold/results.csv")
ap.add_argument("--pool", default="training/fold/pool.csv")
ap.add_argument("--nsample", type=int, default=5)
ap.add_argument("--show", type=int, default=5)
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

units = []
for ln in open(a.queue):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 5:
        units.append((p[1], p[3], p[4]))           # pdb, tag/arm, seed

ifaces = folder.interfaces_by_pdb(a.pool)
rows_done, _units_done = folder.open_csv(a.csv)

no_trunk, partial, no_ifaces = [], [], []
for pdb, arm, seed in units:
    want = sorted(ifaces.get(pdb, ()))
    if not want:
        no_ifaces.append((pdb, seed)); continue
    if not glob.glob(f"{a.trunks}/{pdb}/*_{seed}.npz"):
        no_trunk.append((pdb, seed)); continue
    short = [i for i in want
             if not folder.iface_scored(rows_done, i, seed, arm, a.nsample)]
    if short:
        got = [len(rows_done.get((i, str(seed), arm), set())) for i in short]
        partial.append((pdb, seed, short, got, len(want)))

if not a.quiet:
    print(f"  queued {len(units)} unit(s), nsample {a.nsample}")
    print(f"    no trunk:                        {len(no_trunk)}")
    print(f"    trunk present, scoring short:    {len(partial)}")
    print(f"    structure absent from the pool:  {len(no_ifaces)}")
    for pdb, seed in no_trunk[:a.show]:
        print(f"       no trunk: {pdb} seed {seed}")
    for pdb, seed, short, got, nw in partial[:a.show]:
        d = ", ".join(f"{i} {g}/{a.nsample}" for i, g in zip(short[:3], got[:3]))
        print(f"       short:    {pdb} seed {seed}: {len(short)} of {nw} interface(s) -- {d}")
    for pdb, seed in no_ifaces[:a.show]:
        print(f"       unpooled: {pdb} seed {seed}")

sys.exit(0 if not (no_trunk or partial or no_ifaces) else 1)
