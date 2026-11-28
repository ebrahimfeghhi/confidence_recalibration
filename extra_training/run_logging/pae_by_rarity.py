#!/usr/bin/env python
"""Is the PAE loss already falling on the poses that are RARE-good, or has it collapsed to the
majority class within each structure?

THE QUESTION. --pae-balance-accept reweights the PAE loss between acceptable and unacceptable
poses. Whether it is needed depends on what the UNBALANCED loss already does: if acceptable poses
are fit as well as unacceptable ones even where they are a small minority of a structure's poses,
balancing buys little. An arm with pae_balance_accept 0 IS that baseline.

THREE THINGS THE FIRST VERSION GOT WRONG, all found in review:

  ALL RANKS, NOT ONE. trajectory.csv is byte-identical to trajectory_rank0.csv -- the trainer
  writes it from rank 0 alone (phase_k_train.py, write_traj=is_main) -- so reading it measured a
  third of the poses and silently called it the epoch.

  PAIRED POSES, NOT PAIRED AVERAGES. Averaging each epoch over whatever poses it happened to
  contain lets a composition change masquerade as a loss change. The two epochs are now
  intersected on the exact (pdb, seed, sample) key and only poses present in BOTH are compared,
  with the dropped count reported.

  STRUCTURES, NOT POSES, ARE THE INDEPENDENT UNIT. 69 poses in the rare bucket are not 69
  independent observations; they come from far fewer structures, and the structure count is now
  printed beside the pose count.

RELATIVE AS WELL AS ABSOLUTE. A -0.23 drop from 3.58 is 6.4%; the same -0.23 from 1.4 would be
16%. The relative column is what makes buckets comparable, and it is the column that shows the
rare bucket improving LEAST.
"""
import collections
import csv
import glob
import os
import statistics as st
import sys

ACC = 0.23
BUCKETS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001))


def pose_groups():
    """(pdb, seed, sample) -> good | bad | straddle, and each structure's good fraction.

    THE UNIT OF THE LOSS IS A STRUCTURE-POSE, THE UNIT OF DockQ IS AN INTERFACE, and 82% of
    structures carry more than one. A pose is good only if ALL its interfaces are acceptable, bad
    only if NONE are; one that straddles is in neither, rather than forced into a class.
    """
    dq = collections.defaultdict(list)
    for r in csv.DictReader(open("training/fold/results.csv")):
        try:
            dq[(r["iface"].rsplit("_", 1)[0], r["seed"], r["sample"])].append(float(r["dockq"]))
        except (TypeError, ValueError, KeyError):
            continue
    grp, per_pdb = {}, collections.defaultdict(list)
    for k, v in dq.items():
        g = "good" if all(x >= ACC for x in v) else ("bad" if all(x < ACC for x in v)
                                                     else "straddle")
        grp[k] = g
        per_pdb[k[0]].append(g)
    rar = {p: sum(1 for g in v if g == "good") / len(v) for p, v in per_pdb.items()}
    return grp, rar


def losses(rid):
    """(epoch, pdb, seed, sample) -> mean pae_loss, over EVERY rank's trajectory file."""
    out = collections.defaultdict(list)
    files = sorted(glob.glob(f"training/runs/{rid}/trajectory_rank*.csv"))
    if not files:                                   # older runs wrote only the rank-0 file
        files = [f"training/runs/{rid}/trajectory.csv"]
    for f in files:
        for r in csv.DictReader(open(f)):
            try:
                e, v = int(float(r["epoch"])), float(r["pae_loss"])
            except (TypeError, ValueError, KeyError):
                continue
            if e < 1 or v != v or v == 0.0:
                continue
            out[(e, r["pdb"], r["unit"].rsplit("_", 1)[-1], r["pose"])].append(v)
    print(f"read {len(files)} rank file(s): {', '.join(os.path.basename(f) for f in files)}")
    return {k: sum(v) / len(v) for k, v in out.items()}


def main():
    rid = sys.argv[1] if len(sys.argv) > 1 else open("training/runs/pose5_arm_rid").read().strip()
    grp, rar = pose_groups()
    L = losses(rid)
    per_ep = collections.Counter(k[0] for k in L)
    if not per_ep:
        sys.exit("STOP: no pae_loss rows")
    full = max(collections.Counter(per_ep.values()).items(), key=lambda kv: kv[1])[0]
    whole = sorted(e for e, n in per_ep.items() if n >= full)
    if len(whole) < 2:
        sys.exit(f"STOP: need two complete epochs, have {whole}")
    lo, hi = whole[0], whole[-1]

    a = {k[1:]: v for k, v in L.items() if k[0] == lo}
    b = {k[1:]: v for k, v in L.items() if k[0] == hi}
    both = set(a) & set(b)
    print(f"run {rid}: epoch {lo} vs epoch {hi}, {full} poses per complete epoch")
    print(f"paired on exact (pdb, seed, sample): {len(both)} pose(s); "
          f"dropped {len(a) - len(both)} present only at ep{lo}, "
          f"{len(b) - len(both)} only at ep{hi}\n")

    print(f"{'rare-good bucket':<18} {'poses':>6} {'structs':>8} {'ep' + str(lo):>8} "
          f"{'ep' + str(hi):>8} {'delta':>9} {'rel':>7}   |  same for BAD poses")
    for g_lo, g_hi in BUCKETS:
        line = [f"{g_lo:.0%}-{g_hi:.0%}"]
        cells = []
        for cls in ("good", "bad"):
            v0, v1, pdbs = [], [], set()
            for k in both:
                pdb = k[0]
                if pdb not in rar or not (g_lo <= rar[pdb] < g_hi) or grp.get(k) != cls:
                    continue
                v0.append(a[k]); v1.append(b[k]); pdbs.add(pdb)
            if not v0:
                cells.append(None); continue
            m0, m1 = st.mean(v0), st.mean(v1)
            cells.append((len(v0), len(pdbs), m0, m1, m1 - m0, (m1 - m0) / m0))
        if cells[0] is None and cells[1] is None:
            continue
        g, bd = cells
        def fmt(c):
            return (f"{c[0]:>6} {c[1]:>8} {c[2]:>8.4f} {c[3]:>8.4f} {c[4]:>+9.4f} "
                    f"{c[5]:>+6.1%}") if c else f"{'-':>6} {'-':>8} {'-':>8} {'-':>8} {'-':>9} {'-':>7}"
        print(f"{line[0]:<18} {fmt(g)}  | {fmt(bd)}")
    print("\nGOOD poses are the left block, BAD the right. A COLLAPSE TO THE MAJORITY CLASS means "
          "the bad\nblock improves and the good one does not. The RELATIVE column is the one to "
          "compare across\nbuckets; structs is the independent unit, not poses.")


if __name__ == "__main__":
    main()
