#!/usr/bin/env python
"""Is the PAE loss already falling on the poses that are RARE-good, or has it collapsed to the
majority class within each structure?

THE QUESTION THIS ANSWERS. --pae-balance-accept reweights the PAE loss between acceptable and
unacceptable poses. Whether that term is needed depends on what the UNBALANCED loss is already
doing: if acceptable poses are being fit as well as unacceptable ones even where they are a small
minority of a structure's poses, balancing buys little; if the loss on them is flat or rising while
the majority class falls, balancing is the priority. This arm runs with pae_balance_accept 0, so it
IS the unbalanced baseline and the question is directly measurable rather than hypothetical.

THE UNIT OF THE LOSS IS A STRUCTURE-POSE, THE UNIT OF DockQ IS AN INTERFACE. trajectory.csv logs
pae_loss per (pdb, seed, sample); a structure can carry several interfaces and 82% do, so a pose
can be acceptable for one and not another. Poses are grouped the way the trainer groups them: all
of its interfaces acceptable -> good, none -> bad, otherwise STRADDLE and excluded from both,
rather than forced into one.

RARITY IS PER STRUCTURE, over the structure's own scored poses: the fraction of its poses that are
good. "Rare-good" is the bottom bucket, where a majority-class collapse would show first.
"""
import collections
import csv
import sys

ACC = 0.23
BUCKETS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001))


def main():
    rid = sys.argv[1] if len(sys.argv) > 1 else open("training/runs/pose5_arm_rid").read().strip()
    # DockQ per (pdb, seed, sample, iface)
    dq = collections.defaultdict(list)
    for r in csv.DictReader(open("training/fold/results.csv")):
        try:
            pdb = r["iface"].rsplit("_", 1)[0]
            dq[(pdb, r["seed"], r["sample"])].append(float(r["dockq"]))
        except (TypeError, ValueError, KeyError):
            continue
    # group each pose, then rarity per structure
    grp, per_pdb = {}, collections.defaultdict(list)
    for k, v in dq.items():
        g = "good" if all(x >= ACC for x in v) else ("bad" if all(x < ACC for x in v)
                                                     else "straddle")
        grp[k] = g
        per_pdb[k[0]].append(g)
    rar = {p: sum(1 for g in v if g == "good") / len(v) for p, v in per_pdb.items()}

    # pae_loss per pose, split into the first and last complete epoch seen
    rows = collections.defaultdict(list)
    eps = set()
    for r in csv.DictReader(open(f"training/runs/{rid}/trajectory.csv")):
        try:
            e = int(float(r["epoch"]))
            v = float(r["pae_loss"])
        except (TypeError, ValueError, KeyError):
            continue
        if e < 1 or v != v or v == 0.0:
            continue
        eps.add(e)
        rows[(e, r["pdb"], r["unit"].rsplit("_", 1)[-1], r["pose"])].append(v)
    if not eps:
        sys.exit("STOP: no pae_loss rows")
    lo, hi = min(eps), max(eps)
    # the last epoch may be partial; step back one so both windows are whole epochs
    hi = hi - 1 if hi > lo else hi

    print(f"run {rid}: PAE loss per pose, epoch {lo} vs epoch {hi}, "
          f"grouped at DockQ {ACC}\n")
    print(f"{'rare-good bucket':<20} {'n good':>7} {'n bad':>7} "
          f"{'good ep' + str(lo):>11} {'good ep' + str(hi):>11} {'delta':>9}   "
          f"{'bad ep' + str(lo):>10} {'bad ep' + str(hi):>10} {'delta':>9}")
    for a, b in BUCKETS:
        acc = {("good", lo): [], ("good", hi): [], ("bad", lo): [], ("bad", hi): []}
        for (e, pdb, seed, pose), v in rows.items():
            if e not in (lo, hi) or pdb not in rar or not (a <= rar[pdb] < b):
                continue
            g = grp.get((pdb, seed, pose))
            if g in ("good", "bad"):
                acc[(g, e)].append(sum(v) / len(v))
        if not acc[("good", lo)] and not acc[("bad", lo)]:
            continue
        m = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
        print(f"{f'{a:.0%}-{b:.0%} good':<20} {len(acc[('good', lo)]):>7} "
              f"{len(acc[('bad', lo)]):>7} {m[('good', lo)]:>11.4f} {m[('good', hi)]:>11.4f} "
              f"{m[('good', hi)] - m[('good', lo)]:>+9.4f}   {m[('bad', lo)]:>10.4f} "
              f"{m[('bad', hi)]:>10.4f} {m[('bad', hi)] - m[('bad', lo)]:>+9.4f}")
    print("\nA COLLAPSE TO THE MAJORITY CLASS looks like: in the bottom bucket the bad-pose delta "
          "is clearly\nnegative while the good-pose delta is flat or positive. Balancing is cheap "
          "insurance if the\ntwo deltas are similar, and the priority if they are not.")


if __name__ == "__main__":
    main()
