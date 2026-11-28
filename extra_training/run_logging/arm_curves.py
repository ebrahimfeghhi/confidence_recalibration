#!/usr/bin/env python
"""ONE MARKDOWN TABLE PER ARM: per-epoch PAE loss by pose group AND the total, shipped on top.

WHY A SCRIPT AND NOT A HAND-ASSEMBLED TABLE. These go out beside the per-epoch figure every time
an epoch lands, and typing them by hand is how a stale number survives into a table -- it already
happened once today, when a shipped baseline measured over a narrower pose set was quoted beside
an arm's curve for an hour.

THE SHIPPED ROW IS READ FROM DISK, from the same training/runs/shipped_baseline_* files the figure
draws its lines from, so the table and the plot cannot disagree. Those files hold the UNCROPPED
shipped means over the SAME units the arms train on, grouped by the POOL's interfaces only --
which is what makes them comparable to the trainer's own good/bad split.

ARMS ARE NAMED BY WHAT THEY ARE, not by run id alone: the id is the second thing in the heading,
because "q4x6wn beats pt8f98" is unreadable a week later.
"""
import csv
import glob
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)
import plot_fullset as P                                            # noqa: E402
# total_loss LIVES IN THE PLOTTER, not in plot_fullset, and it is imported rather than
# reimplemented so the table's ALL column is the identical quantity the figure's dotted
# curve draws -- including its 'complete epochs only' rule, which drops an epoch whose
# row count is short so a partial epoch's mean is never shown as the epoch's.
from plot_group_loss_arms import total_loss, unweighted_split                          # noqa: E402


def _read(p):
    try:
        return float(open(p).read().strip())
    except Exception:
        return None


def tm_table(rid, label):
    """Arm M's OWN objective per epoch: the centred TM error and the mean predicted pair score.

    WHY IT NEEDS ITS OWN TABLE. The PAE cross-entropy table says nothing about this arm -- it does
    not train that term, and the figure plots it only so the arm stays comparable with every other
    one. The centred TM error is what is being descended, and it is printed per epoch per rank in
    the run log rather than written to a csv, so it is parsed from there.

    THE MEAN PREDICTED PAIR SCORE IS REPORTED BESIDE IT because the loss cannot see it: centring
    removes any per-pair constant, so a head that drives every s_pred to 0 or to 1 can still show a
    small centred loss. A value drifting to either end means the objective is being satisfied by
    collapse rather than by ordering.
    """
    import re
    rows = {}
    for d, _a in P.chain(rid):
        f = os.path.join(d, "run.log")
        if not os.path.exists(f):
            continue
        for line in io.open(f, errors="replace"):
            m = re.search(r"epoch (\d+) CENTRED TM ERROR: ([0-9.]+) over (\d+) pose\(s\).*?"
                          r"centred against ([0-9.]+) pose\(s\).*?"
                          r"mean predicted pair score ([0-9.]+)", line)
            if m:
                rows[int(m.group(1))] = (float(m.group(2)), int(m.group(3)),
                                         float(m.group(4)), float(m.group(5)))
    if not rows:
        print(f"**{label} — own objective** (`{rid}`) — no epoch line yet\n")
        return
    print(f"**{label} — OWN objective (centred TM error)** (`{rid}`)\n")
    print("| | centred TM error | mean predicted pair score | poses | centred against |")
    print("|---|---|---|---|---|")
    print("| shipped | not measured | not measured | -- | -- |")
    for e in sorted(rows):
        v, n, c, ps = rows[e]
        print(f"| epoch {e} | {v:.6f} | {ps:.4f} | {n} | {c:.1f} |")
    print()


def table(rid, label):
    gl = P.group_loss(P.chain(rid))
    if not gl:
        print(f"**{label}** (`{rid}`) — no epoch yet\n")
        return
    sg = _read("training/runs/shipped_baseline_good")
    sb = _read("training/runs/shipped_baseline_bad")
    sa = _read("training/runs/shipped_baseline_all")
    # THE TOTAL IS NOT good+bad AVERAGED. It is the mean over EVERY scored pose, straddlers
    # included -- 3.5% of poses on this pool belong to neither group because their structure has
    # several interfaces and they are acceptable on some and not others. So the total can sit
    # outside the interval between the two groups, and plot_fullset.total_loss is the same
    # quantity the figure's dotted curve draws.
    tl = total_loss(rid)
    print(f"**{label}** (`{rid}`)\n")
    print("| | acceptable | unacceptable | ALL poses |")
    print("|---|---|---|---|")
    if sg is not None and sb is not None:
        print(f"| shipped | {sg:.4f} | {sb:.4f} | "
              f"{f'{sa:.4f}' if sa is not None else '--'} |")
    for e in sorted(gl):
        _t = tl.get(e)
        print(f"| epoch {e} | {gl[e][0]:.4f} | {gl[e][1]:.4f} | "
              f"{f'{_t:.4f}' if _t is not None else '--'} |")
    print()


def main():
    if len(sys.argv) > 1:
        for spec in sys.argv[1:]:
            lab, _, rid = spec.partition("=")
            table(rid or lab, lab)
        return
    # NO ARGUMENTS: the two arms of the live comparison, each from its own rid file, skipping one
    # that has not started yet rather than printing an empty heading.
    for f, lab in (("training/runs/armL_rid", "Arm L — native head, vanilla PAE"),
                   ("training/runs/armM_rid", "Arm M — native head, TM-weighted PAE")):
        try:
            rid = open(f).read().strip()
        except Exception:
            rid = ""
        if rid:
            if "TM-weighted" in lab:
                gl = unweighted_split(rid)
                print(f"**{lab}** (`{rid}`) -- UNWEIGHTED, rebuilt from the per-pose column\n")
                print("| | acceptable | unacceptable | ALL poses |")
                print("|---|---|---|---|")
                _sh = [(open(f"training/runs/shipped_baseline_{k}").read().strip()
                        if os.path.exists(f"training/runs/shipped_baseline_{k}") else "--")
                       for k in ("good", "bad", "all")]
                print(f"| shipped | {_sh[0]} | {_sh[1]} | {_sh[2]} |")
                tl = total_loss(rid)
                for e in sorted(gl):
                    t = f"{tl[e]:.4f}" if e in tl else "--"
                    print(f"| epoch {e} | {gl[e][0]:.4f} | {gl[e][1]:.4f} | {t} |")
                print()
                continue
            table(rid, lab)
            if "centred TM" in lab:
                tm_table(rid, lab)


if __name__ == "__main__":
    main()
