#!/usr/bin/env python
"""PAE loss on GOOD vs BAD poses. ONE PANEL, nothing else.

THE ONLY QUESTION HERE is how the training loss splits by pose quality and how that differs
between arms. Both panels come from the trainer's own per-epoch `PAE BY POSE GROUP` line, which
exists only when --pae-split-accept is set.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collections
import csv
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import plot_fullset as P                                            # noqa: E402

COLOURS = ["tab:blue", "tab:red", "tab:green"]

def total_loss(rid):
    """epoch -> mean pae_loss over EVERY pose, from every rank's trajectory file.

    WHY THIS BELONGS BESIDE THE SPLIT. The good/bad curves drop poses that STRADDLE the threshold
    across their structure's interfaces -- about 6% here -- because one pose has ONE PAE loss and
    two contradictory labels, so it can go in neither group. This curve carries no labels, so
    nothing is dropped and the two can be read together.

    ALL RANKS. trajectory.csv is byte-identical to trajectory_rank0.csv, so reading it alone
    measures a third of the poses and calls it the epoch.

    COMPLETE EPOCHS ONLY, by requiring the modal row count: the mean over a partial epoch is
    taken over whichever steps have run, which is not the same population.
    """
    acc = collections.defaultdict(list)
    files = sorted(glob.glob(f"training/runs/{rid}/trajectory_rank*.csv")) or \
        [f"training/runs/{rid}/trajectory.csv"]
    for f in files:
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            try:
                e, v = int(float(r["epoch"])), float(r["pae_loss"])
            except (TypeError, ValueError, KeyError):
                continue
            if e >= 1 and v == v and v != 0.0:
                acc[e].append(v)
    if not acc:
        return {}
    full = max(collections.Counter(len(v) for v in acc.values()).items(),
               key=lambda kv: kv[1])[0]
    return {e: sum(v) / len(v) for e, v in acc.items() if len(v) >= full}



def main():
    out = sys.argv[1]
    # "label=rid" or "label=rid@LO-HI" to clip the epochs drawn. A RESUMED ARM SHARES ITS
    # PARENT'S EARLY EPOCHS BY CONSTRUCTION -- the decay fork replays w57hjo's 1-10 exactly -- so
    # drawing both in full puts two identical curves on top of each other and the only part that
    # differs, epochs 11-20, reads as a tail rather than as the comparison.
    arms = []
    for spec in sys.argv[2:]:
        label, rid = spec.split("=", 1)
        lo, hi = None, None
        if "@" in rid:
            rid, rng = rid.split("@", 1)
            a_, _, b_ = rng.partition("-")
            lo = int(a_) if a_ else None
            hi = int(b_) if b_ else None
        arms.append((label, rid, lo, hi))
    fig, ax0 = plt.subplots(1, 1, figsize=(8.5, 5.5))
    ax = [ax0]
    for k, (label, rid, lo, hi) in enumerate(arms):
        gl = P.group_loss(P.chain(rid))
        e = [x for x in sorted(gl)
             if (lo is None or x >= lo) and (hi is None or x <= hi)]
        if not e:
            print(f"  {label}: no epoch in range {lo}-{hi}; skipped")
            continue
        c = COLOURS[k % len(COLOURS)]
        ax[0].plot(e, [gl[x][0] for x in e], "-o", color=c, lw=2, ms=4, label=f"{label} good")
        ax[0].plot(e, [gl[x][1] for x in e], "--s", color=c, lw=1.6, ms=4, alpha=.65,
                   label=f"{label} bad")
        tl = {x: v for x, v in total_loss(rid).items()
              if (lo is None or x >= lo) and (hi is None or x <= hi)}
        et = sorted(tl)
        if et:
            ax[0].plot(et, [tl[x] for x in et], ":^", color=c, lw=1.4, ms=4, alpha=.5,
                       label=f"{label} ALL poses")
    ax[0].set_title("PAE loss by pose group\n(dotted = ALL poses, including the ~6% that straddle)", fontsize=11.5)
    _yl = os.environ.get("YLIM", "").strip()
    if _yl:
        try:
            _a, _b = (float(x) for x in _yl.split(","))
        except ValueError:
            raise SystemExit(f"STOP: YLIM must be 'lo,hi', got {_yl!r}")
        ax[0].set_ylim(_a, _b)
    ax[0].set_ylabel("mean loss per pose")
    for a_ in ax:
        a_.set_xlabel("epoch")
        a_.grid(alpha=.3)
        a_.legend(fontsize=8)
    fig.suptitle(os.environ.get("TITLE", "PAE LOSS BY POSE GROUP"), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .90])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    def _n(r, lo, hi):
        return len([x for x in P.group_loss(P.chain(r))
                    if (lo is None or x >= lo) and (hi is None or x <= hi)])

    print(f"wrote {out}: " + ", ".join(
        f"{l} {_n(r, lo, hi)} epoch(s)"
        + (f" (clipped to {lo if lo is not None else ''}-{hi if hi is not None else ''})"
           if (lo is not None or hi is not None) else "")
        for l, r, lo, hi in arms))


if __name__ == "__main__":
    main()
