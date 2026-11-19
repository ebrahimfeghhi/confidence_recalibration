#!/usr/bin/env python
"""PAE loss on GOOD vs BAD poses. ONE PANEL, nothing else.

THE ONLY QUESTION HERE is how the training loss splits by pose quality and how that differs
between arms. Both panels come from the trainer's own per-epoch `PAE BY POSE GROUP` line, which
exists only when --pae-split-accept is set.
"""
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collections
import csv
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import plot_fullset as P                                            # noqa: E402

# GREY FOR THE REFERENCE ARM, BLUE FOR THE NEW ONE. The arms are drawn predecessor-first, so
# the one being judged is the one that stands out; red read as "bad" rather than "newer".
COLOURS = ["0.55", "tab:blue", "tab:green"]


_RANK_RE = re.compile(r"\[rank 0\] epoch (\d+) rank-loss [^:]*:\s*([+-]?[0-9.]+)")


def rank_loss(rid):
    """epoch -> the arm's mean ranking loss, read from its own run.log.

    WHY NOT THE TRAJECTORY. trajectory_rank0.csv has one row per POSE and records the PAE term
    only; the rel term is a per-interface-group quantity formed at step level, so `loss` there
    equals `pae_loss` exactly even when ranking is on and contributing a real gradient. Reading
    the trajectory would therefore report a ranking loss of zero for an arm that has one.
    """
    out = {}
    for d, _a in P.chain(rid):
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d, "run.log")
        if not os.path.exists(f):
            continue
        for line in io.open(f, errors="replace"):
            m = _RANK_RE.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out

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
    # A SECOND PANEL ONLY IF AN ARM ACTUALLY HAS A RANKING TERM. A PAE-only sweep should not
    # grow an empty axes, and an arm with alpha 0 has no line to draw.
    _rk = {rid: rank_loss(rid) for _, rid, _, _ in arms}
    _any_rank = any(_rk.values())
    if _any_rank:
        fig, _axs = plt.subplots(1, 2, figsize=(13.0, 6.0))
        ax = list(_axs)
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(8.5, 6.0))
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
    if _any_rank:
        for k, (label, rid, lo, hi) in enumerate(arms):
            r = {e: v for e, v in _rk.get(rid, {}).items()
                 if (lo is None or e >= lo) and (hi is None or e <= hi)}
            if not r:
                continue
            er = sorted(r)
            ax[1].plot(er, [r[e] for e in er], "-o", color=COLOURS[k % len(COLOURS)],
                       lw=2, ms=4, label=f"{label} rank-loss")
        ax[1].set_title("HELD-OUT RANKING LOSS term\n(relative regression on true iTM, "
                        "mean-centred per interface)", fontsize=11.5)
        ax[1].set_ylabel("mean rank-loss per interface group")
    for a_ in ax:
        a_.set_xlabel("epoch")
        a_.grid(alpha=.3)
        # LEGEND BELOW THE AXES, NOT INSIDE THEM. Every curve here falls from left to right and
        # the good/bad/ALL bands sit at different heights, so there is no corner that stays empty
        # for the whole run -- an in-axes legend covered the early epochs of whichever arm started
        # highest, which is the part being compared. Outside the axes it cannot overlap at all,
        # and the number of entries grows with the number of arms, so it is laid out in columns.
        # ONE COLUMN PER ARM, BELOW THE AXES. The labels arrive grouped by arm (good, bad,
        # ALL for arm 0, then arm 1), so ncol = number of arms puts each arm in its own column
        # and each pose group on its own row: the legend reads as the grid it is.
        #
        # NO LEGEND TITLE, AND NO AUTOMATIC LAYOUT. A title row rendered at the same height as
        # the "epoch" xlabel and the two overprinted each other ("Arm E (73rf4v)epochrm F");
        # and the automatic rect reserved a band for the legend while the legend itself is
        # positioned in AXES coordinates, so the reserved band stayed empty below it. The
        # margins are set explicitly instead, which is the only way the two agree.
        h, l = a_.get_legend_handles_labels()
        if l:
            per = 3 if any(x.endswith("ALL poses") for x in l) else 2
            a_.legend(h, l, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                      ncol=max(1, len(l) // per), frameon=False,
                      handlelength=2.2, columnspacing=2.5, borderaxespad=0.)
    fig.suptitle(os.environ.get("TITLE", "PAE LOSS BY POSE GROUP"), fontsize=13)
    fig.subplots_adjust(top=.86, bottom=.28, left=.09, right=.97)
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
