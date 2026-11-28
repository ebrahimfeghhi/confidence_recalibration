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
import json
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

def unweighted_split(rid):
    """epoch -> (good, bad) from the per-pose pae_loss_unweighted column.

    WHY NOT THE PRINTED LINE. Under --pae-tm-weight the trainer's "PAE BY POSE GROUP" line is the
    WEIGHTED loss, which is a different quantity from every unweighted arm's and sits far below it
    (epoch 1 printed acceptable 1.3060 against shipped 3.4523). The per-pose column is the plain
    unweighted mean of the same cross-entropy, so the split is rebuilt from it.

    THE POSE INDEX IS MAPPED THROUGH sample_index, not used as a sample id: 99.2% of label files
    permute the two, and reading the pose column directly is the bug that scrambled a whole day of
    val PAE splits once.
    """
    import collections
    import numpy as np
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import label_index as LI
    ACC = 0.23
    # THE POOL COMES FROM THE ARM, NOT FROM A CONSTANT. This was hardcoded to the hard 280, so a
    # full-training-set arm had its good/bad curves built from only the subset of its poses whose
    # interfaces happen to sit in that 180-structure pool -- 2,700 of 34,590 for Arm S -- while the
    # shipped lines beside them were the 280's too. Reading pae_scored_pairs out of the run's own
    # args.json makes the split cover exactly what the arm trains on, and falls back to the old
    # constant when a run predates the field.
    _pool = "training/slice280/pool.csv"
    for _d, _a in P.chain(rid):
        try:
            _j = json.load(open(os.path.join(_d, "args.json")))
            _c = (_j.get("args", _j) or {}).get("pae_scored_pairs")
        except (OSError, ValueError):
            _c = None
        if _c and os.path.exists(_c):
            _pool = _c
            break
    keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'
            for r in csv.DictReader(open(_pool))}
    dq = collections.defaultdict(list)
    for r in csv.DictReader(open("training/fold/results.csv")):
        if r.get("arm") not in (None, "free") or r["iface"] not in keep:
            continue
        try:
            dq[(r["iface"].split("_")[0], int(r["seed"]), int(r["sample"]))].append(float(r["dockq"]))
        except (ValueError, TypeError):
            continue
    cache = {}
    def samp(pdb, unit, pose):
        k = (pdb, unit)
        if k not in cache:
            q = f"training/fold/labels/{pdb}/{unit}"
            if not q.endswith(".npz"):
                q += ".npz"
            cache[k] = (LI.sample_index_array(np.load(q, allow_pickle=True))
                        if os.path.exists(q) else None)
        si = cache[k]
        return int(si[pose]) if si is not None and pose < len(si) else pose
    acc = collections.defaultdict(lambda: ([], []))
    for d, _a in P.chain(rid):
        for f in sorted(glob.glob(os.path.join(d, "trajectory_rank*.csv"))):
            for r in csv.DictReader(open(f)):
                x = r.get("pae_loss_unweighted")
                if x in (None, "", "nan"):
                    continue
                m = re.search(r"free_(\d+)", r.get("unit", "") or "")
                if not m:
                    continue
                try:
                    e, v = int(float(r["epoch"])), float(x)
                except (TypeError, ValueError):
                    continue
                vals = dq.get((r["pdb"], int(m.group(1)), samp(r["pdb"], r["unit"], int(r["pose"]))))
                if not vals:
                    continue
                if all(q >= ACC for q in vals):
                    acc[e][0].append(v)
                elif all(q < ACC for q in vals):
                    acc[e][1].append(v)
    # COMPLETE EPOCHS ONLY. Without this the mean over however many poses an in-progress epoch
    # has written is reported as that epoch's result -- epoch 2 of xv76z4 appeared at 24% written
    # and 0.026 "better" than epoch 1, which was a statement about the first quarter of a fixed
    # deal order, not about the epoch. total_loss already guards this way (modal row count), and
    # the trainer's own printed line cannot have the problem because it fires once per epoch.
    #
    # THE MODE, NOT THE MAX: the last complete epoch and the first partial one differ, and taking
    # the maximum would admit a partial epoch whenever it happened to be the longest.
    import collections as _c
    tot = {e: len(g) + len(b) for e, (g, b) in acc.items()}
    if not tot:
        return {}
    full = _c.Counter(tot.values()).most_common(1)[0][0]
    return {e: (sum(g) / len(g), sum(b) / len(b))
            for e, (g, b) in acc.items()
            if g and b and tot[e] >= 0.98 * full}


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
    # THE WHOLE RESUME CHAIN, not one directory. A resumed arm's early epochs live in its parent,
    # so globbing one run id gave the child only its own epochs -- and when the child held nothing
    # but a PARTIAL epoch, that partial row count became the modal "full" count below and the
    # epoch was drawn as complete. b59ac6 showed exactly that: 5,700 poses of epoch 3 plotted as a
    # finished epoch while the good/bad series, which read the chained log lines, correctly stopped
    # at 2. Chaining fixes both the missing epochs and the degenerate mode.
    # ONE OWNER PER EPOCH, THE NEWEST RUN THAT HAS IT. P.chain is newest-first, and a resume
    # overlap means the PARENT may hold a few rows of the epoch the CHILD then redid in full --
    # i376xz had 537 rows of epoch 14, ud3rf4 wrote all 13,503. Summed, that epoch had 14,040 rows
    # and looked longer than a complete one, which made the completeness guard below reject every
    # genuine epoch and left the curve with a single point. Resolving ownership first keeps the
    # counts honest; it also drops the parent's abandoned partial, which is what it is.
    segs = P.chain(rid)
    owner, files_by = {}, {}
    for _i, (_d, _a) in enumerate(segs):
        fs = sorted(glob.glob(os.path.join(_d, "trajectory_rank*.csv")))
        files_by[_i] = fs
        for f in fs:
            for r in csv.DictReader(open(f)):
                try:
                    e = int(float(r["epoch"]))
                except (TypeError, ValueError, KeyError):
                    continue
                owner.setdefault(e, _i)      # first (newest) wins
    files = [(i, f) for i in sorted(files_by) for f in files_by[i]]
    if not files:
        files = [(0, f) for f in (sorted(glob.glob(f"training/runs/{rid}/trajectory_rank*.csv"))
                                  or [f"training/runs/{rid}/trajectory.csv"])]
        owner = {}
    for _seg, f in files:
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            try:
                if owner and owner.get(int(float(r["epoch"])), _seg) != _seg:
                    continue
            except (TypeError, ValueError, KeyError):
                pass
            try:
                # pae_loss_unweighted WINS WHEN IT EXISTS. A row-weighted or TM-weighted arm writes
                # its trained objective to pae_loss, which is not the same quantity as an
                # unweighted arm's pae_loss; this column is, and it is absent on arms that need no
                # distinction. Without this, Arm N's total curve would be numerically below every
                # other arm's for a reason that has nothing to do with fitting better.
                _u = r.get("pae_loss_unweighted")
                _raw = _u if _u not in (None, "", "nan") else r["pae_loss"]
                e, v = int(float(r["epoch"])), float(_raw)
            except (TypeError, ValueError, KeyError):
                continue
            if e >= 1 and v == v and v != 0.0:
                acc[e].append(v)
    if not acc:
        return {}
    # THE MODE IS NOT ENOUGH ON ITS OWN. With a single epoch present, that epoch IS the mode
    # whatever its length, so a partial one passes. Taking the MAX alongside it is the fix: a real
    # epoch's row count is the largest seen, and 0.98 tolerates the handful of poses a rank can
    # drop. With two or more complete epochs the mode and the max agree and nothing changes.
    # THE MODE, PLUS A FLOOR SO A LONE PARTIAL EPOCH CANNOT BE ITS OWN MODE. With ownership
    # resolved every complete epoch has the same count, so the mode is that count; the extra
    # condition only matters when a single partial epoch is all there is.
    counts = [len(v) for v in acc.values()]
    full = max(collections.Counter(counts).items(), key=lambda kv: kv[1])[0]
    if len(acc) == 1:
        full = max(counts)
        return {}                      # one epoch, and no way to know whether it finished
    return {e: sum(v) / len(v) for e, v in acc.items() if len(v) >= 0.98 * full}



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
    # THE SHIPPED MEANS, READ ONCE AND USED AS EPOCH 0. Only prepended when the arm's range starts
    # at the beginning: an arm plotted from epoch 10 onward did not pass through shipped at 10, and
    # splicing it in would draw a descent that never happened.
    def _ship(name):
        v = os.environ.get(name, "").strip()
        if not v:
            return None
        try:
            return float(v)
        except ValueError:
            raise SystemExit(f"STOP: {name} must be a number, got {v!r}")
    _s0 = (_ship("SHIPPED_GOOD"), _ship("SHIPPED_BAD"), _ship("SHIPPED_ALL"))

    for k, (label, rid, lo, hi) in enumerate(arms):
        # WHICH SOURCE, DECIDED BY THE RUN'S OWN FLAGS. --pae-tm-weight makes the trainer's
        # printed group line the WEIGHTED loss; the per-pose pae_loss_unweighted column is the
        # comparable one, so it is rebuilt from there for those arms and parsed from the log for
        # every other.
        _tmw = False
        for _d, _ar in P.chain(rid):
            if (_ar or {}).get("pae_tm_weight"):
                _tmw = True
        gl = unweighted_split(rid) if _tmw else P.group_loss(P.chain(rid))
        if _tmw:
            print(f"  {label}: good/bad rebuilt from pae_loss_unweighted "
                  f"({len(gl)} epoch(s)); the printed line is weighted for this arm")
        e = [x for x in sorted(gl)
             if (lo is None or x >= lo) and (hi is None or x <= hi)]
        if not e:
            print(f"  {label}: no epoch in range {lo}-{hi}; skipped")
            continue
        c = COLOURS[k % len(COLOURS)]
        # EPOCH 0 IS NOT ON THE SAME MEASUREMENT AS THE ARM, WHENEVER THE ARM TRAINS WITH DROPOUT.
        #
        # The epoch-0 anchor is training/runs/shipped_baseline_{good,bad,all}, computed by
        # shipped_train_pae -- a `--loss-only` single pass with `pair_dropout: None`, i.e. DROPOUT
        # OFF. Every curve above it is the arm's TRAINING loss with --pair-dropout 0.25 ACTIVE,
        # which inflates it from the first epoch on both pose groups. Splicing the two into one
        # line drew a jump at epoch 0->1 that is partly the dropout switch and not the arm, and
        # it reads as "the arm immediately got worse than shipped".
        #
        # SO THE ANCHOR IS DROPPED FOR A DROPOUT ARM AND KEPT FOR A DROPOUT-OFF ONE, decided from
        # the run's own args.json exactly as the TM-weight source is decided just above. An arm
        # trained without dropout IS on the same measurement as the anchor and its descent from
        # shipped is real, so nothing is lost for those runs.
        _drop = 0.0
        for _d, _ar in P.chain(rid):
            try:
                _v = float((_ar or {}).get("pair_dropout") or 0.0)
            except (TypeError, ValueError):
                _v = 0.0
            if _v > 0:
                _drop = max(_drop, _v)
        _at_start = (lo is None or lo <= 1) and _drop <= 0
        if (lo is None or lo <= 1) and _drop > 0:
            print(f"  {label}: epoch-0 shipped anchor OMITTED -- this arm trains with "
                  f"pair_dropout {_drop:g} and the anchor is a dropout-off pass")

        def _pre(xs, ys, v):
            """Prepend (0, shipped) when this curve really does start from the shipped head."""
            if v is None or not _at_start or (xs and xs[0] == 0):
                return xs, ys
            return [0] + list(xs), [v] + list(ys)

        _x, _y = _pre(e, [gl[x][0] for x in e], _s0[0])
        ax[0].plot(_x, _y, "-o", color=c, lw=2, ms=4, label=f"{label} good")
        _x, _y = _pre(e, [gl[x][1] for x in e], _s0[1])
        ax[0].plot(_x, _y, "--s", color=c, lw=1.6, ms=4, alpha=.65, label=f"{label} bad")
        tl = {x: v for x, v in total_loss(rid).items()
              if (lo is None or x >= lo) and (hi is None or x <= hi)}
        et = sorted(tl)
        if et:
            _x, _y = _pre(et, [tl[x] for x in et], _s0[2])
            ax[0].plot(_x, _y, ":^", color=c, lw=1.4, ms=4, alpha=.5,
                       label=f"{label} ALL (incl. straddlers)")
        # THE EPOCH-0 POINT IS MARKED so it cannot be mistaken for a measured epoch of this arm.
        if _at_start:
            for _v in _s0:
                if _v is not None:
                    ax[0].plot([0], [_v], "o", mfc="white", mec="0.25", ms=7, zorder=5)
    # THE SHIPPED HEAD, AS TWO HORIZONTAL LINES. Every arm is judged against it, and reading that
    # off a separate table while looking at the curves is how a 0.003 move gets called a win.
    #
    # SET BY ENV, NOT HARDCODED, because the right baseline depends on which pool the arm trains
    # on -- the hard 280 and val83 have different shipped levels, and a baseline measured over a
    # DIFFERENT pose set is exactly the comparison that has to be avoided. SHIPPED_GOOD/SHIPPED_BAD
    # are the uncropped shipped means over the SAME units the arm trains on.
    # COLOURED BY POSE GROUP, NOT BY ARM: green is the ACCEPTABLE poses, red the unacceptable
    # ones, grey the combined total. The arm's own curves keep the per-arm colouring, so the
    # question "is this arm above or below shipped, and on which group" is answerable by looking
    # at one pair of same-meaning lines rather than by matching a legend entry to a colour.
    #
    # THE TOTAL IS DRAWN TOO. Without it the figure shows two groups moving in opposite directions
    # and no way to see the net -- which is the number that says whether the arm fits better at
    # all. The arm's matching total is its dotted ":^" curve.
    # NO HORIZONTAL SHIPPED LINES ANY MORE: shipped is epoch 0 of each curve above, so a rule at
    # the same height would draw each of those three numbers twice and spend three legend rows on
    # it. One faint line is kept for the TOTAL only, because the net is the number an arm is
    # ultimately judged on and it is useful to see where it crosses.
    if _s0[2] is not None:
        # THE LABEL NO LONGER SAYS "epoch 0", because for a dropout arm the anchor point is not
        # drawn and calling the line epoch 0 would promise a point that is not there. It also
        # now says dropout-off, which is the one thing a reader has to know to compare it with
        # the curves: the line is the shipped head measured WITHOUT dropout.
        ax[0].axhline(_s0[2], color="0.35", lw=1.0, ls=":", alpha=.45, zorder=0,
                      label=f"shipped ALL {_s0[2]:.4f} (dropout off)")
    # THE TOTAL IS THE SAME QUANTITY ON BOTH SIDES, AND THE FIGURE SAYS SO. The arm's dotted
    # ":^" curve and the grey shipped line are both the mean over EVERY scored pose, straddlers
    # included; the acceptable/unacceptable curves and lines both EXCLUDE straddlers, on both
    # sides. Unstated, the natural reading is that the total should sit between the two groups --
    # it does not, because the groups are unequal in size (2,731 against 9,450 on the hard 280)
    # and the straddlers belong to neither.
    #
    # NO HARDCODED STRADDLER PERCENTAGE. The old title said "~6%", which is val83's 4.9%; on the
    # hard 280 it is 9.8% (1,319 of 13,500), because half those structures carry two or more
    # interfaces and a pose can be acceptable on one and not another. STRADDLE_PCT states the live
    # pool's own figure when the caller knows it, and is omitted otherwise.
    _sp = os.environ.get("STRADDLE_PCT", "").strip()
    ax[0].set_title("PAE loss by pose group" + (f"   ({_sp} straddle)" if _sp else ""),
                    fontsize=11.5)
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
            a_.legend(h, l, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.17),
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
