#!/usr/bin/env python
"""The full-set arm, as it runs: training loss by pose group, and val whenever it was scored.

SEGMENTS, NOT ONE RUN. The arm trains in segments so scoring can land on chosen epochs, so each
stretch of epochs lives under a different run id. They are discovered from args.json -- any run
whose manifest is the full-set one -- and stitched by epoch, rather than listed by hand and going
stale the first time a segment is added.

VAL IS THE POINT OF THE PLOT. The in-sample loss falling says the optimiser works; it said that for
the previous arm too, all the way to a held-out result 6 points below shipped. The shipped line is
drawn on the val panel so a point above it is visible immediately.
"""
import collections
import csv
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

MANIFEST = "training/fullset/structures.csv"
SD = os.environ.get("SD", "/tmp/claude-1004/-home-ebrahim-confidence-recalibration/"
                          "d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad")
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/fullset_arm.png"
ACC = 0.23


def _args(rid):
    try:
        d = json.load(open(os.path.join("training/runs", rid, "args.json")))
    except (ValueError, OSError):
        return None
    return d.get("args", d)


def chain(head):
    """ONE arm's segments, newest first, by walking each segment's resume pointer.

    NOT A GLOB ON THE MANIFEST. Every arm that trains the full set matches that, so globbing it
    mixes arms. The version that did drew ONE train curve keyed by epoch, last run id
    alphabetically winning: the zsoftmax arm owned epochs 2, 5-7 and 21-22 and the PAE-only arm
    owned the rest, which put a ~0.15 step at each handover. Those steps read as training spikes
    and were two different arms on one line. A segment's `resume` names its predecessor, so the
    chain of resume pointers IS the arm, and nothing else can join it.
    """
    out, seen = [], set()
    rid = head
    while rid and rid not in seen:
        seen.add(rid)
        a = _args(rid)
        if a is None:
            break
        out.append((os.path.join("training/runs", rid), a))
        m = re.search(r"training/runs/([^/]+)/resume_ep\d+\.pt", a.get("resume") or "")
        rid = m.group(1) if m else None
    return out


def group_loss(segs):
    """epoch -> (good, bad), from the per-epoch line the trainer prints across all ranks.

    REFUSE A COLLISION rather than overwrite one. Segments of one arm train disjoint epoch
    ranges, so two values for an epoch means the caller handed in runs from two arms -- the exact
    failure this function used to absorb silently.
    """
    pat = re.compile(r"epoch (\d+) PAE BY POSE GROUP.*?acceptable ([0-9.]+) over \d+ pose\(s\), "
                     r"unacceptable ([0-9.]+)")
    got, owner = {}, {}
    for d, _ in segs:
        p = os.path.join(d, "run.log")
        if not os.path.exists(p):
            continue
        for line in open(p, errors="ignore"):
            m = pat.search(line)
            if not m:
                continue
            e, v = int(m.group(1)), (float(m.group(2)), float(m.group(3)))
            if e in got and owner[e] != d and got[e] != v:
                sys.exit(f"STOP: epoch {e} claimed by both {owner[e]} and {d} "
                         f"({got[e]} vs {v}). These are not one arm.")
            got[e], owner[e] = v, d
    return got


def ref_val():
    """epoch -> (frac acceptable, mean q) for the preserved PAE-only arm, from its reports."""
    out = {}
    for f in sorted(glob.glob("training/runs/pae_ref_scores/report_fs_ep*.txt")):
        m = re.search(r"ep(\d+)\.txt$", f)
        if not m:
            continue
        for line in open(f, errors="ignore"):
            if line.startswith("VAL (141"):
                n = re.findall(r"([0-9.]+)%\s+([0-9.]+)", line)
                if n:
                    out[int(m.group(1))] = (float(n[0][0]) / 100.0, float(n[0][1]))
                break
    return out


def _bin(cand):
    by_seed = collections.defaultdict(list)
    for sc, ss, d, seed in cand:
        by_seed[seed].append(d)
    return max(sum(1 for d in v if d >= ACC) for v in by_seed.values())


def score_points(prefix):
    """epoch -> dict of everything the panels need, from one side's scored shard dirs."""
    out = {}
    for w in sorted(glob.glob(os.path.join(SD, prefix + "*"))):
        m = re.search(r"ep(\d+)$", w)
        if not m:
            continue
        by = collections.defaultdict(list)
        for f in glob.glob(os.path.join(w, "rows_*.csv")):
            for r in csv.DictReader(open(f)):
                try:
                    by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]),
                                           float(r["dockq"]), r["seed"]))
                except (TypeError, ValueError, KeyError):
                    pass
        if len(by) < 100:
            continue
        o = {i: max(c, key=lambda t: t[0])[2] for i, c in by.items()}
        # MARGIN: score(top pick) - score(best good candidate), over every winnable item. An item
        # the head gets right contributes exactly 0, which is what makes the average go DOWN as it
        # improves rather than only describing the items it still fails.
        mg, mgs = [], []
        for i, c in by.items():
            acc = [t for t in c if t[2] >= ACC]
            if not acc:
                continue
            mg.append(max(t[0] for t in c) - max(t[0] for t in acc))
            mgs.append(max(t[1] for t in c) - max(t[1] for t in acc))
        s_ = {i: max(c, key=lambda t: t[1])[2] for i, c in by.items()}
        bins = {i: _bin(c) for i, c in by.items()}
        thin = [i for i in by if 1 <= bins[i] <= 4]
        easy = [i for i in by if bins[i] == 5]
        n = len(by)
        out[int(m.group(1))] = dict(
            n=n,
            top1=sum(1 for v in o.values() if v >= ACC) / n,
            q=sum(o.values()) / n,
            s_top1=sum(1 for v in s_.values() if v >= ACC) / n,
            s_q=sum(s_.values()) / n,
            easy=(sum(1 for i in easy if o[i] >= ACC) / len(easy)) if easy else float("nan"),
            s_easy=(sum(1 for i in easy if s_[i] >= ACC) / len(easy)) if easy else float("nan"),
            thin_q=(sum(o[i] for i in thin) / len(thin)) if thin else float("nan"),
            s_thin_q=(sum(s_[i] for i in thin) / len(thin)) if thin else float("nan"),
            margin=(sum(mg) / len(mg)) if mg else float("nan"),
            s_margin=(sum(mgs) / len(mgs)) if mgs else float("nan"),
            n_win=len(mg),
            gained=sum(1 for i in by if o[i] >= ACC and s_[i] < ACC),
            lost=sum(1 for i in by if s_[i] >= ACC and o[i] < ACC),
        )
    return out


def _unused_val_points():
    out = {}
    for w in sorted(glob.glob(os.path.join(SD, "val30_fs_ep*"))):
        m = re.search(r"ep(\d+)$", w)
        if not m:
            continue
        by = collections.defaultdict(list)
        for f in glob.glob(os.path.join(w, "rows_*.csv")):
            for r in csv.DictReader(open(f)):
                try:
                    by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]),
                                           float(r["dockq"])))
                except (TypeError, ValueError, KeyError):
                    pass
        if len(by) < 100:
            continue
        o = [max(c, key=lambda t: t[0])[2] for c in by.values()]
        s = [max(c, key=lambda t: t[1])[2] for c in by.values()]
        out[int(m.group(1))] = (sum(1 for x in o if x >= ACC) / len(o), sum(o) / len(o),
                                sum(1 for x in s if x >= ACC) / len(s), sum(s) / len(s))
    return out


def main():
    """Four panels: the loss split, then quality and top-1 on each side against shipped.

    THE EARLIER VERSION HAD EIGHT. Thin bins, easy bin, gained/lost and a text table are all
    derivable from the reports and were crowding the two curves anyone actually reads.
    """
    head = os.environ.get("RID") or open("training/runs/fullset_rid").read().strip()
    gl = group_loss(chain(head))
    ref = group_loss(chain(os.environ.get("REF_RID", "oowhyw")))
    rval = ref_val()
    # TAGP MUST MATCH THE CHAIN. Each arm scores under its own tag so the row dirs cannot
    # accumulate across arms; the plot reads the same namespace the chain wrote.
    tp = os.environ.get("TAGP", "lr4")
    val = score_points(f"val30_{tp}_ep")
    sli = score_points(f"slice280_{tp}_ep")
    if not gl and not val:
        sys.exit("STOP: no full-set segment has written an epoch yet.")
    # TWO ROWS, ONE PER SIDE. Training on top (the loss split plus the slice's two metrics),
    # held-out underneath. One metric per panel: two y-axes on one panel made the slice's curves
    # look like they crossed, which was an artefact of the scales.
    fig, axx = plt.subplots(2, 4, figsize=(21, 9))
    ax = axx.ravel()

    eps = sorted(gl)
    reps = sorted(ref)
    if reps:
        ax[0].plot(reps, [ref[e][0] for e in reps], "--", color="tab:green", lw=1.2, alpha=.55,
                   label="good, PAE-only")
        ax[0].plot(reps, [ref[e][1] for e in reps], "--", color="tab:red", lw=1.2, alpha=.55,
                   label="bad, PAE-only")
    if eps:
        ax[0].plot(eps, [gl[e][0] for e in eps], "-o", color="tab:green", ms=4, label="good")
        ax[0].plot(eps, [gl[e][1] for e in eps], "-o", color="tab:red", ms=4, label="bad")
    if eps or reps:
        ax[0].legend(fontsize=8)
    ax[0].set_title("TRAIN: loss by pose group (every epoch)", fontsize=11.5)
    ax[0].set_ylabel("mean loss per pose")

    if reps:
        ax[4].plot(reps, [ref[e][0] - ref[e][1] for e in reps], "--", color="tab:purple", lw=1.2,
                   alpha=.55, label="PAE-only")
    if eps:
        ax[4].plot(eps, [gl[e][0] - gl[e][1] for e in eps], "-o", color="tab:purple", ms=4,
                   label="this arm")
        ax[4].axhline(0, color="#555", ls="--", lw=1.4)
    if eps or reps:
        ax[4].legend(fontsize=8)
    ax[4].set_title("TRAIN: good minus bad\n(>0 = good poses fit WORSE)", fontsize=11.5)
    ax[4].set_ylabel("difference in loss")

    def panel(a_, pts, key, skey, title, ylab, pct, colour, lower_better=False):
        e = sorted(pts)
        if not e:
            a_.text(.5, .5, "not scored yet", ha="center", va="center",
                    transform=a_.transAxes, color="#666")
        else:
            y = [pts[x][key] for x in e]
            b = pts[e[0]][skey]
            a_.plot(e, y, "-o", color=colour, lw=2, ms=6, label="this arm")
            a_.axhline(b, color="#555", ls="--", lw=1.6,
                       label=f"shipped {b:.1%}" if pct else f"shipped {b:.4f}")
            lo, hi = min(y + [b]), max(y + [b])
            pad = max((hi - lo) * .3, 1e-4)
            a_.set_ylim(lo - pad, hi + pad)
            a_.set_xticks(e)
            a_.legend(fontsize=8)
        a_.set_title(title + ("\n(lower is better, 0 = always right)" if lower_better else ""),
                     fontsize=11.5)
        a_.set_ylabel(ylab)

    panel(ax[1], sli, "q", "s_q", "TRAIN SLICE: mean DockQ", "mean DockQ", False, "tab:orange")
    panel(ax[2], sli, "top1", "s_top1", "TRAIN SLICE: frac acceptable", "frac acceptable", True,
          "tab:orange")
    panel(ax[3], sli, "margin", "s_margin", "TRAIN SLICE: margin", "score gap", False,
          "tab:orange", lower_better=True)
    panel(ax[5], val, "q", "s_q", "HELD-OUT: mean DockQ", "mean DockQ", False, "tab:blue")
    panel(ax[6], val, "top1", "s_top1", "HELD-OUT: frac acceptable", "frac acceptable", True,
          "tab:blue")
    # THE REFERENCE ARM ON THE SAME AXES. Both arms against shipped is the comparison; the panel
    # limits are set from this arm, so the overlay is clipped rather than allowed to rescale it.
    for a_, j in ((ax[5], 1), (ax[6], 0)):
        if rval:
            re_ = sorted(rval)
            a_.plot(re_, [rval[e][j] for e in re_], "--s", color="#777", lw=1.3, ms=3.5,
                    alpha=.8, label="PAE-only arm")
            a_.legend(fontsize=8)
    panel(ax[7], val, "margin", "s_margin", "HELD-OUT: margin", "score gap", False, "tab:blue",
          lower_better=True)

    for a_ in ax:
        a_.grid(alpha=.3)
        a_.set_xlabel("epoch")
    fig.suptitle(f"FULL-SET ARM -- lr 1e-3, dropout 0.25   "
                 f"({len(eps)} epoch(s); val at {sorted(val) or 'none'}, "
                 f"slice at {sorted(sli) or 'none'})", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .92])
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}: {len(eps)} epoch(s), val {sorted(val)}, slice {sorted(sli)}")


if __name__ == "__main__":
    main()
