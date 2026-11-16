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
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_dirs as ED                                               # noqa: E402
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


def _rank(xs):
    """Ranks with ties averaged, so a tied block cannot order itself by file order."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None                      # a flat column has no correlation, not a zero one
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / (va ** 0.5 * vb ** 0.5)


def _spearman(xs, ys):
    return _pearson(_rank(xs), _rank(ys))


def _bin(cand):
    by_seed = collections.defaultdict(list)
    for sc, ss, d, seed in cand:
        by_seed[seed].append(d)
    return max(sum(1 for d in v if d >= ACC) for v in by_seed.values())


def work_dirs(tag, pool_tag, segs):
    """epoch -> shard directory for one side, from BOTH harnesses.

    TWO PLACES, ONE CURVE. shell_scripts/score_checkpoint.sh writes $SD/<side>_<tag>_ep<N>/ for an
    epoch scored between segments; the trainer's own --eval-pool trigger writes
    training/runs/<rid>/<pool>_eval_shards_ep<N>/ for an epoch scored in process. An arm that
    switches from one to the other mid-flight has its early epochs in the first and the rest in
    the second, and a reader that knows only one of them silently truncates the curve at the
    switch. eval_dirs.chain_shard_dirs is the in-run side's own reader, and it RAISES rather than
    guess when two pools wrote one epoch, which is why the pool tag is named here.
    """
    out = {}
    for w in sorted(glob.glob(os.path.join(SD, tag + "*"))):
        m = re.search(r"ep(\d+)$", w)
        if m and os.path.isdir(w):
            out[int(m.group(1))] = w
    rids = [os.path.basename(d) for d, _ in segs]
    inrun = ED.chain_shard_dirs(list(reversed(rids)), "training/runs", tag=pool_tag)
    # THE IN-PROCESS PASS WINS a shared epoch. The two are the same measurement, so a collision is
    # a re-score rather than a conflict; preferring the in-run one keeps the newer harness
    # authoritative once an arm has switched to it.
    out.update(inrun)
    return out


def score_points(work):
    """epoch -> dict of everything the panels need, from {epoch: shard dir}."""
    out = {}
    for e, w in sorted(work.items()):
        by = collections.defaultdict(list)
        rows_f = sorted(glob.glob(os.path.join(w, "rows_*.csv")))
        metas = glob.glob(os.path.join(w, "rows_*.csv.meta.json"))
        for f in rows_f:
            for r in csv.DictReader(open(f)):
                try:
                    by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]),
                                           float(r["dockq"]), r["seed"]))
                except (TypeError, ValueError, KeyError):
                    pass
        # EVERY SHARD, OR NO POINT. The old gate was `len(by) < 100`, which a directory still
        # being written can pass: the pass is sharded three ways and each shard writes its
        # rows_*.csv and then its .meta.json, so a dir read between those is short by a third of
        # its interfaces -- and a point drawn from two shards is a DIFFERENT subset of the pool,
        # which moves the mean without looking wrong. Require the meta marker from all three
        # shards and the same candidate count on every interface, and say out loud what is being
        # skipped rather than dropping it silently.
        if not by:
            continue
        cand = {len(v) for v in by.values()}
        if len(rows_f) != 3 or len(metas) != 3 or len(cand) != 1:
            print(f"  SKIP ep{e}: {len(rows_f)} rows file(s), {len(metas)} meta(s), "
                  f"{len(by)} interface(s), candidates per interface {sorted(cand)} "
                  f"-- not a complete 3-shard pass", file=sys.stderr)
            continue
        # CORRELATION, WITHIN AN ITEM AND POOLED. The job is to rank the candidates of ONE
        # interface, so the mean per-interface Spearman is the one that matches the task; the
        # pooled number is mostly between-interface variance (an easy target scores high on
        # everything) and reads far better than the head deserves. Both are drawn.
        rh, srh, px, spx, py = [], [], [], [], []
        for c in by.values():
            dq = [t[2] for t in c]
            r1 = _spearman([t[0] for t in c], dq)
            r2 = _spearman([t[1] for t in c], dq)
            if r1 is not None:
                rh.append(r1)
            if r2 is not None:
                srh.append(r2)
            px += [t[0] for t in c]
            spx += [t[1] for t in c]
            py += dq
        pr, spr = _spearman(px, py), _spearman(spx, py)
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
        out[e] = dict(
            n=n,
            rho=(sum(rh) / len(rh)) if rh else float("nan"),
            s_rho=(sum(srh) / len(srh)) if srh else float("nan"),
            n_rho=len(rh),
            prho=pr if pr is not None else float("nan"),
            s_prho=spr if spr is not None else float("nan"),
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


def report_points(d, which):
    """epoch -> the four numbers a report_*.txt carries, for an arm with no rows left.

    ROWS ARE THE PREFERRED SOURCE and score_points stays the one that feeds the figure. An arm
    whose shard dirs were not preserved can still be drawn from its reports, but only for what
    the report prints: top-1 and mean q against shipped. Margin and the ipTM-vs-DockQ correlation
    are per-candidate quantities, so they are left absent rather than faked, and their panels say
    so.
    """
    pat = re.compile(r"^" + which + r"\s*\([^)]*\)\s+([0-9.]+)%\s+([0-9.]+)\s+([0-9.]+)%"
                     r"\s+([0-9.]+)", re.M)
    out = {}
    for f in glob.glob(os.path.join(d, "report_*_ep*.txt")):
        m = re.search(r"_ep(\d+)\.txt$", f)
        if not m:
            continue
        hit = pat.search(io.open(f, encoding="utf-8", errors="ignore").read())
        if not hit:
            continue
        t1, q, s1, sq = (float(x) for x in hit.groups())
        out[int(m.group(1))] = dict(n=0, top1=t1 / 100, q=q, s_top1=s1 / 100, s_q=sq,
                                    margin=float("nan"), s_margin=float("nan"),
                                    rho=float("nan"), s_rho=float("nan"))
    return out


def main():
    """Four panels: the loss split, then quality and top-1 on each side against shipped.

    THE EARLIER VERSION HAD EIGHT. Thin bins, easy bin, gained/lost and a text table are all
    derivable from the reports and were crowding the two curves anyone actually reads.
    """
    head = os.environ.get("RID") or open("training/runs/fullset_rid").read().strip()
    gl = group_loss(chain(head))
    # ONE ARM PER FIGURE when REF_RID is empty. Two arms on one set of axes is the right picture
    # for "did the change help"; it is the wrong one for reading an arm's own shape, because the
    # panel limits are set from this arm and the overlay is clipped to them. Both arms keep their
    # own figure by pointing SD at that arm's preserved work dirs and naming its tag -- the 1e-3
    # arm's rows are under training/runs/pae_ref_scores, so its figure is drawn from the same
    # rows_*.csv as this one and not from its reports.
    _rr = os.environ.get("REF_RID", "oowhyw").strip()
    ref = group_loss(chain(_rr)) if _rr else {}
    rval = ref_val() if _rr else {}
    # TAGP MUST MATCH THE CHAIN. Each arm scores under its own tag so the row dirs cannot
    # accumulate across arms; the plot reads the same namespace the chain wrote.
    tp = os.environ.get("TAGP", "lr4")
    segs = chain(head)
    rep = os.environ.get("REPORTS", "").strip()
    if rep:
        # ROWS WIN WHERE THEY EXIST. An arm part-way through a re-score has reports for every
        # epoch it was ever scored at and rows for only the epochs re-run, so the two sources are
        # merged rather than chosen between: q and top-1 come from all of them, margin and rho
        # from the re-scored subset, and the panels draw whichever epochs carry the metric.
        val, sli = report_points(rep, "VAL"), report_points(rep, "TRAIN SLICE")
        val.update(score_points(work_dirs(f"val30_{tp}_ep", "valfull83", segs)))
        sli.update(score_points(work_dirs(f"slice280_{tp}_ep", "slice280", segs)))
    else:
        val = score_points(work_dirs(f"val30_{tp}_ep", "valfull83", segs))
        sli = score_points(work_dirs(f"slice280_{tp}_ep", "slice280", segs))
    if not gl and not val:
        sys.exit("STOP: no full-set segment has written an epoch yet.")
    # TWO ROWS, ONE PER SIDE. Training on top (the loss split plus the slice's two metrics),
    # held-out underneath. One metric per panel: two y-axes on one panel made the slice's curves
    # look like they crossed, which was an artefact of the scales.
    fig, axx = plt.subplots(2, 5, figsize=(26, 9))
    # THE HISTORICAL NUMBERING IS KEPT. ax[0..3] is the training row and ax[4..7] the held-out
    # row, as they were when the grid was 2x4; the correlation pair is appended as ax[8]/ax[9]
    # so adding a column did not silently renumber every panel below.
    top, bot = axx[0], axx[1]
    ax = [top[0], top[1], top[2], top[3], bot[0], bot[1], bot[2], bot[3], top[4], bot[4]]

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
            e = []
        e = [x for x in e if pts[x][key] == pts[x][key]] or e
        if all(pts[x][key] != pts[x][key] for x in e):
            # NOT FAKED, ABSENT. A reports-only arm has no per-candidate rows, so margin and rho
            # cannot be computed for it; say that on the panel instead of drawing a flat zero.
            a_.text(.5, .5, "not in the reports\n(no rows preserved)", ha="center", va="center",
                    transform=a_.transAxes, color="#666")
        else:
            y = [pts[x][key] for x in e]
            b = next(pts[x][skey] for x in e if pts[x][skey] == pts[x][skey])
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
    panel(ax[8], sli, "rho", "s_rho", "TRAIN SLICE: ipTM vs DockQ\n(mean Spearman within an "
          "interface)", "rho", False, "tab:orange")
    panel(ax[9], val, "rho", "s_rho", "HELD-OUT: ipTM vs DockQ\n(mean Spearman within an "
          "interface)", "rho", False, "tab:blue")

    for a_ in ax:
        a_.grid(alpha=.3)
        a_.set_xlabel("epoch")
    fig.suptitle(os.environ.get("TITLE", "FULL-SET ARM") + "   "
                 + f"({len(eps)} epoch(s); val at {sorted(val) or 'none'}, "
                 f"slice at {sorted(sli) or 'none'})", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .92])
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}: {len(eps)} epoch(s), val {sorted(val)}, slice {sorted(sli)}")


if __name__ == "__main__":
    main()
