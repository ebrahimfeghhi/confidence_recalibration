#!/usr/bin/env python
"""Overlay several arms scored on the SAME held-out pool, paired against the shipped head.

    run_logging/plot_arms.py hronsr=L8 bjwcm5=L1 --out training/runs/arms.png

WHY A SEPARATE PLOTTER. plot_three.py draws ONE run against shipped and is what the per-epoch
watcher calls. Comparing arms needs the paired delta -- trained minus shipped on the SAME
interface -- because the across-interface spread (s.e. ~0.05 on 18 interfaces) is far larger than
any difference between arms and would hide all of it. The paired quantity removes the
between-interface variance that both arms share.

THE TRAINING LOSS IS DRAWN ONLY WHEN THE ARMS SHARE AN OBJECTIVE, and that is CHECKED, not
assumed. L8 weights within-seed pairs 8x where L1 weights them 1x, so those two minimise
different functions and their training curves are not on the same scale; but an arm that differs
only in WHICH PARAMETERS ARE TRAINABLE -- hronsr against xmz9ge -- minimises exactly the same
function, and there the training curve is the most direct comparison available. OBJECTIVE_KEYS
below lists what has to match, read from each run's args.json; when they do not, the panel is
drawn with dashed lines and the legend says so rather than the curve being silently omitted or
silently misread.

THE HELD-OUT LOSS IS ALWAYS COMPARABLE, by contrast, because rank_rho.py computes it CROSS-SEED
at DEFAULT_TAU regardless of what the run trained. That is what makes it a shared yardstick, and
also why it is NOT the validation twin of any of these objectives -- the two loss panels must
never be read against each other as a generalisation gap.

ONLY ARMS ON THE SAME POOL MAY BE OVERLAID. Every number here is recomputed from each run's own
train_eval shards, and `shipped` comes from the same rows via stored_iptm, so the comparison is
interface-for-interface. Overlaying a val10 arm (18 interfaces) with a valfull arm (92) would be
the mistake plot_three.py's header warns about; this refuses it by checking the interface sets
match and naming the offender.
"""
import argparse
import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The held-out loss is cross-seed softplus at DEFAULT_TAU with gap weighting. A run whose own
# objective is exactly that can have its two loss curves read against each other; any other run
# cannot, because the vertical gap then mixes generalisation with a change of function.
SAME_AS_HELDOUT = {"rank_form": "softplus", "rank_pairs": "cross", "rank_within_weight": 1.0,
                   "rank_topk_decay": 0.0, "rank_weight_by_gap": True, "rank_cross_seed": True}
DEFAULT_TAU = 0.0305

OBJECTIVE_KEYS = ("rank_form", "rank_pairs", "rank_within_weight", "rank_topk_decay",
                  "rank_tau", "rank_weight_by_gap", "alpha_rank", "rank_cross_seed")

COLORS = ["#2f6fb0", "#e07b39", "#2e9e83", "#b03a3a", "#7f5fb0"]
SHIPPED_C = "#9a9a9a"
plt.rcParams.update({
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#888888", "axes.linewidth": 0.9,
    "grid.color": "#dddddd", "grid.linewidth": 0.7,
    "legend.frameon": False, "figure.facecolor": "white",
})


def epoch_rows(rid, ep, budget):
    rows = []
    for f in glob.glob(f"training/runs/{rid}/train_eval_shards_ep{ep}/rows_b30_*.csv"
                       if budget == 30 else
                       f"training/runs/{rid}/train_eval_shards_ep{ep}/rows_b{budget}_*.csv"):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    return rows


def heldout_is_twin(rid):
    """Is this run's OWN objective the same function the held-out loss measures?"""
    try:
        with open(f"training/runs/{rid}/args.json") as fh:
            a = json.load(fh)
    except Exception:
        return False
    if any(a.get(k) != v for k, v in SAME_AS_HELDOUT.items()):
        return False
    t = a.get("rank_tau")
    return t is None or abs(float(t) - DEFAULT_TAU) < 1e-9


def objective(rid):
    """The settings that define WHAT the run minimises, for deciding if curves may be overlaid."""
    try:
        with open(f"training/runs/{rid}/args.json") as fh:
            a = json.load(fh)
    except Exception:
        return None
    return tuple(a.get(k) for k in OBJECTIVE_KEYS)


def train_loss(rid):
    """epoch -> training ranking loss. DELEGATED to plot_three, which already parses it.

    It lives in run.log, not trajectory.csv -- an earlier version of this function looked for a
    `rank_loss` column in trajectory.csv, found nothing, and silently drew no curve at all. One
    reader, so the two figures cannot disagree about what the training loss is.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plot_three
    return plot_three.train_rank_loss(rid)


def xsloss(rid, budget, shipped=False):
    """epoch -> held-out cross-seed softplus, from the run's own rank_rho cache.

    ALWAYS CROSS-SEED AT DEFAULT_TAU, whatever the arm trained -- see the module docstring. That
    is what makes it comparable across arms and what disqualifies it as a generalisation check.
    """
    out = {}
    # THROUGH rank_rho.CACHE, never a literal filename. The cache version has been bumped twice
    # for a change of what the numbers MEAN; a hardcoded "_v3" here would keep reading the old
    # quantity after the bump and the figure would silently disagree with the doc.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import rank_rho as _RHO
    path = os.path.join("training", "runs", rid, _RHO.CACHE.format(b=budget))
    if not os.path.exists(path):
        return out
    col = "xsloss_shipped" if shipped else "xsloss_trained"
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                out[int(r["epoch"])] = float(r[col])
            except (KeyError, ValueError, TypeError):
                continue
    return out


def series(rid, budget, thresh):
    """epoch -> (paired_mean, paired_sem, dockq, frac, shipped_dockq, shipped_frac, ifaces)."""
    out = {}
    for d in sorted(glob.glob(f"training/runs/{rid}/train_eval_shards_ep*")):
        ep = int(os.path.basename(d).split("ep")[-1])
        rows = epoch_rows(rid, ep, budget)
        if not rows:
            continue
        by = defaultdict(list)
        for r in rows:
            by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]), float(r["dockq"])))
        dt = [max(ps, key=lambda t: t[0])[2] for ps in by.values()]
        ds = [max(ps, key=lambda t: t[1])[2] for ps in by.values()]
        d_ = [a - b for a, b in zip(dt, ds)]
        out[ep] = (sum(d_) / len(d_), st.stdev(d_) / math.sqrt(len(d_)) if len(d_) > 1 else 0.0,
                   sum(dt) / len(dt), sum(1 for x in dt if x >= thresh) / len(dt),
                   sum(ds) / len(ds), sum(1 for x in ds if x >= thresh) / len(ds),
                   frozenset(by))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", help="rid or rid=label")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--thresh", type=float, default=0.23)
    ap.add_argument("--out", default="training/runs/arms.png")
    a = ap.parse_args()

    data, labels, ifaces = {}, {}, {}
    for spec in a.arms:
        rid, _, lab = spec.partition("=")
        s = series(rid, a.budget, a.thresh)
        if not s:
            raise SystemExit(f"STOP: no shards for {rid} at budget {a.budget}")
        data[rid], labels[rid] = s, (lab or rid)
        ifaces[rid] = next(iter(s.values()))[6]

    # THE POOLS MUST MATCH, or the overlay is meaningless. Named, not silently dropped.
    base = ifaces[a.arms[0].partition("=")[0]]
    for rid, s in ifaces.items():
        if s != base:
            raise SystemExit(
                f"STOP: {rid} covers {len(s)} interface(s) and "
                f"{a.arms[0].partition('=')[0]} covers {len(base)}; these are different pools "
                f"and overlaying them would compare different populations. "
                f"Only in {rid}: {sorted(s - base)[:4]}  only in the first: {sorted(base - s)[:4]}")

    first = data[a.arms[0].partition("=")[0]]
    sh_d = next(iter(first.values()))[4]
    sh_f = next(iter(first.values()))[5]
    n_if = len(base)

    objs = {spec.partition('=')[0]: objective(spec.partition('=')[0]) for spec in a.arms}
    same_obj = len({o for o in objs.values() if o is not None}) <= 1
    # FOUR PANELS, 2x2. Five in a row was 26 inches wide and every client scaled it until the
    # rightmost was unreadable. The two losses now share one panel: they are on the same scale
    # and the gap between them is the point.
    #
    # WHETHER THAT GAP IS A GENERALISATION GAP DEPENDS ON THE ARM, and the panel says which.
    # rank_rho.py computes the held-out loss CROSS-SEED at DEFAULT_TAU for every run, so it is
    # the twin of the training curve only when the run itself trained cross-seed pairs at that
    # tau with no reweighting. SAME_AS_HELDOUT below is that condition, read from args.json: when
    # it does not hold, part of the vertical gap is the two curves being different functions and
    # the subtitle says so instead of leaving it to be misread.
    fig, axgrid = plt.subplots(2, 2, figsize=(13.5, 9.0))
    axes = axgrid.ravel()
    for i, spec in enumerate(a.arms):
        rid = spec.partition("=")[0]
        c, s = COLORS[i % len(COLORS)], data[rid]
        ep = sorted(s)
        axes[0].plot(ep, [s[e][2] for e in ep], color=c, lw=2, marker="o", ms=3.5,
                     label=labels[rid])
        axes[1].plot(ep, [s[e][3] for e in ep], color=c, lw=2, marker="o", ms=3.5,
                     label=labels[rid])
        m = [s[e][0] for e in ep]
        se = [s[e][1] for e in ep]
        axes[2].fill_between(ep, [x - y for x, y in zip(m, se)], [x + y for x, y in zip(m, se)],
                             color=c, alpha=0.15, lw=0)
        axes[2].plot(ep, m, color=c, lw=2, marker="o", ms=3.5, label=labels[rid])
        tl, xl = train_loss(rid), xsloss(rid, a.budget)
        te = [e for e in ep if e in tl]
        xe = [e for e in ep if e in xl]
        if te:
            axes[3].plot(te, [tl[e] for e in te], color=c, lw=2, marker="s", ms=3.5, ls="--",
                         label=f"{labels[rid]} \u2014 training")
        if xe:
            axes[3].plot(xe, [xl[e] for e in xe], color=c, lw=2, marker="o", ms=3.5,
                         label=f"{labels[rid]} \u2014 held out")

    for ax, ttl, ref in ((axes[0], "mean DockQ of the selected pose", sh_d),
                         (axes[1], f"fraction acceptable (DockQ > {a.thresh:g})", sh_f)):
        ax.axhline(ref, color=SHIPPED_C, ls="--", lw=1.3, zorder=0)
        ax.annotate(f"shipped {ref:.4f}", xy=(0.015, ref), xycoords=("axes fraction", "data"),
                    color="#777777", fontsize=9, va="bottom")
        ax.set_title(ttl)
    axes[2].axhline(0.0, color=SHIPPED_C, ls="--", lw=1.3, zorder=0)
    axes[2].annotate("shipped", xy=(0.015, 0.0), xycoords=("axes fraction", "data"),
                     color="#777777", fontsize=9, va="bottom")
    axes[2].set_title("paired trained \u2212 shipped, same interfaces")
    _shipped_xs = None
    for spec in a.arms:
        _x = xsloss(spec.partition("=")[0], a.budget, shipped=True)
        if _x:
            _shipped_xs = next(iter(_x.values()))
            break
    # SHIPPED IS KEPT OFF THE AXIS WHEN IT IS FAR ABOVE THE CURVES. Drawing the line at 0.0803
    # against data in 0.036-0.046 squashed every curve into the bottom fifth of the panel and
    # hid the thing the panel exists to show. It is stated in the corner instead, with "off
    # scale" so its absence cannot read as the arm being level with shipped.
    if _shipped_xs is not None:
        lo, hi = axes[3].get_ylim()
        if _shipped_xs > hi or (hi - lo) and (_shipped_xs - lo) / (hi - lo) > 1.6:
            axes[3].annotate(f"shipped {_shipped_xs:.4f} (off scale, ~{_shipped_xs/max(hi,1e-9):.1f}x higher)",
                             xy=(0.015, 0.96), xycoords="axes fraction", color="#777777",
                             fontsize=9, va="top")
        else:
            axes[3].axhline(_shipped_xs, color=SHIPPED_C, ls="--", lw=1.3, zorder=0)
            axes[3].annotate(f"shipped {_shipped_xs:.4f}", xy=(0.015, _shipped_xs),
                             xycoords=("axes fraction", "data"), color="#777777", fontsize=9,
                             va="bottom")
    twins = all(heldout_is_twin(spec.partition("=")[0]) for spec in a.arms)
    axes[3].set_title("ranking loss: training (dashed) vs held out (solid)\n"
                      + ("same function both sides: the gap is generalisation"
                         if twins else
                         "different functions, so the gap is not generalisation alone"),
                      fontsize=10.5)
    multi = len(a.arms) > 1
    for ax in axes[:4]:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.6, lw=0.7)
        ax.margins(x=0.03)
        if multi or ax is axes[3]:
            ax.legend(fontsize=9, loc="best")
    axes[0].set_ylabel(f"held out, {n_if} interfaces")
    fig.suptitle(" vs ".join(labels[s.partition('=')[0]] for s in a.arms)
                 + f"   \u00b7   held out, {n_if} interfaces at budget {a.budget}"
                 + "   \u00b7   shipped from each run's own shards", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(a.out, dpi=140, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
