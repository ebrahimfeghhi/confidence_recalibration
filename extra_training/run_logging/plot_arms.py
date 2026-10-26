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
import re
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TWO HELD-OUT LOSS COLUMNS, AND THEY ANSWER DIFFERENT QUESTIONS.
#   xsloss_*   each run's OWN objective, so it is the twin of that run's training curve. The gap
#              between them is generalisation. NOT comparable between arms training different
#              functions.
#   xscanon_*  cross-seed softplus at DEFAULT_TAU for every arm regardless of what it trained.
#              Comparable across arms; not the twin of anything that trained something else.
# This plotter uses the first against the training curve and the second when overlaying arms, so
# neither question borrows the other's number. An earlier version hardcoded the canonical
# settings for both, which mislabelled every within-seed arm's loss panel as "different
# functions" after rank_rho started computing the own-objective column.

OBJECTIVE_KEYS = ("rank_form", "rank_pairs", "rank_within_weight", "rank_topk_decay",
                  "rank_tau", "rank_weight_by_gap", "alpha_rank", "rank_cross_seed")

COLORS = ["#2f6fb0", "#e07b39", "#2e9e83", "#b03a3a", "#7f5fb0"]
SHIPPED_C = "#9a9a9a"


def tint(c, f=0.55):
    """Lighten a hex colour toward white by fraction f.

    THE TWO LOSS CURVES NEED TO DIFFER BY COLOUR, not only by dash pattern. Drawn in one colour
    with solid/dashed as the only cue, their legend handles were indistinguishable at the size
    the figure renders, so "training" and "held out" could not be told apart -- which is the one
    thing that panel exists to show. Tinting keeps each arm identifiable by hue when several are
    overlaid, while separating its two curves within that hue.
    """
    c = c.lstrip("#")
    r, g, b = (int(c[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02x%02x%02x" % tuple(int(v + (255 - v) * f) for v in (r, g, b))
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


def heldout_is_twin(rid, budget):
    """Does this run's cache carry the own-objective held-out loss?

    Since rank_rho v4 the xsloss_* column IS the run's own objective, so for any run whose cache
    has been rebuilt the answer is yes and the training-vs-held-out gap is generalisation. A
    cache predating v4 holds the canonical loss instead, and for an arm that trained something
    else that gap would mix generalisation with a change of function -- so this checks the cache
    actually in use rather than assuming.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # THE HIGHEST CACHE PRESENT, not the current CACHE constant. Pinning to the constant meant
    # that bumping it to v6 turned every existing run's answer to False overnight, so a v5 cache
    # -- which DOES hold the own-objective loss -- was plotted under the "CANONICAL cross-seed"
    # subtitle. v4 is the version at which the column became the run's own objective.
    path, ver = _cache_path(rid, budget)
    return bool(path) and ver >= 4


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
    out = {}
    for r in plot_three.run_chain(rid):
        out.update(plot_three.train_rank_loss(r))
    return out


def pae_loss(rid):
    """(epochs, train PAE, held-out PAE) over the chain, for a PAE-trained arm.

    WHY THIS PANEL EXISTS BESIDE THE RANKING ONE. plot_arms was written for ranking arms and its
    fourth panel draws the ranking loss unconditionally. Both PAE arms have `rank_loss: false`
    and log no rank-loss line at all, so on them that panel drew NO training curve and, for the
    held-out side, the canonical cross-seed ranking loss -- a yardstick neither arm minimises --
    which reads as a loss comparison while being nothing of the kind. alpha_pae is 1.0 in both,
    so the PAE loss IS the shared objective and the two sides of it are twins.
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plot_three
    return plot_three.pae_curves(rid)


def trained_pae_not_rank(rid):
    """True when this arm minimises PAE and not a ranking loss, read from the chain's head args."""
    try:
        with open(f"training/runs/{_chain(rid)[0]}/args.json") as fh:
            a = json.load(fh)
    except Exception:
        return False
    return (not a.get("rank_loss")) and float(a.get("alpha_pae") or 0.0) > 0.0


def _cache_path(rid, budget):
    """(path, version) of the HIGHEST rank_rho cache version this run has, or (None, None).

    A run stopped before a version bump never gets the new cache unless it is rescored, and
    reading only the current version drew zg5fnw's held-out curve as nothing at all. The version
    comes back with the path because what the columns MEAN depends on it.
    """
    best = (None, None)
    for f in glob.glob(os.path.join("training", "runs", rid, f"rank_rho_b{budget}_v*.csv")):
        m = re.search(r"_v(\d+)\.csv$", f)
        if m and (best[1] is None or int(m.group(1)) > best[1]):
            best = (f, int(m.group(1)))
    return best


def xsloss(rid, budget, shipped=False, canonical=False):
    """epoch -> held-out ranking loss, from the run's own rank_rho cache.

    TWO QUANTITIES LIVE IN ONE COLUMN NAME ACROSS THE v4 BOUNDARY. Before v4 the xsloss_* column
    was always cross-seed at DEFAULT_TAU -- the canonical loss. From v4 it is the run's OWN
    objective, and v5 added a separate canonical column beside it. So for a pre-v4 cache the
    canonical request is served by the old column, and the own-objective request is served by it
    ONLY when the run's objective happens to be the canonical one; otherwise there is no such
    number on disk and this returns empty rather than the wrong curve.
    """
    out = {}
    # THE WHOLE CHAIN, same reason as series(). Each member caches only its own epochs.
    for _r in _chain(rid):
        out.update(_xsloss_one(_r, budget, shipped=shipped, canonical=canonical))
    return out


def _xsloss_one(rid, budget, shipped=False, canonical=False):
    out = {}
    path, ver = _cache_path(rid, budget)
    if not path:
        return out
    if ver >= 5:
        col = ("xscanon_shipped" if shipped else "xscanon_trained") if canonical \
            else ("xsloss_shipped" if shipped else "xsloss_trained")
    elif ver == 4:
        # own-objective only; no canonical column existed yet
        if canonical and not _is_canonical_objective(rid):
            return out
        col = "xsloss_shipped" if shipped else "xsloss_trained"
    else:
        # pre-v4: the column IS the canonical loss
        if not canonical and not _is_canonical_objective(rid):
            return out
        col = "xsloss_shipped" if shipped else "xsloss_trained"
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                out[int(r["epoch"])] = float(r[col])
            except (KeyError, ValueError, TypeError):
                continue
    return out


def _is_canonical_objective(rid):
    """True when the run trained exactly the canonical loss: cross-seed softplus at DEFAULT_TAU.

    Only then are the own-objective and canonical held-out curves the same number, which is what
    lets a pre-v4 cache answer for both.
    """
    try:
        with open(f"training/runs/{rid}/args.json") as fh:
            a = json.load(fh)
    except Exception:
        return False
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "phases"))
    # rank_loss, NOT rank_rho: rank_rho has no module-level DEFAULT_TAU -- it reaches it through a
    # function-local `import rank_loss` every time. Reading it off rank_rho raised AttributeError
    # for every run with an explicit --rank-tau, which is every arm since bjwcm5.
    import rank_loss as _RKL
    tau = a.get("rank_tau")
    tau_ok = tau is None or abs(float(tau) - _RKL.DEFAULT_TAU) < 1e-9
    return (a.get("rank_form") == "softplus" and bool(a.get("rank_cross_seed"))
            and a.get("rank_pairs") in (None, "cross")
            and float(a.get("rank_within_weight") or 1.0) == 1.0
            and not a.get("rank_topk_decay")
            # straddle and weight_by_gap change the function too. Without them, an arm at
            # --rank-straddle 0.23 --rank-pairs cross and the default tau would be called
            # canonical and its own-objective curve plotted on the cross-arm axis.
            and not float(a.get("rank_straddle") or 0.0)
            and bool(a.get("rank_weight_by_gap", True)) and tau_ok)


def _chain(rid):
    """[oldest_ancestor, ..., rid]. DELEGATED to plot_three so there is one definition of it."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plot_three
    return plot_three.run_chain(rid)


def series(rid, budget, thresh):
    """epoch -> (paired_mean, paired_sem, dockq, frac, shipped_dockq, shipped_frac, ifaces).

    OVER THE WHOLE RESUME CHAIN, for the reason plot_three.pae_curves documents: a resume writes
    to a fresh directory while continuing the parent's epoch numbering, so reading one rid gave
    the share arm as epochs 10-20 -- ip8ga4's own shards -- beside the copy arm's 1-7. The two
    curves then shared an x axis while covering disjoint ranges of it, which is the one way this
    overlay can be read as showing a difference that is only a missing history.

    LATER RUNS WIN on a duplicated epoch: a child that recomputed an epoch holds the current
    value. Chain members here have disjoint epochs, so the rule is a guard, not a fix.
    """
    out = {}
    eps = {}
    for r in _chain(rid):
        for d in glob.glob(f"training/runs/{r}/train_eval_shards_ep*"):
            eps[int(os.path.basename(d).split("ep")[-1])] = r
    for ep in sorted(eps):
        rows = epoch_rows(eps[ep], ep, budget)
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
    # EVERY arm, not any: one PAE arm beside one ranking arm has no shared loss to draw, and the
    # ranking panel is then still the honest choice because it at least labels its held-out side
    # as the canonical yardstick.
    pae_mode = all(trained_pae_not_rank(spec.partition("=")[0]) for spec in a.arms)
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
        if pae_mode:
            _pe, _ptr, _pvl = pae_loss(rid)
            tl = dict(zip(_pe, _ptr))
            xl = dict(zip(_pe, _pvl))
        else:
            tl = train_loss(rid)
            # ACROSS ARMS the canonical column is the only comparable one; for ONE arm the
            # own-objective column is what pairs with its training curve.
            xl = xsloss(rid, a.budget, canonical=(len(a.arms) > 1))
        # NOT restricted to `ep`. The loss is recorded every epoch while the DockQ shards are
        # written only by an evaluated epoch, so intersecting with `ep` silently dropped loss
        # points that exist.
        te = sorted(tl)
        xe = sorted(xl)
        if te:
            axes[3].plot(te, [tl[e] for e in te], color=tint(c), lw=2.2, marker="s", ms=4,
                         ls="--", label=f"{labels[rid]} \u2014 training")
        if xe:
            axes[3].plot(xe, [xl[e] for e in xe], color=c, lw=2.2, marker="o", ms=4,
                         label=f"{labels[rid]} \u2014 HELD OUT")

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
    for spec in ([] if pae_mode else a.arms):
        _x = xsloss(spec.partition("=")[0], a.budget, shipped=True,
                    canonical=(len(a.arms) > 1))
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
    twins = (len(a.arms) == 1
             and heldout_is_twin(a.arms[0].partition("=")[0], a.budget))
    if pae_mode:
        # The vertical gap is NOT a clean generalisation gap: the two sides are different pools
        # AND, as of 2026-10-26, the held-out pass still crops while training does too but on a
        # different population. Comparing the SAME side across arms is what this panel supports.
        axes[3].set_title("PAE loss \u2014 light dashed = TRAINING, dark solid = HELD OUT\n"
                          "the shared objective of both arms; compare like side to like side",
                          fontsize=10.5)
        axes[3].set_ylabel("PAE loss")
    else:
        axes[3].set_title("ranking loss \u2014 light dashed = TRAINING, dark solid = HELD OUT\n"
                          + ("same function both sides: the gap is generalisation"
                             if twins else
                             "held out is the CANONICAL cross-seed loss, comparable across arms"),
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
