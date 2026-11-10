#!/usr/bin/env python
"""THE training plot. One file, one definition, one or more arms on the same axes.

    plot_rel_arm.py RID[,RID2,...] [OUT.png]

Four panels, always the same four, because an arm is judged on all of them together:
  PAE loss      the term that dominates the gradient
  rel           the ranking term this arm adds
  rho           median within-interface Spearman -- the ranking quality selection actually uses
  score spread  std(predicted) / std(true iTM); 1.0 means the predictions are as spread as truth

WHERE THE NUMBERS COME FROM, and why none of it lives here: the PAE curve is the ALL-RANKS
batch_n-weighted mean from epoch_curve.epoch_means (trajectory.csv is rank 0's alone, so averaging
its per-pose column reports a third of the run), and rel/rho/spread are per-epoch scalars rank 0
prints to run.log, parsed by epoch_curve.log_series. Both are shared with every other plotter so
no two figures can disagree about what an epoch's value is -- which they did, repeatedly, before
this was centralised.

Several RIDs overlay for an A/B. The chain of each is resolved from args.json, so a resumed arm is
one curve and not several, and partial epochs are drawn HOLLOW so an incomplete mean is never read
as a finished one.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import epoch_curve as EC

PANELS = [("pae", "PAE loss", "PAE loss (all ranks)"),
          ("rel", "relative iTM term", "rel loss"),
          ("rho", "within-interface Spearman vs DockQ", "median rho"),
          ("spread", "score spread  std(pred) / std(true iTM)", "ratio")]
COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple"]


#: a partial epoch holding less than this fraction of a full one is a FRAGMENT, not a data point:
#: the run was stopped partway and a handful of poses says nothing about the epoch.
FRAGMENT = 0.05


def arm(rid):
    """{panel: {epoch: value}}, plus the set of partial epochs."""
    means = EC.epoch_means(rid)
    logs = EC.log_series(rid)
    full = max((n for _v, n, _s in means.values()), default=0)
    # Drop fragments outright. A stopped arm leaves a few hundred poses in the next epoch, and
    # at 3.52 against a 2.94-3.00 curve that single point set the whole y-axis and squashed five
    # real epochs into the bottom sixth of the panel.
    means = {e: v for e, v in means.items() if v[1] >= FRAGMENT * full}
    return ({"pae": {e: v for e, (v, _n, _s) in means.items()}, **logs},
            EC.partial_epochs(means), EC.run_chain(rid))


def main():
    rids = (sys.argv[1] if len(sys.argv) > 1 else "qip9ta").split(",")
    out = sys.argv[2] if len(sys.argv) > 2 else f"figures/rel_arm_{rids[0]}.png"
    data = {r: arm(r) for r in rids}
    if not any(d[0]["pae"] for d in data.values()):
        raise SystemExit(f"STOP: no completed epochs for {rids}")

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for (key, title, ylab), ax in zip(PANELS, axes.ravel()):
        for i, r in enumerate(rids):
            series, partial, _chain = data[r]
            pts = sorted(series.get(key, {}).items())
            if not pts:
                continue
            col = COLORS[i % len(COLORS)]
            full = [(e, v) for e, v in pts if e not in partial]
            if full:
                ax.plot(*zip(*full), "-o", color=col, lw=2, ms=7,
                        label=r if len(rids) > 1 else None)
            for e, v in pts:
                if e in partial:          # hollow: fewer poses behind it
                    ax.plot([e], [v], "o", mfc="none", mec=col, mew=2, ms=7)
        # SCALE TO COMPLETE EPOCHS. A partial one is drawn hollow but must not set the limits:
        # it is a mean over fewer poses and is routinely an outlier.
        done = [v for r in rids for e, v in data[r][0].get(key, {}).items()
                if e not in data[r][1] and v == v]
        if len(done) > 1:
            lo, hi = min(done), max(done)
            m = (hi - lo) * .28 or abs(hi) * .01 or 1e-4
            ax.set_ylim(lo - m, hi + m)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
        ax.grid(alpha=.3); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ONE LEGEND, OUTSIDE THE AXES. A per-panel legend sat at the upper right INSIDE the data
    # area and started covering points as soon as a curve climbed into that corner -- rho reached
    # 0.3000 at epochs 15-16 and spread 0.4370 at epoch 15, both in the top-right of their panels,
    # exactly where the box was. Moving it under the figure cannot collide with anything by
    # construction, which placement heuristics like loc="best" only promise per render.
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    sub = "  vs  ".join(rids) if len(rids) > 1 else rids[0]
    fig.suptitle(f"TRAINING metrics -- {sub}"
                 + ("   (hollow = partial epoch)" if any(d[1] for d in data.values()) else ""),
                 fontsize=13)
    if len(rids) > 1 and handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 6),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.005))
        fig.tight_layout(rect=[0, .045, 1, .94])
    else:
        fig.tight_layout(rect=[0, 0, 1, .94])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"-> {out}")

    for r in rids:
        series, partial, chain = data[r]
        print(f"\n{r}  (chain {' -> '.join(chain)})")
        eps = sorted(series["pae"])
        print(f"  {'ep':>3} " + " ".join(f"{k:>10}" for k, _t, _y in PANELS))
        for e in eps:
            row = " ".join(f"{series.get(k, {}).get(e, float('nan')):>10.4f}"
                           for k, _t, _y in PANELS)
            print(f"  {e:>3} {row}" + ("   PARTIAL" if e in partial else ""))
        done = [e for e in eps if e not in partial]
        if len(done) > 1:
            v = [series["pae"][e] for e in done]
            down = sum(1 for a_, b_ in zip(v, v[1:]) if b_ < a_)
            print(f"  PAE monotonicity: {down} of {len(v) - 1} steps decreased, "
                  f"net {v[-1] - v[0]:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
