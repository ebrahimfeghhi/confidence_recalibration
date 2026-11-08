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


def arm(rid):
    """{panel: {epoch: value}}, plus the set of partial epochs."""
    means = EC.epoch_means(rid)
    logs = EC.log_series(rid)
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
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
        ax.grid(alpha=.3); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(rids) > 1 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)
    sub = "  vs  ".join(rids) if len(rids) > 1 else rids[0]
    fig.suptitle(f"TRAINING metrics -- {sub}"
                 + ("   (hollow = partial epoch)" if any(d[1] for d in data.values()) else ""),
                 fontsize=13)
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
