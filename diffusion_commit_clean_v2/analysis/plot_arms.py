#!/usr/bin/env python
"""Bar plot: selected-pose DockQ, self-templating vs free + compute-matched, per class.

SUMMARY
    Four groups -- all interfaces, then solved, confident decoy and no acceptable pose -- each with two
    bars: the mean DockQ of the selected (top-ipTM) pose under self-templating (blue) and under free +
    compute-matched sampling (gray). Every interface is a dot on both bars. Above each group, a paired
    two-sided Wilcoxon signed-rank test across its interfaces (self-template DockQ vs free + matched
    DockQ of the same interface).

    The per-interface values come from arm_statistics.selected(), i.e. compare_arms.py's own rules, so
    the bar heights equal the "mean DockQ" columns of arm_statistics.md.

INPUTS
    --exp:        the templated experiment. Default experimental_data/self_template_fixed_50_seed.
    --extra-csv:  free results csvs, colon-separated. Default as compare_arms.py.
    --out:        the figure. Default analysis/arm_barplot.png (a .pdf beside it too).

OUTPUTS
    <out> and <out>.pdf; stdout: n, both means and the Wilcoxon p per group

EXAMPLE
    analysis/plot_arms.py --exp experimental_data/self_template_variable_50_seed \\
        --out diffusion_commit_clean_v2/analysis/arm_barplot_variable.png

FUNCTIONS
    wilcoxon(): paired DockQ -> two-sided p (1.0 when every difference is zero)
    main():     assemble the groups, test, draw
"""
import argparse
import os
import random
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from arm_statistics import selected  # noqa: E402  (puts v2 on the path)
import paths  # noqa: E402

GROUPS = [("all", "All interfaces"), ("solved", "Solved"), ("decoy", "Confident decoy"),
          ("nocov", "No acceptable pose")]
BLUE, GRAY = "#2a78d6", "#83827c"            # validated: CVD dE 15.8, both >= 3:1 on the surface
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def wilcoxon(method, matched):
    """in: paired DockQ lists.  out: two-sided Wilcoxon signed-rank p (1.0 if all differences are 0)"""
    if all(a == b for a, b in zip(method, matched)):
        return 1.0
    return stats.wilcoxon(method, matched).pvalue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=f"{paths.DATA_DIR}/self_template_fixed_50_seed")
    ap.add_argument("--extra-csv",
                    default=f"{paths.FREE_DIR}/results.csv:{paths.DATA_DIR}/free_compute_matched/results.csv")
    ap.add_argument("--out", default=f"{HERE}/arm_barplot.png")
    A = ap.parse_args()

    sel, partial, unmatched = selected(A.exp, A.extra_csv)
    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    w, top = 0.36, 1.0
    for x, (key, name) in enumerate(GROUPS):
        v = [(m, t) for c, m, t in sel.values() if key in ("all", c)]
        matched, method = [m for m, _ in v], [t for _, t in v]
        p = wilcoxon(method, matched)
        for dx, vals, col in ((-w / 2, method, BLUE), (w / 2, matched, GRAY)):
            ax.bar(x + dx, statistics.mean(vals), w * 0.94, color=col, alpha=0.35, edgecolor=col,
                   linewidth=1.2, zorder=2)
            ax.scatter([x + dx + rng.uniform(-w * 0.32, w * 0.32) for _ in vals], vals, s=7, color=col,
                       edgecolors="white", linewidths=0.3, alpha=0.8, zorder=3)
            ax.text(x + dx, statistics.mean(vals) + 0.012, f"{statistics.mean(vals):.3f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.text(x, top + 0.04, f"p = {p:.2g}", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(x, top + 0.005, f"n = {len(v)}", ha="center", va="bottom", fontsize=8, color=MUTED)
        print(f"  {name:20s} n={len(v):4d}  self-template {statistics.mean(method):.3f}  "
              f"free + matched {statistics.mean(matched):.3f}  Wilcoxon p={p:.3g}")

    ax.set_xticks(range(len(GROUPS)), [n for _, n in GROUPS], color=INK)
    ax.set_ylabel("DockQ of the selected (top-ipTM) pose", color=INK)
    ax.set_ylim(0, top + 0.12)
    ax.set_yticks([0, 0.23, 0.5, 0.75, 1.0], ["0", "0.23", "0.5", "0.75", "1.0"])
    ax.axhline(0.23, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=BLUE, alpha=0.6, ec=BLUE),
                       plt.Rectangle((0, 0), 1, 1, fc=GRAY, alpha=0.6, ec=GRAY)],
              labels=["Self-template", "Free + compute-matched"], frameon=False, fontsize=8.5,
              loc="upper right", bbox_to_anchor=(1.0, 1.13), ncol=2)
    ax.set_title(f"{os.path.basename(A.exp)}", loc="left", fontsize=9, color=MUTED, pad=18)
    fig.text(0.01, 0.005, f"Bars: mean; dots: interfaces; dashed line: acceptable (DockQ 0.23). "
             f"p: paired two-sided Wilcoxon signed-rank across interfaces.\n"
             f"{partial} unfinished interfaces and {unmatched} unmatched complexes excluded.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(A.out, dpi=200)
    fig.savefig(os.path.splitext(A.out)[0] + ".pdf")
    print(f"  -> {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
