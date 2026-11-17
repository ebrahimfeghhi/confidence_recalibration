#!/usr/bin/env python
"""SELECTION QUALITY AT BUDGETS ABOVE ONE, on its own figure.

WHY SEPARATE. plot_fullset.py answers "is the head better than shipped at picking ONE pose"; these
are the budget-N questions, and bolting them onto that grid made it twelve panels nobody reads.

TWO METRICS, BOTH POOLINGS, BOTH SIDES.
  best-of-N  -- rank the candidates by the head's score, take the N highest, keep the BEST DockQ
                among them, averaged over interfaces. The "if I submitted N models" number. NOT
                the mean of the N picks, which falls as N grows.
  rank of the first acceptable candidate -- how deep you have to read the head's ranking before
                hitting DockQ >= 0.23. Over WINNABLE items only: an item with no acceptable
                candidate has no such rank, and scoring it as 150 or dropping it silently would
                both move the average for reasons unrelated to ranking. LOWER IS BETTER, and the
                median is drawn beside the mean because the distribution is heavy-tailed -- most
                items are rank 1 and a few are in the hundreds.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import plot_fullset as P                                            # noqa: E402


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "figures/selection_budget.png"
    rid = os.environ.get("RID") or open("training/runs/pose5_arm_rid").read().strip()
    tp = os.environ.get("TAGP", "pose5")
    segs = P.chain(rid)
    val = P.score_points(P.work_dirs(f"val30_{tp}_ep", "valfull83", segs))
    sli = P.score_points(P.work_dirs(f"slice280_{tp}_ep", "slice280", segs))

    # THREE LINES PER PANEL, ONE METRIC PER PANEL. The first version put best-of-5, best-of-10,
    # both poolings of each, and two shipped lines on one axis -- seven lines, unreadable. One
    # metric per panel, ours under each pooling, and shipped once.
    COLS = (("q5", "q5_mean", "s_q5", "best-of-5 DockQ", False),
            ("fa_mean", "fa_mean_mp", "s_fa_mean", "rank of the first acceptable pose", True))
    fig, axx = plt.subplots(2, 2, figsize=(14, 9))
    for ri, (pts, sname, colour) in enumerate(((sli, "TRAIN SLICE", "tab:orange"),
                                               (val, "HELD-OUT", "tab:blue"))):
        for ci, (key, mkey, skey, title, lower) in enumerate(COLS):
            a_ = axx[ri][ci]
            e = [x for x in sorted(pts) if pts[x][key] == pts[x][key]]
            if not e:
                a_.text(.5, .5, "not scored yet", ha="center", va="center",
                        transform=a_.transAxes, color="#666")
            else:
                a_.plot(e, [pts[x][key] for x in e], "-o", color=colour, lw=2, ms=5,
                        label="ours (max-pooled)")
                em = [x for x in e if pts[x][mkey] == pts[x][mkey]]
                if em:
                    a_.plot(em, [pts[x][mkey] for x in em], "--^", color=colour, lw=1.4, ms=4,
                            alpha=.6, label="ours (mean-pooled)")
                b = pts[e[0]][skey]
                a_.axhline(b, color="#555", ls="--", lw=1.8,
                           label=f"shipped {b:.1f}" if lower else f"shipped {b:.4f}")
                a_.set_xticks(e)
                a_.legend(fontsize=8)
            extra = ""
            if lower and e:
                extra = (f"\nwinnable n={pts[e[0]]['n_win_fa']}, median rank "
                         f"{pts[e[-1]]['fa_med']:.0f} vs shipped {pts[e[-1]]['s_fa_med']:.0f}"
                         f" -- LOWER IS BETTER")
            a_.set_title(f"{sname}: {title}{extra}", fontsize=11)
            a_.set_ylabel("rank" if lower else "mean DockQ")
            a_.set_xlabel("epoch")
            a_.grid(alpha=.3)
    fig.suptitle(os.environ.get("TITLE", "SELECTION AT BUDGET N") + f"  ({rid})", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .93])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}: val {sorted(val)}, slice {sorted(sli)}")


if __name__ == "__main__":
    main()
