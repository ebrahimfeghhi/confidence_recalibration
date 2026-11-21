#!/usr/bin/env python
"""TWO ARMS ON ONE SET OF AXES, for "did the change help".

ONE FIGURE PER ARM is right for reading an arm's own shape, which is what plot_fullset.py does.
It is the wrong picture for a controlled comparison: two arms at the SAME learning rate differing
only in the loss belong on the same axes, because the question is the gap between them and the
eye cannot hold a gap across two files.

MAX-POOLED IS SOLID, MEAN-POOLED DASHED. The max is what shipped ipTM is, so it is the
comparable headline and the shipped reference line is the same quantity; the mean is what the
ranking arms optimise. Both are drawn because on these arms they disagree.

Usage: plot_two_arms.py OUT.png "label=RID:TAG" "label=RID:TAG"
  TAG is the $SD scratchpad prefix for an arm scored between segments, or any string for an arm
  whose shard dirs live in its own run directory -- chain_shard_dirs finds those from the run id.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import plot_fullset as P                                            # noqa: E402

COLOURS = ["tab:blue", "tab:red", "tab:green"]


def load(spec):
    label, rest = spec.split("=", 1)
    rid, tag = rest.split(":", 1)
    segs = P.chain(rid)
    return dict(
        label=label, rid=rid,
        val=P.score_points(P.work_dirs(f"val30_{tag}_ep", "valfull83", segs)),
        sli=P.score_points(P.work_dirs(f"slice280_{tag}_ep", "slice280", segs)),
    )


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    if len(specs) < 2:
        sys.exit("STOP: give at least two arms as label=RID:TAG")
    arms = [load(s) for s in specs]
    fig, axx = plt.subplots(2, 3, figsize=(19, 9.5))
    rows = (("sli", "TRAIN SLICE (in-sample)"), ("val", "HELD-OUT"))
    cols = (("q", "q_mean", "mean DockQ", False),
            ("top1", "top1_mean", "frac acceptable", True),
            ("rho", "rho_mean", "ipTM vs DockQ (mean Spearman within an interface)", False))
    for ri, (side, sname) in enumerate(rows):
        for ci, (key, mkey, cname, pct) in enumerate(cols):
            a_ = axx[ri][ci]
            drawn = False
            for k, arm in enumerate(arms):
                pts = arm[side]
                e = [x for x in sorted(pts) if pts[x][key] == pts[x][key]]
                if not e:
                    continue
                drawn = True
                c = COLOURS[k % len(COLOURS)]
                a_.plot(e, [pts[x][key] for x in e], "-o", color=c, lw=2, ms=5,
                        label=f"{arm['label']} (max)")
                em = [x for x in sorted(pts) if pts[x][mkey] == pts[x][mkey]]
                if em:
                    a_.plot(em, [pts[x][mkey] for x in em], "--^", color=c, lw=1.3, ms=4,
                            alpha=.55, label=f"{arm['label']} (mean)")
                # SHIPPED ONCE. It is the same baseline for every arm -- drawing it per arm would
                # suggest each had its own.
                if k == 0:
                    b = next(pts[x]["s_" + key] for x in e
                             if pts[x]["s_" + key] == pts[x]["s_" + key])
                    a_.axhline(b, color="#555", ls="--", lw=1.6,
                               label=f"shipped {b:.1%}" if pct else f"shipped {b:.4f}")
            if not drawn:
                a_.text(.5, .5, "not scored yet", ha="center", va="center",
                        transform=a_.transAxes, color="#666")
            a_.set_title(f"{sname}: {cname}", fontsize=10.5)
            a_.set_xlabel("epoch")
            a_.grid(alpha=.3)
            a_.legend(fontsize=7.5)
    fig.suptitle(os.environ.get("TITLE", "TWO ARMS"), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .94])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}: " + ", ".join(
        f"{a['label']} val {sorted(a['val'])} slice {sorted(a['sli'])}" for a in arms))


if __name__ == "__main__":
    main()
