#!/usr/bin/env python
"""ONE TRAINING-LOSS CURVE ACROSS A WHOLE RESUME CHAIN, with the handovers marked.

    plot_chain_loss.py <out.png> <rid> [--title T]

The rid is the LAST run of the chain; `plot_fullset.chain` walks the --resume links back to the
start, so `qkicr2` draws fshpzx -> 7mo52r -> ggd8uv -> qkicr2 as a single line over epochs 1..N.
Vertical dashes mark where one run handed over to the next, because a stitched curve hides the
fact that four separate launches produced it and a discontinuity at a handover is a real finding
rather than noise.

WHY THE TOTAL AND NOT THE GOOD/BAD SPLIT. The split needs --pae-split-accept, which this chain
never passed (it predates the flag), so `group_loss` is empty for it and the usual two-group
figure cannot be drawn at all. The total is read per POSE from each run's trajectory.csv and
averaged per epoch, which is defined for every run.

NO SHIPPED LINE, DELIBERATELY. training/runs/shipped_baseline_all is the shipped head's mean over
the UNCROPPED training pool, and this chain trains on 256-token crops -- the two are different
quantities and drawing one as a rule under the other invites exactly the comparison that must not
be made. Pass --shipped VALUE to add a line if a cropped baseline is ever measured.
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import plot_fullset as P                                            # noqa: E402
from plot_group_loss_arms import total_loss                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("rid")
    ap.add_argument("--title", default="")
    ap.add_argument("--shipped", type=float, default=None)
    a = ap.parse_args()

    chain = [d.split("/")[-1] for d, _ in P.chain(a.rid)][::-1]      # oldest first
    tl = total_loss(a.rid)
    if not tl:
        sys.exit(f"no per-epoch loss for {a.rid}")
    E = sorted(tl)

    # WHICH RUN OWNS WHICH EPOCH, read from each run's own trajectory rather than inferred from
    # the resume epoch numbers -- a resume that re-ran an epoch would otherwise be mislabelled.
    owner = {}
    for rid in chain:
        try:
            for e in total_loss(rid):
                owner.setdefault(e, rid)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.plot(E, [tl[e] for e in E], "-o", color="tab:blue", lw=2, ms=4.5,
            label="training PAE loss (mean over poses)")
    bounds, seen = [], set()
    for e in E:
        o = owner.get(e)
        if o and o not in seen:
            seen.add(o)
            bounds.append((e, o))
    for e, o in bounds[1:]:
        ax.axvline(e - 0.5, color="0.6", lw=1.0, ls="--", zorder=0)
    for i, (e, o) in enumerate(bounds):
        end = bounds[i + 1][0] - 1 if i + 1 < len(bounds) else E[-1]
        ax.annotate(f"{o}\nep {e}-{end}", xy=(e, ax.get_ylim()[1]),
                    xytext=(e + 0.1, 0.97), textcoords=("data", "axes fraction"),
                    fontsize=8, color="0.35", va="top")
    if a.shipped is not None:
        ax.axhline(a.shipped, color="0.35", lw=1.2, ls=":", label=f"shipped {a.shipped:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training PAE loss")
    ax.grid(alpha=.25)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(a.title or f"{' -> '.join(chain)} -- stitched training loss")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}: epochs {E[0]}-{E[-1]} over {len(chain)} run(s)")
    for i, (e, o) in enumerate(bounds):
        end = bounds[i + 1][0] - 1 if i + 1 < len(bounds) else E[-1]
        print(f"  {o:8s} epochs {e}-{end}")


if __name__ == "__main__":
    main()
