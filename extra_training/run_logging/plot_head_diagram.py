#!/usr/bin/env python
"""The confidence head as shipped, and as modified by --pose-perblock-share.

DRAWN FROM THE SOURCE, not from memory. protenix/model/modules/confidence.py:
  198  z_init  = linear_no_bias_s1(s_inputs) + linear_no_bias_s2(s_inputs)
  202  z_trunk = z_init + z_trunk
  297  z_pair  = z_trunk + linear_no_bias_d(one_hot(cdist(x_rep, x_rep)))    256 x 39  <- POSE
  305          +          linear_no_bias_d_wo_onehot(cdist(x_rep, x_rep))    256 x  1  <- POSE
  310  s_single, z_pair = pairformer_stack(s_trunk, z_pair)                  n_blocks = 4
  330  pae_pred = linear_no_bias_pae(pae_ln(z_pair))                         64 x 256

THE RE-INJECTION IS DRAWN AS A BUS, not as curved arrows into each block. The two projections
write onto one horizontal line that runs under the stack, and each block taps it at a (+) node.
That is the point being made: there is ONE pair of matrices and four read points, so the arrows are
reused weights rather than copies, and gradients from all four taps land on the same 10,240 numbers.
An earlier version used four curved arrows fanning out of the projection boxes, which read as four
separate things.

AND IT ADDS ACCESS, NOT INFORMATION. cdist over one representative atom per token is all the pose
ever becomes, in both panels. No backbone frame, no orientation. If the head's deficit is
orientation neither panel can reach it; --frame-direction is the pathway that would.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

POSE, FROZ, TRAIN = "#c0392b", "#9aa0a6", "#1f6fb4"
POSE_BG, TRAIN_BG, FROZ_BG, BLK_BG = "#fdf0ee", "#eaf2fb", "#f6f7f8", "#e9eff7"


def box(ax, x, y, w, h, text, fc="#fff", ec="#5a5a5a", fs=9, lw=1.3, bold=False, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.5",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha=ha, va="center", fontsize=fs, zorder=4,
            fontweight="bold" if bold else "normal", color="#1a1a1a", linespacing=1.45)


def arr(ax, p, q, color="#5a5a5a", lw=1.5, ls="-", z=2, ms=12):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, linestyle=ls, zorder=z, shrinkA=1, shrinkB=2))


def plus(ax, x, y, color=POSE, r=0.80, fs=11):
    ax.add_patch(Circle((x, y), r, fc="#ffffff", ec=color, lw=1.6, zorder=5))
    ax.text(x, y - 0.04, "+", ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold", zorder=6)


def draw(ax, perblock, label, title, sub, subcol):
    ax.set_xlim(0, 100); ax.set_ylim(0, 31); ax.axis("off")
    ax.text(0.0, 29.9, label, fontsize=14, fontweight="bold", va="top", color="#1a1a1a")
    ax.text(3.6, 29.9, title, fontsize=11.5, va="top", color="#1a1a1a")

    LANE = 20.0          # the main left-to-right lane
    BUS = 8.2            # the pose bus

    # ---------- inputs (frozen) ----------
    box(ax, 0.5, 23.6, 14.5, 3.3, "s_inputs", fc=FROZ_BG, ec=FROZ, fs=9.5)
    box(ax, 0.5, 19.2, 14.5, 3.3, "z_trunk   (frozen trunk)", fc=FROZ_BG, ec=FROZ, fs=9.5)
    arr(ax, (15.0, 25.25), (22.0, 21.4), FROZ, lw=1.4)
    arr(ax, (15.0, 20.85), (22.0, 20.6), FROZ, lw=1.4)

    # ---------- the pose ----------
    box(ax, 0.5, 10.2, 14.5, 4.4,
        "x_pred_coords\ncdist over the representative atom", fc=POSE_BG, ec=POSE, fs=8.6)
    ax.text(7.75, 9.3, "THE POSE: distances only, no orientation",
            ha="center", va="top", fontsize=8, color=POSE, style="italic")

    # ---------- the two trainable projections ----------
    box(ax, 18.0, 12.9, 17.5, 3.0, "linear_no_bias_d            256x39", fc=TRAIN_BG, ec=TRAIN,
        lw=2.0, fs=8.6)
    box(ax, 18.0, 9.4, 17.5, 3.0, "linear_no_bias_d_wo_onehot  256x1", fc=TRAIN_BG, ec=TRAIN,
        lw=2.0, fs=8.6)
    arr(ax, (15.0, 13.0), (18.0, 14.4), POSE, lw=1.5)
    arr(ax, (15.0, 11.6), (18.0, 10.9), POSE, lw=1.5)
    ax.text(26.75, 17.0, "10,240 parameters -- the ONLY ones trained",
            ha="center", va="bottom", fontsize=8.8, color=TRAIN, fontweight="bold")

    # ---------- blocks ----------
    xs, bw = [45.0, 56.0, 67.0, 78.0], 9.2
    for i, x in enumerate(xs):
        box(ax, x, 18.4, bw, 5.6, f"pairformer\nblock {i + 1}\n\nfrozen", fc=BLK_BG, ec=FROZ,
            fs=8.6)

    # z_pair, then the lane through the stack
    box(ax, 22.0, 18.4, 9.0, 5.6, "z_pair\n[N,N,256]", fc="#ffffff", ec="#5a5a5a", bold=True,
        fs=9.5)
    plus(ax, 38.0, LANE + 1.2)
    arr(ax, (31.0, LANE + 1.2), (37.2, LANE + 1.2))
    arr(ax, (38.8, LANE + 1.2), (45.0, LANE + 1.2))
    for i in range(3):
        if perblock:
            plus(ax, xs[i] + bw + 0.9, LANE + 1.2)
            arr(ax, (xs[i] + bw, LANE + 1.2), (xs[i] + bw + 0.1, LANE + 1.2))
            arr(ax, (xs[i] + bw + 1.7, LANE + 1.2), (xs[i + 1], LANE + 1.2))
        else:
            arr(ax, (xs[i] + bw, LANE + 1.2), (xs[i + 1], LANE + 1.2))

    # ---------- the pose bus ----------
    taps = [38.0] + ([xs[i] + bw + 0.9 for i in range(3)] if perblock else [])
    ax.plot([26.75, max(taps)], [BUS, BUS], color=POSE, lw=2.6, solid_capstyle="round", zorder=2)
    # the drop from the two projections onto the bus: a plain line, not an arrow --
    # a zero-length arrowhead divides by zero in FancyArrowPatch.
    ax.plot([26.75, 26.75], [9.4, BUS], color=POSE, lw=2.6,
            solid_capstyle="round", zorder=2)
    for t in taps:
        ax.plot([t, t], [BUS, LANE + 1.2 - 0.85], color=POSE, lw=2.0, zorder=2)
        arr(ax, (t, LANE + 1.2 - 2.0), (t, LANE + 1.2 - 0.85), POSE, lw=2.0)
        ax.add_patch(Circle((t, BUS), 0.34, fc=POSE, ec=POSE, zorder=4))

    # ---------- readout ----------
    box(ax, 89.5, 18.4, 9.6, 5.6, "pae_ln\n+\nlinear_no_bias_pae\n64x256", fc=FROZ_BG, ec=FROZ,
        fs=8.0)
    arr(ax, (xs[-1] + bw, LANE + 1.2), (89.5, LANE + 1.2))
    box(ax, 89.5, 12.6, 9.6, 3.6, "PAE\nipTM\nselection", fc="#ffffff", ec="#5a5a5a", fs=8.6,
        bold=True)
    arr(ax, (94.3, 18.4), (94.3, 16.3))

    # BELOW THE BUS. At y=5.9 this box overlapped the bus (y=8.2) and the vertical taps.
    ax.text(57.0, 3.1, sub, ha="center", va="center", fontsize=9.4, color=subcol,
            linespacing=1.55,
            bbox=dict(fc="#ffffff", ec=subcol, lw=1.2, boxstyle="round,pad=0.55"))


fig, axes = plt.subplots(2, 1, figsize=(17.5, 11.0))
draw(axes[0], False, "A", "Confidence head AS SHIPPED",
     "ONE injection point. After it, nothing sees the coordinates again --\n"
     "blocks 2-4 work on a representation where the pose is already mixed into the trunk.",
     "#4a4a4a")
draw(axes[1], True, "B", "WITH --pose-perblock-share",
     "FOUR injection points, ONE pair of matrices. The bus is read, not copied --\n"
     "zero new parameters, and gradients from all four taps land on the same 10,240 numbers.",
     POSE)
fig.suptitle("FT3 confidence head: where the pose enters.      "
             "blue = trainable (10,240 params, 0.052% of the head)      grey = frozen      "
             "red = the pose pathway",
             fontsize=10.8, y=0.988, color="#1a1a1a")
fig.tight_layout(rect=(0, 0, 1, 0.962))
fig.savefig("training/runs/head_diagram.png", dpi=160)
print("wrote training/runs/head_diagram.png")
