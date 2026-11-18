#!/usr/bin/env python
"""Bar plots: the four six-site arms side by side at epochs 10 and 20.

ONE METRIC PER FIGURE, four panels each: pooling (max / mean) x pool (held-out / train slice).
Within a panel, one cluster per arm and one bar per epoch, with the SHIPPED head drawn as a
dashed line.

EVERY NUMBER IS RECOMPUTED FROM THE PER-CANDIDATE ROWS, not copied from a log or a json. The
metric jsons carry q and top-1 but no rho, and they carry only the headline pooling, so reading
them would mean mixing two sources for one figure. score_points() in plot_fullset is the same
reader the line plots use -- it requires all three shards and an equal candidate count per
interface before it returns a point, so a half-written pass is skipped rather than drawn short.
The shipped values come from the stored_iptm column of those same rows, so our bar and its
baseline are always the same interfaces.

A MISSING PASS IS A GAP, NOT A ZERO: Arm D has no epoch 20 yet, and that bar is left out with
the arm's label annotated instead.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_fullset as PF

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(ROOT)

# label -> rid file. The rid files are what the launch queues wrote, so the arms are named by
# what they ARE and the id is looked up rather than typed.
ARMS = [
    ("Arm A\n6 sites", "training/runs/pose6_paeonly_rid"),
    ("Arm B\n+ pae_ln", "training/runs/pose6_paeln_rid"),
    ("Arm C\n+ LoRA r4", "training/runs/pose6_lora4_rid"),
    ("Arm D\n+ LoRA r8", "training/runs/pose6_lora8_rid"),
]
EPOCHS = [10, 20]
SIDES = [("valfull83", "HELD-OUT  (141 interfaces)"),
         ("slice280", "TRAIN SLICE  (280 in-sample interfaces)")]
# metric -> (max-pooled key, mean-pooled key, shipped key, axis label, percent?)
METRICS = [
    ("q",    "q",    "q_mean",    "s_q",    "mean DockQ of the top-1 pick", False),
    ("top1", "top1", "top1_mean", "s_top1", "frac acceptable at top 1", True),
    ("rho",  "rho",  "rho_mean",  "s_rho",  "within-interface Spearman (ipTM vs DockQ)", False),
]
COL = {10: "tab:blue", 20: "tab:orange"}


def load():
    """(label, rid) -> {side: {epoch: point}}, recomputed from rows."""
    out = []
    for lab, ridf in ARMS:
        try:
            rid = open(ridf).read().strip()
        except OSError:
            print(f"  NO RID FILE for {lab.replace(chr(10), ' ')} ({ridf}) -- arm skipped")
            continue
        if not rid:
            print(f"  EMPTY RID FILE for {lab.replace(chr(10), ' ')} -- arm skipped")
            continue
        per = {}
        for tag, _ in SIDES:
            work = {}
            for e in EPOCHS:
                d = f"training/runs/{rid}/{tag}_eval_shards_ep{e}"
                if os.path.isdir(d):
                    work[e] = d
            per[tag] = PF.score_points(work) if work else {}
        got = {t: sorted(per[t]) for t in per}
        print(f"  {lab.replace(chr(10), ' ')} = {rid}: "
              + "  ".join(f"{t} {got[t]}" for t in got))
        out.append((lab, rid, per))
    return out


def bars(ax, data, key, ship_key, ylab, pct, title):
    xs, labs, nmiss = [], [], []
    w = 0.38
    vals = []
    ships = []
    for i, (lab, _rid, per) in enumerate(data):
        labs.append(lab)
        for j, e in enumerate(EPOCHS):
            p = per.get(title_tag, {}).get(e)
            x = i + (j - 0.5) * w
            if p is None or p.get(key) != p.get(key):      # absent or NaN
                nmiss.append((x, lab, e))
                continue
            v = p[key] * (100.0 if pct else 1.0)
            vals.append(v)
            ax.bar(x, v, w * 0.92, color=COL[e], zorder=3,
                   label=f"epoch {e}" if i == 0 else None)
            ax.text(x, v, f"{v:.1f}" if pct else f"{v:.4f}", ha="center", va="bottom",
                    fontsize=7, zorder=4)
            if p.get(ship_key) == p.get(ship_key):
                ships.append(p[ship_key] * (100.0 if pct else 1.0))
        xs.append(i)
    # SHIPPED, FROM THE SAME ROWS. Every pass scores the same pool, so these agree; the spread is
    # printed rather than assumed, and a disagreement above a rounding step would show up here.
    if ships:
        s = sum(ships) / len(ships)
        if max(ships) - min(ships) > (0.05 if pct else 0.0005):
            print(f"    NOTE {title}/{key}: shipped varies across passes "
                  f"{min(ships):.4f}..{max(ships):.4f}; drawing the mean")
        ax.axhline(s, ls="--", lw=1.4, color="0.35", zorder=2,
                   label=f"shipped ({s:.1f})" if pct else f"shipped ({s:.4f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel(ylab + (" (%)" if pct else ""), fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    # A BAR CHART STARTING AT ZERO HIDES THESE DIFFERENCES: the arms sit within a couple of
    # percent of each other and of shipped, so the panel is zoomed and every tick is labelled.
    # THE FLOOR SITS BELOW THE SMALLEST BAR, not on it -- clamping to min() drew the lowest arm
    # as a zero-height sliver, which reads as "no data" rather than "lowest".
    lo = min(vals + ships) if (vals or ships) else 0.0
    hi = max(vals + ships) if (vals or ships) else 1.0
    rng = (hi - lo) or (1.0 if pct else 0.01)
    ax.set_ylim(lo - rng * 0.45, hi + rng * 0.30)
    ax.set_xlim(-0.65, len(labs) - 0.35)
    for x, _lab, e in nmiss:
        ax.text(x, lo - rng * 0.40, f"ep{e}\nnot yet", ha="center", va="bottom",
                fontsize=6.5, color="0.45", style="italic", zorder=4)


def main():
    print("loading, recomputed from per-candidate rows:")
    data = load()
    if not data:
        print("STOP: no arm had readable rows")
        return 1
    for name, kmax, kmean, kship, ylab, pct in METRICS:
        fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.0))
        for r, (pool_key, pool_lab) in enumerate([(kmax, "max-pooled ipTM (shipped convention)"),
                                                  (kmean, "mean-pooled ipTM (what these arms "
                                                          "optimise)")]):
            for c, (tag, side_lab) in enumerate(SIDES):
                global title_tag
                title_tag = tag
                bars(ax[r][c], data, pool_key, kship, ylab, pct,
                     f"{side_lab}\n{pool_lab}")
        # ONE LEGEND FOR THE FIGURE. Per-axes legends sat on top of Arm D, which is the one arm
        # with a missing bar and therefore the one that most needed to be readable.
        h, l = ax[0][0].get_legend_handles_labels()
        seen, hh, ll = set(), [], []
        for a, b in zip(h, l):
            if b.startswith("shipped"):      # its value differs per panel; say so generically
                b = "shipped head (dashed, per-panel value in the axis)"
            if b in seen:
                continue
            seen.add(b)
            hh.append(a)
            ll.append(b)
        fig.legend(hh, ll, loc="upper right", fontsize=8, ncol=3,
                   bbox_to_anchor=(0.995, 0.995), framealpha=0.9)
        fig.suptitle(f"{ylab.upper()} -- four six-site arms at epochs 10 and 20",
                     fontsize=11, x=0.30)
        fig.tight_layout(rect=(0, 0, 1, 0.945))
        out = f"figures/arms_bar_{name}.png"
        fig.savefig(out, dpi=135)
        plt.close(fig)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
