#!/usr/bin/env python
"""Training-pool metrics per run, side by side: mean DockQ and fraction acceptable.

Reads the `epoch N TRAIN:` lines the trainer prints, so it needs nothing the run does not already
write and can be re-run while the runs are still going.
"""
import argparse, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"epoch (\d+) TRAIN: mean_dockq ([0-9.]+)\s+frac_acc ([0-9.]+)")
COLORS = ["#1f77b4", "#e67e22", "#8e44ad", "#16a085", "#c0392b"]
FULL_STEPS = 18     # 17 full + 1 partial per epoch at accum 85 on 30 structures


def series(root, rid):
    log = os.path.join(root, rid, "run.log")
    if not os.path.exists(log):
        return None
    xs, dq, fa = [], [], []
    for m in LINE.finditer(open(log, errors="replace").read()):
        xs.append(int(m.group(1))); dq.append(float(m.group(2))); fa.append(float(m.group(3)))
    return xs, dq, fa


def pae_by_epoch(root, rid, full_steps=FULL_STEPS):
    """Mean PAE loss per epoch OVER EVERY POSE, incomplete epochs dropped.

    Per-step PAE swings from 1.40 to 2.53 inside one epoch, because a step is 84 poses from a
    handful of structures and structures differ wildly in difficulty. A 3-step mean is therefore
    noise: a partial epoch 6 read 2.0903 where the complete one reads 2.2214, which is large
    enough to invert a comparison between two arms -- and did.

    WEIGHTED BY batch_n, not a mean of the step means. The 18th step of an epoch is the
    remainder -- 72 poses against 84 -- and averaging step means gives it 1/18 of the weight when
    it holds 72/1500 = 4.8% of the data. Those 72 poses are a handful of structures and their loss
    ranged 1.21-2.78 across krah8z's 29 epochs against ~1.99 for the epoch, so the unweighted mean
    carried about +/-0.02 of pure tail noise and read epoch 29 as a run low (1.9499) where the
    pose-weighted number is 1.9558 and the trend is flat. Same statistic as the early-stop rule
    now uses, so the curve and the stop decision can no longer disagree in sign.
    """
    traj = os.path.join(root, rid, "trajectory.csv")
    if not os.path.exists(traj):
        return [], []
    import collections, csv
    d = collections.defaultdict(lambda: [0.0, 0, 0])   # loss*poses, poses, steps
    with open(traj) as fh:
        for r in csv.DictReader(fh):
            if r.get("step_taken") != "1":
                continue
            try:
                e, v, n = int(r["epoch"]), float(r["batch_pae_loss"]), int(r["batch_n"])
            except (ValueError, KeyError, TypeError):
                continue
            d[e][0] += v * n
            d[e][1] += n
            d[e][2] += 1
    xs = sorted(e for e, v in d.items() if v[2] >= full_steps and v[1] > 0)
    return xs, [d[e][0] / d[e][1] for e in xs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run id or run_id=label")
    ap.add_argument("--root", default="training/runs")
    ap.add_argument("--stock-dockq", type=float, default=None)
    ap.add_argument("--stock-frac", type=float, default=None)
    ap.add_argument("--ceiling-dockq", type=float, default=None)
    ap.add_argument("--ceiling-frac", type=float, default=None)
    ap.add_argument("--oracle-dockq", type=float, default=None)
    ap.add_argument("--oracle-frac", type=float, default=None)
    ap.add_argument("--title", default="training pool, 77 interfaces")
    ap.add_argument("--out", default="train_eval.png")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160)
    panels = [(axes[0], 1, "mean DockQ", a.stock_dockq, a.ceiling_dockq, a.oracle_dockq),
              (axes[1], 2, "fraction DockQ >= 0.23", a.stock_frac, a.ceiling_frac, a.oracle_frac)]
    drew = False
    for ci, spec in enumerate(a.runs):
        rids, _, label = spec.partition("=")
        # A RESUMED ARM IS ONE CURVE. --resume writes a NEW run dir continuing at epoch N+1, so
        # `a+b=label` concatenates them in epoch order rather than drawing the same experiment as
        # two unrelated series.
        ids = [r for r in rids.split("+") if r]
        xs, dq, fa = [], [], []
        for rid in ids:
            s = series(a.root, rid)
            if s is None:
                sys.exit(f"no run.log for {rid}")
            xs += s[0]; dq += s[1]; fa += s[2]
        if xs:
            order = sorted(range(len(xs)), key=lambda i: xs[i])
            xs = [xs[i] for i in order]; dq = [dq[i] for i in order]; fa = [fa[i] for i in order]
        rid = ids[0]
        # THE LOSS PANEL IS INDEPENDENT of the metric panels: it comes from trajectory.csv, which
        # fills every step, while the metric lands only on even epochs. An arm with no eval point
        # yet still has a loss curve worth seeing.
        px, py = [], []
        for _r in ids:
            _px, _py = pae_by_epoch(a.root, _r)
            px += _px; py += _py
        if px:
            _o = sorted(range(len(px)), key=lambda i: px[i])
            px = [px[i] for i in _o]; py = [py[i] for i in _o]
        if px:
            axes[2].plot(px, py, marker="o", ms=4.5, lw=1.8, color=COLORS[ci % len(COLORS)],
                         label=f"{label or rid}  (n={len(px)})")
            drew = True
        if not xs:
            if not a.quiet:
                print(f"  {rid}: no TRAIN points yet"
                      + (f", {len(px)} loss epoch(s)" if px else ""))
            continue
        drew = True
        for ax, idx, _lab, *_ in panels:
            ys = dq if idx == 1 else fa
            ax.plot(xs, ys, marker="o", ms=4.5, lw=1.8, color=COLORS[ci % len(COLORS)],
                    label=f"{label or rid}  (n={len(xs)})")
        if not a.quiet:
            print(f"  {rid}: {len(xs)} point(s), last epoch {xs[-1]} = "
                  f"dockq {dq[-1]:.4f}  frac {fa[-1]:.4f}"
                  + (f"  |  PAE ep{px[-1]} {py[-1]:.4f}" if px else ""))
    if not drew:
        sys.exit("no points to plot yet")

    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("mean PAE loss")
    axes[2].set_title("complete epochs only", fontsize=9, color="#666")
    axes[2].grid(alpha=0.25, lw=0.6)
    axes[2].legend(loc="upper right", fontsize=9, frameon=False)

    for ax, _idx, ylab, stock, ceil, orac in panels:
        for v, lab, c in ((stock, "shipped head", "#888"),
                          (ceil, "perfect-PAE ceiling", "#c0392b"),
                          (orac, "oracle", "#27ae60")):
            if v is not None:
                ax.axhline(v, ls="--", lw=1.1, color=c)
                ax.annotate(f"{lab}  {v:.3f}", xy=(0.995, v), xycoords=("axes fraction", "data"),
                            ha="right", va="bottom", fontsize=8, color=c)
        ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.25, lw=0.6); ax.set_ylim(bottom=0)
        ax.legend(loc="center left", fontsize=9, frameon=False)
    fig.suptitle(a.title)
    fig.tight_layout()
    fig.savefig(a.out)
    if not a.quiet:
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
