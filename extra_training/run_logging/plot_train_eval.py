#!/usr/bin/env python
"""frac-acceptable on the TRAINING pool, per run, at the epochs it was evaluated.

Reads the `epoch N TRAIN:` lines the trainer prints, so it needs nothing the run does not already
write and can be re-run while the runs are still going.
"""
import argparse, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"epoch (\d+) TRAIN: mean_dockq ([0-9.]+)\s+frac_acc ([0-9.]+)")

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+", help="run id or run_id=label")
ap.add_argument("--root", default="training/runs")
ap.add_argument("--metric", default="frac_acc", choices=["frac_acc", "mean_dockq"])
ap.add_argument("--stock", type=float, default=None)
ap.add_argument("--ceiling", type=float, default=None, help="perfect-PAE ceiling")
ap.add_argument("--oracle", type=float, default=None)
ap.add_argument("--title", default="")
ap.add_argument("--out", default="train_eval.png")
a = ap.parse_args()

fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
COLORS = ["#1f77b4", "#e67e22", "#8e44ad", "#16a085"]
for ci, spec in enumerate(a.runs):
    rid, _, label = spec.partition("=")
    log = os.path.join(a.root, rid, "run.log")
    if not os.path.exists(log):
        sys.exit(f"no {log}")
    xs, ys = [], []
    for m in LINE.finditer(open(log, errors="replace").read()):
        xs.append(int(m.group(1)))
        ys.append(float(m.group(3) if a.metric == "frac_acc" else m.group(2)))
    if not xs:
        print(f"  {rid}: no TRAIN points yet, skipping")
        continue
    ax.plot(xs, ys, marker="o", ms=4.5, lw=1.8, color=COLORS[ci % len(COLORS)],
            label=f"{label or rid}  (n={len(xs)})")
    print(f"  {rid}: {len(xs)} point(s), last epoch {xs[-1]} = {ys[-1]:.4f}")

for v, lab, c in ((a.stock, "shipped head", "#888"),
                  (a.ceiling, "perfect-PAE ceiling", "#c0392b"),
                  (a.oracle, "oracle", "#27ae60")):
    if v is not None:
        ax.axhline(v, ls="--", lw=1.1, color=c)
        ax.annotate(f"{lab}  {v:.3f}", xy=(0.995, v), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color=c)

ax.set_xlabel("epoch")
ax.set_ylabel("fraction DockQ >= 0.23" if a.metric == "frac_acc" else "mean DockQ")
ax.set_title(a.title or "training-pool metric, 77 interfaces")
ax.grid(alpha=0.25, lw=0.6)
ax.legend(loc="center left", fontsize=9, frameon=False)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(a.out)
print(f"wrote {a.out}")
