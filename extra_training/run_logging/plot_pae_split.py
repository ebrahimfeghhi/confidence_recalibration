#!/usr/bin/env python
"""The PAE loss split by pose group, per epoch, for an overfit arm.

    python -I run_logging/plot_pae_split.py <rid> [out.png]

WHY ITS OWN PLOTTER. plot_rel_arm draws the four ranking panels, and on a PAE-only arm every one
of them is empty: the scoring that feeds them is gated on the ranking loss. What this arm produces
instead is the per-epoch PAE loss of the acceptable poses against the unacceptable ones, which is
the whole question -- whether both groups are being fit or one is being sacrificed.

READS spread.r<rank>.csv, which the trainer writes, so the figure cannot disagree with the log.
Rows predating the columns are tolerated: a resumed run's header is written once at file
creation, so early rows can be short, and a short row is skipped rather than guessed at.

ALL-RANK FIGURES. The trainer all-reduces the sums and counts before writing them, so a row
covers the whole epoch's ~5,865 poses rather than one rank's third. It did not always: the first
four epochs of 7c8koq were rank 0 alone, and with acceptable poses at ~3.5% of the population that
third moved the acceptable count 51-80 and its mean 4.26-4.75 on the shuffle alone.

NO COUNT PANEL. It existed to show that a moving denominator could move a mean, which was a real
risk at a third of the epoch and is not at the whole of it. The counts remain in the csv and in
the printed line, where the one thing still worth watching -- unclassified staying at 0 -- shows.

THE TWO CURVES ARE NOT COMPARABLE IN LEVEL, only in trend. The groups differ in size by ~27x
(69 acceptable poses against 1,866 at epoch 1) and in composition, so the gap's absolute value
says little; what matters is whether the acceptable curve falls, and whether the gap closes.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RID = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: plot_pae_split.py <rid> [out.png]")
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/pae_split_{RID}.png"
SRC = f"training/runs/{RID}/spread.r0.csv"
if not os.path.exists(SRC):
    sys.exit(f"STOP: {SRC} does not exist. Is --pae-split-accept set on this run?")

rows = []
with open(SRC) as fh:
    for r in csv.DictReader(fh):
        if r.get("pae_acc") in (None, "", "nan") or r.get("epoch") in (None, ""):
            continue
        try:
            rows.append({k: float(r[k]) for k in
                         ("epoch", "pae_acc", "pae_unacc", "n_pae_acc", "n_pae_unacc",
                          "n_pae_straddle", "n_pae_unclassified", "pae_acc_minus_unacc")})
        except (TypeError, ValueError, KeyError):
            continue        # a row written before these columns existed
if not rows:
    sys.exit(f"STOP: {SRC} holds no row with a PAE split yet.")
rows.sort(key=lambda d: d["epoch"])
ep = [d["epoch"] for d in rows]

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

ax[0].plot(ep, [d["pae_acc"] for d in rows], "-o", color="tab:green", lw=2, ms=5,
           label=f"acceptable (n~{int(rows[-1]['n_pae_acc'])}/epoch)")
ax[0].plot(ep, [d["pae_unacc"] for d in rows], "-o", color="tab:red", lw=2, ms=5,
           label=f"unacceptable (n~{int(rows[-1]['n_pae_unacc'])}/epoch)")
ax[0].set_title("PAE loss by pose group (cropped batches, ALL ranks)", fontsize=11.5)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("mean PAE loss per pose"); ax[0].legend(fontsize=9)

ax[1].axhline(0.0, color="#888", lw=1.2, ls="--")
ax[1].plot(ep, [d["pae_acc_minus_unacc"] for d in rows], "-o", color="tab:purple", lw=2, ms=5)
ax[1].set_title("acceptable minus unacceptable\n(>0 = the good poses fit WORSE)", fontsize=11.5)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("difference in PAE loss")

for a_ in ax:
    a_.grid(alpha=.3)
fig.suptitle(f"PAE loss split by pose group -- {RID}   "
             f"(DockQ 0.23, {len(rows)} epoch(s))", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .92])
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"-> {OUT}")
for d in rows:
    print(f"  ep{int(d['epoch']):<3d} acc {d['pae_acc']:.4f} ({int(d['n_pae_acc'])})   "
          f"unacc {d['pae_unacc']:.4f} ({int(d['n_pae_unacc'])})   "
          f"diff {d['pae_acc_minus_unacc']:+.4f}   "
          f"straddle {int(d['n_pae_straddle'])}  unclass {int(d['n_pae_unclassified'])}")
