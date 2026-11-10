#!/usr/bin/env python
"""Plot what Protenix's interface crop does to the iTM label, from crop_itm_rank.py's CSVs.

THREE PANELS, AND THE THIRD IS THE ONE THAT DECIDES ANYTHING.

  1. rho delta: mean_c Spearman(cropped iTM, DockQ) - Spearman(uncropped iTM, DockQ). How much of
     the label's ORDERING survives the crop.
  2. regret delta: mean_c [oracle - DockQ(crop's pick)] - [oracle - DockQ(uncropped pick)].
     Regret is what we actually deploy against -- only the argmax is ever used -- and this project
     has watched rho improve four times without selected DockQ following, so the two panels can
     and probably will disagree. Positive means the crop picks worse poses.
  3. regret delta against GT interface retention. The expected failure mode is a crop whose random
     anchor lands away from the antibody-antigen interface: get_interface_token accepts ANY pair
     of distinct chains, heavy-light packing included. If that is the mechanism, the damage should
     concentrate at low retention, and this panel shows it directly.

Usage:
    python -I run_logging/plot_crop_itm_rank.py training/runs/crop_itm/rank.csv \
        figures/crop_itm_rank.png
"""
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(pattern):
    files = sorted(glob.glob(pattern)) or sorted(glob.glob(pattern.replace(".csv", ".s*.csv")))
    if not files:
        raise SystemExit(f"STOP: no CSV matched {pattern}")
    rows = []
    for f in files:
        rows += list(csv.DictReader(open(f)))
    return rows, files


def main(src, out):
    rows, files = load(src)
    f = lambda k: np.array([float(r[k]) for r in rows])              # noqa: E731
    rho_d, reg_d, ret = f("delta"), f("regret_delta"), f("retention_mean")
    ok = ~(np.isnan(rho_d) | np.isnan(reg_d))
    rho_d, reg_d, ret = rho_d[ok], reg_d[ok], ret[ok]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, v, lab, col in ((axes[0], rho_d, "Spearman(iTM, DockQ) delta", "#4878a8"),
                            (axes[1], reg_d, "DockQ regret delta", "#d1495b")):
        ax.hist(v, bins=40, color=col, edgecolor="white", lw=.5)
        ax.axvline(0, color="k", lw=1.5)
        ax.axvline(v.mean(), color="#2e8b57", ls="--", lw=2,
                   label=f"mean {v.mean():+.4f}")
        ax.set_xlabel(f"cropped - uncropped   ({lab})")
        ax.set_ylabel("interfaces")
        ax.grid(alpha=.3)
        ax.legend(fontsize=9, frameon=False, loc="upper center",
                  bbox_to_anchor=(.5, -.16))
    axes[0].set_title("ordering: negative = crop loses ranking signal", fontsize=11)
    axes[1].set_title("selection: POSITIVE = crop picks worse poses", fontsize=11)

    axes[2].scatter(ret, reg_d, s=12, alpha=.45, color="#d1495b", edgecolors="none")
    axes[2].axhline(0, color="k", lw=1.5)
    # binned median, so the trend is readable through the scatter
    if len(ret) > 40:
        edges = np.linspace(ret.min(), ret.max(), 9)
        mid = .5 * (edges[1:] + edges[:-1])
        med = [np.median(reg_d[(ret >= lo) & (ret < hi)]) if ((ret >= lo) & (ret < hi)).sum() >= 5
               else np.nan for lo, hi in zip(edges[:-1], edges[1:])]
        axes[2].plot(mid, med, "-o", color="#2e8b57", lw=2, ms=6, label="binned median")
        axes[2].legend(fontsize=9, frameon=False, loc="upper center",
                       bbox_to_anchor=(.5, -.16))
    axes[2].set_xlabel("fraction of the interface's scored tokens kept")
    axes[2].set_ylabel("DockQ regret delta")
    axes[2].set_title("does damage track losing the interface?", fontsize=11)
    axes[2].grid(alpha=.3)

    n_crops = rows[0].get("n_crops", "?")
    cs = rows[0].get("crop_size", "?")
    fig.suptitle(f"Protenix interface crop vs the iTM label -- {len(rho_d)} interfaces, "
                 f"{n_crops} crops each, crop_size {cs}", fontsize=13)
    fig.tight_layout(rect=[0, .02, 1, .93])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)

    print(f"{len(rows)} interface(s) from {len(files)} file(s); {len(rho_d)} usable")
    for nm, v in (("rho delta", rho_d), ("regret delta", reg_d)):
        print(f"  {nm:13s} mean {v.mean():+.4f}  median {np.median(v):+.4f}  "
              f"sd {v.std():.4f}  p10 {np.percentile(v,10):+.4f}  p90 {np.percentile(v,90):+.4f}  "
              f"worse on {100*(v>0).mean() if 'regret' in nm else 100*(v<0).mean():.0f}% of interfaces")
    print(f"  retention     mean {ret.mean():.3f}  p10 {np.percentile(ret,10):.3f}")
    print(f"  regret: uncropped mean {f('regret_full')[ok].mean():.4f}  "
          f"cropped mean {f('regret_crop_mean')[ok].mean():.4f}  "
          f"crop-to-crop sd {f('regret_crop_std')[ok].mean():.4f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "training/runs/crop_itm/rank.csv",
         sys.argv[2] if len(sys.argv) > 2 else "figures/crop_itm_rank.png")
