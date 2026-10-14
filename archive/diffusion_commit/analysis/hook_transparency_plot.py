#!/usr/bin/env python
"""Is the mid-loop hook transparent? Plot arm P (plain N recycles) against arm S (hook re-injects the
templates the fold was ALREADY using).

Both arms must be the same run: arm S splices values that cannot change the answer. In fp32 two
identical folds agree to ~0.001 DockQ, so anything materially off the diagonal means the act of
decoding mid-loop perturbs the fold by itself -- which would confound any later measurement of what
self-templated poses do, since the effect we are chasing is the same size.

    hook_transparency_plot.py --s <arm S csv> --p <arm P csv> --out <png> [--label "fp32"]
"""
import argparse
import collections
import csv
import statistics as st
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")

THRESH = 0.02          # above this, a difference is not reproducibility noise


def load(path, arm, into):
    for r in csv.DictReader(open(path)):
        into[(r["iface"], int(r["sample"]))][arm] = (float(r["dockq"]), float(r["iptm"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--p", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    A = ap.parse_args()

    D = collections.defaultdict(dict)
    load(A.p, "P", D)
    load(A.s, "S", D)
    pts = [(k[0], k[1], v["P"][0], v["S"][0], v["P"][1], v["S"][1])
           for k, v in sorted(D.items()) if "P" in v and "S" in v]
    diffs = [abs(s - p) for _, _, p, s, _, _ in pts]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ifaces = sorted({p[0] for p in pts})
    cmap = plt.get_cmap("tab10")
    colour = {f: cmap(i % 10) for i, f in enumerate(ifaces)}

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))

    # --- left: P vs S, one point per pose ------------------------------------------------------
    ax = axes[0]
    lo, hi = -0.03, 1.03
    ax.plot([lo, hi], [lo, hi], color="#9ca3af", lw=1, zorder=1)
    # a band showing what reproducibility noise alone would look like
    ax.fill_between([lo, hi], [lo - THRESH, hi - THRESH], [lo + THRESH, hi + THRESH],
                    color="#9ca3af", alpha=0.18, lw=0, zorder=0,
                    label=f"±{THRESH} (noise band)")
    for f in ifaces:
        sub = [(p, s) for i, _, p, s, _, _ in pts if i == f]
        ax.scatter(*zip(*sub), s=46, alpha=0.9, color=colour[f], edgecolor="white",
                   linewidth=0.7, zorder=3, label=f)
    ax.set_xlabel("arm P: plain 11 recycles, DockQ")
    ax.set_ylabel("arm S: hook re-injects identical templates, DockQ")
    ax.set_title(f"Every pose should sit on the diagonal\n"
                 f"{sum(1 for d in diffs if d > THRESH)} of {len(diffs)} poses off by >{THRESH}",
                 fontsize=10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # --- right: the same differences, sorted, so the scale is unmissable -----------------------
    ax = axes[1]
    order = sorted(pts, key=lambda r: abs(r[3] - r[2]))
    xs = range(len(order))
    ax.bar(xs, [abs(r[3] - r[2]) for r in order],
           color=[colour[r[0]] for r in order], width=0.82)
    ax.axhline(THRESH, color="#c2412d", lw=1, ls="--",
               label=f"{THRESH} = above reproducibility noise")
    ax.axhline(0.001, color="#6b7280", lw=1, ls=":",
               label="0.001 = fp32 repeat-fold agreement")
    ax.set_yscale("symlog", linthresh=0.001)
    ax.set_xlabel("pose, sorted by disagreement")
    ax.set_ylabel("|arm S DockQ - arm P DockQ|")
    ax.set_title(f"median {st.median(diffs):.4f}   max {max(diffs):.4f}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Mid-loop hook transparency check" + (f" -- {A.label}" if A.label else ""),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(A.out, dpi=160)
    print(f"wrote {A.out}")
    print(f"  poses {len(diffs)} | median {st.median(diffs):.4f} | max {max(diffs):.4f} "
          f"| >{THRESH}: {sum(1 for d in diffs if d > THRESH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
