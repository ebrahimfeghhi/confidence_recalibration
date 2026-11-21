#!/usr/bin/env python
"""Token-length distribution of the training set, with the pair-cost consequence of a crop.

WHY BOTH PANELS. A histogram alone answers "how long are the structures" but not the question a
crop is actually asked to answer, which is "what would it save". The pairformer's cost is
dominated by O(N^2) pair operations, so a structure's share of the epoch goes as tokens^2, not as
tokens -- the top decile by length carries far more than a tenth of the compute. The right panel
is that: for each cap, sum(min(N, cap)^2) / sum(N^2), i.e. what fraction of the uncropped pair
cost survives if every structure longer than the cap were cropped TO it.

WHAT IT IS NOT. Cropping to a fixed token budget is NOT what the union-interface crop does -- that
one keeps a structure-specific token set and lands wherever it lands (measured at 57% of tokens at
a 2.5 A margin, 74% at 5 A). The cap curve is an upper bound on what any crop of that size could
save, and a way to read which caps are even worth building for.

    python -I run_logging/plot_token_hist.py figures/token_hist.png
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANIFEST = "train_val_splits/cache_train_structures_extended.csv"
# The live arm's skip list (launch_fullhead_5site.sh). Seven of these are memory exclusions, so
# leaving them in would bias the right tail upward -- exactly the tail the crop question is about.
SKIP = set("4zto,7m7i,9b9j,9dzf,9dzg,9e7x,9mx1,9q5m".split(","))
CKPT_THRESH = 1250          # --ckpt-token-threshold on the live arm
CAPS = (256, 384, 512, 640, 768, 896, 1024, 1250, 1536)


def main(out):
    rows = list(csv.DictReader(open(MANIFEST)))
    tok = np.array([int(r["tokens"]) for r in rows if r["pdb"] not in SKIP])
    n_drop = len(rows) - len(tok)
    tot = float(np.sum(tok.astype(float) ** 2))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))

    ax.hist(tok, bins=np.arange(0, tok.max() + 64, 64), color="#4878a8",
            edgecolor="white", lw=.6)
    for v, c, lab in ((np.median(tok), "#d1495b", f"median {np.median(tok):.0f}"),
                      (CKPT_THRESH, "#2e8b57", f"ckpt threshold {CKPT_THRESH}")):
        ax.axvline(v, color=c, ls="--", lw=2, label=lab)
    ax.set_xlabel("tokens"); ax.set_ylabel("structures")
    ax.set_title(f"training set token lengths  (n={len(tok)})", fontsize=11)
    ax.grid(alpha=.3)
    # BELOW THE AXES, NOT INSIDE. At the upper right the box sat on top of the ckpt-threshold
    # line at 1250 and the right tail of the histogram; the same collision the rel-arm plot had.
    ax.legend(fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(.5, -.16), ncol=2)

    caps = np.array(CAPS, dtype=float)
    cost = np.array([np.sum(np.minimum(tok, c).astype(float) ** 2) / tot for c in caps])
    frac = np.array([(tok <= c).mean() for c in caps])
    ax2.plot(caps, 100 * cost, "-o", color="#d1495b", lw=2, ms=7,
             label="pair cost kept  sum(min(N,cap)^2)/sum(N^2)")
    ax2.plot(caps, 100 * frac, "-s", color="#4878a8", lw=2, ms=6,
             label="structures already under the cap")
    ax2.axvline(CKPT_THRESH, color="#2e8b57", ls="--", lw=2)
    ax2.set_xlabel("token cap"); ax2.set_ylabel("percent")
    ax2.set_title("what a crop to that cap would cost and cover", fontsize=11)
    ax2.set_ylim(0, 102); ax2.grid(alpha=.3)
    # The lower-right box had the 1250 line running straight through it.
    ax2.legend(fontsize=9, frameon=False, loc="upper center",
               bbox_to_anchor=(.5, -.16), ncol=1)

    fig.suptitle("Token lengths and the pair-cost consequence of cropping", fontsize=13)
    fig.tight_layout(rect=[0, .02, 1, .93])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=130)

    q = [10, 25, 50, 75, 90, 95, 99]
    print(f"{len(rows)} manifest rows, {len(tok)} after the skip list ({n_drop} dropped)")
    print("  min %d   " % tok.min()
          + "   ".join(f"p{p} {np.percentile(tok, p):.0f}" for p in q)
          + f"   max {tok.max()}")
    print(f"  mean {tok.mean():.1f}   over the ckpt threshold: "
          f"{(tok > CKPT_THRESH).sum()} structure(s), {100 * (tok > CKPT_THRESH).mean():.1f}%")
    print("\n  cap   under cap   pair cost kept")
    for c, f, k in zip(CAPS, frac, cost):
        print(f"  {c:5d}   {100*f:7.1f}%   {100*k:12.1f}%")
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "figures/token_hist.png")
