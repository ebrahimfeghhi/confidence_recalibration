#!/usr/bin/env python
"""Held-out valfull, 2 x 3: the two ipTM variants by the three things worth reading.

ROWS are the two ipTM variants the eval produces from ONE forward pass -- MEAN-pooled over the
confidence head's tokens, and MAX-pooled. The arm trains and selects by the MEAN; the MAX is the
readout nothing optimises. They are drawn as separate rows rather than overlaid because they have
traded places more than once and an overlay invites reading the better one as "the" result.

COLUMNS are mean selected DockQ, fraction acceptable (DockQ >= 0.23), and the within-interface
Spearman of the score against true DockQ. The first two are what section 1's target is stated in.
The third is why the first two do not move: it is a ranking metric over ALL candidates, and it
improves while selection does not.

SHIPPED IS A DASHED REFERENCE, not a series -- its ipTM is fixed, so it cannot move between
epochs. Its own MEAN/MAX split is real (the stored head has both poolings too) and is drawn per
row, so each panel compares like with like.

TIES ARE RESOLVED BY EXPECTATION, via select_tied.expected_pick: stored_iptm is 4dp and 11 of 92
interfaces tie, so max() would report whichever pose the shard glob happened to return first.
This differs from phase_k_eval's own rule by 0.0003 on the shipped baseline -- see select_tied.

    python -I run_logging/plot_val_2x3.py [run_id] [out.png]
"""
import csv, glob, math, os, re, statistics as st, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
import eval_dirs                                                             # noqa: E402
import plot_three                                                            # noqa: E402
import select_tied                                                           # noqa: E402

RID = sys.argv[1] if len(sys.argv) > 1 else "mieumn"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/val_2x3_{RID}.png"
ACC = 0.23
RIDS = list(plot_three.run_chain(RID))
RUNS = os.path.join(ROOT, "training", "runs")


def spearman(xs, ys):
    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                out[o[k]] = (i + j) / 2 + 1
            i = j + 1
        return out
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


# idx 0 = MAX-pooled ipTM, 1 = MEAN-pooled ipTM, 2 = shipped ipTM, 3 = true DockQ
eps, ours = [], {"MEAN": defaultdict(list), "MAX": defaultdict(list)}
ship = {"MEAN": defaultdict(list), "MAX": defaultdict(list)}
# NAME THE POOL. A run with --extra-eval writes several pools per epoch, and an unqualified read
# drew egw5wq's IN-SAMPLE numbers as held-out at epoch 10. This plot is the HELD-OUT one, so it
# asks for valfull and gets nothing rather than something wrong if that pool is absent.
_POOL = os.environ.get("VAL_POOL_TAG", "valfull")
for ep, d in sorted(eval_dirs.chain_shard_dirs(RIDS, RUNS, tag=_POOL).items()):
    by = defaultdict(list)
    for f in glob.glob(os.path.join(d, "rows_b*_s*.csv")):
        for r in csv.DictReader(open(f)):
            vm, vs = r.get("iptm_mean"), r.get("stored_iptm")
            if not vm or not vs:
                continue
            by[r["iface"]].append((float(r["iptm"]), float(vm), float(vs), float(r["dockq"])))
    if not by:
        continue
    eps.append(ep)
    for name, oi in (("MEAN", 1), ("MAX", 0)):
        dq, fa, rh = [], [], []
        sdq, sfa, srh = [], [], []
        for v in by.values():
            a, b = select_tied.expected_pick(v, oi, 3, ACC)
            dq.append(a); fa.append(b)
            # the shipped column carries ONE number per pose; its MEAN/MAX split is the same
            # stored score, so the shipped reference is identical across rows BY CONSTRUCTION
            # and is drawn once per row only so each panel is self-contained.
            a, b = select_tied.expected_pick(v, 2, 3, ACC)
            sdq.append(a); sfa.append(b)
            truth = [p[3] for p in v]
            rh.append(spearman([p[oi] for p in v], truth))
            srh.append(spearman([p[2] for p in v], truth))
        ours[name]["dq"].append(sum(dq) / len(dq))
        ours[name]["fa"].append(sum(fa) / len(fa))
        ours[name]["rho"].append(st.median(rh))
        ship[name]["dq"].append(sum(sdq) / len(sdq))
        ship[name]["fa"].append(sum(sfa) / len(sfa))
        ship[name]["rho"].append(st.median(srh))

if not eps:
    sys.exit(f"  no eval rows for {RID}")

COLS = [("dq", "mean selected DockQ", "#1b6ca8"),
        ("fa", f"frac acceptable (>= {ACC})", "#2e8b57"),
        ("rho", "median Spearman vs DockQ", "#8e44ad")]
fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.8))
for ri, name in enumerate(("MEAN", "MAX")):
    for ci, (key, lab, col) in enumerate(COLS):
        ax = axes[ri][ci]
        y, s = ours[name][key], ship[name][key]
        ax.plot(eps, y, "o-", color=col, lw=2.5, ms=6.5, zorder=4, label="ours")
        ax.axhline(s[-1], ls="--", color="#555", lw=1.8, zorder=3,
                   label=f"shipped {s[-1]:.4f}")
        ax.set_title(f"{name}-pooled ipTM  --  {lab}", fontsize=12.5, weight="bold", pad=8)
        ax.set_xlabel("epoch", fontsize=11)
        ax.set_xticks(eps)
        ax.grid(alpha=.25)
        ax.tick_params(labelsize=10)
        if ci == 0:
            ax.legend(fontsize=10, framealpha=.95)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        lo = min(min(y), s[-1]); hi = max(max(y), s[-1]); pad = max((hi - lo) * .35, 1e-4)
        ax.set_ylim(lo - pad, hi + pad)
        if key == "fa":
            ax.annotate(f"step = 1 interface = {1/92:.4f}",
                        xy=(0.5, 0.92), xycoords="axes fraction", ha="center",
                        fontsize=9.5, color="#555", style="italic")
        ax.annotate(f"{y[-1]:+.4f} vs shipped {s[-1]:+.4f}\n= {y[-1]-s[-1]:+.4f}",
                    xy=(0.5, 0.03), xycoords="axes fraction", ha="center", fontsize=10,
                    color=("#1a7f37" if y[-1] > s[-1] else "#c0392b"), weight="bold")
fig.suptitle(f"held-out valfull, 92 interfaces, 150 candidates  --  {' -> '.join(RIDS)}"
             f"  (epochs {eps[0]}-{eps[-1]})", fontsize=13.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(OUT, dpi=140)
print(f"  {len(eps)} eval epoch(s) -> {OUT}")
hdr = f"  {'ep':>3}" + "".join(f"{n+' '+k:>16}" for n in ("MEAN", "MAX") for k in ("dq", "fa", "rho"))
print(hdr)
for i, e in enumerate(eps):
    print(f"  {e:>3}" + "".join(f"{ours[n][k][i]:>16.4f}" for n in ("MEAN", "MAX")
                                for k in ("dq", "fa", "rho")))
print("  shipped" + "".join(f"{ship[n][k][-1]:>16.4f}" for n in ("MEAN", "MAX")
                            for k in ("dq", "fa", "rho")))
