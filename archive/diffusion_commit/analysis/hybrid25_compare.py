#!/usr/bin/env python
"""hybrid25: does self-templating 25 donor seeds beat the 25 free seeds it came from?

THE COMPARISON, matched on donors rather than on compute. hybrid25 templated on the five poses of
each free seed 300000-24 and re-folded as 300025-49. So the fair free baseline is selection over
seeds 300000-24 ONLY -- the exact seeds the templates came from -- not over the whole 50-seed pool.
Using all 50 would give the free arm twice the draws and measure seed count, not templating.

SELECTION IS BY THE ARM'S OWN CONFIDENCE: the pose with the highest pair ipTM, and we report that
pose's DockQ. Not the oracle best. An earlier version of this comparison in the session reported an
ipTM value in a DockQ column (tuples built as (dockq, iptm), then max() then [1]), which inflated the
apparent gain; hence the explicit key= everywhere below.

Also asks whether confidence tracks correctness in each arm: Pearson and Spearman of ipTM against
DockQ over every pose. That is the question behind the whole project -- ipTM can rise because the
structure improved, or because a template manufactured self-consistency.

    hybrid25_compare.py [--out <png dir>]
"""
import argparse
import collections
import csv
import os
import statistics as st
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")

import paths          # noqa: E402
from fmt import table  # noqa: E402

DONOR_SEEDS = set(range(300000, 300025))     # what hybrid25 templated FROM
H25 = f"{paths.DATA_DIR}/archive/hybrid25/results.csv"
ACCEPT = 0.23


def load():
    """(free poses, templated poses) as {iface: [(iptm, dockq)]}, plus each interface's class."""
    free = collections.defaultdict(list)
    for r in csv.DictReader(open(str(paths.PXM2025_POOL_CSV))):
        if int(r["seed"]) in DONOR_SEEDS:
            free[f"{r['pdb']}_{r['chain1']}{r['chain2']}"].append(
                (float(r["pair_iptm"]), float(r["dockq"])))
    tmpl, cls = collections.defaultdict(list), {}
    for r in csv.DictReader(open(H25)):
        tmpl[r["interface"]].append((float(r["iptm"]), float(r["dockq"])))
        cls[r["interface"]] = r["cls"]
    return free, tmpl, cls


def selected(poses):
    """DockQ of the pose this arm's own ipTM picks."""
    return max(poses, key=lambda p: p[0])[1]


def corr(poses):
    xs = [p[0] for p in poses]
    ys = [p[1] for p in poses]
    try:
        from scipy.stats import pearsonr, spearmanr
        return pearsonr(xs, ys)[0], spearmanr(xs, ys)[0]
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{paths.DATA_DIR}/archive/hybrid25")
    A = ap.parse_args()

    free, tmpl, cls = load()
    ifaces = sorted(set(free) & set(tmpl))
    rows = []
    for i in ifaces:
        f, t = selected(free[i]), selected(tmpl[i])
        rows.append([i, cls[i], len(free[i]), len(tmpl[i]), f, t, t - f])
    rows.sort(key=lambda r: -r[6])

    print(table(["interface", "class", "free poses", "tmpl poses",
                 "free SELECTED", "tmpl SELECTED", "gain"], rows,
                align="l l r r r r r",
                dec={"free SELECTED": 3, "tmpl SELECTED": 3, "gain": 3}))
    print(f"\n  n = {len(rows)} interfaces "
          f"(free seeds 300000-24 vs templated 300025-49)\n")

    per = []
    for c in ("decoy", "nocov", "solved", "ALL"):
        sub = [r for r in rows if c == "ALL" or r[1] == c]
        if not sub:
            continue
        g = [r[6] for r in sub]
        per.append([c, len(sub), st.mean([r[4] for r in sub]), st.mean([r[5] for r in sub]),
                    st.mean(g), sum(1 for x in g if x > 0.05), sum(1 for x in g if x < -0.05),
                    sum(1 for r in sub if r[4] >= ACCEPT), sum(1 for r in sub if r[5] >= ACCEPT)])
    print(table(["class", "n", "free sel", "tmpl sel", "mean gain", "better", "worse",
                 f"free ok(>={ACCEPT})", "tmpl ok"], per, align="l r r r r r r r r",
                dec={"free sel": 3, "tmpl sel": 3, "mean gain": 3}))

    allfree = [p for i in ifaces for p in free[i]]
    alltmpl = [p for i in ifaces for p in tmpl[i]]
    fr, fs = corr(allfree)
    tr, ts = corr(alltmpl)
    print("\n  ipTM vs DockQ over every pose -- does confidence track correctness?")
    print(table(["arm", "poses", "Pearson r", "Spearman rho"],
                [["free (300000-24)", len(allfree), fr, fs],
                 ["templated (300025-49)", len(alltmpl), tr, ts]],
                align="l r r r", dec={"Pearson r": 3, "Spearman rho": 3}))

    # PER-INTERFACE correlation. The pooled r above is dominated by between-interface spread in
    # difficulty; what matters for rescuing one interface is whether ipTM ranks ITS OWN poses, so the
    # correlation has to be computed within each interface separately.
    crow, undef = [], []
    for i in ifaces:
        fr_i, _ = corr(free[i])
        tr_i, _ = corr(tmpl[i])
        # a near-constant DockQ (every pose in the same basin) makes r undefined, not zero
        if any(x != x for x in (fr_i, tr_i)):
            undef.append(i)
            continue
        crow.append([i, cls[i], fr_i, tr_i, tr_i - fr_i])
    crow.sort(key=lambda r: -r[4])
    print("\n  ipTM vs DockQ WITHIN each interface (125 poses each)")
    print(table(["class", "n", "free r", "tmpl r", "mean change", "r improved", "r worsened"],
                [[c, len([r for r in crow if c == "ALL" or r[1] == c]),
                  st.mean([r[2] for r in crow if c == "ALL" or r[1] == c]),
                  st.mean([r[3] for r in crow if c == "ALL" or r[1] == c]),
                  st.mean([r[4] for r in crow if c == "ALL" or r[1] == c]),
                  sum(1 for r in crow if (c == "ALL" or r[1] == c) and r[4] > 0.05),
                  sum(1 for r in crow if (c == "ALL" or r[1] == c) and r[4] < -0.05)]
                 for c in ("decoy", "nocov", "solved", "ALL")
                 if [r for r in crow if c == "ALL" or r[1] == c]],
                align="l r r r r r r",
                dec={"free r": 3, "tmpl r": 3, "mean change": 3}))
    if undef:
        print(f"  {len(undef)} interfaces excluded: DockQ is constant across their poses, so r is "
              f"undefined ({', '.join(undef[:4])}{' ...' if len(undef) > 4 else ''})")

    plot(rows, allfree, alltmpl, (fr, tr), A.out)
    plot_corr(crow, A.out)
    return 0


def plot(rows, allfree, alltmpl, rs, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    colour = {"decoy": "#c2412d", "nocov": "#6b7280", "solved": "#2f6f4e"}

    # --- selected DockQ, free vs templated -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, zorder=1)
    ax.axhline(ACCEPT, color="#9ca3af", lw=0.6, ls=":", zorder=1)
    ax.axvline(ACCEPT, color="#9ca3af", lw=0.6, ls=":", zorder=1)
    for c in ("solved", "nocov", "decoy"):
        pts = [(r[4], r[5]) for r in rows if r[1] == c]
        if pts:
            ax.scatter(*zip(*pts), s=34, alpha=0.8, label=f"{c} (n={len(pts)})",
                       color=colour[c], edgecolor="white", linewidth=0.6, zorder=3)
    up = sum(1 for r in rows if r[6] > 0.05)
    dn = sum(1 for r in rows if r[6] < -0.05)
    ax.set_xlabel("free selected DockQ  (seeds 300000-24)")
    ax.set_ylabel("templated selected DockQ  (seeds 300025-49)")
    ax.set_title("Self-templating vs the seeds it templated from\n"
                 f"above the line: {up}   below: {dn}   n={len(rows)}", fontsize=10)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p1 = f"{outdir}/hybrid25_selected_scatter.png"
    fig.savefig(p1, dpi=160)

    # --- ipTM vs DockQ per pose, both arms -----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, (poses, name, r) in zip(axes, ((allfree, "free (300000-24)", rs[0]),
                                          (alltmpl, "templated (300025-49)", rs[1]))):
        ax.scatter([p[0] for p in poses], [p[1] for p in poses], s=5, alpha=0.18,
                   color="#2f4f7f", linewidth=0)
        ax.axhline(ACCEPT, color="#9ca3af", lw=0.6, ls=":")
        ax.set_title(f"{name}\nn={len(poses)} poses,  Pearson r = {r:.3f}", fontsize=10)
        ax.set_xlabel("pair ipTM")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("DockQ")
    fig.suptitle("Does confidence track correctness?", fontsize=11)
    fig.tight_layout()
    p2 = f"{outdir}/hybrid25_iptm_vs_dockq.png"
    fig.savefig(p2, dpi=160)
    print(f"\n  wrote {p1}\n  wrote {p2}")


def plot_corr(crow, outdir):
    """Per-interface ipTM-DockQ correlation, free vs templated. One point per interface."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colour = {"decoy": "#c2412d", "nocov": "#6b7280", "solved": "#2f6f4e"}
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot([-1, 1], [-1, 1], color="#9ca3af", lw=1, zorder=1)
    ax.axhline(0, color="#d1d5db", lw=0.6, zorder=1)
    ax.axvline(0, color="#d1d5db", lw=0.6, zorder=1)
    for c in ("solved", "nocov", "decoy"):
        pts = [(r[2], r[3]) for r in crow if r[1] == c]
        if pts:
            ax.scatter(*zip(*pts), s=34, alpha=0.8, label=f"{c} (n={len(pts)})",
                       color=colour[c], edgecolor="white", linewidth=0.6, zorder=3)
    up = sum(1 for r in crow if r[4] > 0.05)
    dn = sum(1 for r in crow if r[4] < -0.05)
    ax.set_xlabel("free: ipTM-DockQ r within interface")
    ax.set_ylabel("templated: ipTM-DockQ r within interface")
    ax.set_title("Does confidence rank an interface's OWN poses better after templating?\n"
                 f"above the line: {up}   below: {dn}   n={len(crow)}", fontsize=10)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = f"{outdir}/hybrid25_corr_scatter.png"
    fig.savefig(p, dpi=160)
    print(f"  wrote {p}")


if __name__ == "__main__":
    raise SystemExit(main())
