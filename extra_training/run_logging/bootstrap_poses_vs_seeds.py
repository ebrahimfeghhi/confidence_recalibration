#!/usr/bin/env python
"""SAME POSE COUNT, SPENT TWO WAYS: 1 pose x many seeds, or 5 poses x a fifth of the seeds.

THE QUESTION. A seed costs one trunk pass and yields 5 diffusion poses. If ipTM cannot rank poses
inside a seed, then 4 of every 5 poses are unselectable and the budget should buy seeds instead.
This holds the SELECTABLE POOL SIZE fixed and varies only how it was assembled, so any difference
is attributable to seed diversity versus pose diversity and not to pool size.

THE TWO CONDITIONS, per chain pair with S >= 10 seeds, k = S // 5, N = 5k poses in BOTH:
  1 pose x 5k seeds    draw 5k seeds without replacement, 1 random pose each -> 5k poses, 5k trunks
  5 poses x k seeds    draw k  seeds without replacement, all 5 poses        -> 5k poses,  k trunks
  1 pose x k seeds     draw k  seeds without replacement, 1 random pose each ->  k poses,  k trunks

THE TWO COMPARISONS THE THREE ARMS GIVE, and they answer different questions:
  arm 1 vs arm 2   POSE-MATCHED, 5k poses each -- is seed diversity or pose diversity the better
                   way to assemble a selectable pool of a given size? Arm 1 costs 5x the trunk
                   passes, so a win there is not yet actionable.
  arm 2 vs arm 3   COMPUTE-MATCHED, k trunk passes each -- what do the extra 4 poses per seed
                   actually buy once you have paid for the seed? This is the comparison that maps
                   onto a fixed 50-seed budget, and arm 3 has a FIFTH of the pool, so any win for
                   arm 2 is bought with poses that cost almost nothing.

WHAT IS REPORTED, per bootstrap replicate: each metric is computed per chain pair and then averaged
over chain pairs, giving one number per replicate; 300 replicates give the distribution drawn as a
box. SELECTED means the pose with the highest ipTM in that replicate's pool; ORACLE means the best
pose in it, i.e. the ceiling a perfect reranker would reach on the same pool.
  mean DockQ                       selected and oracle
  fraction DockQ >= 0.23           selected and oracle   (acceptable)
  fraction DockQ >= 0.49           selected and oracle   (medium)

ORIGINAL 5 POSES ONLY. The regenerated poses are a different diffusion seed and ran -0.0076 lower
in mean DockQ, so including them would make "5 poses per seed" mean something different in sabdab
than in the other two pools.
"""
import argparse
import collections
import csv
import functools
import os

print = functools.partial(print, flush=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESH = (0.23, 0.49)
DS = {"pxm": "pxm", "foldbench": "foldbench", "sabdab": "sabdab (train+val)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=f"{ROOT}/training/fold/abag_all_results.csv")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--min-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import numpy as np

    # dataset -> chain pair -> seed -> {pose: (dockq, iptm)}
    raw = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(dict)))
    for r in csv.DictReader(open(a.master)):
        if r["pose_set"] != "orig":
            continue
        ds = DS.get(r["dataset"])
        if ds is None:
            continue
        try:
            raw[ds][r["interface"]][r["seed"]][int(r["pose"])] = (
                float(r["dockq"]), float(r["iptm"]))
        except (ValueError, KeyError, TypeError):
            continue

    # keep chain pairs with >= min_seeds complete 5-pose seeds; pack to arrays [S, 5]
    pool = collections.defaultdict(list)
    for ds, ifaces in raw.items():
        for iface, seeds in ifaces.items():
            good = [s for s, d in seeds.items() if len(d) == 5]
            if len(good) < a.min_seeds:
                continue
            dq = np.array([[seeds[s][k][0] for k in sorted(seeds[s])] for s in good])
            ip = np.array([[seeds[s][k][1] for k in sorted(seeds[s])] for s in good])
            pool[ds].append((iface, dq, ip))

    for ds in pool:
        S = [d.shape[0] for _, d, _ in pool[ds]]
        print(f"{ds:20s} chain pairs {len(pool[ds]):5d}   seeds/pair min {min(S)} "
              f"med {int(np.median(S))} max {max(S)}   "
              f"N poses per replicate med {5 * (int(np.median(S)) // 5)}")

    rng = np.random.default_rng(a.seed)
    METRICS = ["mean DockQ", f"frac DockQ >= {THRESH[0]}", f"frac DockQ >= {THRESH[1]}"]
    CONDS = ["1 pose x 5k seeds", "5 poses x k seeds", "1 pose x k seeds"]
    # dataset -> (cond, sel|orc) -> metric -> [replicate values]
    res = collections.defaultdict(lambda: collections.defaultdict(
        lambda: collections.defaultdict(list)))

    for ds, items in pool.items():
        for b in range(a.boot):
            acc = collections.defaultdict(lambda: collections.defaultdict(list))
            for _iface, dq, ip in items:
                S = dq.shape[0]
                k = S // 5
                n_seed_arm = k
                n_pose_arm = 5 * k
                # arm 1: n_pose_arm seeds, one random pose each
                si = rng.choice(S, size=n_pose_arm, replace=False)
                pj = rng.integers(0, 5, size=n_pose_arm)
                d1, i1 = dq[si, pj], ip[si, pj]
                # arm 2: n_seed_arm seeds, all 5 poses
                sj = rng.choice(S, size=n_seed_arm, replace=False)
                d2, i2 = dq[sj].ravel(), ip[sj].ravel()
                # arm 3: the SAME k seeds as arm 2, one random pose each -- same trunk passes,
                # a fifth of the poses. Sharing arm 2's seed draw makes the pair differ only in
                # how many poses per seed are kept, which is the whole question.
                pk = rng.integers(0, 5, size=n_seed_arm)
                d3, i3 = dq[sj, pk], ip[sj, pk]
                for cond, (d, i) in zip(CONDS, ((d1, i1), (d2, i2), (d3, i3))):
                    sel = float(d[int(i.argmax())])
                    orc = float(d.max())
                    acc[(cond, "selected")]["mean DockQ"].append(sel)
                    acc[(cond, "oracle")]["mean DockQ"].append(orc)
                    for t, lab in zip(THRESH, METRICS[1:]):
                        acc[(cond, "selected")][lab].append(1.0 if sel >= t else 0.0)
                        acc[(cond, "oracle")][lab].append(1.0 if orc >= t else 0.0)
            for key, m in acc.items():
                for lab, v in m.items():
                    res[ds][key][lab].append(float(np.mean(v)))
        print(f"  {ds}: {a.boot} replicate(s) done")

    print()
    rows = []
    for ds in sorted(res):
        print(f"=== {ds}   ({len(pool[ds])} chain pairs, {a.boot} bootstraps)")
        print(f"{'condition':22} {'which':9} " + " ".join(f"{m:>24}" for m in METRICS))
        for cond in CONDS:
            for which in ("selected", "oracle"):
                cells = []
                for m in METRICS:
                    v = np.array(res[ds][(cond, which)][m])
                    cells.append(f"{v.mean():.4f} [{np.percentile(v, 2.5):.4f},"
                                 f"{np.percentile(v, 97.5):.4f}]")
                    rows.append({"dataset": ds, "condition": cond, "which": which,
                                 "metric": m, "mean": v.mean(),
                                 "lo95": np.percentile(v, 2.5),
                                 "hi95": np.percentile(v, 97.5)})
                print(f"{cond:22} {which:9} " + " ".join(f"{c:>24}" for c in cells))
        # the paired difference that is the point: pose arm minus seed arm, per replicate
        for lab, ia, ib in (("POSE-MATCHED    arm1 - arm2", 0, 1),
                            ("COMPUTE-MATCHED arm2 - arm3", 1, 2)):
            print(lab)
            for which in ("selected", "oracle"):
                cells = []
                for m in METRICS:
                    d = (np.array(res[ds][(CONDS[ia], which)][m])
                         - np.array(res[ds][(CONDS[ib], which)][m]))
                    star = ("*" if (np.percentile(d, 2.5) > 0)
                            or (np.percentile(d, 97.5) < 0) else " ")
                    cells.append(f"{d.mean():+.4f} [{np.percentile(d, 2.5):+.4f},"
                                 f"{np.percentile(d, 97.5):+.4f}]{star}")
                    rows.append({"dataset": ds, "condition": lab, "which": which,
                                 "metric": m, "mean": d.mean(),
                                 "lo95": np.percentile(d, 2.5),
                                 "hi95": np.percentile(d, 97.5)})
                print(f"{'':22} {which:9} " + " ".join(f"{c:>24}" for c in cells))
        print()

    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"rows -> {a.out}")

    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        dss = sorted(res)
        fig, axes = plt.subplots(len(dss), 3, figsize=(20, 5.2 * len(dss)))
        if len(dss) == 1:
            axes = axes.reshape(1, -1)
        cols = {("1 pose x 5k seeds", "selected"): "#2c7fb8",
                ("1 pose x 5k seeds", "oracle"): "#a6cee3",
                ("5 poses x k seeds", "selected"): "#d95f0e",
                ("5 poses x k seeds", "oracle"): "#fdbf6f",
                ("1 pose x k seeds", "selected"): "#238b45",
                ("1 pose x k seeds", "oracle"): "#a1d99b"}
        order = [(c, w) for c in CONDS for w in ("selected", "oracle")]
        for ri, ds in enumerate(dss):
            for ci, m in enumerate(METRICS):
                ax = axes[ri, ci]
                data = [res[ds][k][m] for k in order]
                bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
                for patch, k in zip(bp["boxes"], order):
                    patch.set_facecolor(cols[k])
                for med in bp["medians"]:
                    med.set_color("black")
                ax.set_xticks(range(1, len(order) + 1))
                short = {CONDS[0]: "1 pose\n5k seeds", CONDS[1]: "5 poses\nk seeds",
                         CONDS[2]: "1 pose\nk seeds"}
                ax.set_xticklabels([f"{short[k[0]]}\n{k[1]}" for k in order], fontsize=12)
                ax.tick_params(axis="y", labelsize=12)
                ax.set_title(f"{ds} -- {m}", fontsize=16)
                ax.grid(axis="y", alpha=0.3)
                if ci == 0:
                    ax.set_ylabel("mean over chain pairs", fontsize=14)
        fig.suptitle(
            f"Blue 1 pose x 5k seeds (5k poses, 5k trunks)   |   "
            f"Orange 5 poses x k seeds (5k poses, k trunks)   |   "
            f"Green 1 pose x k seeds (k poses, k trunks)\n"
            f"blue vs orange = pose-matched, orange vs green = compute-matched.   "
            f"{a.boot} bootstraps, chain pairs with >= {a.min_seeds} seeds, "
            f"original 5 poses only", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(a.fig, dpi=130)
        print(f"figure -> {a.fig}")


if __name__ == "__main__":
    main()
