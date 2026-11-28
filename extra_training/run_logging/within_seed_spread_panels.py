#!/usr/bin/env python
"""Is the selector BLIND within a seed, or does it see and mis-read?

Flat top-1 with a rising oracle (figures/poses_vs_seeds_matched.png) has three readings that the
oracle-vs-selected plot cannot separate:
  blind              -- the score barely moves among a seed's 5 poses
  noisy              -- it moves, but the movement is uncorrelated with quality
  confidently wrong  -- it moves, and the movement is anti-correlated with quality
Only the first is a conditioning-capacity problem, i.e. the premise the pose-injection arms were
built on. So this figure reports the two quantities that do separate them, on the same data as the
bootstrap figure (training/fold/abag_all_results.csv, pose_set=orig, the 5 original poses).

PANEL A -- variance fraction, which is the unitless form of "spread". Raw spread is not comparable
across metrics because DeepRank-Ab's predicted DockQ is on its own scale, so per chain pair we take
the within-seed share of the total sum of squares, SS_within / (SS_within + SS_between). DockQ sets
the reference: its within-seed share is how much of the available quality variation lives inside a
seed and is therefore reachable for free. A score whose share is near zero next to DockQ's is blind.

PANEL B -- Spearman against DockQ, within seeds (mean over the seed's own 5 poses) next to between
seeds (over seed-level means). Near-zero within with healthy between is "noisy"; negative within is
"confidently wrong". Panel A cannot tell those apart and Panel B cannot tell blind from noisy
(a near-constant score has an undefined within-seed ordering), which is why both are needed.

ONE BOX PER METRIC PER DATASET, over chain pairs rather than one pooled bar, because difficulty
strata behave differently and a single number hides that. Chain pairs below --min-seeds seeds are
dropped: with one seed there is no between-seed term at all, so their absence would otherwise tilt
the surviving set toward high-variation items.

THE TIE CHECK comes first in the printout. If ipTM is coarse enough that a seed's 5 samples tie,
the within-seed pick is arbitrary and the head will look blind without being blind -- that is a
different diagnosis with a different fix, so the exact-tie and near-tie rates are reported before
any of the panels are read.
"""
import argparse
import collections
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))

DS = {"pxm": "pxm", "foldbench": "foldbench", "sabdab": "sabdab (train+val83)"}
METRICS = ["DockQ", "ipTM", "DeepRank-Ab"]
COLOR = {"DockQ": "#737373", "ipTM": "#2c7fb8", "DeepRank-Ab": "#d95f0e"}


def var_frac_within(vals_by_seed):
    """SS_within / SS_total over a chain pair's poses, or None if there is no variation."""
    import numpy as np
    allv = np.concatenate(vals_by_seed)
    gm = allv.mean()
    sst = float(((allv - gm) ** 2).sum())
    if sst <= 0:
        return None
    ssw = float(sum(((v - v.mean()) ** 2).sum() for v in vals_by_seed))
    return ssw / sst


def spear(x, y):
    """Spearman rho, None when either side is constant (rho undefined, not zero)."""
    import numpy as np
    from scipy import stats as sps
    if len(x) < 3:
        return None
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    r = sps.spearmanr(x, y).statistic
    return None if r != r else float(r)


def load_deeprank(roles_csv, val_roles_csv, score_dirs):
    """(pdb, antigen model chain) -> {(seed, sample): predicted_dockq}, plus the antigen role map.

    DeepRank-Ab scores the whole antibody against ONE antigen chain, so a single score row covers
    both the H-antigen and the L-antigen chain pair of that antigen.

    TWO CHAIN NAMINGS, AND THE JOIN NEEDS BOTH. The interface letters in the results CSVs are REF
    chains, while the antigen in a score filename is that chain's MODEL name -- for 10fj they are
    permuted (model A = ref B = H, model C = ref A = ag0), so looking the interface letters up in a
    model-keyed table silently loses more than half the chain pairs. This mirrors
    training/runs/deeprank_ab_cpu/score_training.py:26, which keys roles by ref_chain and emits the
    model chain, so the two sides agree by construction rather than by coincidence.
    """
    role = {}
    for p in (roles_csv, val_roles_csv):
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                role.setdefault(r["pdb"], {})[r["ref_chain"]] = (r["role"].strip(),
                                                                 r["model_chain"])
    sc = collections.defaultdict(dict)
    for d in score_dirs:
        for f in glob.glob(os.path.join(d, "*.csv")):
            try:
                with open(f) as fh:
                    rows = list(csv.DictReader(fh))
            except OSError:
                continue
            for r in rows:
                try:
                    sc[(r["pdb"], r["antigen"])][(int(r["seed"]), int(r["sample"]))] = \
                        float(r["predicted_dockq"])
                except (KeyError, ValueError, TypeError):
                    continue
    return sc, role


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=f"{ROOT}/training/fold/abag_all_results.csv")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles_all.csv")
    ap.add_argument("--val-roles", default=f"{ROOT}/validation/fold/chain_roles_all83.csv")
    ap.add_argument("--score-dirs", default=f"{ROOT}/training/runs/deeprank_ab_cpu/scores,"
                                            f"{ROOT}/training/runs/deeprank_ab_cpu/validation/scores")
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--min-seeds", type=int, default=10)
    ap.add_argument("--tie-tol", type=float, default=1e-4)
    ap.add_argument("--out-fig", default=f"{ROOT}/figures/within_seed_spread_panels.png")
    ap.add_argument("--out-csv", default=f"{ROOT}/docs/within_seed_spread_panels.csv")
    a = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- master, orig poses only, exactly the bootstrap figure's input ----------------------
    cell = collections.defaultdict(dict)        # (ds, iface) -> seed -> {pose: (dockq, iptm)}
    for r in csv.DictReader(open(a.master)):
        if r["pose_set"] != "orig":
            continue
        ds = r["dataset"]
        if ds not in DS:
            continue
        try:
            seed, pose = int(r["seed"]), int(r["pose"])
            dq, ip = float(r["dockq"]), float(r["iptm"])
        except (ValueError, TypeError):
            continue
        cell[(ds, r["structure"], r["interface"])].setdefault(seed, {})[pose] = (dq, ip)

    sc, role = load_deeprank(a.roles, a.val_roles, a.score_dirs.split(","))

    rows = []
    tie_exact = tie_near = tie_tot = 0
    for (ds, pdb, iface), seeds in cell.items():
        full = {s: v for s, v in seeds.items() if len(v) == int(a.poses)}
        if len(full) < int(a.min_seeds):
            continue
        order = sorted(full)
        dq_by = [np.array([full[s][p][0] for p in range(a.poses)]) for s in order]
        ip_by = [np.array([full[s][p][1] for p in range(a.poses)]) for s in order]

        for v in ip_by:
            tie_tot += 1
            if len(np.unique(v)) < len(v):
                tie_exact += 1
            if float(np.ptp(v)) < a.tie_tol:
                tie_near += 1

        # DeepRank: the antigen chain of this interface, if it was scored
        # DeepRank scoring was its own job queue, so a chain pair can be scored on a SUBSET of the
        # seeds ipTM has. Dropping the whole chain pair for one missing seed cost 6 of every 10 of
        # them, so instead each metric is read on the seeds it actually has and the per-chain-pair
        # seed count is carried in the csv.
        dr_by, dr_dq = None, None
        letters = iface.split("_")[-1]
        ro = role.get(pdb) or {}
        ags = [m for c in letters for rl, m in [ro.get(c, ("", ""))] if rl.startswith("ag")]
        if len(ags) == 1 and (pdb, ags[0]) in sc:
            tbl = sc[(pdb, ags[0])]
            keep = [i for i, s in enumerate(order)
                    if all((s, p) in tbl for p in range(a.poses))]
            if len(keep) >= int(a.min_seeds):
                dr_by = [np.array([tbl[(order[i], p)] for p in range(a.poses)]) for i in keep]
                dr_dq = [dq_by[i] for i in keep]
        rec_dr_seeds = 0 if dr_by is None else len(dr_by)

        rec = {"dataset": ds, "structure": pdb, "interface": iface, "seeds": len(order),
               "seeds_deeprank": rec_dr_seeds}
        for nm, by, ref in (("DockQ", dq_by, dq_by), ("ipTM", ip_by, dq_by),
                            ("DeepRank-Ab", dr_by, dr_dq)):
            rec[f"vf_{nm}"] = "" if by is None else var_frac_within(by)
            if by is None or nm == "DockQ":
                continue
            wi = [spear(b, d) for b, d in zip(by, ref)]
            wi = [x for x in wi if x is not None]
            rec[f"sw_{nm}"] = float(np.mean(wi)) if wi else ""
            rec[f"sb_{nm}"] = spear(np.array([b.mean() for b in by]),
                                    np.array([d.mean() for d in ref]))
        rows.append(rec)

    print(f"chain pairs kept {len(rows)} (>= {a.min_seeds} complete {a.poses}-pose seeds)")
    print(f"TIE CHECK over {tie_tot} seeds: exact ipTM tie among the 5 samples "
          f"{tie_exact} ({100.0 * tie_exact / max(tie_tot, 1):.2f}%), "
          f"spread < {a.tie_tol:g} {tie_near} ({100.0 * tie_near / max(tie_tot, 1):.2f}%)")
    print()

    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    cols = ["dataset", "structure", "interface", "seeds", "seeds_deeprank"] + \
           [f"vf_{m}" for m in METRICS] + \
           [f"{p}_{m}" for m in ("sw", "sb") for m in ("ipTM", "DeepRank-Ab") for p in ("",)] + \
           [f"sw_{m}" for m in ("ipTM", "DeepRank-Ab")] + \
           [f"sb_{m}" for m in ("ipTM", "DeepRank-Ab")]
    cols = list(dict.fromkeys([c for c in cols if c and not c.startswith("_")]))
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def col(ds, key):
        return np.array([r[key] for r in rows
                         if r["dataset"] == ds and isinstance(r.get(key), float)])

    dss = [d for d in DS if any(r["dataset"] == d for r in rows)]
    fig, axes = plt.subplots(2, len(dss), figsize=(6.0 * len(dss), 10.0), squeeze=False)

    for j, ds in enumerate(dss):
        axA = axes[0][j]
        data, labs, cols_ = [], [], []
        for m in METRICS:
            v = col(ds, f"vf_{m}")
            if v.size == 0:
                continue
            data.append(v)
            labs.append(f"{m}\nn={v.size}")
            cols_.append(COLOR[m])
        bp = axA.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp["boxes"], cols_):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("black")
        axA.set_xticklabels(labs, fontsize=12)
        axA.set_ylim(-0.02, 1.02)
        axA.axhline(0, color="black", lw=0.8)
        axA.grid(axis="y", alpha=0.3)
        axA.set_title(f"{DS[ds]}\nA. within-seed share of variance", fontsize=15)
        if j == 0:
            axA.set_ylabel("SS$_{within}$ / SS$_{total}$   per chain pair", fontsize=14)
        axA.tick_params(axis="y", labelsize=12)
        for i, v in enumerate(data, 1):
            axA.text(i, 1.0, f"med {np.median(v):.2f}", ha="center", va="top", fontsize=11)

        axB = axes[1][j]
        data, labs, cols_ = [], [], []
        for m in ("ipTM", "DeepRank-Ab"):
            for p, tag in (("sw", "within seed"), ("sb", "between seed")):
                v = col(ds, f"{p}_{m}")
                if v.size == 0:
                    continue
                data.append(v)
                labs.append(f"{m}\n{tag}\nn={v.size}")
                cols_.append(COLOR[m])
        bp = axB.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c, lab in zip(bp["boxes"], cols_, labs):
            patch.set_facecolor(c)
            patch.set_alpha(0.75 if "within" in lab else 0.3)
        for med in bp["medians"]:
            med.set_color("black")
        axB.set_xticklabels(labs, fontsize=11)
        axB.set_ylim(-1.05, 1.05)
        axB.axhline(0, color="black", lw=1.0)
        axB.grid(axis="y", alpha=0.3)
        axB.set_title("B. Spearman vs DockQ", fontsize=15)
        if j == 0:
            axB.set_ylabel("Spearman rho   per chain pair", fontsize=14)
        axB.tick_params(axis="y", labelsize=12)
        for i, v in enumerate(data, 1):
            axB.text(i, 1.02, f"med {np.median(v):+.2f}", ha="center", va="top", fontsize=11)

    fig.suptitle("Blind, noisy, or confidently wrong: within-seed behaviour of each selector "
                 f"(orig {a.poses} poses, chain pairs with >= {a.min_seeds} seeds)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(a.out_fig), exist_ok=True)
    fig.savefig(a.out_fig, dpi=140)
    print(f"figure -> {a.out_fig}")
    print(f"rows   -> {a.out_csv}")
    print()

    hdr = f"{'dataset':>22s} {'metric':>14s} {'med vf':>8s} {'med rho within':>15s} " \
          f"{'med rho between':>16s} {'n':>6s}"
    print(hdr)
    for ds in dss:
        for m in METRICS:
            vf = col(ds, f"vf_{m}")
            sw = col(ds, f"sw_{m}")
            sb = col(ds, f"sb_{m}")
            if vf.size == 0:
                continue
            print(f"{DS[ds]:>22s} {m:>14s} {np.median(vf):8.3f} "
                  f"{(np.median(sw) if sw.size else float('nan')):15.3f} "
                  f"{(np.median(sb) if sb.size else float('nan')):16.3f} {vf.size:6d}")


if __name__ == "__main__":
    main()
