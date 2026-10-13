#!/usr/bin/env python
"""Fraction of interfaces that are acceptable / decoy / no-acceptable, versus seed depth.

    ./val_seed_depth_curve.py --out .../val_seed_depth                       # sabdab_val2
    ./val_seed_depth_curve.py --arm "all new" --out .../val_seed_depth_allnew
    ./val_seed_depth_curve.py --dataset pxm2025 --weighting interface \
        --arm-source sabdab --out .../pxm_seed_depth                         # a benchmark

WEIGHTING. `interface` (the default) weights every interface equally, which is how these numbers are
normally reported. `cluster` averages within each cluster then over clusters equally, which stops a
cluster that contributes two representatives from counting twice; it is available with --weighting
cluster and is the stricter choice for the validation set, where 13 of 40 clusters have two.

ARM SOURCE. `split-csv` reads the `kind` column of our frozen validation split. `sabdab` derives it
for any structure by looking its PDB id up in SAbDab2: a structure is `mixed` when its ab_ag_cluster
also holds pre-2021-09-30 structures, `all new` when it does not, and is dropped from an arm-filtered
plot when SAbDab2 does not contain it at all.

Writes <out>.png, <out>.pdf and <out>.csv.

CLASSES, per interface, for a given draw of k seeds:
  no acceptable  no pose in the draw reaches DockQ --thresh        a sampling failure
  decoy          an acceptable pose is in the draw but the top pair-ipTM pose is below it
                                                                   a selection failure
  acceptable     the top pair-ipTM pose is above it

WEIGHTING is the frozen convention: fractions are computed within each cluster, then averaged over
clusters with equal weight, so a cluster contributing two representatives counts once.

DRAWS. k=1 enumerates all --universe single-seed draws exactly; there are only 100 of them, so
sampling would be strictly worse than counting. k=--universe has exactly one possible draw and
therefore zero spread by construction. In between, --boots draws without replacement under a fixed
seed. The band is +-1 sd ACROSS DRAWS -- the spread of the estimate at that depth, not a confidence
interval on a population.

The x axis is log-scaled: the depths are 1, 5, 10 then multiples of 10, and on a linear axis the
region where the curves actually move is compressed into the first centimetre.
"""
import argparse
import collections
import csv
import math
import os
import random
import statistics as st
import sys

sys.path[:0] = ["/home/ebrahim/tts_abag"]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter  # noqa: E402
import paths                             # noqa: E402

CLASSES = ("acceptable", "decoy", "no acceptable")
COLOURS = {"acceptable": "#2d7d46", "decoy": "#c2691f", "no acceptable": "#8c2d3f"}


SPLITS = ("/home/ebrahim/tts_abag/extra_training/data/zenodo_splits/splits_final/abag_split.csv")
CUT = "2021-09-30"


def sabdab_kind_by_pdb():
    """{pdb4: 'mixed'|'all new'} from SAbDab2; absent pdbs are simply not in the dict."""
    pre, post, pdb2cl = (collections.defaultdict(set), collections.defaultdict(set),
                         collections.defaultdict(set))
    for r in csv.DictReader(open(SPLITS)):
        p, c = r["PDB_ID"][-4:].lower(), str(r["ab_ag_cluster"])
        pdb2cl[p].add(c)
        (post if (r["PDBdepo"] or "")[:10] > CUT else pre)[c].add(p)
    out = {}
    for p, cs in pdb2cl.items():
        out[p] = "mixed" if any(pre[c] and post[c] for c in cs) else "all new"
    return out


def load(dataset, jobs_manifest, split_csv, universe, weighting, arm_source):
    """out: ({group: [ {seed: [(dockq, iptm), ...]} per interface ]}, {group: kind}, dropped)

    `group` is a cluster under --weighting cluster and the interface key itself under `interface`,
    so the averaging below is one code path."""
    d = paths.DATASETS[dataset]
    if arm_source == "split-csv":
        pdb2cl = {m["pdb"]: m["cluster"] for m in csv.DictReader(open(jobs_manifest))}
        kind_by_group = {r["cluster"]: r["kind"] for r in csv.DictReader(open(split_csv))}
    else:
        pdb2cl, kind_by_group = None, None
        sab = sabdab_kind_by_pdb()
    # TWO SCHEMAS. A fold directory's results.csv is iface,seed,...,dockq,iptm. A benchmark's pooled
    # table (paths.DATASETS[ds].free_pool) is pdb,chain1,chain2,seed,...,dockq,pair_iptm,tier -- and
    # it is the right source for a benchmark, because the arm's own results.csv holds only that arm's
    # seeds. Rows at tier `suspect` are excluded; see the pooled table's INDEX.md.
    src = str(d.free_pool) if getattr(d, "free_pool", None) else os.path.join(str(d.free),
                                                                             "results.csv")
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(src) as fh:
        rd = csv.DictReader(fh)
        pooled = "pair_iptm" in (rd.fieldnames or [])
        for r in rd:
            if pooled and r.get("tier") == "suspect":
                continue
            try:
                dq = float(r["dockq"])
                ip = float(r["pair_iptm"] if pooled else r["iptm"])
            except Exception:
                continue
            if math.isnan(dq) or math.isnan(ip):
                continue
            key = f"{r['pdb']}_{r['chain1']}{r['chain2']}" if pooled else r["iface"]
            rows[key][int(r["seed"])].append((dq, ip))
    print(f"   source: {src}")
    byclust, kind, dropped = collections.defaultdict(list), {}, 0
    for k, sd in rows.items():
        pdb = k.rsplit("_", 1)[0]
        if len(sd) < universe:
            dropped += 1
            continue
        if weighting == "cluster":
            g = pdb2cl.get(pdb) if pdb2cl else None
            if g is None:
                dropped += 1
                continue
            kind[g] = kind_by_group.get(g) if kind_by_group else sab.get(pdb)
        else:
            g = k
            kind[g] = kind_by_group.get(pdb2cl.get(pdb)) if pdb2cl else sab.get(pdb)
        byclust[g].append({s: sd[s] for s in sorted(sd)[:universe]})
    return byclust, kind, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sabdab_val2")
    ap.add_argument("--jobs-manifest",
                    default="/home/ebrahim/tts_abag/extra_training/val2_jobs/manifest.csv")
    ap.add_argument("--split-csv",
                    default="/home/ebrahim/tts_abag/extra_training/results/val_split_proposal.csv")
    ap.add_argument("--universe", type=int, default=100)
    ap.add_argument("--depths", default="1,5,10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--boots", type=int, default=50)
    ap.add_argument("--thresh", type=float, default=0.23)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="all", choices=["all", "all new", "mixed"])
    ap.add_argument("--weighting", default="interface", choices=["interface", "cluster"],
                    help="interface: weight every interface equally, the field convention and the "
                         "default. cluster: average within cluster then over clusters equally, which "
                         "stops a cluster contributing two representatives from counting twice")
    ap.add_argument("--arm-source", default="split-csv", choices=["split-csv", "sabdab"],
                    help="where `all new` vs `mixed` comes from")
    ap.add_argument("--label", default=None, help="overrides the title's dataset name")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    byclust, kind, dropped = load(a.dataset, a.jobs_manifest, a.split_csv, a.universe,
                                  a.weighting, a.arm_source)
    clusters = sorted(c for c in byclust if a.arm == "all" or kind.get(c) == a.arm)
    if not clusters:
        raise SystemExit(f"STOP: no clusters match arm {a.arm!r}")
    nif = sum(len(byclust[c]) for c in clusters)
    rng = random.Random(a.seed)

    def classify(sd, seeds):
        poses = [p for s in seeds for p in sd[s]]
        oracle = max(p[0] for p in poses)
        chosen = max(poses, key=lambda p: p[1])[0]
        if oracle < a.thresh:
            return "no acceptable"
        return "acceptable" if chosen >= a.thresh else "decoy"

    def fracs(pick):
        out = {}
        for cls in CLASSES:
            out[cls] = st.mean([
                st.mean([1.0 if classify(sd, pick(sd)) == cls else 0.0 for sd in byclust[c]])
                for c in clusters])
        return out

    depths = [int(x) for x in a.depths.split(",") if 0 < int(x) <= a.universe]
    print(f"{a.dataset} arm={a.arm} weighting={a.weighting}: {len(clusters)} groups, {nif} interfaces"
          + (f" ({dropped} interfaces dropped for < {a.universe} seeds)" if dropped else ""))
    table = []
    for k in depths:
        if k == 1:
            # THE i-TH SEED OF EACH INTERFACE, not a shared seed id. Benchmarks do not share a seed
            # set across interfaces -- pxm2025 has 25 distinct seed sets -- so indexing by position
            # is the only definition that works for both, and it still enumerates every single-seed
            # draw exactly rather than sampling them.
            draws = [fracs(lambda sd, i=i: [sorted(sd)[i]]) for i in range(a.universe)]
        elif k >= a.universe:
            draws = [fracs(lambda sd: list(sd))]
        else:
            draws = [fracs(lambda sd: rng.sample(list(sd), k)) for _ in range(a.boots)]
        row = {"seeds": k, "draws": len(draws)}
        for cls in CLASSES:
            v = [d[cls] for d in draws]
            row[cls] = st.mean(v)
            row[cls + " sd"] = st.stdev(v) if len(v) > 1 else 0.0
        table.append(row)
        print(f"  {k:3d} seeds ({len(draws):3d} draws): " + "  ".join(
            f"{cls} {100*row[cls]:.1f}+-{100*row[cls+' sd']:.1f}%" for cls in CLASSES))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        for r in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    xs = [r["seeds"] for r in table]
    for cls in CLASSES:
        ys = [100 * r[cls] for r in table]
        es = [100 * r[cls + " sd"] for r in table]
        ax.plot(xs, ys, "-o", color=COLOURS[cls], label=cls, ms=4, lw=1.8, zorder=3)
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                        color=COLOURS[cls], alpha=0.18, lw=0, zorder=2)
    ax.set_xscale("log")
    # EVERY depth gets a tick, but only a legible subset gets a label: on a log axis 50 through
    # 100 sit within a few millimetres of one another and the labels overprint.
    label_at = {1, 5, 10, 20, 30, 50, 100}
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) if x in label_at else "" for x in xs]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("seeds per interface (log scale)")
    ax.set_ylabel("% of interfaces")
    unit = "clusters" if a.weighting == "cluster" else "interfaces"
    arm = f"all {unit}" if a.arm == "all" else f"{a.arm} {unit}"
    name = a.label or a.dataset
    shown = (f"{len(clusters)} clusters, {nif} interfaces" if a.weighting == "cluster"
             else f"{nif} interfaces")
    ax.set_title(f"{name}, {arm}: {shown}", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.text(0.005, 0.015,
             f"x: seeds drawn per interface.  y: % of interfaces in each class, "
             + ("averaged within cluster then over clusters equally.\n"
                if a.weighting == "cluster" else "every interface weighted equally.\n") +
             f"band = ±1 sd over draws.  k=1 enumerates all {a.universe} single-seed draws; "
             f"k={a.universe} has one possible draw; otherwise {a.boots} draws without replacement.",
             fontsize=6.5, color="#555", va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(a.out + ".png")
    fig.savefig(a.out + ".pdf")
    print(f"\nwrote {a.out}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
