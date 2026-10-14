#!/usr/bin/env python
"""Pose-to-pose scatter within a fold: does self-templating tighten a wrong donor more than a right one?

SUMMARY
    A fold makes 5 samples. Their pose-to-pose agreement -- here the mean pairwise DockQ over the 10
    pairs, each pair averaged over both directions, on ONE benchmark interface -- is the fold's scatter
    (high agreement = low scatter). For every templated fold of an experiment and every benchmark
    interface of its complex, this measures the agreement of the templated fold AND of the free fold its
    donor came from (the donor's seed, its 5 samples), with each fold's top ipTM and the donor's DockQ
    against the crystal, so templated and free can be compared donor by donor.

    Pose-to-pose DockQ is OST's (eval/ost_dockq.py, the project's reference; --method ost, the default),
    or dockq_torch.dockq_matrix, the pipeline's build-step-6 implementation (--method torch). They differ
    when a pose has a stray terminal OXT atom in the interface: OST's clean-up removes OXT, dockq_torch
    keeps it (7zoz A05 sample 2: one misplaced OXT added 10 contacts and moved one pair 0.96 -> 0.58).
    So --method torch here leaves OXT out, as OST does. Each
    sample's chains for the interface come from that sample's own crystal mapping -- a templated fold's
    scorer json (ost_json/<k>.json or pxm_json/<k>.json), a free pose's chain_maps.csv row -- so a copy
    that moves between samples is followed.

    INTEGRITY. A templated pose is used only if its ipTM, re-read from its own confidence json at the
    mapped chains (pool_index.pair_iptm_at), equals the ipTM results.csv recorded for it; a fold failing
    that on any sample is skipped and counted. (Early pxm2025 folds shared run directories and some
    poses were overwritten; their rows are right but their files may not be.)

INPUTS
    --dataset:  pxm2025 or foldbench. Default pxm2025.
    --exp:      the templated experiment. Default the dataset's self_template_fixed_50_seed.
    --poses:    extra run-directory roots holding its folds, colon-separated (carried folds). Optional.
    --jobs:     worker processes. Default 32. CPU only.
    --pdbs:     comma-separated complexes to restrict to. Default all.
    --method:   ost or torch. Default ost.
    --device:   for --method torch: cpu or cuda[:n]. Default cpu.
    --out:      the per-fold csv. Default <exp>/fold_scatter.csv.

OUTPUTS
    <out>: one row per (templated fold, interface): tag, donor, donor DockQ, agreement / top ipTM / mean
           DockQ of the templated fold and of the donor's free fold
    stdout: agreement change (templated minus free) by donor correctness and interface class

EXAMPLE
    analysis/fold_scatter.py --dataset foldbench --jobs 32

FUNCTIONS
    first_copy():  a {reference: model} map with copy-N names folded onto the first copy's
    agreement():   5 poses with their chains -> mean pairwise DockQ, both directions averaged
    agreement_ost(): the same with OST, each pair's json kept under <out>_ost/
    fold_rows():   one templated fold -> its rows, or a skip reason
    main():        find the folds, measure, write, summarise
"""
import argparse
import collections
import csv
import glob
import json
import os
import re
import statistics
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "eval"))

import paths  # noqa: E402
import pool_index  # noqa: E402
from score_experiment import ACCEPTABLE, arm_defaults, classes_at, pool_by_seed  # noqa: E402

G = {}   # per-worker read-only state, set by init()


def first_copy(cmap):
    """in: {reference chain: model chain}.  out: "<X>-<n>" renamed "<X>" where X itself is unmapped"""
    out = {}
    for ref, mdl in cmap.items():
        head, _, tail = ref.rpartition("-")
        out[head if head and tail.isdigit() and head not in cmap else ref] = mdl
    return out


def agreement(items):
    """in: [(pose, ab chain, ag chain)] x 5.  out: mean pairwise DockQ over the pairs, each pair the
    mean of both directions"""
    import torch
    torch.set_num_threads(1)
    from dockq_torch import dockq_matrix
    M = dockq_matrix(items, device=G.get("device", "cpu"), verbose=False, skip=("OXT",))
    n = len(items)
    return statistics.mean((M[i][j] + M[j][i]) / 2 for i in range(n) for j in range(i + 1, n))


def agreement_ost(items, out_dir):
    """in: [(pose, ab chain, ag chain)] x 5, a directory for OST's jsons.
    out: mean pairwise DockQ, each pair the mean of both directions; an interface OST does not report
    (no contact in the reference pose) counts as 0, as in the pipeline"""
    import ost_dockq
    os.makedirs(out_dir, exist_ok=True)
    n, vals = len(items), []
    for i in range(n):
        for j in range(i + 1, n):
            pair = []
            for a, b in ((i, j), (j, i)):          # model a, reference b
                res = ost_dockq.score(items[a][0], items[b][0], f"{out_dir}/{a}_{b}.json")
                v = ost_dockq.interface(res, items[b][1], items[b][2])
                pair.append(float(v["dockq"]) if v and v.get("dockq") is not None else 0.0)
            vals.append(sum(pair) / 2)
    return statistics.mean(vals)


def measure(items, key):
    """in: [(pose, ab chain, ag chain)] x 5, a key naming the fold.  out: its agreement by the chosen
    method (G["method"]); OST's jsons go under G["ost_dir"]/<key>/"""
    if G["method"] == "ost":
        return agreement_ost(items, f"{G['ost_dir']}/{key}")
    return agreement(items)


def init(state):
    G.update(state)


def fold_rows(task):
    """in: (run directory, pdb, seed, donor pose, tag).  out: (rows, skip reason or "")"""
    rdir, pdb, seed, donor, tag = task
    m = re.search(r"/(\d+)-[^/]+/predictions/[^/]+_sample_(\d+)\.cif$", donor)
    if not m:
        return [], "donor path unreadable"
    dseed, dsample = m.group(1), int(m.group(2))
    rows = []
    for iface in G["ifaces"][pdb]:
        ab, ag = G["roles"][iface]
        tmpl, free = [], []
        for k in range(5):
            cif = f"{rdir}/predictions/{pdb}_sample_{k}.cif"
            js = f"{rdir}/{G['json']}/{k}.json"
            if not (os.path.exists(cif) and os.path.exists(js)):
                return [], "pose or scorer json missing"
            d = json.load(open(js))
            cmap = first_copy(d["chain_mapping"] if "chain_mapping" in d else d["ref_to_model_chain_mapping"])
            if ab not in cmap or ag not in cmap:
                return [], "interface chain unmapped in a templated pose"
            want = G["rows"].get((iface, seed, tag, k))
            if want is None:
                return [], "no results.csv row"
            got = [pool_index.pair_iptm_at(cif, cmap[ab], cmap[ag]),
                   pool_index.pair_iptm_at(cif, cmap[ag], cmap[ab])]
            if min(abs(g - want[0]) for g in got) > 1e-3:
                return [], "ipTM on disk does not match results.csv"
            tmpl.append((cif, cmap[ab], cmap[ag], want))
            fmap = G["maps"].get((pdb, dseed, k))
            fcif = paths.pool_pose(pdb, dseed, k, G["free"])
            fv = G["pool"].get((iface, dseed, k))
            if not fmap or ab not in fmap or ag not in fmap or fv is None:
                return [], "free source pose unmapped or unscored"
            free.append((fcif, fmap[ab], fmap[ag], fv))
        if (iface, dseed, dsample) not in G["pool"]:
            return [], "donor pose unscored in the pool"
        # OST keeps one json per pose pair: under the templated fold's run directory name, and under
        # free_<seed> for the donor's free fold, so a free fold shared by several donors is computed once
        ta = measure([t[:3] for t in tmpl], f"{pdb}/{os.path.basename(rdir)}")
        fa = measure([f[:3] for f in free], f"{pdb}/free_{dseed}")
        rows.append([pdb, iface, G["cls"].get(iface, ""), tag, dseed, dsample,
                     f"{G['pool'][(iface, dseed, dsample)][1]:.4f}",
                     f"{ta:.4f}", f"{max(t[3][0] for t in tmpl):.4f}",
                     f"{statistics.mean(t[3][1] for t in tmpl):.4f}",
                     f"{fa:.4f}", f"{max(f[3][0] for f in free):.4f}",
                     f"{statistics.mean(f[3][1] for f in free):.4f}"])
    return rows, ""


HEADER = ["pdb", "iface", "class", "tag", "donor_seed", "donor_sample", "donor_dockq",
          "tmpl_agreement", "tmpl_top_iptm", "tmpl_mean_dockq",
          "free_agreement", "free_top_iptm", "free_mean_dockq"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--poses", default="")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--pdbs", default=None)
    ap.add_argument("--method", choices=("ost", "torch"), default="ost")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    ds, exp, _ = arm_defaults(A.dataset)
    A.exp = A.exp or exp
    A.out = A.out or f"{A.exp}/fold_scatter.csv"

    pool, first = pool_by_seed(50, str(ds.pool_csv))
    ifaces = collections.defaultdict(list)
    for i in pool:
        ifaces[i.split("_")[0]].append(i)
    rows_csv, folds = {}, {}
    for r in csv.DictReader(open(f"{A.exp}/results.csv")):
        rows_csv[(r["iface"], r["seed"], r["arm"], int(r["sample"]))] = (float(r["iptm"]), float(r["dockq"]))
        folds[(r["donor_pose"], r["seed"])] = (r["iface"].split("_")[0], r["arm"])
    roles = pool_index.load_roles(str(ds.chain_roles))
    state = dict(
        ifaces={p: sorted(i for i in v if i in roles) for p, v in ifaces.items()}, roles=roles,
        rows=rows_csv, cls=classes_at(pool, first), free=str(ds.free),
        json="ost_json" if ds.scorer == "ost" else "pxm_json",
        maps={k: first_copy(v) for k, v in pool_index.load_chain_maps(str(ds.chain_maps)).items()},
        pool={(i, s, x): (ip, dq) for i, rs in pool.items() for s, x, dq, ip in rs},
        method=A.method, device=A.device, ost_dir=f"{os.path.splitext(A.out)[0]}_ost")

    tasks, seen = [], set()
    for root in [f"{A.exp}/poses"] + [p for p in A.poses.split(":") if p]:
        for rj in sorted(glob.glob(f"{root}/*/*/run.json")):
            j = json.load(open(rj))
            src = (j.get("templates") or [{}])[0].get("source_pose")
            key = (src, str(j.get("seed")))
            if (src and key in folds and key not in seen and len(j["templates"]) == 1
                    and (not A.pdbs or folds[key][0] in A.pdbs.split(","))):
                seen.add(key)
                tasks.append((os.path.dirname(rj), folds[key][0], key[1], src, folds[key][1]))
    print(f"  {len(tasks)} templated folds on disk of {len(folds)} in results.csv", flush=True)

    import multiprocessing as mp
    out, skips = [], collections.Counter()
    with mp.Pool(A.jobs, initializer=init, initargs=(state,)) as p:
        for n, (rows, why) in enumerate(p.imap_unordered(fold_rows, tasks, 4), 1):
            out += rows
            skips[why] += bool(why)
            if n % 500 == 0 or n == len(tasks):
                print(f"    {n}/{len(tasks)} folds  rows {len(out)}  skipped {sum(skips.values())}", flush=True)
    with open(A.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(sorted(out))
    print(f"  -> {A.out}  ({len(out)} rows; skipped folds: {dict((k, v) for k, v in skips.items() if k)})")

    print("\n  agreement = mean pairwise DockQ of a fold's 5 samples (higher = less scatter)")
    print(f"  {'donor':16s} {'class':7s} {'n':>5s} {'free agreement':>15s} {'templated':>10s} "
          f"{'change':>8s} {'top-ipTM change':>16s}")
    for label, test in (("correct donor", lambda d: d >= ACCEPTABLE), ("wrong donor", lambda d: d < ACCEPTABLE)):
        for k in ("all", "decoy", "solved", "nocov"):
            v = [r for r in out if test(float(r[6])) and (k == "all" or r[2] == k)]
            if v:
                fa = statistics.median(float(r[10]) for r in v)
                ta = statistics.median(float(r[7]) for r in v)
                ch = statistics.median(float(r[7]) - float(r[10]) for r in v)
                ic = statistics.median(float(r[8]) - float(r[11]) for r in v)
                print(f"  {label:16s} {k:7s} {len(v):5d} {fa:15.3f} {ta:10.3f} {ch:+8.3f} {ic:+16.3f}")
    print("  (medians over fold x interface rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
