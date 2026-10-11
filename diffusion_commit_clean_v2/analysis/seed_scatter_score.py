#!/usr/bin/env python
"""Score the 50-sample refolds: are a seed's poses discrete docking sites or random scatter?

SUMMARY
    For every finished fold of the seed_scatter_50 experiment (seed_scatter_select.py picked the seeds)
    and every benchmark interface of its complex, this builds the pose-to-pose DockQ matrix over the
    fold's 50 new samples PLUS the seed's 5 original free poses (55 x 55, each pair the mean of both
    directions, dockq_torch with OXT left out as OST does). Each sample's chains come from its own
    crystal mapping (the scorer's pxm_json for new poses, chain_maps.csv for the originals), so a copy
    that moves between samples is followed. Incremental: an interface whose npz exists is skipped.

    Per (fold, interface), over the 50 new samples:
      agreement      mean pairwise DockQ (higher = less scatter)
      clusters       groups of near-copies: connected components of the graph joining poses with
                     DockQ >= 0.5 (a pose belongs to a group if it matches ANY member; matching only
                     a group's first pose split real members off as false one-offs)
      top_share      share of the 50 in the largest cluster; singletons = poses near no other pose
      singleton_iptm median ipTM of the singletons; singleton_nearest = their median best DockQ to any
                     other pose (near 0 = a placement no other draw comes close to)
      mid_pairs      share of pairs with DockQ 0.23-0.5 -- neither the same pose nor a different one.
                     Discrete sites give few of these; random scatter gives few near-copies at all
      orig_repro     how many of the 5 original poses have a near-copy (DockQ >= 0.5) among the new 50:
                     whether the refold reproduced the original trunk
    plus the original 5-pose agreement at this interface (from the matrix) and crystal DockQ / ipTM of the
    top-ipTM new pose. EVALUATION reads the crystal only through results.csv.

INPUTS
    --exp:     the experiment. Default experimental_data/seed_scatter_50 (pxm2025), else
               <dataset data>/seed_scatter_50.
    --device:  cpu or cuda[:n]. Default cpu.
    --jobs:    worker processes. Default 16.
    --dataset: pxm2025 or foldbench. Default pxm2025. Sets the pool, chain maps, roles and the
               scorer json directory (pxm_json for pxm2025, ost_json for FoldBench).

OUTPUTS
    <exp>/matrices/<pdb>_<seed>_<iface>.npz: dockq (55, 55) symmetrised, source (new | orig), sample, iptm
    <exp>/scatter_summary.csv: one row per (fold, interface), the columns above
    stdout: progress, then medians by pick (decoy_top_iptm, decoy_best_dockq, committed, scattered; a
            row counts under every pick its label holds). pick = the decoy pick on THIS interface if
            any, else the fold's scatter pick(s), else "other"

EXAMPLE
    analysis/seed_scatter_score.py --jobs 16

FUNCTIONS
    folds():      finished run directories -> [(rdir, pdb, seed)]
    one():        one (fold, interface) -> npz on disk, or a skip reason
    clusters():   a similarity matrix and an order -> cluster sizes
    summarise():  every npz -> scatter_summary.csv rows
    main():       score what is new, then summarise
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import paths  # noqa: E402
import pool_index  # noqa: E402
from fold_scatter import first_copy  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experimental_data")
SAME = 0.5          # DockQ at which two poses count as the same pose
ACC = 0.23
G = {}


def folds(exp, n_sample, jname):
    """in: the experiment, samples per fold, the scorer json directory name.
    out: [(run dir, pdb, seed)] with all poses and jsons"""
    out = []
    for rj in sorted(glob.glob(f"{exp}/poses/*/*/run.json")):
        j = json.load(open(rj))
        rdir = os.path.dirname(rj)
        pdb = j["pdb"]
        if (len(glob.glob(f"{rdir}/predictions/{pdb}_sample_*.cif")) == n_sample
                and len(glob.glob(f"{rdir}/{jname}/*.json")) == n_sample):
            out.append((rdir, pdb, str(j["seed"])))
    return out


def one(task):
    """in: (run dir, pdb, seed, interface).  out: skip reason or "" """
    rdir, pdb, seed, iface = task
    dst = f"{G['exp']}/matrices/{pdb}_{seed}_{iface}.npz"
    if os.path.exists(dst):
        return ""
    ab, ag = G["roles"][iface]
    items, src, smp, ipt = [], [], [], []
    jdir = f"{rdir}/{G['json']}"       # folder.py moves the scorer's jsons here when the fold is done
    for k in range(G["n"]):
        cmap = json.load(open(f"{jdir}/{k}.json"))
        cmap = first_copy(cmap.get("chain_mapping") or cmap["ref_to_model_chain_mapping"])
        if ab not in cmap or ag not in cmap:
            return "new pose unmapped"
        cif = f"{rdir}/predictions/{pdb}_sample_{k}.cif"
        items.append((cif, cmap[ab], cmap[ag]))
        src.append("new"); smp.append(k)
        ipt.append(G["rows"].get((iface, seed, k), (np.nan, np.nan))[0])
    for k in range(5):
        fmap = G["maps"].get((pdb, seed, k))
        if not fmap or ab not in fmap or ag not in fmap:
            continue
        items.append((paths.pool_pose(pdb, seed, k, G["free"]), fmap[ab], fmap[ag]))
        src.append("orig"); smp.append(k)
        ipt.append(G["pool"].get((iface, seed, k), (np.nan, np.nan))[0])
    import torch
    torch.set_num_threads(1)
    from dockq_torch import dockq_matrix
    M = np.asarray(dockq_matrix(items, device=G["device"], verbose=False, skip=("OXT",)), dtype=np.float32)
    M = (M + M.T) / 2
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, dockq=M, source=np.array(src), sample=np.array(smp, dtype=np.int16),
                        iptm=np.array(ipt, dtype=np.float32))
    return ""


def clusters(S):
    """in: similarity matrix.  out: cluster sizes, largest first (connected components at SAME)"""
    n, seen, size = len(S), set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, k = [i], 0
        seen.add(i)
        while stack:
            j = stack.pop()
            k += 1
            for m in np.where(S[j] >= SAME)[0]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        size.append(k)
    return sorted(size, reverse=True)


def summarise(exp, sel, rows):
    """in: the experiment, {(pdb, seed, iface): pick label}, results rows.
    out: summary rows (also written to csv)"""
    out = []
    for f in sorted(glob.glob(f"{exp}/matrices/*.npz")):
        pdb, seed, iface = os.path.basename(f)[:-4].split("_", 2)
        d = np.load(f)
        new = np.where(d["source"] == "new")[0]
        org = np.where(d["source"] == "orig")[0]
        S = d["dockq"][np.ix_(new, new)]
        iu = np.triu_indices(len(new), 1)
        pairs = S[iu]
        ipt = d["iptm"][new]
        order = list(np.argsort(-np.nan_to_num(ipt, nan=-1)))
        sizes = clusters(S)
        off = S - np.eye(len(new)) * 2                    # each pose's best match to ANOTHER pose
        best = off.max(1)
        lone = best < SAME
        repro = int((d["dockq"][np.ix_(org, new)].max(1) >= SAME).sum()) if len(org) else -1
        top = order[0]
        cdq = rows.get((iface, seed, int(d["sample"][new][top])), (np.nan, np.nan))[1]
        pick = sel.get((pdb, seed, iface), "")
        so = d["dockq"][np.ix_(org, org)]
        oagr = so[np.triu_indices(len(org), 1)].mean() if len(org) > 1 else np.nan
        out.append([pdb, seed, iface, pick, f"{oagr:.4f}", len(new), f"{pairs.mean():.4f}",
                    len(sizes), f"{sizes[0] / len(new):.3f}", int(lone.sum()),
                    f"{np.nanmedian(ipt[lone]):.4f}" if lone.any() else "",
                    f"{np.median(best[lone]):.4f}" if lone.any() else "",
                    f"{((pairs >= ACC) & (pairs < SAME)).mean():.3f}", f"{(pairs >= SAME).mean():.3f}",
                    len(org), repro, f"{ipt[top]:.4f}", f"{cdq:.4f}",
                    f"{np.mean([rows.get((iface, seed, int(k)), (0, 0))[1] >= ACC for k in d['sample'][new]]):.3f}"])
    head = ["pdb", "seed", "iface", "pick", "orig_agreement", "n_new", "agreement", "clusters",
            "top_share", "singletons", "singleton_iptm", "singleton_nearest", "mid_pairs", "same_pairs", "n_orig", "orig_repro",
            "top_iptm", "top_crystal_dockq", "share_acceptable"]
    with open(f"{exp}/scatter_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(out)
    return head, out


def init(state):
    G.update(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None,
                    help="default experimental_data/seed_scatter_50 (pxm2025), <dataset data>/seed_scatter_50 otherwise")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--nsample", type=int, default=50)
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    A = ap.parse_args()
    ds = paths.DATASETS[A.dataset]
    A.exp = A.exp or (f"{DATA}/seed_scatter_50" if A.dataset == "pxm2025" else f"{ds.data}/seed_scatter_50")
    jname = "ost_json" if ds.scorer == "ost" else "pxm_json"

    pool = pool_index.load_pool(str(ds.pool_csv))
    roles = pool_index.load_roles(str(ds.chain_roles))
    rows = {}
    if os.path.exists(f"{A.exp}/results.csv"):
        for r in csv.DictReader(open(f"{A.exp}/results.csv")):
            if r["dockq"] and r["iptm"]:
                rows[(r["iface"], r["seed"], int(r["sample"]))] = (float(r["iptm"]), float(r["dockq"]))
    # the scatter picks name interfaces in anchor letters, so they label a whole fold (pdb, seed); the
    # decoy picks name benchmark interfaces (reference letters, as here), so they label one interface
    sel_seed, sel_iface = collections.defaultdict(set), collections.defaultdict(set)
    for r in csv.DictReader(open(f"{A.exp}/selection.csv")):
        if r["pick"].startswith("decoy"):
            sel_iface[(r["pdb"], r["seed"], r["iface"])].add(r["pick"])
        else:
            sel_seed[(r["pdb"], r["seed"])].add(r["pick"])
    state = dict(exp=A.exp, json=jname, roles=roles, rows=rows, n=A.nsample, device=A.device, free=str(ds.free),
                 maps={k: first_copy(v) for k, v in pool_index.load_chain_maps(str(ds.chain_maps)).items()},
                 pool={(i, s, x): (ip, dq) for i, rs in pool.items() for s, x, dq, ip in rs})
    ifaces = collections.defaultdict(list)
    for i in pool:
        if i in roles:
            ifaces[i.split("_")[0]].append(i)

    F = folds(A.exp, A.nsample, jname)
    tasks = [(rdir, pdb, seed, i) for rdir, pdb, seed in F for i in sorted(ifaces[pdb])]
    todo = [t for t in tasks if not os.path.exists(f"{A.exp}/matrices/{t[1]}_{t[2]}_{t[3]}.npz")]
    print(f"  folds finished: {len(F)}   fold x interface: {len(tasks)}   new to score: {len(todo)}", flush=True)
    skips = collections.Counter()
    if todo:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(A.jobs, initializer=init, initargs=(state,)) as p:
            for why in p.imap_unordered(one, todo, 1):
                skips[why] += bool(why)
    if any(skips.values()):
        print(f"  skipped: {dict((k, v) for k, v in skips.items() if k)}")

    sel = {}
    for pdb, seed in {(p, s) for p, s, _ in sel_iface} | set(sel_seed):
        for i in ifaces[pdb]:
            picks = sel_iface.get((pdb, seed, i)) or sel_seed.get((pdb, seed)) or {"other"}
            sel[(pdb, seed, i)] = "+".join(sorted(picks))
    head, out = summarise(A.exp, sel, rows)
    print(f"  wrote {A.exp}/scatter_summary.csv ({len(out)} rows)")
    for k in ("decoy_top_iptm", "decoy_best_dockq", "decoy_committed_correct", "committed", "scattered"):
        v = [r for r in out if k in r[3].split("+")]
        if not v:
            continue
        med = lambda c: statistics.median(float(r[head.index(c)]) for r in v)  # noqa: E731
        print(f"  {k:23s} n={len(v):4d}  orig agreement {med('orig_agreement'):.3f}  new agreement "
              f"{med('agreement'):.3f}  clusters {med('clusters'):.0f}  top share {med('top_share'):.2f}  "
              f"singletons {med('singletons'):.0f}  mid pairs {med('mid_pairs'):.3f}  "
              f"originals reproduced {med('orig_repro'):.0f}/5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
