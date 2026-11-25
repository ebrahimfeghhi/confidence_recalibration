"""Untrained pose-consensus selection on OUR OWN Protenix poses, using AbPoolRank's own code.

WHY. Breadth-over-Depth's selector reads four scalars per pose: within-pool percentiles of
geometric pose-similarity support, of source-balanced support, and of epitope-Jaccard support,
plus log pool size. Its gain over deep single-predictor sampling is mostly candidate coverage
(+5.1 of +6.5 points by its own oracle decomposition), but it never reports plain consensus
selection on a DEEP SINGLE-PREDICTOR pool: Pcons, 3D-Jury and MULTICOM-PSS were applied to the
mixed pools only, and the two rankers it did run on deep pools (VoroIF-GNN, DeepRank-Ab) lost to
native confidence on Protenix-v2 by 7 and 10 points.

So the question this answers is narrow and untested: on 150 Protenix poses per interface, does
geometric/epitope consensus beat the shipped ipTM head? No training, no GPU, no fitted
parameters -- the percentiles ARE the score.

TWO THINGS TO KNOW ABOUT THE FEATURES ON A SINGLE-SOURCE POOL.
  * srcbal_pct is not merely degenerate, it is NOISE. With one source every pose's own source is
    excluded, support is all-NaN -> nan_to_num -> 0, and percentile() of a constant vector
    returns argsort(argsort(const))/(n-1), i.e. the ramp 0..1 in INPUT ORDER. Verified by running
    their code. It is therefore zeroed here, and the bundled checkpoint is scored both ways so
    the size of that artefact is visible rather than assumed.
  * log_pool is identical for every pose in a pool and cannot change an argmax at fixed pool size.

THE ANTIGEN ASSEMBLY. describe() treats every chain outside antibody_chains as the antigen, so
for a structure with two antigen chains the epitope spans both. 80 of our 82 structures have one
antigen chain; the 2 that do not are reported separately rather than silently averaged in.

Selection ties are broken exactly as phase_k_eval does -- lowest seed, then lowest sample --
because 23 of 141 interfaces have a tied top stored_iptm and the tie decides a different DockQ.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from val_two_tables import load  # noqa: E402

ABPOOLRANK_SRC = ("/tmp/claude-1004/-home-ebrahim-confidence-recalibration/"
                  "d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad/abpoolrank/src")
sys.path.insert(0, ABPOOLRANK_SRC)
from abpoolrank.select.descriptors import describe  # noqa: E402
from abpoolrank.select.features import pool_features  # noqa: E402
from abpoolrank.select.pairwise import pair_matrices  # noqa: E402

POSES = "validation/fold/poses"
ROLES = "validation/fold/chain_roles_all83.csv"
RESTRICT = "training/valfull83/pool_noleak.csv"


def antibody_chains(roles_path):
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(roles_path)):
        if r["role"] in ("H", "L"):
            out[r["pdb"]].append(r["model_chain"])
    return out


def n_antigen_chains(roles_path):
    out = collections.Counter()
    for r in csv.DictReader(open(roles_path)):
        if r["role"].startswith("ag"):
            out[r["pdb"]] += 1
    return out


def pose_path_index(pdb, poses_root):
    """(seed, sample) -> cif path. The seed directory carries a random suffix."""
    idx = {}
    for p in glob.glob(f"{poses_root}/{pdb}/*/predictions/{pdb}_sample_*.cif"):
        seed = int(Path(p).parts[-3].split("-")[0])
        sample = int(Path(p).stem.rsplit("_", 1)[1])
        idx[seed, sample] = p
    return idx


def argmax_like_eval(scores, seeds, samples):
    """Highest score, then LOWEST seed, then LOWEST sample -- phase_k_eval.py:795."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], seeds[i], samples[i]))
    return order[0]


def candidates_from_results(path, require_trunk=True):
    """iface -> [(iptm, iptm, iptm, dockq, seed, sample), ...] from a folder results csv.

    Same tuple layout val_two_tables.load returns, so one_interface is unchanged: index 0 is the
    SHIPPED baseline score and index 3 the DockQ label. For the training pool the baseline is the
    fold's own `iptm` column, which IS the shipped head -- those rows were produced by folding
    with it, before any arm existed.
    """
    out = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if require_trunk and r.get("has_trunk") not in ("1", "True", "true"):
            continue
        try:
            ip, dq = float(r["iptm"]), float(r["dockq"])
            out[r["iface"]].append((ip, ip, ip, dq, int(r["seed"]), int(r["sample"])))
        except (KeyError, ValueError):
            continue
    return out


def one_interface(args):
    iface, cands, ab, poses_root = args
    pdb = iface.split("_")[0]
    idx = pose_path_index(pdb, poses_root)
    rows = [c for c in cands if (c[4], c[5]) in idx]
    if len(rows) < 10:
        return iface, None, f"only {len(rows)} of {len(cands)} candidates have a pose file", []
    paths = [idx[c[4], c[5]] for c in rows]
    descs = [describe(p, antibody_chains=tuple(ab)) for p in paths]
    live = [i for i, d in enumerate(descs) if d is not None]
    if len(live) < 10:
        return iface, None, f"only {len(live)} usable descriptors", []
    rows = [rows[i] for i in live]
    mats = pair_matrices([descs[i] for i in live], channels=("geom_sim", "epi_jac"))
    n = len(rows)
    X = pool_features(mats["geom_sim"], mats["epi_jac"], np.zeros(n, int))
    seeds = [r[4] for r in rows]
    samples = [r[5] for r in rows]
    dockq = [r[3] for r in rows]
    ship = [r[0] for r in rows]
    per_pose = [
        {"iface": iface, "seed": seeds[i], "sample": samples[i],
         "pss_pct": float(X[i, 0]), "srcbal_pct": float(X[i, 1]),
         "epi_pct": float(X[i, 2]), "log_pool": float(X[i, 3]),
         "dockq": dockq[i], "iptm": ship[i]}
        for i in range(n)
    ]
    out = {
        "iface": iface, "n": n,
        "dockq_shipped": dockq[argmax_like_eval(ship, seeds, samples)],
        "dockq_pss": dockq[argmax_like_eval(list(X[:, 0]), seeds, samples)],
        "dockq_epi": dockq[argmax_like_eval(list(X[:, 2]), seeds, samples)],
        "dockq_both": dockq[argmax_like_eval(list(X[:, 0] + X[:, 2]), seeds, samples)],
        "dockq_oracle": max(dockq),
    }
    return iface, out, None, per_pose


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rid", default="7mo52r")
    ap.add_argument("--epoch", type=int, default=16)
    ap.add_argument("--from-results", default="",
                    help="read candidates from a folder results csv instead of a run's eval "
                         "shards; the TRAINING pool has no eval shards of its own")
    ap.add_argument("--poses", default=POSES)
    ap.add_argument("--roles", default=ROLES)
    ap.add_argument("--restrict", default=RESTRICT,
                    help="pool csv limiting which interfaces are scored; '' means all of them")
    ap.add_argument("--min-candidates", type=int, default=10,
                    help="skip an interface with fewer usable poses; the training pool's "
                         "741 depth-1 interfaces offer 5 and cannot separate anything")
    ap.add_argument("--limit", type=int, default=0, help="first N interfaces only, for a pilot")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--features-out", default="",
                    help="per-pose feature table: iface, seed, sample, the four AbPoolRank "
                         "features, dockq and the shipped iptm. Persisting it is what makes "
                         "scoring with a trained selector cheap -- the all-pairs superposition "
                         "is the whole cost and it does not depend on the selector.")
    a = ap.parse_args()

    raw = (candidates_from_results(a.from_results) if a.from_results
           else load(a.rid, a.epoch))
    if a.restrict:
        keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}'
                for r in csv.DictReader(open(a.restrict))}
        raw = {k: v for k, v in raw.items() if k in keep}
    cands = {k: v for k, v in raw.items() if len(v) >= a.min_candidates}
    print(f"interfaces offered {len(raw)}, with >= {a.min_candidates} poses: {len(cands)}",
          flush=True)
    ab = antibody_chains(a.roles)
    order = sorted(cands)
    if a.limit:
        order = order[:a.limit]
    jobs = [(i, cands[i], ab[i.split("_")[0]], a.poses) for i in order]

    import multiprocessing as mp
    results, skipped, feats = [], [], []
    with mp.Pool(a.workers) as pool:
        for iface, out, why, per_pose in pool.imap_unordered(one_interface, jobs, chunksize=1):
            feats.extend(per_pose)
            if out is None:
                skipped.append((iface, why))
            else:
                results.append(out)
            done = len(results) + len(skipped)
            if done % 10 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    if not results:
        print("no interface produced a result")
        return 1
    cols = ["iface", "n", "dockq_shipped", "dockq_pss", "dockq_epi", "dockq_both", "dockq_oracle"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(sorted(results, key=lambda r: r["iface"]))
    print(f"\nscored {len(results)} interface(s), skipped {len(skipped)}")
    for iface, why in skipped[:8]:
        print(f"  skipped {iface}: {why}")
    for th in (0.23, 0.49, 0.80):
        print(f"\n=== Success@1 at DockQ >= {th} ({len(results)} interfaces) ===")
        base = statistics.fmean(1.0 * (r["dockq_shipped"] >= th) for r in results)
        for key in ("dockq_shipped", "dockq_pss", "dockq_epi", "dockq_both", "dockq_oracle"):
            v = statistics.fmean(1.0 * (r[key] >= th) for r in results)
            tag = "" if key == "dockq_shipped" else f"  delta {v - base:+.4f}"
            print(f"  {key:>14} {v:.4f}{tag}")
    if a.features_out:
        fcols = ["iface", "seed", "sample", "pss_pct", "srcbal_pct", "epi_pct", "log_pool",
                 "dockq", "iptm"]
        with open(a.features_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fcols)
            w.writeheader(); w.writerows(feats)
        print(f"per-pose features ({len(feats)} rows) -> {a.features_out}")
    print(f"\nrows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
