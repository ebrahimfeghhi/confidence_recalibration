#!/usr/bin/env python
"""Rank the poses of each confident-decoy interface by how much they resemble its low-ipTM poses. Written 2026-10-02.

For each confident-decoy interface (score_experiment.classes_at == "decoy", free-50 pool):
  low set  = poses with pair ipTM < T (default 0.2)
  badness  = a pose's mean (or max: AGG) pose-to-pose DockQ to the low set (itself excluded), from pose_dockq_matrices
  candidates = poses with pair ipTM >= T
Reports per interface: Spearman(badness, crystal DockQ) over the candidates, and the crystal DockQ of the
lowest-badness candidate vs the top-ipTM pose and the best available (oracle). Crystal DockQ is evaluation only.

JOIN. The matrices name interfaces in anchor chain letters, the pool in crystal letters. Each crystal interface is
matched to the matrix of the same complex whose per-pose pair ipTM agrees (|diff| < TOL) on the most of its poses;
the match must be unique (strictly more agreeing poses than the runner-up) and cover >= MIN_COVER of the poses.
Only poses that agree are used. Interfaces that fail are listed, not guessed.
Usage: python analysis/low_anchor_badness.py [T] [mean|max] -> experimental_results/low_anchor_badness_<T>[_max].csv + stdout tables"""
import collections, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import score_experiment as se, paths

T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
AGG = sys.argv[2] if len(sys.argv) > 2 else "mean"
assert AGG in ("mean", "max"), AGG
SUFFIX = "" if AGG == "mean" else f"_{AGG}"
TOL, MIN_COVER = 1e-3, 0.9
MATS = os.path.join(str(paths.DATA_DIR), "pose_dockq_matrices")
OUT = os.path.join(os.path.dirname(str(paths.DATA_DIR)), "experimental_results")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else float("nan")


rows = []
for name in ("foldbench", "pxm2025"):
    pool, first = se.pool_by_seed(50, str(paths.DATASETS[name].pool_csv))
    cls = se.classes_at(pool, first)
    mats = collections.defaultdict(list)
    for f in glob.glob(f"{MATS}/{name}/*.npz"):
        z = np.load(f)
        mats[os.path.basename(f).split("_")[0]].append(
            (os.path.basename(f)[:-4], z["dockq"], {(s, int(k)): j for j, (s, k) in enumerate(zip(z["seed"], z["sample"]))},
             z["pair_iptm"]))
    fails = collections.Counter()
    for i in sorted(x for x, c in cls.items() if c == "decoy"):
        pdb = i.split("_")[0]
        pr = [r for r in pool[i] if r[0] in first[pdb]]
        hits = []
        for mname, M, idx, ip in mats.get(pdb, []):
            ok = [(r, idx[(r[0], r[1])]) for r in pr if (r[0], r[1]) in idx and abs(ip[idx[(r[0], r[1])]] - r[3]) < TOL]
            hits.append((len(ok), mname, M, ok))
        hits.sort(key=lambda h: -h[0])
        if not hits:
            fails["no matrix for complex"] += 1; print(f"  FAIL {name} {i}: no matrix"); continue
        if hits[0][0] < MIN_COVER * len(pr) or (len(hits) > 1 and hits[1][0] == hits[0][0]):
            fails["no unique matrix match"] += 1
            print(f"  FAIL {name} {i}: best {hits[0][1]} agrees on {hits[0][0]}/{len(pr)}"
                  + (f", runner-up {hits[1][1]} {hits[1][0]}" if len(hits) > 1 else ""))
            continue
        n_ok, mname, M, ok = hits[0]
        low = [j for r, j in ok if r[3] < T]
        cand = [(r, j) for r, j in ok if r[3] >= T]
        top = max(ok, key=lambda x: x[0][3])[0]
        base = dict(dataset=name, iface=i, matrix=mname, n_matched=n_ok, n_pool=len(pr), n_low=len(low),
                    n_cand=len(cand), top_iptm=round(top[3], 3), top_dockq=round(top[2], 3),
                    oracle_dockq=round(max(r[2] for r, _ in ok), 3))
        if not low or len(cand) < 2:
            rows.append({**base, "spearman": "", "pick_dockq": "", "pick_iptm": "", "cand_frac_acceptable": ""}); continue
        bad = np.array([getattr(np, AGG)([M[j, k] for k in low if k != j]) for _, j in cand])
        dq = np.array([r[2] for r, _ in cand])
        p = int(np.argmin(bad))
        rows.append({**base, "spearman": round(spearman(bad, dq), 3), "pick_dockq": round(dq[p], 3),
                     "pick_iptm": round(cand[p][0][3], 3), "cand_frac_acceptable": round(float((dq >= se.ACCEPTABLE).mean()), 3)})
    print(f"  {name}: join failures {dict(fails) or 0}")

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/low_anchor_badness_{T}{SUFFIX}.csv", "w") as fh:
    keys = list(rows[0])
    fh.write(",".join(keys) + "\n")
    for r in rows:
        fh.write(",".join(str(r[k]) for k in keys) + "\n")

A = se.ACCEPTABLE
print(f"\n| dataset | decoy interfaces scored | median Spearman(badness, crystal DockQ) | Spearman < 0 | "
      f"top-ipTM pick acceptable | lowest-badness pick acceptable | random candidate pick acceptable (expected) | mean DockQ top-ipTM pick | mean DockQ lowest-badness pick | mean oracle DockQ |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name in ("foldbench", "pxm2025"):
    s = [r for r in rows if r["dataset"] == name and r["spearman"] != ""]
    n = len(s)
    sp = np.array([r["spearman"] for r in s], dtype=float)
    print(f"| {name} | {n} (of {sum(r['dataset'] == name for r in rows)} joined) | {np.nanmedian(sp):.3f} | "
          f"{int((sp < 0).sum())}/{n} | {sum(r['top_dockq'] >= A for r in s)}/{n} | {sum(r['pick_dockq'] >= A for r in s)}/{n} | {sum(r['cand_frac_acceptable'] for r in s):.1f}/{n} | "
          f"{np.mean([r['top_dockq'] for r in s]):.3f} | {np.mean([r['pick_dockq'] for r in s]):.3f} | "
          f"{np.mean([r['oracle_dockq'] for r in s]):.3f} |")
print(f"-> {OUT}/low_anchor_badness_{T}{SUFFIX}.csv")
