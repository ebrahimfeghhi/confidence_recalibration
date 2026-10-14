#!/usr/bin/env python
"""Is a pose's DockQ to the high-confidence poses a better ranker than its own ipTM? Written 2026-10-02.

Per interface of the free-50 pool (score_experiment.pool_by_seed) with at least one pose of pair ipTM > HI (default 0.8):
  anchors   = poses with pair ipTM > HI (the teaching signal; no crystal)
  rest      = every other pose (the band to fix)
  pseudo    = a rest pose's max pose-to-pose DockQ to any anchor (crystal-frame matrices,
              pose_dockq_matrices/<dataset>_crystal). A pose absent from the matrix has no contact at this
              interface (pose_to_pose_dockq.py drops those), so it cannot share an anchor's placement: pseudo = 0.
Evaluation only (crystal DockQ, acceptable = >= 0.23), over the rest poses:
  error rate = over pairs (acceptable, not acceptable), the fraction where the bad pose scores higher (ties 0.5),
               for ipTM and for pseudo. 0 = perfect order, 0.5 = chance. (= 1 - AUC.)
  pick       = is the top-scoring rest pose acceptable, by ipTM and by pseudo; random = mean acceptable fraction.
Also, over ALL poses of every anchored interface: Spearman(score, crystal DockQ) per interface, for ipTM and for
  pseudo_all = max DockQ to any anchor OTHER than the pose itself (an interface's only anchor scores 1).
Usage: python analysis/anchor_distance_vs_iptm.py [HI] -> experimental_results/anchor_distance_vs_iptm_<HI>.csv + tables"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import score_experiment as se, paths

HI = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
A = se.ACCEPTABLE
MATS = os.path.join(str(paths.DATA_DIR), "pose_dockq_matrices")
OUT = os.path.join(os.path.dirname(str(paths.DATA_DIR)), "experimental_results")


def avg_rank(x):
    """ranks with ties averaged"""
    order = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[order] = np.arange(len(x))
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return sums[inv] / cnt[inv]


def spearman(a, b):
    ra, rb = avg_rank(np.asarray(a, float)), avg_rank(np.asarray(b, float))
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() and rb.std() else float("nan")


def error_rate(score, good):
    """fraction of (good, bad) pairs where the bad one scores higher; ties count half. (wrong, ties, pairs)"""
    g, b = score[good], score[~good]
    if not len(g) or not len(b):
        return None
    diff = g[:, None] - b[None, :]
    return float((diff < 0).sum()), float((diff == 0).sum()), diff.size


rows = []
for name in ("foldbench", "pxm2025"):
    pool, first = se.pool_by_seed(50, str(paths.DATASETS[name].pool_csv))
    for i in sorted(pool):
        pr = [r for r in pool[i] if r[0] in first[i.split("_")[0]]]
        if not pr or max(r[3] for r in pr) <= HI:
            continue
        f = f"{MATS}/{name}_crystal/{i}.npz"
        if not os.path.exists(f):
            print(f"  SKIP {name} {i}: no crystal-frame matrix"); continue
        z = np.load(f)
        idx = {(s, int(k)): j for j, (s, k) in enumerate(zip(z["seed"], z["sample"]))}
        for r in pr:   # the matrix and the pool must describe the same poses
            if (r[0], r[1]) in idx:
                assert abs(z["pair_iptm"][idx[(r[0], r[1])]] - r[3]) < 1e-4, (i, r)
        anchors = [idx[(r[0], r[1])] for r in pr if r[3] > HI and (r[0], r[1]) in idx]
        n_anchor = sum(r[3] > HI for r in pr)
        rest = [r for r in pr if r[3] <= HI]
        base = dict(dataset=name, iface=i, n_pool=len(pr), n_anchor=n_anchor, n_anchor_in_matrix=len(anchors),
                    anchor_frac_acceptable=round(np.mean([r[2] >= A for r in pr if r[3] > HI]), 3), n_rest=len(rest))
        if anchors:   # every pose, anchors included: nearest OTHER anchor
            aset = set(anchors)
            def pall(r):
                if (r[0], r[1]) not in idx:
                    return 0.0
                j = idx[(r[0], r[1])]; others = [a for a in anchors if a != j]
                return float(z["dockq"][j, others].max()) if others else 1.0
            dq_all = [r[2] for r in pr]
            base["rho_iptm_all"] = round(spearman([r[3] for r in pr], dq_all), 3)
            base["rho_pseudo_all"] = round(spearman([pall(r) for r in pr], dq_all), 3)
        if not anchors or len(rest) < 2:
            rows.append({**base, "kind": "no anchors in matrix" if not anchors else "no rest"}); continue
        dq = np.array([r[2] for r in rest]); good = dq >= A
        ip = np.array([r[3] for r in rest])
        ps = np.array([z["dockq"][idx[(r[0], r[1])], anchors].max() if (r[0], r[1]) in idx else 0.0 for r in rest])
        kind = "mixed" if good.any() and not good.all() else ("all good" if good.all() else "all bad")
        out = {**base, "kind": kind, "n_rest_good": int(good.sum()), "random_pick": round(float(good.mean()), 3),
               "iptm_pick_ok": int(dq[np.argmax(ip)] >= A), "pseudo_pick_ok": int(dq[np.argmax(ps)] >= A),
               "iptm_pick_dockq": round(float(dq[np.argmax(ip)]), 3), "pseudo_pick_dockq": round(float(dq[np.argmax(ps)]), 3),
               "oracle_dockq": round(float(dq.max()), 3)}
        for key, s in (("iptm", ip), ("pseudo", ps)):
            e = error_rate(s, good)
            if e:
                out[f"{key}_wrong"], out[f"{key}_ties"], out["pairs"] = e
                out[f"{key}_err"] = round((e[0] + 0.5 * e[1]) / e[2], 3)
        rows.append(out)

os.makedirs(OUT, exist_ok=True)
keys = list(dict.fromkeys(k for r in rows for k in r))
with open(f"{OUT}/anchor_distance_vs_iptm_{HI}.csv", "w") as fh:
    fh.write(",".join(keys) + "\n")
    for r in rows:
        fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

print(f"\nInterfaces with >= 1 pose above ipTM {HI}, by what their remaining poses (ipTM <= {HI}) look like")
print("| dataset | interfaces with an anchor | anchors acceptable (pooled) | mixed rest (good and bad) | rest all good | rest all bad | other |")
print("|---|---|---|---|---|---|---|")
for name in ("foldbench", "pxm2025"):
    R = [r for r in rows if r["dataset"] == name]
    na = sum(r["n_anchor"] for r in R); aa = sum(r["anchor_frac_acceptable"] * r["n_anchor"] for r in R)
    k = lambda x: sum(r["kind"] == x for r in R)
    print(f"| {name} | {len(R)} | {aa:.0f}/{na} ({aa / na:.3f}) | {k('mixed')} | {k('all good')} | {k('all bad')} | "
          f"{len(R) - k('mixed') - k('all good') - k('all bad')} |")

print(f"\nMixed interfaces only: ranking the rest poses by ipTM vs by DockQ to the anchors (crystal DockQ grades)")
print("| dataset | mixed interfaces | (good, bad) pairs | error rate ipTM, pooled | error rate pseudo, pooled | "
      "median per-interface error ipTM | median per-interface error pseudo | pseudo better / worse / tied (interfaces) | "
      "top pick acceptable: ipTM | top pick acceptable: pseudo | random pick (expected) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for name in ("foldbench", "pxm2025"):
    M = [r for r in rows if r["dataset"] == name and r["kind"] == "mixed"]
    if not M:
        print(f"| {name} | 0 |"); continue
    P = sum(r["pairs"] for r in M)
    pooled = lambda key: sum(r[f"{key}_wrong"] + 0.5 * r[f"{key}_ties"] for r in M) / P
    ie, pe = np.array([r["iptm_err"] for r in M]), np.array([r["pseudo_err"] for r in M])
    print(f"| {name} | {len(M)} | {P} | {pooled('iptm'):.3f} | {pooled('pseudo'):.3f} | {np.median(ie):.3f} | "
          f"{np.median(pe):.3f} | {(pe < ie).sum()} / {(pe > ie).sum()} / {(pe == ie).sum()} | "
          f"{sum(r['iptm_pick_ok'] for r in M)}/{len(M)} | {sum(r['pseudo_pick_ok'] for r in M)}/{len(M)} | "
          f"{sum(r['random_pick'] for r in M):.1f}/{len(M)} |")
print(f"\nEvery interface with an anchor, ALL its poses: Spearman(score, crystal DockQ) per interface")
print("| dataset | interfaces | rho defined for both | median rho ipTM | median rho DockQ-to-anchor | "
      "DockQ-to-anchor higher / lower / equal | mean rho ipTM | mean rho DockQ-to-anchor |")
print("|---|---|---|---|---|---|---|---|")
for name in ("foldbench", "pxm2025"):
    R = [r for r in rows if r["dataset"] == name and "rho_iptm_all" in r]
    D = [r for r in R if not (np.isnan(r["rho_iptm_all"]) or np.isnan(r["rho_pseudo_all"]))]
    a, b = np.array([r["rho_iptm_all"] for r in D]), np.array([r["rho_pseudo_all"] for r in D])
    print(f"| {name} | {len(R)} | {len(D)} | {np.median(a):.3f} | {np.median(b):.3f} | {(b > a).sum()} / {(b < a).sum()} / "
          f"{(b == a).sum()} | {a.mean():.3f} | {b.mean():.3f} |")
print(f"-> {OUT}/anchor_distance_vs_iptm_{HI}.csv")
