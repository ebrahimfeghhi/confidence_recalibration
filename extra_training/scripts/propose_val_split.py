#!/usr/bin/env python
"""Propose a 40-cluster post-cutoff validation split, auditing the existing 21 under the new rules.

    ./propose_val_split.py --out ../results/val_split_proposal.csv

RULES, applied identically to retained and newly sampled clusters:
  R1  the representative's token count <= --tok-cap (default 1000). Cache cost grows as N^2:
      z_trunk is [N, N, 256] in bf16, 96% of the npz and incompressible, so one 1,200-token
      structure costs as much as six at 500.
  R2  the cluster withholds <= --post-cap post-cutoff structures from training (default 5). This is
      the real training cost of holding a cluster out; its pre-cutoff structures are withheld too but
      those are the ones Protenix already fits.
  R3  composition target --n-allnew all-new + --n-mixed mixed. "All new" means no structure in the
      cluster predates 2021-09-30, so no relative of it can have been in Protenix's training pool.
  R4  no two clusters whose representatives share a deposition date and token count -- same study,
      different cluster id, which is redundancy the cluster id does not catch.

SELECTION IS BLIND TO PREDICTION SCORES. The representative is the smallest post-cutoff structure in
the cluster by token count; token count is a property of the input, not of any prediction. Sampling
is stratified by token count into one bin per needed cluster, one cluster drawn per bin, under a
fixed seed, so the set spans the size range rather than clustering at the cheap end.

THE EXISTING 21 GET NO PRIVILEGE. They came from phase B's frozen draw (date-stratified, seeded,
pre-folding, no oracle), which makes them admissible candidates -- but phase B never controlled
token count, cluster population or redundancy, so each is re-tested here and retained only on merit.
"""
import argparse, collections, csv, os, random, sys

REPO = "/home/ebrahim/tts_abag/extra_training"
SPLIT = f"{REPO}/data/zenodo_splits/splits_final/abag_split.csv"
CUT = "2021-09-30"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-cap", type=int, default=1000)
    ap.add_argument("--post-cap", type=int, default=5)
    ap.add_argument("--n-allnew", type=int, default=24)
    ap.add_argument("--n-mixed", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=100, help="seed depth, for the cache estimate")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", default=f"{REPO}/results/val_split_proposal.csv")
    a = ap.parse_args()

    train = {r["cluster"] for r in csv.DictReader(open(f"{REPO}/results/train_clusters.csv"))}
    val = {r["cluster"] for r in csv.DictReader(open(f"{REPO}/results/val_clusters.csv"))}
    pool = train | val

    def seqlen(r):
        return (len(r["Hseq"] or "") + len(r["Lseq"] or "")
                + sum(len(s) for s in (r["agresolvedseqs"] or "").split("/") if s))

    pre, post, best = (collections.defaultdict(set), collections.defaultdict(set), {})
    for r in csv.DictReader(open(SPLIT)):
        c = str(r["ab_ag_cluster"])
        if c not in pool:
            continue
        newer = (r["PDBdepo"] or "")[:10] > CUT
        (post if newer else pre)[c].add(r["PDB_ID"])
        if newer:
            t = seqlen(r)
            # 150 tokens is the floor below which a "complex" is a peptide stub
            if t >= 150 and (c not in best or t < best[c][0]):
                best[c] = (t, r["PDB_ID"], (r["PDBdepo"] or "")[:10])

    # REAL token counts where we have them: the proxy underestimates assemblies with repeated
    # chains by up to 2x (9dzf: 1081 proxy vs 2289 real), and that error is 4x in cache bytes.
    real = collections.defaultdict(list)
    for m in csv.DictReader(open(f"{REPO}/val_jobs/manifest.csv")):
        if m["post_cutoff"] == "True":
            real[m["cluster"]].append((int(m["n_tokens"]), m["pdb"], m["deposited"]))
    folded = {c: min(v) for c, v in real.items()}

    gb = lambda t: t * t * 512 / 1024**3 * a.seeds * 1.04     # z_trunk + ~4% other tensors

    cand = {}
    for c, (t, p, dep) in best.items():
        if c in folded:                      # prefer the measured value over the proxy
            t, p, dep = folded[c]
        cand[c] = dict(cluster=c, kind="all new" if not pre[c] else "mixed", pdb=p, tokens=t,
                       deposited=dep, gb=gb(t), post_held=len(post[c]), pre_held=len(pre[c]),
                       folded=c in folded)

    passing = {c: v for c, v in cand.items()
               if v["tokens"] <= a.tok_cap and v["post_held"] <= a.post_cap}

    # R4 within the candidate set: same deposition date and token count means same study
    seen, dropped_r4 = {}, []
    for c, v in sorted(passing.items(), key=lambda kv: (kv[1]["deposited"], kv[1]["tokens"], kv[0])):
        k = (v["deposited"], v["tokens"])
        if k in seen:
            dropped_r4.append((c, seen[k]))
        else:
            seen[k] = c
    for c, _ in dropped_r4:
        passing.pop(c, None)

    rng = random.Random(a.seed)

    def stratified(pool_, kind, n, exclude):
        c = sorted([v for k, v in pool_.items() if k not in exclude and v["kind"] == kind],
                   key=lambda v: v["tokens"])
        if len(c) < n:
            return c
        out = []
        for i in range(n):
            lo = i * len(c) // n
            hi = max(lo + 1, (i + 1) * len(c) // n)
            out.append(rng.choice(c[lo:hi]))
        return out

    retained = [passing[c] for c in sorted(passing, key=int) if passing[c]["folded"]]
    for v in retained:
        v["origin"] = "retained"
    nan = sum(1 for v in retained if v["kind"] == "all new")
    nmx = len(retained) - nan
    ex = {v["cluster"] for v in retained}
    added = (stratified(passing, "all new", max(0, a.n_allnew - nan), ex)
             + stratified(passing, "mixed", max(0, a.n_mixed - nmx), ex))
    for v in added:
        v["origin"] = "sampled"
    sel = retained + added

    rejected = [v for c, v in cand.items() if c not in passing and v["folded"]]
    print(f"eligible after R1 (tok<={a.tok_cap}) and R2 (post<={a.post_cap}): "
          f"{sum(1 for v in passing.values() if v['kind']=='all new')} all-new, "
          f"{sum(1 for v in passing.values() if v['kind']=='mixed')} mixed")
    if dropped_r4:
        print(f"R4 dropped {len(dropped_r4)}: " +
              ", ".join(f"cluster {c} (duplicate of {k})" for c, k in dropped_r4))
    if rejected:
        print("already-folded clusters REJECTED by the rules:")
        for v in sorted(rejected, key=lambda v: -v["tokens"]):
            why = []
            if v["tokens"] > a.tok_cap: why.append(f"R1 {v['tokens']} tok")
            if v["post_held"] > a.post_cap: why.append(f"R2 withholds {v['post_held']}")
            print(f"   cluster {v['cluster']:>5} {v['pdb']} -- {', '.join(why)} "
                  f"({v['gb']:.0f} GB saved)")
    tk = sorted(v["tokens"] for v in sel)
    print()
    print(f"SELECTED {len(sel)} clusters: {sum(1 for v in sel if v['kind']=='all new')} all-new, "
          f"{sum(1 for v in sel if v['kind']=='mixed')} mixed; "
          f"{sum(1 for v in sel if v['origin']=='retained')} retained, "
          f"{sum(1 for v in sel if v['origin']=='sampled')} sampled")
    print(f"   tokens min/median/max {tk[0]}/{tk[len(tk)//2]}/{tk[-1]}")
    print(f"   cache at {a.seeds} seeds: {sum(v['gb'] for v in sel):.0f} GB")
    print(f"   withholds {sum(v['post_held'] for v in sel)} post-cutoff and "
          f"{sum(v['pre_held'] for v in sel)} pre-cutoff structures from training")
    print(f"   already folded: {sum(1 for v in sel if v['folded'])} of {len(sel)}")

    cols = ["cluster", "kind", "origin", "pdb", "tokens", "deposited", "gb", "post_held",
            "pre_held", "folded"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for v in sorted(sel, key=lambda v: (v["kind"], -v["tokens"])):
            w.writerow({**v, "gb": round(v["gb"], 1)})
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
