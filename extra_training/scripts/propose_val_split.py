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
  R4  REPORTED, NOT ENFORCED. Clusters whose representatives share a deposition date and token
      count are flagged for review. An earlier version DROPPED one of each such pair, which was
      wrong: 8t04 (cluster 1833) and 8t05 (cluster 1882) share 2023-05-31 and 448 tokens yet differ
      in heavy chain, light chain AND antigen sequence -- distinct systems, not duplicates. A drop
      now requires identical folded input, which the cluster id already enforces within a cluster.

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
    # a short set silently passing review is the failure this guards: the stratified picker returns
    # fewer than asked when a stratum is empty, and the summary would simply read "37 clusters".
    ap.add_argument("--allow-short", action="store_true",
                    help="permit fewer clusters than requested instead of failing")
    a = ap.parse_args()

    train = {r["cluster"] for r in csv.DictReader(open(f"{REPO}/results/train_clusters.csv"))}
    val = {r["cluster"] for r in csv.DictReader(open(f"{REPO}/results/val_clusters.csv"))}
    pool = train | val

    # PHASE A'S ELIGIBILITY, not "every row in the cluster". phase_a_define_train_set.py:118,262
    # keeps only rows with a polypeptide antigen; without it a hapten-only pairing can be chosen as
    # a cluster's representative, and Protenix would be asked to fold a complex that is not an
    # antibody-protein complex at all.
    POLY = ("PROTEIN", "PEPTIDE")

    def is_poly(r):
        return any(t in POLY for t in str(r["agtypes"]).split("/"))

    # PHASE G'S OWN TOKEN COUNT. phase_g_build_val_jobs.py:154-165 folds Hseq, Lseq and the
    # agEXPECTEDseqs entries whose agtypes is polypeptide. agresolvedseqs -- what this used before --
    # is the resolved subsequence and is SHORTER, so it under-prices the fold; the error is squared
    # in cache bytes.
    def job_tokens(r):
        n = len(r["Hseq"] or "") + len(r["Lseq"] or "")
        types = str(r["agtypes"]).split("/")
        seqs = str(r["agexpectedseqs"]).split("/")
        for i, t in enumerate(types):
            if t in POLY and i < len(seqs) and seqs[i]:
                n += len(seqs[i])
        return n

    pre, post, best = (collections.defaultdict(set), collections.defaultdict(set), {})
    for r in csv.DictReader(open(SPLIT)):
        c = str(r["ab_ag_cluster"])
        if c not in pool:
            continue
        if not is_poly(r):
            continue
        newer = (r["PDBdepo"] or "")[:10] > CUT
        (post if newer else pre)[c].add(r["PDB_ID"])
        if newer:
            t = job_tokens(r)
            # 150 tokens is the floor below which a "complex" is a peptide stub
            if t >= 150 and (c not in best or t < best[c][0]):
                best[c] = (t, r["PDB_ID"], (r["PDBdepo"] or "")[:10], r["INSTANCE"])

    # REAL token counts where we have them: the proxy underestimates assemblies with repeated
    # chains by up to 2x (9dzf: 1081 proxy vs 2289 real), and that error is 4x in cache bytes.
    real = collections.defaultdict(list)
    for m in csv.DictReader(open(f"{REPO}/val_jobs/manifest.csv")):
        if m["post_cutoff"] == "True":
            real[m["cluster"]].append((int(m["n_tokens"]), m["pdb"], m["deposited"],
                                       m["instance"]))
    folded = {c: min(v) for c, v in real.items()}

    gb = lambda t: t * t * 512 / 1024**3 * a.seeds * 1.04     # z_trunk + ~4% other tensors

    cand = {}
    for c, (t, p, dep, inst) in best.items():
        src = "job json"
        if c in folded:                      # the built job's own count beats any estimate
            t, p, dep, inst = folded[c]
        else:
            src = "phase-G formula on abag_split"
        cand[c] = dict(cluster=c, kind="all new" if not pre[c] else "mixed", pdb=p, instance=inst,
                       tokens=t, deposited=dep, gib=gb(t), post_held=len(post[c]),
                       pre_held=len(pre[c]), folded=c in folded, token_source=src)

    passing = {c: v for c, v in cand.items()
               if v["tokens"] <= a.tok_cap and v["post_held"] <= a.post_cap}

    # R4 REPORTED. Shared date and size is evidence of a shared study, which bears on how
    # independent two clusters are -- it is not evidence that the folded inputs are duplicates.
    seen, flagged = {}, []
    for c, v in sorted(passing.items(), key=lambda kv: (kv[1]["deposited"], kv[1]["tokens"], kv[0])):
        k = (v["deposited"], v["tokens"])
        if k in seen:
            flagged.append((c, seen[k], v["deposited"], v["tokens"]))
        else:
            seen[k] = c

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
    if flagged:
        print(f"R4 FLAGGED for review (shared study, nothing dropped):")
        for c, k, d, t in flagged:
            print(f"   clusters {c} and {k}: both {d}, both {t} tokens -- check dependence")
    if rejected:
        print("already-folded clusters REJECTED by the rules:")
        for v in sorted(rejected, key=lambda v: -v["tokens"]):
            why = []
            if v["tokens"] > a.tok_cap: why.append(f"R1 {v['tokens']} tok")
            if v["post_held"] > a.post_cap: why.append(f"R2 withholds {v['post_held']}")
            print(f"   cluster {v['cluster']:>5} {v['pdb']} -- {', '.join(why)} "
                  f"({v['gib']:.0f} GiB saved)")
    tk = sorted(v["tokens"] for v in sel)
    print()
    print(f"SELECTED {len(sel)} clusters: {sum(1 for v in sel if v['kind']=='all new')} all-new, "
          f"{sum(1 for v in sel if v['kind']=='mixed')} mixed; "
          f"{sum(1 for v in sel if v['origin']=='retained')} retained, "
          f"{sum(1 for v in sel if v['origin']=='sampled')} sampled")
    print(f"   tokens min/median/max {tk[0]}/{tk[len(tk)//2]}/{tk[-1]}")
    gib = sum(v["gib"] for v in sel)
    print(f"   cache at {a.seeds} seeds: {gib:.0f} GiB = {gib*1024**3/1e9:.0f} GB decimal")
    nest = sum(1 for v in sel if v["token_source"] != "job json")
    print(f"   token counts: {len(sel)-nest} from built job jsons, {nest} from the phase-G formula "
          f"(must be re-priced from a real job json before folding)")
    print(f"   withholds {sum(v['post_held'] for v in sel)} post-cutoff and "
          f"{sum(v['pre_held'] for v in sel)} pre-cutoff structures from training")
    print(f"   already folded: {sum(1 for v in sel if v['folded'])} of {len(sel)}")

    # FAIL rather than quietly deliver a short set
    want = a.n_allnew + a.n_mixed
    got_an = sum(1 for v in sel if v["kind"] == "all new")
    got_mx = len(sel) - got_an
    if (got_an, got_mx) != (a.n_allnew, a.n_mixed) and not a.allow_short:
        raise SystemExit(f"STOP: asked for {a.n_allnew} all-new + {a.n_mixed} mixed, got "
                         f"{got_an} + {got_mx}. Too few candidates survive R1/R2, or a token "
                         f"stratum is empty. Loosen the caps or pass --allow-short deliberately.")
    cols = ["cluster", "kind", "origin", "pdb", "instance", "tokens", "token_source", "deposited",
            "gib", "post_held", "pre_held", "folded"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for v in sorted(sel, key=lambda v: (v["kind"], -v["tokens"])):
            w.writerow({**v, "gib": round(v["gib"], 1)})
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
