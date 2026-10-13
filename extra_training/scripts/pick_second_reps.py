#!/usr/bin/env python
"""Choose the second representative for each validation cluster that has one.

    ./pick_second_reps.py                       # dry run
    ./pick_second_reps.py --out ../results/val_split_second_reps.csv

RULE, fixed in advance and blind to predictions: among a cluster's other eligible post-cutoff
structures, prefer one whose ANTIGEN SEQUENCE DIFFERS from the first representative's, then the
smallest by token count. Antigen identity and token count are properties of the input and the
reference; no DockQ, ipTM or other prediction enters, so the choice cannot be tuned to a result.

Eligibility is phase A's and phase G's, not a looser local rule: the row must carry a polypeptide
antigen (agtypes includes PROTEIN or PEPTIDE) and the job must be at least --min-tokens, counted the
way phase G counts it -- Hseq + Lseq + the agEXPECTEDseqs entries whose agtypes is polypeptide.

The first representative is read from val2_jobs/manifest.csv, which is what was actually built and
folded, not from the proposal csv.

OUTPUT is a cluster,instance,pdb,tokens,deposited,diff_antigen csv. Feed it to phase G as
--instances TOGETHER WITH the 40 first representatives: --instances rebuilds pool.csv for the whole
job set, so passing only the seconds would strip the 40 already-folded structures of their interface
definitions.
"""
import argparse
import collections
import csv
import os

REPO = "/home/ebrahim/tts_abag/extra_training"
POLY = ("PROTEIN", "PEPTIDE")
CUT = "2021-09-30"


def job_tokens(r):
    n = len(r["Hseq"] or "") + len(r["Lseq"] or "")
    types = str(r["agtypes"]).split("/")
    seqs = str(r["agexpectedseqs"]).split("/")
    for i, t in enumerate(types):
        if t in POLY and i < len(seqs) and seqs[i]:
            n += len(seqs[i])
    return n


def antigens(r):
    return {s for s in (r["agexpectedseqs"] or "").split("/") if s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=f"{REPO}/val2_jobs/manifest.csv")
    ap.add_argument("--splits", default=f"{REPO}/data/zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--min-tokens", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    first = {m["cluster"]: m["pdb"] for m in csv.DictReader(open(a.manifest))}
    firstinst = {m["cluster"]: m["instance"] for m in csv.DictReader(open(a.manifest))}

    bypdb = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.splits)):
        c = str(r["ab_ag_cluster"])
        if c not in first or (r["PDBdepo"] or "")[:10] <= CUT:
            continue
        if not any(t in POLY for t in str(r["agtypes"]).split("/")):
            continue
        t = job_tokens(r)
        if t < a.min_tokens:
            continue
        p = r["PDB_ID"][-4:].lower()
        if p not in bypdb[c]:
            bypdb[c][p] = (t, r["INSTANCE"], (r["PDBdepo"] or "")[:10], antigens(r))

    gib = lambda t: t * t * 512 / 1024 ** 3 * a.seeds * 1.04
    sec = lambda n: 0.00182 * n ** 1.57 * a.seeds

    out, tot_gib, tot_s = [], 0.0, 0.0
    for c in sorted(first, key=int):
        fp = first[c]
        fag = bypdb[c].get(fp, (0, "", "", set()))[3]
        others = {p: v for p, v in bypdb[c].items() if p != fp}
        if not others:
            continue
        # different antigen first (False sorts before True), then smallest
        p, (t, inst, dep, ag) = sorted(others.items(), key=lambda kv: (bool(kv[1][3] & fag), kv[1][0]))[0]
        out.append(dict(cluster=c, instance=inst, pdb=p, tokens=t, deposited=dep,
                        diff_antigen="yes" if not (ag & fag) else "no"))
        tot_gib += gib(t)
        tot_s += sec(t)

    nd = sum(1 for r in out if r["diff_antigen"] == "yes")
    print(f"{len(out)} of {len(first)} clusters have a second eligible structure")
    print(f"   {nd} get a DIFFERENT antigen, {len(out)-nd} only the same one")
    print(f"   cache at {a.seeds} seeds: {tot_gib:.0f} GiB")
    print(f"   fold time: {tot_s/3600:.0f} GPU-h -> {tot_s/3/3600:.1f} h on 3 cards")
    for r in out:
        print(f"   cluster {r['cluster']:>5}  first {first[r['cluster']]}  second {r['pdb']}  "
              f"{r['tokens']:4d} tok  diff-antigen {r['diff_antigen']}")

    if a.out:
        # THE 40 FIRST REPRESENTATIVES ARE INCLUDED. phase G --instances rebuilds pool.csv for the
        # whole job set; handing it only the seconds would strip the 40 folded structures of their
        # interface definitions and the fold would score nothing for them.
        rows = [dict(cluster=c, instance=firstinst[c], pdb=first[c], tokens="", deposited="",
                     diff_antigen="", rank="first") for c in sorted(first, key=int)]
        rows += [{**r, "rank": "second"} for r in out]
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["cluster", "instance", "pdb", "rank", "tokens",
                                               "deposited", "diff_antigen"], lineterminator="\n",
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out}: {len(first)} first + {len(out)} second = {len(rows)} instances")


if __name__ == "__main__":
    main()
