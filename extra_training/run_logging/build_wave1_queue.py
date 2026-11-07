#!/usr/bin/env python
"""Build the Wave 1 fold queue, with the first 40-60 structures DELIBERATELY DIVERSIFIED.

WHY THE ORDER MATTERS. The pilot is the first look at whether the new folds carry signal, and the
cheapest possible queue -- smallest structures first -- would make it unrepresentative in exactly
the ways that matter: small structures are mostly single-domain antigens, and the cheapest
candidate of a cluster is not a random member of it. A pilot of the 50 smallest would answer a
question nobody asked.

SO THE PILOT IS ROUND-ROBINED over the axes the expansion is varying:
  cluster category   pre-only / mixed / post-only
  cutoff status      pre-cutoff / post-cutoff
  token quartile     computed over the whole wave-1 candidate set
  antigen cluster    never twice in the pilot until every family has appeared once
After the pilot the remainder is ordered smallest-first, because by then throughput matters more
than representativeness.

Writes a queue of (pdb, seeds, category, cutoff, tokens, ab_cluster, ag_cluster) rows. Nothing is
folded here.
"""
import argparse, collections, csv, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
_src = open(os.path.join(ROOT, "phases", "phase_j0_select_structures.py")).read()
_g = {"__file__": os.path.join(ROOT, "phases", "phase_j0_select_structures.py"), "__name__": "nm"}
exec(compile(_src.split("def main(")[0], "j0", "exec"), _g)
chain_seqs, eligible, job_tokens, ag_key = _g["chain_seqs"], _g["eligible"], _g["job_tokens"], _g["ag_key"]
CUT = "2021-09-30"

ap = argparse.ArgumentParser()
ap.add_argument("--drop-list", required=True, help="degenerate training structures, one pdb per line")
ap.add_argument("--pilot", type=int, default=50)
ap.add_argument("--out", required=True)
a = ap.parse_args()

idx = json.load(open(os.path.join(ROOT, "building_msa/work/pdb_seqs/seq_to_pdb_index.json")))
MR = os.path.join(ROOT, "building_msa/work/mmcif_msa")
tc = {r["cluster"] for r in csv.DictReader(open(os.path.join(ROOT, "train_val_splits/train_clusters.csv")))}
vc = {r["cluster"] for r in csv.DictReader(open(os.path.join(ROOT, "train_val_splits/val_split_proposal_clusters.csv")))}
tfold = set(os.listdir(os.path.join(ROOT, "training/fold/trunks")))
drop = set(open(a.drop_list).read().split())
RETAIN = tfold - drop


def msa_ok(seq):
    """THE FILES, not the directory. Raised in review: os.path.isdir passes on a directory that
    exists but holds no alignment, or holds a zero-byte one left by an interrupted search, and the
    failure would only surface partway through 36 GPU-hours of folding."""
    if seq not in idx:
        return False
    d = os.path.join(MR, idx[seq])
    if not os.path.isdir(d):
        return False
    for n in ("pairing.a3m", "non_pairing.a3m"):
        try:
            if os.path.getsize(os.path.join(d, n)) <= 0:
                return False
        except OSError:
            return False
    return True


def usable(r):
    ok, _ = eligible(r)
    if not ok or job_tokens(r) > 2560:
        return False
    ss = chain_seqs(r)
    return bool(ss) and all(msa_ok(s) for s in ss)


U = collections.defaultdict(lambda: {"pre": 0, "post": 0})
AV = collections.defaultdict(list)
RETCL = collections.defaultdict(set)
for r in csv.DictReader(open(os.path.join(ROOT, "SAbDab2_data/zenodo_splits/splits_final/abag_split.csv"))):
    cl = str(r.get("ab_ag_cluster")); p = r["PDB_ID"][-4:].lower()
    if cl not in tc or cl in vc:
        continue
    k = "post" if (r.get("PDBdepo") or "")[:10] > CUT else "pre"
    if usable(r):
        U[cl][k] += 1
        if p not in tfold:
            # THE INSTANCE IS RECORDED, not just the pdb. A pdb carries several instances -- one
            # per antibody/antigen chain pairing -- and they do NOT all pass the gates: 7pa7 has
            # 10, of which 4 have a protein antigen and the 6 smallest are ion- or ligand-only.
            # Downstream anything that re-derives "the row for this pdb" by taking the cheapest
            # instance picks an ineligible one and reports a false failure. Raised by the
            # preflight doing exactly that.
            AV[cl].append({"p": p, "t": job_tokens(r), "k": k, "ab": str(r.get("ab_cluster")),
                           "ag": ag_key(r.get("agclusters")), "seq": set(chain_seqs(r)),
                           "instance": r["INSTANCE"]})
    if p in RETAIN:
        RETCL[cl].add(p)


def lab(d):
    return "mixed" if (d["pre"] and d["post"]) else ("pre-only" if d["pre"] else "post-only")


# ---- pick per cluster, honouring the targets and the novelty preference
occ_ag = set(); occ_ab = set(); occ_seq = set()
for r in csv.DictReader(open(os.path.join(ROOT, "SAbDab2_data/zenodo_splits/splits_final/abag_split.csv"))):
    if r["PDB_ID"][-4:].lower() in RETAIN:
        occ_ag.add(ag_key(r.get("agclusters"))); occ_ab.add(str(r.get("ab_cluster")))
        for s in chain_seqs(r):
            occ_seq.add(s)

picked = []
for cl, d in sorted(U.items()):
    L = lab(d)
    target = 1 if L == "pre-only" else 2
    k = max(0, target - len(RETCL[cl]))
    if not k:
        continue
    pool = {x["p"]: x for x in AV[cl]}.values()           # dedup by pdb
    # NOVELTY FIRST, then post-cutoff for mixed/post-only, then cheapest.
    def novelty(x):
        return -((x["ag"] not in occ_ag) * 4 + (x["ab"] not in occ_ab) * 2
                 + (not (x["seq"] & occ_seq)) * 1)
    pool = sorted(pool, key=lambda x: (novelty(x), x["k"] != "post" if L != "pre-only" else 0, x["t"]))
    take = list(pool)[:k]
    # when two are taken, prefer the pair that differs most
    if k == 2 and len(pool) > 2:
        cands = list(pool)[:12]
        best = None
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                x, y = cands[i], cands[j]
                sc = (x["ag"] != y["ag"]) * 2 + (x["ab"] != y["ab"]) + (not (x["seq"] & y["seq"]))
                tot = x["t"] + y["t"]
                if best is None or (sc, -tot) > (best[0], -best[1]):
                    best = (sc, tot, [x, y])
        take = best[2]
    for x in take:
        picked.append({"pdb": x["p"], "instance": x["instance"], "seeds": 10, "category": L,
                       "cutoff": x["k"], "tokens": x["t"], "ab_cluster": x["ab"],
                       "ag_cluster": x["ag"], "ab_ag_cluster": cl})
        occ_ag.add(x["ag"]); occ_ab.add(x["ab"]); occ_seq |= x["seq"]

# ---- DIVERSIFIED PILOT: round-robin over (category, cutoff, token quartile), antigen-unique first
tks = sorted(x["tokens"] for x in picked)
q = [tks[int(len(tks) * f)] for f in (0.25, 0.5, 0.75)]
def quart(t):
    return 0 if t <= q[0] else 1 if t <= q[1] else 2 if t <= q[2] else 3
buckets = collections.defaultdict(list)
for x in picked:
    buckets[(x["category"], x["cutoff"], quart(x["tokens"]))].append(x)
for v in buckets.values():
    v.sort(key=lambda x: x["tokens"])
pilot, seen_ag = [], set()
keys = sorted(buckets)
while len(pilot) < min(a.pilot, len(picked)):
    moved = False
    for key in keys:
        if len(pilot) >= a.pilot:
            break
        b = buckets[key]
        # prefer a structure whose antigen family is not yet in the pilot
        pick = next((i for i, x in enumerate(b) if x["ag_cluster"] not in seen_ag), 0 if b else None)
        if pick is None:
            continue
        x = b.pop(pick); seen_ag.add(x["ag_cluster"]); pilot.append(x); moved = True
    if not moved:
        break
rest = sorted((x for v in buckets.values() for x in v), key=lambda x: x["tokens"])
queue = pilot + rest
for i, x in enumerate(queue):
    x["order"] = i
    x["pilot"] = i < len(pilot)

with open(a.out, "w") as f:
    json.dump({"n": len(queue), "pilot_n": len(pilot), "units": sum(x["seeds"] for x in queue),
               "queue": queue}, f, indent=1)
print(f"queue: {len(queue)} structures, {sum(x['seeds'] for x in queue)} units -> {a.out}")
print(f"pilot: {len(pilot)} structures")
for nm, sub in (("PILOT", pilot), ("FULL QUEUE", queue)):
    c = collections.Counter((x["category"], x["cutoff"]) for x in sub)
    tq = collections.Counter(quart(x["tokens"]) for x in sub)
    print(f"\n  {nm}: {len(sub)} structures")
    print(f"    by category/cutoff: {dict(sorted(c.items()))}")
    print(f"    by token quartile:  {dict(sorted(tq.items()))}")
    print(f"    distinct antigen clusters: {len({x['ag_cluster'] for x in sub})}")
    print(f"    distinct ab clusters:      {len({x['ab_cluster'] for x in sub})}")
    print(f"    tokens: min {min(x['tokens'] for x in sub)} median "
          f"{sorted(x['tokens'] for x in sub)[len(sub)//2]} max {max(x['tokens'] for x in sub)}")
