#!/usr/bin/env python
"""Build the selection CSV for the 30 new Wave 1 VALIDATION structures.

The instance is recovered by matching each already-built job json's chain sequences against
the candidate instances, NOT by token count. Picking "the row for this pdb" by token count is
the bug that produced 21 false chain-mapping failures on the training side (7pa7 has 10
instances, 4 with a protein antigen and the 6 cheapest ion-only) and recurred twice after that.
A structure with several instances is only resolved when exactly one reproduces the job.
"""
import argparse, csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
_src = open(os.path.join(ROOT, "phases", "phase_j0_select_structures.py")).read()
_g = {"__file__": os.path.join(ROOT, "phases", "phase_j0_select_structures.py"), "__name__": "nm"}
exec(compile(_src.split("def main(")[0], "j0", "exec"), _g)
chain_seqs, eligible, job_tokens = _g["chain_seqs"], _g["eligible"], _g["job_tokens"]

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="validation/fold/queue.tsv")
ap.add_argument("--splits", default="SAbDab2_data/zenodo_splits/splits_final/abag_split.csv")
ap.add_argument("--jobs-dir", default="validation/jobs")
ap.add_argument("--suffix", default="-final-updated.json")
ap.add_argument("--exclude-folded", default="validation/fold/trunks",
                help="structures already folded (the old benchmark) are not part of this selection")
ap.add_argument("--out", default="validation/full_val/val30_structures.csv")
a = ap.parse_args()

# the 30 are the queue's structures minus whatever already has trunks
want = []
for ln in open(a.queue):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 5 and p[1] not in want:
        want.append(p[1])
folded = set(os.listdir(a.exclude_folded)) if os.path.isdir(a.exclude_folded) else set()
want = [p for p in want if p not in folded]
print(f"queue structures not yet folded: {len(want)}")

cand = {}
for r in csv.DictReader(open(a.splits)):
    pdb = r["PDB_ID"][len("pdb_0000"):] if r["PDB_ID"].startswith("pdb_0000") else None
    if pdb in want:
        cand.setdefault(pdb, []).append(r)

out, problems, ties = [], [], []
order = {}
for _p, _rs in cand.items():
    for _r in _rs:
        order.setdefault(_r["INSTANCE"], len(order))
for pdb in want:
    rows = cand.get(pdb, [])
    if not rows:
        problems.append((pdb, "no instance in the splits")); continue
    jj = os.path.join(a.jobs_dir, f"{pdb}{a.suffix}")
    if not os.path.exists(jj):
        problems.append((pdb, f"no job json at {jj}")); continue
    obj = json.load(open(jj))
    job = [c["proteinChain"]["sequence"] for c in obj[0]["sequences"] if "proteinChain" in c]
    hits = [r for r in rows if list(chain_seqs(r)) == job]
    if len(hits) != 1:
        # fall back to a set comparison before giving up: order can differ between builders
        hits = [r for r in rows if sorted(chain_seqs(r)) == sorted(job)]
    if len(hits) > 1:
        # GENUINE TIE: several instances carry identical sequences, i.e. duplicate copies of the
        # same complex in the assembly. The sequences cannot separate them, so break the tie the
        # way build_wave1_jobs.py did when it wrote this very job json -- eligible and MSA-covered
        # first, then cheapest by job_tokens, then splits-file order. Choosing the other copy
        # would be a valid label too; what matters is that the selection reproduces the fold.
        hits.sort(key=lambda r: (0 if eligible(r) else 1, job_tokens(r), order[r["INSTANCE"]]))
        ties.append((pdb, len(hits), hits[0]["INSTANCE"]))
        hits = hits[:1]
    if len(hits) != 1:
        problems.append((pdb, f"{len(hits)} of {len(rows)} instance(s) reproduce the job's "
                              f"{len(job)} chain(s)")); continue
    r = hits[0]
    out.append({"cluster": r.get("ab_ag_cluster", ""), "pdb": pdb, "instance": r["INSTANCE"],
                "ab_cluster": r.get("ab_cluster", ""), "agclusters": r.get("agclusters", ""),
                "tokens": job_tokens(r), "n_chains": len(job), "resolution": r.get("resolution", ""),
                "method": r.get("method", ""), "msa_ok": "1",
                "split": r.get("ab_ag_split", ""), "eligible": "1" if eligible(r) else "0"})

multi = sum(1 for p in want if len(cand.get(p, [])) > 1)
print(f"resolved {len(out)} of {len(want)}; {multi} had several instances to choose between")
if ties:
    print(f"ties broken on identical sequences ({len(ties)}):")
    for pdb, n, inst in ties:
        print(f"   {pdb}: {n} identical instance(s) -> {inst}")
if problems:
    print(f"PROBLEMS ({len(problems)}):")
    for p, why in problems:
        print(f"   {p}: {why}")
if len(out) != len(want):
    raise SystemExit("STOP: not every structure resolved to exactly one instance; "
                     "nothing written. Resolve the above before folding.")

os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print(f"wrote {a.out}")
print(f"  ineligible by phase_j0's own rule: {sum(1 for r in out if r['eligible'] == '0')}")
from collections import Counter
print(f"  splits: {dict(Counter(r['split'] for r in out))}")
