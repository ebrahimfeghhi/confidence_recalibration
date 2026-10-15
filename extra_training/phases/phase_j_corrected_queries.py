#!/usr/bin/env python
"""PHASE J -- write the corrected-query FASTA and the sequence -> directory index both sides share.

WHY AN INDEX AND NOT A CONVENTION. The MSA store's directory names come from the FASTA header
(`phase_f_pair_and_split.py` renames the query header and keys its output on it), while a job json
holds an absolute path to that directory. Those two have to agree, and the first attempt let them
disagree silently: the FASTA was written with headers like `22gi_H_corrected` while the corrected
jobs reused the OLD directory's basename, `c978_8h8q_H_7d8e89`. Phase F would have searched all 155
queries, written them under its own names, and every job would have pointed at a directory that does
not exist -- a whole search wasted, discovered at fold time.

ONE SEQUENCE CAN SERVE SEVERAL ROLES. 155 unique corrected sequences cover 195 roles, so a header
derived from "the first structure that uses this sequence" would leave the other structures' jobs
pointing at a directory named after a different structure. The index is keyed on the SEQUENCE for
that reason, which is also what the MSA store is keyed on -- one search per unique sequence, however
many roles share it.

WRITES
  query.fasta      one record per unique corrected sequence, header = the directory name
  seq_to_dir.json  {corrected sequence: directory name}, read by phase_j_corrected_jobs.py so the
                   paths it writes cannot drift from the names phase F will use
"""
import argparse
import collections
import csv
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def dir_name(seq, roles):
    """A stable directory name for one corrected sequence.

    Carries a readable role for a human and a hash of the SEQUENCE for uniqueness, so two structures
    sharing a corrected sequence share one directory -- which is correct, since they share one search
    -- and two different sequences can never collide however similar their roles look.
    """
    h = hashlib.sha256(seq.encode()).hexdigest()[:8]
    return f"{sorted(roles)[0]}_corrected_{h}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", nargs="+", required=True,
                    help="one or more --native-gaps csvs from phase_j_job_cifs.py")
    ap.add_argument("--out-dir", default=f"{ROOT}/building_msa/queries_corrected")
    a = ap.parse_args()

    per = collections.defaultdict(set)
    for g in a.gaps:
        for r in csv.DictReader(open(g)):
            if r.get("kind") == "omission" and r.get("corrected_query"):
                per[r["corrected_query"].strip().upper()].add(f"{r['pdb']}_{r['role']}")
    if not per:
        raise SystemExit(f"STOP: no omission rows with a corrected query in {a.gaps}")

    os.makedirs(a.out_dir, exist_ok=True)
    idx = {seq: dir_name(seq, roles) for seq, roles in per.items()}
    if len(set(idx.values())) != len(idx):
        raise SystemExit("STOP: two sequences hashed to the same directory name")

    fa = os.path.join(a.out_dir, "query.fasta")
    with open(fa + ".part", "w") as fh:
        for seq in sorted(per):
            fh.write(f">{idx[seq]}\n{seq}\n")
    os.replace(fa + ".part", fa)

    ij = os.path.join(a.out_dir, "seq_to_dir.json")
    with open(ij + ".part", "w") as fh:
        json.dump(idx, fh, indent=1, sort_keys=True)
    os.replace(ij + ".part", ij)

    roles = sum(len(v) for v in per.values())
    L = sorted(len(s) for s in per)
    print(f"unique corrected sequences: {len(per)}  covering {roles} role(s)")
    print(f"  lengths: min {L[0]}, median {L[len(L)//2]}, max {L[-1]}")
    print(f"  sequences serving more than one role: {sum(1 for v in per.values() if len(v) > 1)}")
    print(f"wrote {fa}")
    print(f"wrote {ij} -- phase_j_corrected_jobs.py MUST read this rather than inventing names")


if __name__ == "__main__":
    main()
