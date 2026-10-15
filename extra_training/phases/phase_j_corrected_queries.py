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
    ap.add_argument("--controls", type=int, default=5,
                    help="how many UNCHANGED queries to add to the batch as the conversion control. "
                         "They must be in the SAME batch: phase D's intermediates are per batch and "
                         "positional, so a separate control batch could not have its environmental "
                         "pass run after UniRef30 is freed -- the full D+E+F equivalence check would "
                         "need another 96 GB download. 0 disables them")
    ap.add_argument("--msa-root", default=f"{ROOT}/building_msa/work/mmcif_msa",
                    help="the existing store, to pick control queries whose MSA is already there")
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

    # THE CONTROLS, CHOSEN TO SPAN THE RANGE rather than taken at random: one query can pass by luck
    # and five similar ones barely better. Shortest, longest, median, and two more by length so a
    # length-dependent failure cannot hide. Each must already have a stored MSA to compare against.
    controls = {}
    if a.controls:
        import glob
        have = []
        for d in sorted(os.listdir(a.msa_root)):
            pa = os.path.join(a.msa_root, d, "pairing.a3m")
            if not os.path.exists(pa):
                continue
            q = []
            with open(pa) as fh:
                for line in fh:
                    if line.startswith(">"):
                        if q:
                            break
                        continue
                    q.append(line.strip())
            seq = "".join(q).replace("-", "").upper()
            if seq and seq not in per:          # UNCHANGED: must not be one of the corrected ones
                have.append((len(seq), seq, d))
        have.sort()
        # SPAN THE RANGE THE CORRECTED QUERIES OCCUPY, not the store's. Chosen from the whole store,
        # the picks were a 5-residue peptide and a 2,710-residue antigen -- neither says anything
        # about 111-234 residue antibody chains, and the 5-residue one would not even share this
        # batch: phase_d_search_uniref.sh routes queries under 20 residues to a separate batch
        # (query_short.fasta), so its intermediates would live elsewhere and the environmental pass
        # for it could not be resumed after UniRef30 is freed -- the exact trap this change exists to
        # avoid. 2,710 was also 9mx1, the one structure protenix-v2 cannot fold at all.
        lo = min(len(x) for x in per)
        hi = max(len(x) for x in per)
        span = [h for h in have if lo <= h[0] <= hi]
        if len(span) >= a.controls:
            have = span
            print(f"control candidates within the corrected range {lo}-{hi} aa: {len(have)}")
        else:
            print(f"WARNING: only {len(span)} candidates in {lo}-{hi} aa; falling back to the whole "
                  f"store, so a control may not be representative")
        if len(have) < a.controls:
            raise SystemExit(f"STOP: only {len(have)} usable control candidates in {a.msa_root}")
        picks = {0, len(have) - 1, len(have) // 2}
        step = max(1, len(have) // (a.controls + 1))
        i = step
        while len(picks) < a.controls and i < len(have):
            picks.add(i); i += step
        for i in sorted(picks)[:a.controls]:
            L, seq, d = have[i]
            controls[seq] = f"CONTROL_{d}"
        print(f"\ncontrol queries (UNCHANGED, searched in the SAME batch so their intermediates are "
              f"retained with the rest): {len(controls)}")
        for seq, nm in sorted(controls.items(), key=lambda t: len(t[0])):
            print(f"   {len(seq):4d} aa  {nm}")

    fa = os.path.join(a.out_dir, "query.fasta")
    with open(fa + ".part", "w") as fh:
        for seq in sorted(per):
            fh.write(f">{idx[seq]}\n{seq}\n")
        for seq, nm in sorted(controls.items()):
            fh.write(f">{nm}\n{seq}\n")       # same batch; see --controls
    os.replace(fa + ".part", fa)
    if controls:
        cj = os.path.join(a.out_dir, "controls.json")
        with open(cj + ".part", "w") as fh:
            json.dump({nm: {"sequence": seq, "stored_msa_dir": nm[len("CONTROL_"):]}
                       for seq, nm in controls.items()}, fh, indent=1, sort_keys=True)
        os.replace(cj + ".part", cj)
        print(f"wrote {cj} -- which stored MSA each control must be compared against")

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
