#!/usr/bin/env python
"""Label every protein entity of every complex as antibody, antigen, or neither -- the known input.

SUMMARY
    Selection needs to know which chains are antibodies and which antigens, to decide which chain pairs
    can be interfaces. That is taken as KNOWN, as a user would supply it: the benchmark's own
    annotation of each interface's antibody and antigen chain (pxm2025/chain_roles.csv, with entity ids
    from pxm2025/interfaces.csv). Nothing else about the crystal is read, and nothing downstream reads
    the crystal until evaluation.

    An entity the benchmark does not annotate -- a light chain whose heavy chain carries the scored
    interface, an extra protein in the assembly -- gets role "none" and is left out of interfaces,
    exactly as the benchmark defines them. It is still folded and templated like every other chain.

    MATCHING an annotated entity to the job json's protein entities: by sequence length, which is
    unique within the complex for 310 of the 315 annotated entities; the 5 ties are broken by the
    entity id. Anything else raises -- the entity ids alone are not the job's order (9hvk lists a ligand
    second).

INPUTS
    --pdbs:  a csv with a pdb column, or comma-separated complexes. Required.
    --out:   the csv to write. Required.

OUTPUTS
    <out>: pdb, entity, role, sequence -- one row per protein entity; entity is its index among the job
           json's protein entities, the order Protenix names chains in; role is antibody, antigen or
           none

EXAMPLE
    chain_roles.py --pdbs exp/poses.csv --out exp/chain_roles.csv

FUNCTIONS
    annotated(): the benchmark files -> {pdb: {(entity id, length): role}}
    main():      every complex, one csv
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402


def annotated():
    """in: nothing.  out: {pdb: {(entity id, sequence length): "antibody" | "antigen"}}"""
    ifc = {r["id"]: r for r in csv.DictReader(open(f"{paths.PXM2025_DIR}/interfaces.csv"))}
    out = collections.defaultdict(dict)
    for r in csv.DictReader(open(str(paths.PXM2025_CHAIN_ROLES))):
        i = ifc[r["interface"]]
        ent = {i["chain_id_1"]: (int(i["entity_id_1"]), int(i["seq_length_1"])),
               i["chain_id_2"]: (int(i["entity_id_2"]), int(i["seq_length_2"]))}
        for ch, role in ((r["ab_chain"], "antibody"), (r["ag_chain"], "antigen")):
            if out[r["pdb"]].get(ent[ch], role) != role:
                raise SystemExit(f"{r['pdb']}: entity {ent[ch][0]} annotated as both roles")
            out[r["pdb"]][ent[ch]] = role
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbs", required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    pdbs = (sorted({r["pdb"] for r in csv.DictReader(open(A.pdbs))}) if os.path.exists(A.pdbs)
            else sorted(A.pdbs.split(",")))
    labels, rows, counts = annotated(), [], collections.Counter()
    for pdb in pdbs:
        obj = json.load(open(f"{paths.PXM2025_JOBS}/{pdb}-final-updated.json"))
        obj = obj[0] if isinstance(obj, list) else obj
        prot = [(n, e["proteinChain"]["sequence"]) for n, e in enumerate(obj["sequences"])
                if "proteinChain" in e]
        role = ["none"] * len(prot)
        for (eid, length), r in labels[pdb].items():
            hit = [k for k, (_, s) in enumerate(prot) if len(s) == length]
            if len(hit) > 1:
                hit = [k for k in hit if prot[k][0] == eid - 1]
            if len(hit) != 1:
                raise SystemExit(f"{pdb}: annotated entity {eid} ({length} res) matches {len(hit)} "
                                 f"job entities")
            role[hit[0]] = r
        if "antibody" not in role or "antigen" not in role:
            raise SystemExit(f"{pdb}: no annotated antibody or antigen")
        for k, (_, s) in enumerate(prot):
            rows.append((pdb, k, role[k], s))
            counts[role[k]] += 1
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    with open(A.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "entity", "role", "sequence"])
        w.writerows(rows)
    print(f"  {len(pdbs)} complexes, {len(rows)} protein entities: {counts['antibody']} antibody, "
          f"{counts['antigen']} antigen, {counts['none']} unannotated  -> {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
