#!/usr/bin/env python
"""Label every protein entity of every complex as antibody or antigen, from its input sequence alone.

SUMMARY
    Selection needs to know which chains are antibodies, to decide which chain pairs can be
    antibody-antigen interfaces. That used to be read out of the crystal structures' role file, which
    put the crystal on the selection path. This reads only what a user supplies: the sequences in the
    job json.

    An entity is an ANTIBODY if ANARCI numbers an immunoglobulin variable domain in it (heavy, light
    or single-domain), and an ANTIGEN otherwise. Checked once against the crystal roles, as an
    evaluation, on the 123 complexes of the 50-seed pool: 123 of 123 agree.

    RUN IT WITH paths.ANARCI_PYTHON. Under the Protenix venv's Biopython, ANARCI numbers nothing and
    every entity would read as an antigen; main() refuses a complex without an antibody for that reason.

INPUTS
    --pdbs:     comma-separated complexes, or a csv with a pdb column. Required.
    --out:      the csv to write. Required.
    --dataset:  which job jsons to read. Default pxm2025.

OUTPUTS
    <out>: pdb, entity, role, sequence -- one row per protein entity, entity = its index among the
           job json's protein entities (the order Protenix names chains in)

EXAMPLE
    $ANARCI_PYTHON chain_roles.py --pdbs exp/poses.csv --out exp/chain_roles.csv

FUNCTIONS
    entity_roles(): one complex's job json -> [(entity, role, sequence)]
    main():         every complex, one csv
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

ANTIBODY, ANTIGEN = "antibody", "antigen"


def entity_roles(job_json):
    """in: a job json path.  out: [(entity index, role, sequence)] over its protein entities"""
    from anarci import anarci
    obj = json.load(open(job_json))
    obj = obj[0] if isinstance(obj, list) else obj
    seqs = [e["proteinChain"]["sequence"] for e in obj["sequences"] if "proteinChain" in e]
    numbered, _, _ = anarci([(str(i), s) for i, s in enumerate(seqs)], scheme="imgt", output=False)
    return [(i, ANTIBODY if numbered[i] else ANTIGEN, s) for i, s in enumerate(seqs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdbs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", choices=sorted(paths.DATASET_NAMES), default="pxm2025")
    A = ap.parse_args()

    if os.path.exists(A.pdbs):
        pdbs = sorted({r["pdb"] for r in csv.DictReader(open(A.pdbs))})
    else:
        pdbs = sorted(A.pdbs.split(","))
    jobs, _, _ = paths.dataset(A.dataset)
    rows, bad = [], []
    for pdb in pdbs:
        er = entity_roles(f"{jobs}/{pdb}-final-updated.json")
        roles = {r for _, r, _ in er}
        if roles != {ANTIBODY, ANTIGEN}:
            bad.append(f"{pdb}: {sorted(roles)}")
        rows += [(pdb, i, r, s) for i, r, s in er]
    if bad:
        raise SystemExit(f"{len(bad)} complexes lack an antibody or an antigen: {bad[:5]}")
    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    with open(A.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "entity", "role", "sequence"])
        w.writerows(rows)
    n_ab = sum(1 for r in rows if r[2] == ANTIBODY)
    print(f"  {len(pdbs)} complexes, {len(rows)} protein entities: {n_ab} antibody, "
          f"{len(rows) - n_ab} antigen  -> {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
