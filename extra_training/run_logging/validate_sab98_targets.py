"""Enumerate instance-level targets for the SAbDab2 test set and VALIDATE every one.

WHY IT EXISTS. The first pass built one target per PDB file and preflighted only that
paths resolve and that pool chain names appear in the reference. That is not what the
measurement depends on. Five defects got through it: structures holding several distinct
antibodies (the paper's unit is ONE antibody instance -- "the antibody is {H, L}"), one
heavy chain paired with two light chains, one antibody folded with two antigen copies,
an interface pool built as an unfiltered antibody x antigen cross-product, and one
target (9jie) whose chains were mapped to the wrong crystallographic copy so that NO
pooled pair touches. This reports all of it at once, per target, before anything is built.

THE CHAIN MAPPING IS BY AUTHOR NAME, NOT BY SEQUENCE. map_pool_by_sequence.py picks, for
each job chain, the unused reference subchain with the highest identity. With two
identical Fab copies in the asymmetric unit that choice is arbitrary, and picking copy 1's
heavy chain together with copy 2's antigen yields a target with no contacts -- exactly
9jie. SAbDab2 publishes the author chain id per instance, and gemmi gives the author name
of each subchain, so the mapping is a lookup. SAbDab2 suffixes a chain contributing
several segments (B1/B2 of author B), which is why a failed lookup retries without the
trailing digits.

Report only. Writes nothing into any fold directory.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

import gemmi

csv.field_size_limit(10 ** 9)
CONTACT = 5.0


def subchains_by_auth(path: Path):
    """{auth chain name: [(subchain_id, sequence, [(x,y,z), ...]), ...]}"""
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    out = collections.defaultdict(list)
    for ch in st[0]:
        for sub in ch.subchains():
            res = [r for r in sub]
            if len(res) < 5:
                continue
            seq = gemmi.one_letter_code([r.name for r in res]).upper()
            pts = [(a.pos.x, a.pos.y, a.pos.z) for r in res for a in r]
            out[ch.name].append((sub.subchain_id(), seq, pts))
    return out


def resolve(auth_map, name):
    """A SAbDab2 chain id -> its reference subchains, retrying without a segment suffix."""
    if name in auth_map:
        return auth_map[name]
    bare = re.sub(r"\d+$", "", name)
    return auth_map.get(bare, [])


def touches(a_pts, b_pts, cutoff=CONTACT):
    c2 = cutoff * cutoff
    for x1, y1, z1 in a_pts:
        for x2, y2, z2 in b_pts:
            if (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2 < c2:
                return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--roles", type=Path, required=True, help="roles_sab98.json: one row per antibody instance")
    p.add_argument("--refs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="JSON report + target definitions")
    p.add_argument("--copy-identity", type=float, default=0.95,
                   help="heavy chains at or above this identity are crystallographic copies of one "
                        "antibody; one representative is kept, because keeping both would add "
                        "correlated near-duplicate targets")
    a = p.parse_args()

    roles = json.loads(a.roles.read_text())
    by_pdb = collections.defaultdict(list)
    for r in roles:
        by_pdb[r["pdb"]].append(r)

    targets, failures = [], []
    for pdb in sorted(by_pdb):
        ref = a.refs / f"{pdb}.cif"
        if not ref.exists():
            failures.append({"pdb": pdb, "instance": "*", "why": "no reference cif"})
            continue
        auth = subchains_by_auth(ref)
        kept_h: list[str] = []
        for inst in by_pdb[pdb]:
            hsubs = resolve(auth, inst["H"])
            if not hsubs:
                failures.append({"pdb": pdb, "instance": inst["H"],
                                 "why": f"heavy chain {inst['H']!r} absent from the reference "
                                        f"(auth chains: {sorted(auth)})"})
                continue
            hseq = hsubs[0][1]
            dup = next((s for s in kept_h
                        if len(s) and sum(x == y for x, y in zip(s, hseq)) / min(len(s), len(hseq))
                        >= a.copy_identity), None)
            if dup is not None:
                failures.append({"pdb": pdb, "instance": inst["H"],
                                 "why": "crystallographic copy of an instance already kept"})
                continue
            kept_h.append(hseq)

            ab_names = [inst["H"]] + ([inst["L"]] if inst["L"] else [])
            ag_names = [x for x in re.split(r"[;,|/\s]+", inst["ags"]) if x]
            ab, ag, missing = [], [], []
            for nm, bucket in [(n, ab) for n in ab_names] + [(n, ag) for n in ag_names]:
                subs = resolve(auth, nm)
                if not subs:
                    missing.append(nm)
                bucket.extend(subs)
            if missing:
                failures.append({"pdb": pdb, "instance": inst["H"],
                                 "why": f"chain(s) absent from the reference: {missing}"})
                continue
            pairs = [(x[0], y[0]) for x in ab for y in ag
                     if x[0] != y[0] and touches(x[2], y[2])]
            if not pairs:
                failures.append({"pdb": pdb, "instance": inst["H"],
                                 "why": f"no antibody-antigen chain pair within {CONTACT} A "
                                        f"(ab {[x[0] for x in ab]}, ag {[y[0] for y in ag]})"})
                continue
            targets.append({
                "target": f"{pdb}_{len(kept_h) - 1}",
                "pdb": pdb,
                "H": inst["H"], "L": inst["L"], "ags": ag_names,
                "ab_subchains": [x[0] for x in ab],
                "ag_subchains": [y[0] for y in ag],
                "pool": pairs,
                "n_tokens": sum(len(x[1]) for x in ab) + sum(len(y[1]) for y in ag),
            })

    a.out.write_text(json.dumps({"targets": targets, "failures": failures}, indent=1))
    pdbs = {t["pdb"] for t in targets}
    print(f"instance rows read: {len(roles)} over {len(by_pdb)} structures")
    print(f"VALID targets: {len(targets)} over {len(pdbs)} structures  "
          f"-> {len(targets) * 20} units at 20 seeds")
    print(f"pool rows (contacting only): {sum(len(t['pool']) for t in targets)}")
    drop = collections.Counter(f["why"].split("(")[0].strip() for f in failures)
    print("dropped instances:")
    for why, n in drop.most_common():
        print(f"  {n:>4}  {why}")
    lost = sorted(set(by_pdb) - pdbs)
    print(f"structures with NO valid target: {len(lost)} {lost}")
    tok = sorted(t["n_tokens"] for t in targets)
    if tok:
        print(f"tokens per target: min {tok[0]} median {tok[len(tok)//2]} max {tok[-1]}")
    print(f"report -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
