#!/usr/bin/env python
"""PHASE J -- write the CORRECTED job jsons for the structures on the refold list.

WHAT IT CHANGES. One thing: a role's `proteinChain.sequence` is replaced by the corrected query
`phase_j_job_cifs.py --native-gaps` computed for it, which restores the internal native residues the
original query dropped. Everything else in the json -- the other chains, counts, constraints,
template settings -- is copied through untouched, because a refold has to differ from the original in
exactly the input that was wrong and nothing else.

WHAT IT REFUSES TO TOUCH. Only runs classified `omission`. A `linker` belongs to neither role and an
`unresolved` or `ambiguous` run has not been judged, so a structure carrying one of those in the role
being corrected is SKIPPED with its reason, not partially corrected. That is why 8dpl is absent from
the output: its 115-136 run begins TVSSGTGG, the canonical end of a VH, so where the domain stops and
the tether starts is a residue-level decision nobody has taken.

MATCHED BY SEQUENCE, NOT BY POSITION. The role -> chain assignment lives in the splits row, not in
the json, and the json's chain order is not guaranteed to follow it. So the chain to correct is found
by an EXACT match against the old query string recorded in the gap report. A json with no exact match,
or more than one, is refused -- that means the report and the json disagree about what was folded,
which is not something to resolve by guessing.
"""
import argparse
import collections
import csv
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ACTIONABLE = "omission"
BLOCKING = ("unresolved", "ambiguous", "linker?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", required=True, help="the --native-gaps csv from phase_j_job_cifs.py")
    ap.add_argument("--jobs-dir", default=f"{ROOT}/training/jobs",
                    help="the ORIGINAL job jsons, read only")
    ap.add_argument("--out-dir", default=f"{ROOT}/training/jobs_corrected")
    ap.add_argument("--report", default=None, help="csv of what was corrected and what was skipped")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.gaps)))
    if not rows:
        raise SystemExit(f"STOP: {a.gaps} has no rows")
    if "corrected_query" not in rows[0]:
        raise SystemExit(f"STOP: {a.gaps} has no corrected_query column; it predates "
                         "phase_j_job_cifs.py's corrected-query emission. Regenerate it.")

    by_pdb = collections.defaultdict(list)
    for r in rows:
        by_pdb[r["pdb"]].append(r)

    os.makedirs(a.out_dir, exist_ok=True)
    done, skipped = [], []
    for pdb, rs in sorted(by_pdb.items()):
        blocked = sorted({r["kind"] for r in rs if r["kind"] in BLOCKING})
        if blocked:
            skipped.append(dict(pdb=pdb, why=f"carries {','.join(blocked)} run(s); needs a "
                                             "residue-level decision before it can be corrected"))
            continue
        act = [r for r in rs if r["kind"] == ACTIONABLE and r["corrected_query"]]
        if not act:
            skipped.append(dict(pdb=pdb, why="no actionable omission with a corrected query"))
            continue
        src = os.path.join(a.jobs_dir, f"{pdb}-final-updated.json")
        if not os.path.exists(src):
            skipped.append(dict(pdb=pdb, why=f"no job json at {src}"))
            continue
        doc = json.load(open(src))

        # one corrected sequence per ROLE; a role appears once per run, and every run of a role
        # carries the same corrected_query because it restores all of that role's omissions at once
        per_role = {}
        for r in act:
            prev = per_role.get(r["role"])
            if prev is not None and prev != (r["old_query"], r["corrected_query"]):
                per_role[r["role"]] = None          # disagreement: refuse this role
            else:
                per_role[r["role"]] = (r["old_query"], r["corrected_query"])
        if any(v is None for v in per_role.values()):
            skipped.append(dict(pdb=pdb, why="two runs of one role disagree on the corrected query"))
            continue

        changed, why = 0, None
        for role, (old, new) in sorted(per_role.items()):
            hits = [(ti, si) for ti, task in enumerate(doc) for si, sq in enumerate(task["sequences"])
                    if "proteinChain" in sq
                    and (sq["proteinChain"].get("sequence") or "").strip().upper() == old.upper()]
            if len(hits) != 1:
                why = (f"role {role}: {len(hits)} chain(s) in the json match the old query exactly "
                       "(need 1) -- the report and the json disagree about what was folded")
                break
            ti, si = hits[0]
            doc[ti]["sequences"][si]["proteinChain"]["sequence"] = new
            changed += 1
        if why:
            skipped.append(dict(pdb=pdb, why=why))
            continue

        out = os.path.join(a.out_dir, f"{pdb}-final-updated.json")
        tmp = out + ".part"
        with open(tmp, "w") as fh:
            json.dump(doc, fh, indent=1)
        os.replace(tmp, out)                        # atomic: a reader never sees a half-written job
        done.append(dict(pdb=pdb, roles=";".join(sorted(per_role)), n_roles=changed,
                         residues_restored=sum(int(r["n_missing"]) for r in act),
                         old_lens=";".join(str(len(v[0])) for _k, v in sorted(per_role.items())),
                         new_lens=";".join(str(len(v[1])) for _k, v in sorted(per_role.items()))))

    print(f"corrected job jsons written: {len(done)} -> {a.out_dir}")
    print(f"  roles corrected   : {sum(d['n_roles'] for d in done)}")
    print(f"  residues restored : {sum(d['residues_restored'] for d in done)}")
    print(f"\nSKIPPED: {len(skipped)}")
    for sk in skipped:
        print(f"   {sk['pdb']}: {sk['why']}")
    if a.report:
        with open(a.report, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["pdb", "roles", "n_roles", "residues_restored",
                                               "old_lens", "new_lens"], lineterminator="\n")
            w.writeheader(); w.writerows(done)
        sp = a.report.replace(".csv", "_skipped.csv")
        with open(sp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["pdb", "why"], lineterminator="\n")
            w.writeheader(); w.writerows(skipped)
        print(f"\nwrote {a.report} and {sp}")


if __name__ == "__main__":
    main()
