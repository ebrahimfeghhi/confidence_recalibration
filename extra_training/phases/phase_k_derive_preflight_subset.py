#!/usr/bin/env python
"""Derive an alignment marker for a SUBSET of an already-verified manifest. Seconds, no GPU.

WHY THIS EXISTS. phase_k_alignment_preflight.py checks each structure independently -- it proves
that structure's cached trunk features match the dataset's, atom for atom. So a marker over 491
structures already proves it for any 377 of them. But the marker records only a COUNT and a digest
of the whole unit list, so the trainer, which compares that digest, cannot tell a subset from an
unrelated population and refuses it. Re-running costs 22.4 s/structure -- 2.3 h for 377 -- to
recompute checks that have already passed.

This closes that gap without weakening anything. It refuses unless it can RECOMPUTE the parent's
digest from the parent's own selection file, which is what proves the parent marker describes the
population this tool thinks it does, and unless every structure in the child manifest is in that
parent set. Then it writes the child's marker with the child's own digest and a record of where
the evidence came from.

WHAT IT DOES NOT DO. It does not look at a single trunk or label. Its output is only as good as
the parent's, and it says so in the marker: `derived_from` names the parent file, its digest and
its structure count, so anything reading the marker can see the check was inherited rather than
performed. If the parent is stale -- trunks rebuilt since -- this inherits that staleness.
Re-run the real preflight if the cache has been rebuilt.

  phases/phase_k_derive_preflight_subset.py \\
    --parent training/runs/9mfb4y/alignment_preflight.json \\
    --parent-selection train_val_splits/cache_train_structures.csv \\
    --selection training/solved_cut/structures.csv \\
    --manifest-skip '9mx1,...' --seeds 500000-500009 \\
    --out training/solved_cut/alignment_preflight.json
"""
import argparse
import csv
import hashlib
import json
import os
import time


def population(selection, skip, lo, hi):
    """(sorted pdbs, digest) exactly as phase_k_alignment_preflight and phase_k_train compute it.

    The row filter is theirs too: a row without a numeric `tokens` is not a unit, and the skip
    list is matched lowercased.
    """
    rows = [r for r in csv.DictReader(open(selection))
            if r.get("tokens", "").isdigit() and r["pdb"].lower() not in skip]
    pdbs = sorted({r["pdb"] for r in rows})
    units = sorted(f"{p}/free_{s}.npz" for p in pdbs for s in range(lo, hi + 1))
    return pdbs, hashlib.sha256("\n".join(units).encode()).hexdigest()[:16], len(units)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", required=True, help="an alignment marker with ok+complete")
    ap.add_argument("--parent-selection", required=True,
                    help="the manifest csv that parent marker was built from")
    ap.add_argument("--selection", required=True, help="the subset manifest csv")
    ap.add_argument("--manifest-skip", default="")
    ap.add_argument("--seeds", default="500000-500009")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    skip = {x.strip().lower() for x in a.manifest_skip.split(",") if x.strip()}
    lo, hi = (int(x) for x in a.seeds.split("-"))

    par = json.load(open(a.parent))
    if not par.get("ok"):
        raise SystemExit(f"STOP: {a.parent} does not record ok=true, so it proves nothing.")
    if not par.get("complete"):
        raise SystemExit(f"STOP: {a.parent} does not record complete=true. It checked "
                         f"{par.get('n_structures')} of {par.get('n_manifest_structures', '?')}.")
    if par.get("seeds") != a.seeds:
        raise SystemExit(f"STOP: {a.parent} was built over seeds {par.get('seeds')!r}, not "
                         f"{a.seeds!r}. A unit it never checked cannot be inherited.")

    p_pdbs, p_digest, p_units = population(a.parent_selection, skip, lo, hi)
    # THE PARENT'S DIGEST MUST BE REPRODUCIBLE FROM ITS OWN SELECTION FILE. This is the step that
    # makes the derivation evidence rather than assertion: if it matches, the marker provably
    # describes this exact set of structures, so containment in that set means something.
    if p_digest != par.get("manifest_digest"):
        raise SystemExit(
            f"STOP: recomputing {a.parent_selection} under --manifest-skip/--seeds gives digest "
            f"{p_digest} over {len(p_pdbs)} structure(s), but {a.parent} records "
            f"{par.get('manifest_digest')}. Either the selection file changed since the marker "
            f"was written or it is not the file the marker came from, and in both cases the "
            f"marker's structure set is unknown. Re-run the real preflight.")
    if par.get("n_structures") != len(p_pdbs):
        raise SystemExit(f"STOP: {a.parent} records n_structures={par.get('n_structures')} but "
                         f"{a.parent_selection} yields {len(p_pdbs)} under this skip list.")

    c_pdbs, c_digest, c_units = population(a.selection, skip, lo, hi)
    outside = sorted(set(c_pdbs) - set(p_pdbs))
    if outside:
        raise SystemExit(f"STOP: {len(outside)} structure(s) in {a.selection} were never checked "
                         f"by {a.parent}: {outside[:8]}. Run the real preflight for those.")

    marker = {
        "ok": True,
        "complete": True,
        "n_manifest_structures": len(c_pdbs),
        "n_structures": len(c_pdbs),
        "limit": 0,
        "manifest_digest": c_digest,
        "n_units": c_units,
        "manifest_selection": a.selection,
        "manifest_skip": a.manifest_skip,
        "seeds": a.seeds,
        "checked_seed": par.get("checked_seed"),
        "mismatches": [],
        "missing_trunks": [],
        "schema": par.get("schema", 2),
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seconds": 0.0,
        # PROVENANCE, SO NOTHING READING THIS THINKS THE CHECK HAPPENED HERE.
        "derived": True,
        "derived_from": {
            "parent": os.path.abspath(a.parent),
            "parent_selection": os.path.abspath(a.parent_selection),
            "parent_digest": par.get("manifest_digest"),
            "parent_n_structures": par.get("n_structures"),
            "parent_written": par.get("written"),
            "note": ("every structure here was checked individually by the parent run; this file "
                     "recomputed the parent's digest from its own selection csv to confirm the "
                     "parent describes that population, then confirmed containment. No trunk or "
                     "label was read. Re-run phase_k_alignment_preflight.py if the cache has "
                     "been rebuilt since the parent."),
        },
    }
    with open(a.out, "w") as fh:
        json.dump(marker, fh, indent=1)
    print(f"parent {a.parent}: digest {p_digest} reproduced over {len(p_pdbs)} structure(s)")
    print(f"child  {a.selection}: {len(c_pdbs)} structure(s), {c_units} unit(s), "
          f"digest {c_digest}, all inside the parent")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
