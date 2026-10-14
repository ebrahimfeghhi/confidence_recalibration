#!/usr/bin/env python
"""PHASE J -- turn job_cifs' --native-gaps report into the committed REFOLD LIST.

WHY A PHASE AND NOT A ONE-LINER. This writes `train_val_splits/refold_internal_omissions*.csv`, which
decides which structures spend GPU time. An artifact that decides compute has to be reproducible from
a file in the repository, not from a heredoc someone typed once.

WHAT IT CONSUMES. `phase_j_job_cifs.py --native-gaps <csv>`, one row per internal native run that a
folded query omits, already classified:

  omission  a real piece of this role that the query dropped because it was unresolved. 11zw's heavy
            query omits native 159-165, SKSTSGG. The fold therefore joined two residues 10.78 A apart
            as neighbours, and no label-side repair can add the tokens back -- only a refold can.
  linker    the synthetic tether of a single-chain construct, which belongs to NEITHER role. 8h3c's
            is GGGGSGGGGSGGGGS, 8dpl's TVSSGTGGSGGGGSGGGGSGGG. Restoring one would splice a linker
            into an antibody domain, so these are EXCLUDED and recorded separately rather than
            dropped silently.

WHAT IT WRITES. One row per structure with every run, its native span and the exact sequence to
restore, so each case is reviewable rather than asserted, plus the fold and disk cost of acting on it.
"""
import argparse
import collections
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FIELDS = ["pdb", "n_roles", "roles", "n_runs", "n_residues", "runs", "restored"]


def summarise(rows):
    """One row per structure, from the per-run rows of a --native-gaps csv."""
    per = collections.defaultdict(list)
    for r in rows:
        per[r["pdb"]].append(r)
    out = []
    for pdb in sorted(per):
        rs = per[pdb]
        out.append(dict(
            pdb=pdb,
            n_roles=len({r["role"] for r in rs}),
            roles=";".join(sorted({r["role"] for r in rs})),
            n_runs=len(rs),
            n_residues=sum(int(r["n_missing"]) for r in rs),
            runs=";".join(f"{r['role']}:{r['native_from']}-{r['native_to']}" for r in rs),
            restored=";".join(r["sequence"] for r in rs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", required=True,
                    help="the --native-gaps csv from phase_j_job_cifs.py")
    ap.add_argument("--out", required=True, help="the refold list to write")
    ap.add_argument("--linkers", default=None,
                    help="where to record the runs EXCLUDED as inter-domain linkers")
    ap.add_argument("--seeds", type=int, default=10,
                    help="seeds per structure, for the cost: 10 for training, 100 for validation")
    ap.add_argument("--mb-per-unit", type=float, default=280.0,
                    help="trunk bytes per fold, for the disk cost. Measure it: "
                         "find <fold>/trunks -name '*.npz' -printf '%s\\n' | awk '{s+=$1;n++} "
                         "END{print s/n/1048576}'")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.gaps)))
    if not rows:
        raise SystemExit(f"STOP: {a.gaps} has no rows. Run phase_j_job_cifs.py --native-gaps first; "
                         "an empty report and a missing one must not look the same.")
    needed = {"pdb", "role", "native_from", "native_to", "n_missing", "sequence", "kind"}
    missing = needed - set(rows[0])
    if missing:
        raise SystemExit(f"STOP: {a.gaps} lacks {sorted(missing)}; it predates the native-position "
                         "mapping and cannot be read as a refold list.")

    om = [r for r in rows if r["kind"] == "omission"]
    lk = [r for r in rows if r["kind"].startswith("linker")]
    out = summarise(om)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(out)
    if a.linkers:
        with open(a.linkers, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(lk)

    kinds = collections.Counter(r["kind"] for r in rows)
    print(f"{a.gaps}: {len(rows)} run(s) over {len({r['pdb'] for r in rows})} structure(s)")
    for k, v in kinds.most_common():
        print(f"   {v:5d}  {k}")
    print(f"\nREFOLD LIST -> {a.out}")
    print(f"  structures : {len(out)}")
    print(f"  runs       : {sum(r['n_runs'] for r in out)}")
    print(f"  residues   : {sum(r['n_residues'] for r in out)}")
    d = collections.Counter(r["n_residues"] for r in out)
    print(f"  residues per structure: {dict(sorted(d.items()))}")
    print(f"\nCOST at {a.seeds} seeds per structure")
    print(f"  folds      : {len(out) * a.seeds}")
    print(f"  new trunks : {len(out) * a.seeds * a.mb_per_unit / 1024:.0f} GB "
          f"(the same again is reclaimable from the superseded units)")
    if a.linkers:
        print(f"\nEXCLUDED as inter-domain linkers -> {a.linkers}: {len(lk)}")
        for r in lk:
            print(f"   {r['pdb']} {r['role']} native {r['native_from']}-{r['native_to']}: "
                  f"{r['sequence']}")


if __name__ == "__main__":
    main()
